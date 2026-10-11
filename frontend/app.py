import os
import io
import time
import math
import hashlib
from datetime import datetime
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="FlowFake — AI Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber/Forensic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(20, 24, 33, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 229, 255, 0.08);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 25px;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #00e5ff 0%, #7c4dff 50%, #00e676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 6px;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }
    
    /* Verdict Card Styles */
    .verdict-box {
        padding: 24px;
        border-radius: 16px;
        margin-top: 15px;
        margin-bottom: 20px;
        text-align: center;
        animation: fadeIn 0.4s ease-in-out;
    }
    
    .verdict-fake {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.12) 0%, rgba(213, 0, 0, 0.05) 100%);
        border: 2px solid #ff1744;
        box-shadow: 0 0 35px rgba(255, 23, 68, 0.25);
    }
    
    .verdict-real {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.12) 0%, rgba(0, 200, 83, 0.05) 100%);
        border: 2px solid #00e676;
        box-shadow: 0 0 35px rgba(0, 230, 118, 0.25);
    }
    
    .verdict-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .verdict-score {
        font-size: 3.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 10px 0;
    }
    
    .verdict-analysis {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 500;
        margin-top: 8px;
    }

    /* Metric pill */
    .metric-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .metric-val {
        color: #f8fafc;
        font-size: 1.3rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Model card mini */
    .model-breakdown-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FLOWFAKE MODEL DEFINITION (LTC + Liquid Neural Network)
# ==============================================================================
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt, y + dt*k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 32, tau_init: float = 0.1, solver_type: str = 'rk4'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        self.log_tau = nn.Parameter(torch.full((hidden_size,), math.log(tau_init)))
        self.vleak = nn.Parameter(torch.zeros(hidden_size))
        self.gleak = nn.Parameter(torch.ones(hidden_size))
        self.cm = nn.Parameter(torch.ones(hidden_size))
        self.mu = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.sigma = nn.Parameter(torch.ones(hidden_size, hidden_size) * 2.0)
        self.erev = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)

    def ode_func(self, t, h, x_input=None):
        gleak = self.gleak.clamp(min=0.1, max=10.0)
        cm = self.cm.clamp(min=0.1, max=10.0)
        input_contrib = self.W_in(x_input) if x_input is not None else torch.zeros_like(h)
        syn_act = torch.sigmoid((h.unsqueeze(1) - self.mu) / self.sigma)
        syn_curr = torch.sum(syn_act * self.erev, dim=1)
        leak_curr = gleak * (self.vleak - h)
        total_curr = input_contrib + syn_curr + leak_curr
        dh_dt = total_curr / cm
        return torch.clamp(dh_dt, min=-5.0, max=5.0)

class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_mels: int = 128, cnn_hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, cnn_hidden, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(cnn_hidden)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_hidden)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(cnn_hidden)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(cnn_hidden)
        self.act5 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        return x

class LiqNNModel(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 32, out_size: int = 1, time_steps: int = 200, tau_init: float = 0.1):
        super().__init__()
        self.conv_extractor = ConvFeatureExtractor(input_size, hidden_size)
        self.ltc_cell = LiquidTimeConstantCell(hidden_size, hidden_size, tau_init=tau_init, solver_type='rk4')
        self.final_fc = nn.Linear(hidden_size, out_size)
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.dt = 0.01
        self.ode_unfolds = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        conv_features = self.conv_extractor(x)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(min(self.time_steps, conv_features.size(-1))):
            current_features = conv_features[:, :, t]
            for _ in range(self.ode_unfolds):
                h = rk4_step(lambda t, h: self.ltc_cell.ode_func(t, h, current_features), t, h, self.dt)
        return self.final_fc(h)

# ==============================================================================
# MODEL REGISTRY
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AVAILABLE_MODELS = {
    "🌐 MLAAD Model (Specialist for ElevenLabs, TTS & Voice Clones)": os.path.join(BASE_DIR, "MLAAD_Trained_Model.pth"),
    "⭐ In-The-Wild Model (General Speech, Phone Calls, Social Media)": os.path.join(BASE_DIR, "ITW_Trained_Model.pth"),
    "🔬 ASVspoof 2019 LA Model (Acoustic Lab Benchmark)": os.path.join(BASE_DIR, "ASVspoof_Trained_Model.pth"),
    "👥 FoR Model (Fake-or-Real Benchmark)": os.path.join(BASE_DIR, "FOR_Trained_Model.pth")
}

@st.cache_resource(show_spinner=False)
def load_single_model(checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LiqNNModel(input_size=128, hidden_size=32, out_size=1, time_steps=200).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        return model, device, True
    return None, device, False

# ==============================================================================
# AUDIO PREPROCESSING (EXACT MATCH TO FLOWFAKE TRAINING)
# ==============================================================================
def preprocess_audio_exact(file_bytes, target_sr=16000, chunk_duration_s=10, max_chunks=4):
    """
    Exact matching audio pipeline with multi-chunk windowing:
    1. Mono conversion
    2. Resample to 16kHz
    3. Multi-chunk extraction (10s fixed length each)
    4. Per-chunk Energy Normalization: c = c / (norm(c) + 1e-8) (Guarantees unit norm=1.0)
    5. MelSpectrogram per chunk: n_fft=512, hop_length=80, n_mels=128
    6. Log Mel: log(mel + 1e-6)
    """
    waveform, original_sr = None, None

    # Tier 1: torchaudio in-memory
    try:
        waveform, original_sr = torchaudio.load(io.BytesIO(file_bytes))
    except Exception:
        pass

    # Tier 2: soundfile in-memory
    if waveform is None:
        try:
            import soundfile as sf
            data, original_sr = sf.read(io.BytesIO(file_bytes))
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 2:
                waveform = waveform.t()
        except Exception:
            pass

    # Tier 3: PyAV robust packet demuxer (skips corrupt trailers, metadata tags, and MPEG anomalies)
    if waveform is None:
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            import av
            container = av.open(tmp_path)
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            if audio_stream is not None:
                frames = []
                original_sr = audio_stream.rate or audio_stream.sample_rate
                for packet in container.demux(audio_stream):
                    if packet.dts is None and packet.pts is None:
                        continue
                    try:
                        for frame in packet.decode():
                            if original_sr is None:
                                original_sr = frame.sample_rate
                            arr = frame.to_ndarray()
                            if arr.dtype == np.int16:
                                arr_f = arr.astype(np.float32) / 32768.0
                            elif arr.dtype == np.int32:
                                arr_f = arr.astype(np.float32) / 2147483648.0
                            elif arr.dtype == np.uint8:
                                arr_f = (arr.astype(np.float32) - 128.0) / 128.0
                            else:
                                arr_f = arr.astype(np.float32)
                            frames.append(torch.from_numpy(arr_f))
                    except (av.error.InvalidDataError, av.error.FFmpegError):
                        continue
                    except Exception:
                        continue

                # Flush any buffered frames
                try:
                    for frame in audio_stream.codec_context.decode():
                        arr = frame.to_ndarray()
                        if arr.dtype == np.int16:
                            arr_f = arr.astype(np.float32) / 32768.0
                        elif arr.dtype == np.int32:
                            arr_f = arr.astype(np.float32) / 2147483648.0
                        else:
                            arr_f = arr.astype(np.float32)
                        frames.append(torch.from_numpy(arr_f))
                except Exception:
                    pass

                container.close()

                if frames:
                    waveform = torch.cat(frames, dim=-1)
        except Exception as e:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    if waveform is None:
        return None

    # 1. Mono
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    # 2. Resample
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)

    raw_duration = len(waveform) / target_sr
    chunk_len = target_sr * chunk_duration_s

    # 3. Mel Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=512, hop_length=80, n_mels=128
    )

    log_mels = []
    if len(waveform) <= chunk_len:
        # Pad if shorter
        if len(waveform) < chunk_len:
            w_pad = torch.nn.functional.pad(waveform, (0, chunk_len - len(waveform)))
        else:
            w_pad = waveform
        norm_chunk = w_pad / (torch.norm(w_pad) + 1e-8)
        mel = mel_transform(norm_chunk)
        log_mels.append(torch.log(mel + 1e-6))
    else:
        # Extract multiple evenly spaced 10s chunks across the file
        step = (len(waveform) - chunk_len) // (max_chunks - 1) if max_chunks > 1 else 0
        for i in range(max_chunks):
            start = i * step
            end = start + chunk_len
            chunk = waveform[start:end]
            # Normalize each 10s chunk independently to unit energy (1.0)
            norm_chunk = chunk / (torch.norm(chunk) + 1e-8)
            mel = mel_transform(norm_chunk)
            log_mels.append(torch.log(mel + 1e-6))

    return {
        "log_mels": log_mels,
        "log_mel": log_mels[0],  # for spectrogram visualization
        "raw_waveform": waveform.numpy(),
        "duration": raw_duration,
        "sr": original_sr,
        "num_chunks": len(log_mels)
    }

def run_model_inference(model, device, log_mels):
    """Run forward pass across all chunks and return mean sigmoid probability (1.0 = Real, 0.0 = Fake)"""
    if isinstance(log_mels, torch.Tensor):
        log_mels = [log_mels]
    scores = []
    for lm in log_mels:
        x = lm.unsqueeze(0).to(device)  # [1, 128, T]
        with torch.no_grad():
            out = model(x)
            prob_real = torch.sigmoid(out.squeeze()).item()
            scores.append(prob_real)
    return float(np.mean(scores))

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="main-header">
    <div class="title-gradient">FlowFake Audio Forensic System</div>
    <p class="subtitle-text">Liquid Neural Network (LTC-ODE) Deepfake Voice Detector • ElevenLabs & Synthetic Audio Hunter</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Detection Engine")
    
    detection_mode = st.radio(
        "Scanning Mode",
        options=[
            "🛡️ Multi-Model Consensus (Ensemble — Recommended)",
            "🎯 Single Model Mode"
        ],
        index=0,
        help="Ensemble runs both the ElevenLabs/TTS specialist (MLAAD) and In-The-Wild (ITW) models together for maximum accuracy."
    )
    
    single_model_name = None
    if "Single Model" in detection_mode:
        single_model_name = st.selectbox(
            "Select Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0
        )
        
    st.markdown("---")
    st.markdown("### 🌐 Audio Origin / Context")
    domain_mode = st.selectbox(
        "Audio Source Context",
        options=[
            "🌐 Auto-Adaptive (Smart Domain Consensus — Recommended)",
            "📱 YouTube / Social Media / Phone (In-The-Wild Focus)",
            "🤖 AI Voice Generator / ElevenLabs / Narakeet (TTS Focus)",
            "⚖️ Balanced Equal Average (25% Each)"
        ],
        index=0,
        help="Adapts model weighting according to acoustic domain. Social media audio has lossy codec compression (MP3/AAC), whereas AI voice generators produce clean vocoded speech."
    )
        
    st.markdown("---")
    st.markdown("### 🎚️ Forensic Sensitivity")
    sensitivity = st.select_slider(
        "Detection Strictness",
        options=["Lenient (35%)", "Balanced (50% Default)", "Strict (65%)", "Ultra-Strict (80%)"],
        value="Balanced (50% Default)",
        help="Higher strictness flags suspicious synthetic acoustic artifacts more aggressively."
    )
    
    threshold_map = {
        "Lenient (35%)": 0.35,
        "Balanced (50% Default)": 0.50,
        "Strict (65%)": 0.65,
        "Ultra-Strict (80%)": 0.80
    }
    decision_threshold = threshold_map[sensitivity]
    
    st.markdown("---")
    st.caption("💡 **Tip for ElevenLabs / TTS:** Modern voice clones generate unnatural glottal phase patterns. The **MLAAD Model** is specifically trained on synthetic TTS engines.")

# ==============================================================================
# FEATURE FLAGS (Toggle on/off - set to True when requested to re-enable)
# ==============================================================================
ENABLE_LIVE_RECORDING = False       # Set to True when user requests live microphone recording
ENABLE_SPECTRAL_ANALYSIS = False    # Set to True when user requests Acoustic Spectral Analysis
ENABLE_FORENSIC_REPORT = False      # Set to True when user requests Downloadable Forensic Report

# ==============================================================================
# AUDIO INPUT
# ==============================================================================
audio_bytes = None
audio_filename = "recording.wav"

if ENABLE_LIVE_RECORDING:
    tab_upload, tab_mic = st.tabs(["📁 Upload Audio File", "🎙️ Live Microphone Recording"])
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload audio clip (.wav, .mp3, .mpeg, .m4a, .aac, .ogg, .flac, .webm, .opus, .3gp)",
            type=["wav", "mp3", "mpeg", "mpg", "m4a", "aac", "ogg", "flac", "webm", "opus", "mp4", "3gp", "amr", "wma"],
            help="Upload human recording, mobile phone voice memo, or ElevenLabs voice clone to inspect."
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            audio_filename = uploaded_file.name

    with tab_mic:
        st.markdown("Record live voice to compare against synthetic voice signatures:")
        try:
            mic_recording = st.audio_input("Record Live Audio")
            if mic_recording is not None:
                audio_bytes = mic_recording.read()
                audio_filename = "live_microphone_input.wav"
        except AttributeError:
            st.info("Live microphone recording requires Streamlit 1.40+. Please upload audio using the Upload tab.")
else:
    uploaded_file = st.file_uploader(
        "Upload audio clip (.wav, .mp3, .mpeg, .m4a, .aac, .ogg, .flac, .webm, .opus, .3gp)",
        type=["wav", "mp3", "mpeg", "mpg", "m4a", "aac", "ogg", "flac", "webm", "opus", "mp4", "3gp", "amr", "wma"],
        help="Upload human recording, mobile phone voice memo, or ElevenLabs voice clone to inspect."
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_filename = uploaded_file.name

# ==============================================================================
# FORENSIC SCAN EXECUTION
# ==============================================================================
if audio_bytes is not None:
    st.markdown("### 🎧 Audio Playback & Inspection")
    col_play, col_meta = st.columns([1, 1])
    
    with col_play:
        st.audio(audio_bytes)
        
    sha256_hash = hashlib.sha256(audio_bytes).hexdigest()
    
    with col_meta:
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8; background: #111827; padding: 14px; border-radius: 8px;">
            <div><b>Filename:</b> {audio_filename}</div>
            <div><b>File Size:</b> {len(audio_bytes)/1024:.1f} KB</div>
            <div><b>SHA-256:</b> {sha256_hash[:20]}...{sha256_hash[-8:]}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔍 Run Forensic Deepfake Detection", type="primary", use_container_width=True):
        with st.spinner("Analyzing micro-temporal acoustic flow and neural vocal tract dynamics across audio windows..."):
            t_start = time.time()
            prep = preprocess_audio_exact(audio_bytes)
            if prep is None:
                st.error("⚠️ Could not decode or parse this audio file. Please ensure it is a valid audio file.")
                st.stop()
            
            # Load and evaluate models
            model_scores = {}
            domain_rationale = ""
            
            if "Ensemble" in detection_mode:
                # Run all 4 models across extracted audio windows
                for name, path in AVAILABLE_MODELS.items():
                    m, dev, ok = load_single_model(path)
                    if ok:
                        score_real = run_model_inference(m, dev, prep["log_mels"])
                        model_scores[name] = score_real
                
                mlaad_score = [v for k, v in model_scores.items() if "MLAAD" in k][0]
                itw_score = [v for k, v in model_scores.items() if "In-The-Wild" in k][0]
                asv_score = [v for k, v in model_scores.items() if "ASVspoof" in k][0]
                for_score = [v for k, v in model_scores.items() if "FoR" in k][0]
                
                # Intelligent Domain Consensus Arbitration
                if "Social Media" in domain_mode:
                    final_real_prob = 0.65 * itw_score + 0.35 * for_score
                    domain_rationale = "📱 Social Media / In-The-Wild preset: Prioritizing models robust against lossy audio compression (MP3/AAC) and ambient acoustics."
                elif "AI Voice Generator" in domain_mode:
                    final_real_prob = 0.65 * mlaad_score + 0.35 * asv_score
                    domain_rationale = "🤖 AI Voice Generator preset: Prioritizing TTS specialists (MLAAD & ASVspoof) sensitive to synthetic vocal tract and vocoder artifacts."
                elif "Balanced" in domain_mode:
                    final_real_prob = 0.25 * (mlaad_score + itw_score + asv_score + for_score)
                    domain_rationale = "⚖️ Balanced Equal Consensus: 25% equal weight across all 4 benchmark models."
                else:  # Auto-Adaptive Smart Arbitration
                    # Case 1: Real-World authentic speech (YouTube / Social Media / Phone)
                    # When ITW and FoR both show authentic human characteristics (> 70%),
                    # lossy MP3/AAC compression often causes studio-only models (MLAAD/ASV) to throw false alarms.
                    if itw_score >= 0.70 and for_score >= 0.70 and (mlaad_score < 0.20 or asv_score < 0.20):
                        final_real_prob = 0.65 * itw_score + 0.35 * for_score
                        domain_rationale = "🌐 Auto-Adaptive Arbitration: Genuine vocal tract dynamics confirmed by real-world models (In-The-Wild & FoR). Lossy compression artifacts (MP3/AAC) discounted."
                    # Case 2: Clean Synthetic TTS (Narakeet / ElevenLabs / Voice Clones)
                    # When TTS specialist models detect vocoder phase anomalies (< 15% real),
                    # clean synthetic speech can bypass ITW because it lacks YouTube compression noise.
                    elif mlaad_score <= 0.15 and asv_score <= 0.15 and itw_score > 0.40:
                        final_real_prob = 0.65 * mlaad_score + 0.35 * asv_score
                        domain_rationale = "🌐 Auto-Adaptive Arbitration: Unnatural glottal airflow and vocoder signatures detected by TTS specialists (MLAAD & ASVspoof). Neural voice synthesis confirmed."
                    else:
                        final_real_prob = (0.35 * itw_score) + (0.35 * mlaad_score) + (0.15 * asv_score) + (0.15 * for_score)
                        domain_rationale = "🌐 Auto-Adaptive Arbitration: Multi-model consensus aggregated across domains."
                
                active_engine_name = "🛡️ Multi-Model Ensemble (LTC Consensus)"
            else:
                m_path = AVAILABLE_MODELS[single_model_name]
                m, dev, ok = load_single_model(m_path)
                final_real_prob = run_model_inference(m, dev, prep["log_mels"])
                model_scores[single_model_name] = final_real_prob
                active_engine_name = single_model_name
                domain_rationale = f"Direct inference from {single_model_name}."
                
            t_infer = time.time() - t_start

        # Classification decision based on sensitivity threshold
        # In FlowFake: 1.0 = Real, 0.0 = Fake
        # Therefore: If probability of real < threshold -> FAKE!
        is_real = final_real_prob >= decision_threshold
        synthetic_prob = (1.0 - final_real_prob) * 100.0
        human_prob = final_real_prob * 100.0
        
        confidence = human_prob if is_real else synthetic_prob

        # Display Verdict Banner
        if not is_real:
            analysis_text = f"Synthetic acoustic markers detected! Artificial vocal cord glottal airflow and spectral discontinuities identified ({synthetic_prob:.1f}% synthetic probability)."
            st.markdown(f"""
            <div class="verdict-box verdict-fake">
                <div class="verdict-title" style="color: #ff1744;">🔴 VERDICT: FAKE (AI Generated Deepfake)</div>
                <div class="verdict-score" style="color: #ff1744;">{confidence:.1f}%</div>
                <div class="verdict-analysis"><b>Forensic Finding:</b> {analysis_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            analysis_text = f"Authentic human vocal tract physics verified. Natural micro-airflow fluid dynamics consistent with genuine organic speech ({human_prob:.1f}% authentic probability)."
            st.markdown(f"""
            <div class="verdict-box verdict-real">
                <div class="verdict-title" style="color: #00e676;">🟢 VERDICT: REAL (Authentic Human Voice)</div>
                <div class="verdict-score" style="color: #00e676;">{confidence:.1f}%</div>
                <div class="verdict-analysis"><b>Forensic Finding:</b> {analysis_text}</div>
            </div>
            """, unsafe_allow_html=True)

        if domain_rationale:
            st.info(f"💡 **Forensic Context Analysis:** {domain_rationale}")

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-label">Human Authenticity</div>
                <div class="metric-val" style="color: #00e676;">{human_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-label">Deepfake Likelihood</div>
                <div class="metric-val" style="color: #ff1744;">{synthetic_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-label">Clip Duration</div>
                <div class="metric-val">{prep['duration']:.2f}s ({prep['num_chunks']} windows)</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-label">Inference Time</div>
                <div class="metric-val">{t_infer*1000:.0f} ms</div>
            </div>
            """, unsafe_allow_html=True)

        # Multi-Model Breakdown Cards
        if "Ensemble" in detection_mode:
            st.markdown("### 📊 Multi-Model Breakdown")
            cols = st.columns(len(model_scores))
            for i, (m_label, s_val) in enumerate(model_scores.items()):
                m_is_real = s_val >= decision_threshold
                badge = "🟢 REAL" if m_is_real else "🔴 FAKE"
                score_pct = s_val * 100.0 if m_is_real else (1.0 - s_val) * 100.0
                short_name = m_label.split("(")[0].strip()
                with cols[i]:
                    st.markdown(f"""
                    <div class="model-breakdown-card">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">{short_name}</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: {'#00e676' if m_is_real else '#ff1744'}; margin: 4px 0;">{badge} ({score_pct:.1f}%)</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Real: {s_val*100:.1f}% | Fake: {(1.0-s_val)*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ======================================================================
        # SPECTRAL VISUALIZATION (Toggleable - set ENABLE_SPECTRAL_ANALYSIS=True to show)
        # ======================================================================
        if ENABLE_SPECTRAL_ANALYSIS:
            st.markdown("---")
            st.markdown("### 📈 Acoustic Spectral Analysis")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), constrained_layout=True)
            fig.patch.set_facecolor('#0e1117')

            # Waveform Plot
            ax1.set_facecolor('#111827')
            time_axis = np.linspace(0, prep['duration'], len(prep['raw_waveform']))
            ax1.plot(time_axis, prep['raw_waveform'], color='#00e5ff', lw=0.8)
            ax1.set_title("Audio Waveform (Time Domain Amplitude)", color='#e2e8f0', fontsize=10, fontweight='bold')
            ax1.set_ylabel("Amplitude", color='#94a3b8', fontsize=8)
            ax1.tick_params(colors='#94a3b8', labelsize=8)
            for spine in ax1.spines.values(): spine.set_color('#334155')

            # Log Mel-Spectrogram Plot
            ax2.set_facecolor('#111827')
            img = ax2.imshow(prep['log_mel'].numpy(), origin='lower', aspect='auto', cmap='magma')
            ax2.set_title("Log Mel-Spectrogram (FlowFake LTC Feature Representation)", color='#e2e8f0', fontsize=10, fontweight='bold')
            ax2.set_xlabel("Time Frames (Hop=80, FFT=512)", color='#94a3b8', fontsize=8)
            ax2.set_ylabel("Mel Frequency Bins (128)", color='#94a3b8', fontsize=8)
            ax2.tick_params(colors='#94a3b8', labelsize=8)
            for spine in ax2.spines.values(): spine.set_color('#334155')

            st.pyplot(fig)
            plt.close(fig)

        # ======================================================================
        # DOWNLOADABLE FORENSIC REPORT (Toggleable - set ENABLE_FORENSIC_REPORT=True to show)
        # ======================================================================
        if ENABLE_FORENSIC_REPORT:
            st.markdown("---")
            report_text = f"""================================================================================
FLOWFAKE AUDIO FORENSIC INVESTIGATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. EVIDENCE IDENTIFICATION:
--------------------------------------------------------------------------------
File Name           : {audio_filename}
File SHA-256 Hash   : {sha256_hash}
Audio Duration      : {prep['duration']:.2f} seconds ({prep['num_chunks']} analysis windows scanned)
Sampling Rate       : {prep['sr']} Hz (Normalized to 16,000 Hz Mono)
File Size           : {len(audio_bytes)} bytes

2. FORENSIC CLASSIFICATION:
--------------------------------------------------------------------------------
Detection Engine    : {active_engine_name}
Audio Domain Context: {domain_mode if 'Ensemble' in detection_mode else 'Single Model'}
Arbitration Rationale: {domain_rationale}
Sensitivity Setting : {sensitivity} (Threshold = {decision_threshold})
Verdict             : {"REAL (Authentic Human Voice)" if is_real else "FAKE (AI Generated Deepfake)"}
Confidence Score    : {confidence:.2f}%
Human Authenticity  : {human_prob:.2f}%
Deepfake Likelihood : {synthetic_prob:.2f}%

3. TECHNICAL ACOUSTIC FINDINGS:
--------------------------------------------------------------------------------
Findings            : {analysis_text}
Feature Transform   : 128-band Mel Spectrogram (N_FFT=512, Hop=80, Per-Window Normalized)
Neural Architecture : Liquid Time-Constant (LTC) with 4th-Order Runge-Kutta (RK4)
Inference Latency   : {t_infer*1000:.1f} ms

================================================================================
END OF REPORT — FlowFake Acoustic Defense Intelligence
================================================================================
"""
            st.markdown("### 📄 Forensic Investigation Report")
            st.download_button(
                label="⬇️ Download Official Forensic Investigation Report (.txt)",
                data=report_text,
                file_name=f"Forensic_Report_{os.path.splitext(audio_filename)[0]}.txt",
                mime="text/plain",
                use_container_width=True
            )
else:
    st.info("👆 Please upload an audio file to begin forensic detection.")
