# FlowFake: Liquid Networks for Audio Deepfake Detection 🌊🤖

[![Status](https://img.shields.io/badge/Status-Accepted-success)](#)
[![Venue](https://img.shields.io/badge/ICML%202026-ML4Audio%20Workshop-blue)](#)
[![Task](https://img.shields.io/badge/Task-Audio%20Deepfake%20Detection-orange)](#)
[![Params](https://img.shields.io/badge/Model-~34K%20Parameters-purple)](#)

> **Paper:** *[FlowFake: Liquid Networks for Audio Deepfake Detection](https://arxiv.org/abs/2606.19579)* 📄  
> **Status:** Accepted at the **ICML 2026 ML4Audio Workshop** 🏆

FlowFake is a lightweight audio deepfake detector built with **Liquid Time-Constant (LTC)** dynamics 💧.
The core design goal is cross-dataset robustness: learning **universal temporal speech dynamics** instead of memorizing dataset-specific spectral fingerprints.

## Why FlowFake? 🤔

Many audio deepfake detectors fail under cross-dataset evaluation because they overfit static spectral artifacts.
FlowFake instead models synthetic speech as **multi-timescale trajectory anomalies** in hidden-state dynamics:

- ⚡ frame-level spectral discontinuities
- 🎙️ prosodic inconsistencies
- ⏳ abnormal temporal evolution

Natural speech follows physical articulatory dynamics; synthetic speech often breaks them.
FlowFake explicitly models this mismatch in continuous time.

## Method at a Glance 🔍

- **Backbone:** Conv1D front-end + LTC-ODE temporal module
- **Temporal modeling:** adaptive per-neuron time constants (continuous-time dynamics)
- **Solver:** RK4 with **two ODE unfolds per frame**
- **Capacity:** ~34K parameters
- **Output:** bonafide vs spoof

### Multi-timescale Dynamics 📈

- **Fast neurons** capture short-timescale artifacts (~10–100 ms)
- **Slow neurons** capture long-timescale prosodic cues (~100 ms–2 s)

This division helps preserve generalization under distribution shift.

## Architecture 🏗️

### Processing Pipeline

```text
Audio Waveform 🔊
→ Log-Mel Spectrogram 📊
→ Conv1D Feature Extractor 🧩
→ LTC-ODE Temporal Module 🌊
→ RK4 Solver ⚙️
→ Classifier 🧠
→ Bonafide / Spoof ✅/❌
```

### LTC Cell Components

The LTC cell combines:

- external input current (**I_ext**)
- recurrent synaptic current (**I_syn**)
- leak current (**I_leak**)

### Figure

![FlowFake Architecture Placeholder](https://github.com/GhostRider2023/FlowFake/blob/main/Assets/Architecture.jpg)

## Repository Layout

```text
FlowFake/
├── ASV19/     # ASVspoof LA training + cross-dataset tests
├── FOR/       # Fake-or-Real training + cross-dataset tests
├── ITW/       # In-The-Wild training + cross-dataset tests
└── MLAADv1/   # MLAAD training + cross-dataset tests
```

## Quick Start

### 1) Environment

Use Python 3.9+ and install core dependencies:

```bash
pip install torch torchaudio numpy scipy scikit-learn tqdm
```

### 2) Train on a dataset

Example entry points in this repository:

```bash
python ASV19/ASV_TRAINED.py --la_root /path/to/ASVspoof2019_LA
python MLAADv1/MLAAD_TRAIN.py --mlaad_root /path/to/MLAAD
python FOR/train_FOR.py --root /path/to/FOR
python ITW/train_ITW.py --root /path/to/ITW --csv meta.csv
```

### 3) Run cross-dataset evaluation

Cross-dataset evaluation scripts are provided in each dataset folder (e.g., `test_on_LA.py`, `test_on_FOR.py`, `test_on_MLAAD.py`).

### 4) Launch the Interactive Web UI

An interactive Streamlit forensic detection interface is provided for drag-and-drop audio inspection:

```bash
# On Windows:
run_frontend.bat

# Or directly with Streamlit:
streamlit run frontend/app.py
```

## Citation

If you use FlowFake in your research, please cite:

```bibtex
@inproceedings{flowfake2026,
  title     = {FlowFake: Liquid Networks for Audio Deepfake Detection},
  author    = {Shivaay Dhondiyal, Divyansh Sharma},
  booktitle = {ICML 2026 Workshop on Machine Learning for Audio (ML4Audio)},
  year      = {2026}
}
```
