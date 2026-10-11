import random

all_seeds = [5555, 6789, 10007, 999, 42, 63, 89, 101, 256, 333, 512, 777, 4096, 8080, 9001, 12345, 20000]

selected_seeds = random.sample(all_seeds, 3)

print("Selected seeds:", selected_seeds)