from pathlib import Path
import numpy as np


PROCESSED_DIR = Path("processed")

# this file is used to check the preprocessed data files for sanity before training the VAE model. 
# It prints out the shape, dtype, min/max values, and active ratios of the piano roll data for each split (train/validation/test).

def check_split(split_name):
    path = PROCESSED_DIR / f"{split_name}.npy"

    if not path.exists():
        print(f"{split_name}: missing file {path}")
        return

    data = np.load(path)

    window_active_ratios = data.mean(axis=(1, 2))

    print(split_name)
    print("shape:", data.shape)
    print("dtype:", data.dtype)
    print("min:", data.min(), "max:", data.max())
    print("overall active ratio:", data.mean())
    print("window active ratio min:", window_active_ratios.min())
    print("window active ratio mean:", window_active_ratios.mean())
    print("window active ratio max:", window_active_ratios.max())
    print()


for split in ["train", "validation", "test"]:
    check_split(split)