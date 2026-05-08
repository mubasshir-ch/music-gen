import numpy as np

for split in ["train", "validation", "test"]:
    data = np.load(f"processed/{split}.npy")
    print(split)
    print("shape:", data.shape)
    print("dtype:", data.dtype)
    print("min:", data.min(), "max:", data.max())
    print("active ratio:", data.mean())
    print()