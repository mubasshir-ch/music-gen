import numpy as np
import torch
from torch.utils.data import Dataset

# this file defines the PianoRollDataset class, 
# which is a PyTorch Dataset that loads piano roll data from .npy files.

class PianoRollDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.from_numpy(x)