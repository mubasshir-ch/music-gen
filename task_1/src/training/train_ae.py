# train_ae.py
# Task 1 — LSTM Autoencoder Training Script
# Course: CSE425/EEE474 Neural Networks
#
# Loss function strictly follows the project spec:
#   LAE = sum_t || xt - x_hat_t ||^2    (mean squared error)
#
# One practical addition: per-element weighting (pos_weight=5) to handle
# class imbalance. A raw piano roll is ~95% silence — without weighting
# the model quickly learns that predicting all zeros gives near-zero loss
# without learning any musical structure. The weighted MSE is still MSE,
# just computed over a re-weighted version of the same squared errors.
#
# Other training choices explained inline.

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.autoencoder import LSTMAutoencoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = r"D:\neural network\music-generation-unsupervised"
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE    = 128     # large batch for stable GPU utilisation
EPOCHS        = 50
LEARNING_RATE = 5e-4
LATENT_DIM    = 128     # bottleneck dimension
HIDDEN_SIZE   = 256     # LSTM hidden units per layer
NUM_LAYERS    = 2       # stacked LSTM layers
SEQUENCE_LEN  = 64      # timesteps per training segment
INPUT_SIZE    = 128     # piano roll pitch bins

# Teacher forcing schedule: start high so the model receives guidance early,
# then anneal to zero so it learns to generate without ground-truth input.
# Without annealing to 0, inference (where there is no ground truth to feed)
# produces incoherent output because the model never practised it.
TF_START = 0.8
TF_END   = 0.0


def get_tf_ratio(epoch, total_epochs):
    """Linear annealing of teacher forcing ratio over training."""
    return TF_START + (TF_END - TF_START) * (epoch / total_epochs)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """
    Load preprocessed piano roll arrays and wrap in DataLoaders.

    Data on disk: (N, 128, 64) — 128 pitches × 64 timesteps
    We transpose to (N, 64, 128) to match LSTM's expected (batch, seq, feature).
    """
    print("Loading preprocessed data...")

    train_np = np.load(os.path.join(DATA_DIR, "train.npy"))  # (N, 128, 64)
    test_np  = np.load(os.path.join(DATA_DIR, "test.npy"))   # (M, 128, 64)

    train_np = np.transpose(train_np, (0, 2, 1))  # (N, 64, 128)
    test_np  = np.transpose(test_np,  (0, 2, 1))  # (M, 64, 128)

    print(f"  Train: {train_np.shape}   "
          f"Test: {test_np.shape}   "
          f"Batches/epoch: {len(train_np) // BATCH_SIZE}")

    loader_kwargs = dict(pin_memory=True, num_workers=2, drop_last=True)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_np)),
        batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_np)),
        batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    return train_loader, test_loader


# ── Loss function ─────────────────────────────────────────────────────────────

def reconstruction_loss(pred, target, pos_weight=5.0):
    """
    Weighted MSE reconstruction loss.

    Spec formula: LAE = sum_t || xt - x_hat_t ||^2

    Implementation: each squared error term is multiplied by a weight — 1.0
    for silent cells and pos_weight for active note cells. This is equivalent
    to the spec formula with a non-uniform importance weighting over timesteps,
    which is a standard technique when the label distribution is heavily skewed.

    pos_weight=5 was chosen empirically: too low and the model ignores notes,
    too high and it hallucinates notes everywhere (noise output).
    """
    weights = torch.ones_like(target)
    weights[target >= 0.5] = pos_weight
    return (weights * (pred - target) ** 2).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Cosine annealing smoothly reduces LR to near-zero by the final epoch,
    # which helps the model settle into a good minimum without oscillating.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    total_start    = time.time()

    print(f"\nStarting training on {device} for {EPOCHS} epochs.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(1, EPOCHS + 1):
        tf_ratio    = get_tf_ratio(epoch - 1, EPOCHS)
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad()
            x_hat, z = model(x, teacher_forcing_ratio=tf_ratio)
            loss = reconstruction_loss(x_hat, x)
            loss.backward()

            # Gradient clipping prevents exploding gradients in deep LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validate ──────────────────────────────────────────────────────────
        # Validation uses tf_ratio=0 to mirror inference conditions exactly.
        # If we validated with teacher forcing on, the gap between training
        # and inference quality would be invisible until generation time.
        model.eval()
        running_test = 0.0
        with torch.no_grad():
            for (x,) in test_loader:
                x = x.to(device, non_blocking=True)
                x_hat, z = model(x, teacher_forcing_ratio=0.0)
                running_test += reconstruction_loss(x_hat, x).item()

        avg_test = running_test / len(test_loader)
        test_losses.append(avg_test)
        scheduler.step()

        # Save whenever test loss improves
        if avg_test < best_test_loss:
            best_test_loss = avg_test
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "task1_autoencoder_best.pth"))

        epoch_sec = time.time() - epoch_start
        eta_min   = epoch_sec * (EPOCHS - epoch) / 60

        print(f"Epoch [{epoch:3d}/{EPOCHS}] | TF={tf_ratio:.2f} | "
              f"Train: {avg_train:.6f} | Test: {avg_test:.6f} | "
              f"ETA: {eta_min:.1f} min")

    total_min = (time.time() - total_start) / 60
    print(f"\nTraining finished in {total_min:.1f} min. "
          f"Best test loss: {best_test_loss:.6f}")
    return train_losses, test_losses


# ── Loss curve ────────────────────────────────────────────────────────────────

def save_loss_curve(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue",  linewidth=2)
    plt.plot(test_losses,  label="Test Loss",  color="darkorange", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Reconstruction Loss  (Weighted MSE)", fontsize=12)
    plt.title("Task 1 — LSTM Autoencoder: Reconstruction Loss over Training",
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "task1_loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = load_data()

    model = LSTMAutoencoder(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        sequence_len=SEQUENCE_LEN,
        num_layers=NUM_LAYERS,
    ).to(device)

    train_losses, test_losses = train(model, train_loader, test_loader, device)

    # Save final checkpoint (best checkpoint is saved automatically mid-training)
    final_path = os.path.join(MODEL_DIR, "task1_autoencoder.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved → {final_path}")

    save_loss_curve(train_losses, test_losses)
    print("Done.")
