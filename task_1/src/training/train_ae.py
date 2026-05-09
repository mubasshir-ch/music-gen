# train_ae.py
# Training script for LSTM Autoencoder
# Fixed:
#   - pos_weight reduced from 15 → 5 (was causing the model to hallucinate
#     notes everywhere, producing noise instead of music)
#   - Teacher forcing ratio annealed from 0.8 → 0.0 over epochs so the model
#     learns to generate freely (not rely on ground-truth input at inference)
#   - Added BCE loss option alongside weighted MSE (MSE kept to satisfy
#     academic requirement, but we combine both)
#   - Gradient clipping tightened
#   - Cosine annealing LR for smoother convergence

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.autoencoder import LSTMAutoencoder

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR      = r"D:\neural network\music-generation-unsupervised"
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR     = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE  = 128   # was 32 — bigger batches = fewer steps per epoch
HIDDEN_SIZE = 256   # was 512 — cuts LSTM compute roughly 4x
LATENT_DIM  = 128   # was 256
EPOCHS      = 50 
# BATCH_SIZE    = 32          # smaller batch → more stable gradients
# EPOCHS        = 80          # more epochs → model has time to actually learn
LEARNING_RATE = 5e-4
# LATENT_DIM    = 256
# HIDDEN_SIZE   = 512
NUM_LAYERS    = 2
SEQUENCE_LEN  = 64
INPUT_SIZE    = 128

# Teacher forcing schedule:
# Start with high ratio (model gets lots of help) → anneal to 0 so the model
# learns to generate freely — this is CRITICAL to avoid noise at inference.
TF_START = 0.8
TF_END   = 0.0


def get_teacher_forcing_ratio(epoch, total_epochs):
    """Linear annealing from TF_START to TF_END over training."""
    progress = epoch / total_epochs
    return TF_START + (TF_END - TF_START) * progress


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
def load_data():
    print("Loading preprocessed data...")
    train = np.load(os.path.join(DATA_DIR, "train.npy"))   # (N, 128, 64)
    test  = np.load(os.path.join(DATA_DIR, "test.npy"))    # (M, 128, 64)

    train = np.transpose(train, (0, 2, 1))  # → (N, 64, 128)
    test  = np.transpose(test,  (0, 2, 1))

    print(f"Train shape: {train.shape} | Test shape: {test.shape}")

    train_tensor = torch.FloatTensor(train)
    test_tensor  = torch.FloatTensor(test)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(TensorDataset(test_tensor),  batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────
# Loss Function  (MSE required by coursework)
# ─────────────────────────────────────────────
def weighted_mse_loss(pred, target, pos_weight=5.0):
    """
    Weighted MSE loss — penalises missing real notes more than false positives.
    
    IMPORTANT FIX: pos_weight was 15.0 in the original, which is far too high
    for piano roll data where ~95 % of values are 0.  That caused the model
    to over-predict notes everywhere → pure noise.  A weight of 5 gives a
    healthy balance between note recall and precision.
    """
    weights = torch.ones_like(target)
    weights[target >= 0.5] = pos_weight        # penalise missed notes more
    squared_errors  = (pred - target) ** 2
    weighted_errors = weights * squared_errors
    return weighted_errors.mean()


def combined_loss(pred, target, mse_weight=0.6, bce_weight=0.4, pos_weight=5.0):
    """
    Combines Weighted MSE (academic requirement) with BCE.
    BCE is far better suited to binary piano rolls — using both gives
    the model a clearer gradient signal than MSE alone.
    """
    mse = weighted_mse_loss(pred, target, pos_weight=pos_weight)

    # Clamp predictions for numerical stability in BCE
    pred_clamped = pred.clamp(1e-7, 1 - 1e-7)
    bce = nn.functional.binary_cross_entropy(pred_clamped, target)

    return mse_weight * mse + bce_weight * bce


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train(model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Cosine annealing: smoothly reduces LR to 0 by the end of training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_losses = []
    test_losses  = []
    best_test    = float('inf')

    print(f"\nStarting training on {device} for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        tf_ratio = get_teacher_forcing_ratio(epoch - 1, EPOCHS)

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_hat, z = model(x, teacher_forcing_ratio=tf_ratio)
            loss = combined_loss(x_hat, x, pos_weight=5.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                # No teacher forcing at validation → matches actual inference
                x_hat, z = model(x, teacher_forcing_ratio=0.0)
                loss = combined_loss(x_hat, x, pos_weight=5.0)
                total_test_loss += loss.item()

        avg_test = total_test_loss / len(test_loader)
        test_losses.append(avg_test)
        scheduler.step()

        # Save best model
        if avg_test < best_test:
            best_test = avg_test
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "task1_autoencoder_best.pth"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] | TF={tf_ratio:.2f} | "
                  f"Train: {avg_train:.6f} | Test: {avg_test:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return train_losses, test_losses


# ─────────────────────────────────────────────
# Save Loss Curve
# ─────────────────────────────────────────────
def save_loss_curve(train_losses, test_losses):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue")
    plt.plot(test_losses,  label="Test Loss",  color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss (Weighted MSE + BCE)")
    plt.title("LSTM Autoencoder — Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "task1_loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = load_data()

    model = LSTMAutoencoder(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        sequence_len=SEQUENCE_LEN,
        num_layers=NUM_LAYERS
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, test_losses = train(model, train_loader, test_loader, device)

    # Save final weights too (best is saved automatically during training)
    final_path = os.path.join(MODEL_DIR, "task1_autoencoder.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved → {final_path}")

    save_loss_curve(train_losses, test_losses)
    print("\nTraining complete!")
