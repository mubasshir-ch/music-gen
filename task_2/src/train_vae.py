from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import PianoRollDataset
from vae_model import LSTMVAE


# ============== Config ===============

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "processed" / "train.npy"
VAL_PATH = PROJECT_ROOT / "processed" / "validation.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64         # adjust this based on your GPU memory and training speed requirements
TOTAL_EPOCHS = 52       # adjust this based on how long you want to train and how the losses are progressing
LR = 1e-3               # reduce this if the training is unstable or the losses are not decreasing

INPUT_DIM = 88          # number of piano keys (MIDI pitches 21 to 108)
HIDDEN_DIM = 128        # size of LSTM hidden layers
LATENT_DIM = 64         # size of the latent vector (bottleneck)
NUM_LAYERS = 2          # number of LSTM layers in the encoder and decoder
SEQ_LEN = 128           # length of the input piano roll windows (must match the window size used in preprocessing)

KL_WARMUP_EPOCHS = 30   # KL-Annealing: number of epochs to warm up the KL divergence weight (beta) from 0 to MAX_BETA
MAX_BETA = 0.01         # KL-Annealing: maximum weight for the KL divergence term in the VAE loss (try values between 0.01 and 0.1 to start with)

RESUME = True 
RESUME_PATH = CHECKPOINT_DIR / "latest_lstm_vae.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Pos weight helper

def estimate_pos_weight(dataset, max_batches=100, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_positive = 0
    total_cells = 0

    for i, x in enumerate(loader):
        total_positive += x.sum().item()
        total_cells += x.numel()

        if i + 1 >= max_batches:
            break

    total_negative = total_cells - total_positive
    pos_weight = total_negative / max(total_positive, 1)
    pos_weight = min(pos_weight, 30.0)

    return pos_weight


# Loss functions
# KL(q(z|x) || N(0, I))
def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

# KL-annealing part
def get_beta(epoch):
    return min(MAX_BETA, MAX_BETA * epoch / KL_WARMUP_EPOCHS)

def vae_loss(logits, x, mu, logvar, recon_criterion, beta):
    recon_loss = recon_criterion(logits, x)
    kl_loss = kl_divergence(mu, logvar)

    total_loss = recon_loss + beta * kl_loss   # # VAE loss = reconstruction loss + beta * KL divergence
    return total_loss, recon_loss, kl_loss

# Train / validate
def train_one_epoch(model, loader, recon_criterion, optimizer, beta):
    model.train()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0

    for x in tqdm(loader, desc="Training", leave=False):
        x = x.to(DEVICE)

        optimizer.zero_grad()                           # zero the gradients before the backward pass

        logits, mu, logvar = model(x)                   # forward pass through the VAE model to get the output logits and the latent distribution parameters (mu and logvar)    

        loss, recon_loss, kl_loss = vae_loss(
            logits,
            x,
            mu,
            logvar,
            recon_criterion,
            beta,
        )

        loss.backward()                                 # compute the gradients of the loss with respect to the model parameters

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradient clipping to prevent exploding gradients (clip the norm of the gradients to a maximum value)

        optimizer.step()                                # update the model parameters using the computed gradients and the optimization algorithm (Adam in this case)

        batch_size = x.size(0)                          # get the actual batch size (last batch may be smaller)

        # accumulate the total loss, reconstruction loss, and KL loss for this batch, weighted by the batch size, to compute the average losses later
        total_loss_sum += loss.item() * batch_size
        recon_loss_sum += recon_loss.item() * batch_size
        kl_loss_sum += kl_loss.item() * batch_size

    n = len(loader.dataset)

    return (
        total_loss_sum / n,
        recon_loss_sum / n,
        kl_loss_sum / n,
    )

# The validate function is similar to train_one_epoch but runs in evaluation mode (model.eval()) and does not compute gradients or update the model parameters. 
# It also uses torch.no_grad() to disable gradient tracking for efficiency. It computes and returns the average total loss, reconstruction loss, and KL loss on the validation set.

def validate(model, loader, recon_criterion, beta):
    model.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0

    with torch.no_grad():
        for x in tqdm(loader, desc="Validation", leave=False):
            x = x.to(DEVICE)

            logits, mu, logvar = model(x)

            loss, recon_loss, kl_loss = vae_loss(
                logits,
                x,
                mu,
                logvar,
                recon_criterion,
                beta,
            )

            batch_size = x.size(0)

            total_loss_sum += loss.item() * batch_size
            recon_loss_sum += recon_loss.item() * batch_size
            kl_loss_sum += kl_loss.item() * batch_size

    n = len(loader.dataset)

    return (
        total_loss_sum / n,
        recon_loss_sum / n,
        kl_loss_sum / n,
    )


# Checkpointing and plotting

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    history,
    best_val_loss,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
            "config": {
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
                "num_layers": NUM_LAYERS,
                "seq_len": SEQ_LEN,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "kl_warmup_epochs": KL_WARMUP_EPOCHS,
                "max_beta": MAX_BETA,
            },
        },
        path,
    )


def plot_history(history):
    epochs = range(1, len(history["train_total"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_total"], label="Train total loss")
    plt.plot(epochs, history["val_total"], label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total VAE loss")
    plt.legend()
    plt.title("LSTM VAE Total Loss")
    plt.savefig(PLOT_DIR / "lstm_vae_total_loss.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_recon"], label="Train reconstruction loss")
    plt.plot(epochs, history["val_recon"], label="Validation reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.legend()
    plt.title("LSTM VAE Reconstruction Loss")
    plt.savefig(PLOT_DIR / "lstm_vae_recon_loss.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_kl"], label="Train KL loss")
    plt.plot(epochs, history["val_kl"], label="Validation KL loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL divergence")
    plt.legend()
    plt.title("LSTM VAE KL Loss")
    plt.savefig(PLOT_DIR / "lstm_vae_kl_loss.png", dpi=300)
    plt.close()


# The main function orchestrates the entire training process: 
# it loads the datasets, 
# initializes the model and optimizer, 
# optionally resumes from a checkpoint, 
# and runs the training loop for the specified number of epochs.

def main():
    print("Using device:", DEVICE)

    train_dataset = PianoRollDataset(TRAIN_PATH)
    val_dataset = PianoRollDataset(VAL_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    pos_weight_value = estimate_pos_weight(train_dataset)
    print("Estimated pos_weight:", pos_weight_value)

    pos_weight = torch.tensor(pos_weight_value, device=DEVICE)
    recon_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # initialize the LSTM VAE model with the specified architecture and move it to the configured device (CPU or GPU).
    model = LSTMVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    # initialize the Adam optimizer with the model parameters and the specified learning rate (LR).
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 1
    best_val_loss = float("inf")

    history = {
        "train_total": [],
        "val_total": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": [],
        "beta": [],
    }

    # if RESUME is True and the specified checkpoint file exists, load the model state, optimizer state, epoch number, training history, and best validation loss from the checkpoint to resume training from where it left off. Otherwise, start fresh training from epoch 1.
    if RESUME and RESUME_PATH.exists():
        checkpoint = torch.load(
            RESUME_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh VAE training")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        beta = get_beta(epoch)

        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS}")
        print(f"Beta: {beta:.4f}")

        # train the model for one epoch on the training set and compute the average total loss, reconstruction loss, and KL loss for the epoch. 
        # The train_one_epoch function runs in training mode (model.train()) and computes gradients to update the model parameters.
        train_total, train_recon, train_kl = train_one_epoch(
            model,
            train_loader,
            recon_criterion,
            optimizer,
            beta,
        )

        # validate the model on the validation set and compute the average total loss, reconstruction loss, and KL loss for the epoch.
        val_total, val_recon, val_kl = validate(
            model,
            val_loader,
            recon_criterion,
            beta,
        )

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)
        history["beta"].append(beta)

        print(f"Train total: {train_total:.6f}")
        print(f"Val total:   {val_total:.6f}")
        print(f"Train recon: {train_recon:.6f}")
        print(f"Val recon:   {val_recon:.6f}")
        print(f"Train KL:    {train_kl:.6f}")
        print(f"Val KL:      {val_kl:.6f}")

        latest_path = CHECKPOINT_DIR / "latest_lstm_vae.pt"

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            history,
            best_val_loss,
        )

        print(f"Saved latest checkpoint to {latest_path}")

        if val_total < best_val_loss:
            best_val_loss = val_total

            best_path = CHECKPOINT_DIR / "best_lstm_vae.pt"

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                history,
                best_val_loss,
            )

            print(f"Saved best checkpoint to {best_path}")

        plot_history(history)
        print("Updated VAE plots.")

    print("\nVAE training finished.")


if __name__ == "__main__":
    main()