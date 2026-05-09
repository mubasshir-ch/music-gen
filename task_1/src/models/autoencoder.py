# autoencoder.py
# Task 1 — LSTM Autoencoder for Single-Genre Music Generation
# Course: CSE425/EEE474 Neural Networks
#
# Architecture implements the spec equations exactly:
#   Encoder : z = f_phi(X)
#   Decoder : X_hat = g_theta(z)
#
# Design decisions and why:
#   - Encoder uses the final hidden state of a 2-layer LSTM as the
#     sequence summary, then projects it to latent_dim via a linear layer.
#   - Decoder initialises LSTM hidden and cell states from z (not from zeros),
#     which lets z carry full musical context into every generated timestep.
#   - dropout= is intentionally NOT set inside nn.LSTM — that parameter
#     triggers a cuDNN kernel path that deadlocks on many CUDA 11/12 builds
#     when batch_first=True. Explicit nn.Dropout layers are used instead.
#   - The decoder runs a single batched LSTM call over the whole sequence
#     (not a Python timestep loop), which avoids kernel-launch overhead
#     that made training appear frozen.

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    LSTM Encoder: maps a piano roll sequence X to a fixed-size latent vector z.

    Implements: z = f_phi(X)

    Input  shape: (batch, seq_len=64, input_size=128)
    Output shape: (batch, latent_dim=128)
    """

    def __init__(self, input_size=128, hidden_size=256,
                 latent_dim=128, num_layers=2):
        super().__init__()

        # 2-layer LSTM reads the full 64-step piano roll
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # dropout= omitted on purpose — see module docstring
        )

        self.dropout = nn.Dropout(0.3)

        # Projects hidden_size → latent_dim to form z
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # Run LSTM; we only need the final hidden state
        _, (hidden, _) = self.lstm(x)     # hidden: (num_layers, batch, H)
        h = self.dropout(hidden[-1])       # top layer: (batch, H)
        z = self.fc(h)                     # (batch, latent_dim)
        return z


class Decoder(nn.Module):
    """
    LSTM Decoder: reconstructs a piano roll sequence from latent vector z.

    Implements: X_hat = g_theta(z)

    Input  shape: (batch, latent_dim=128)
    Output shape: (batch, seq_len=64, output_size=128)

    During training, teacher forcing is applied: the ground-truth previous
    timestep is fed as LSTM input with probability teacher_forcing_ratio,
    and zeros otherwise. At inference time teacher_forcing_ratio=0 so the
    model must rely entirely on the latent state.
    """

    def __init__(self, latent_dim=128, hidden_size=256,
                 output_size=128, sequence_len=64, num_layers=2):
        super().__init__()

        self.sequence_len = sequence_len
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_size  = output_size

        # Two separate projections — one for h0, one for c0
        # This gives each cell state its own learned initialisation from z
        self.init_h = nn.Linear(latent_dim, num_layers * hidden_size)
        self.init_c = nn.Linear(latent_dim, num_layers * hidden_size)

        # LSTM input = previous piano roll timestep (teacher forced or zeros)
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout      = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid      = nn.Sigmoid()   # piano roll values in [0, 1]

    def _init_hidden(self, z):
        """Derive LSTM h0 and c0 from latent vector z."""
        batch = z.size(0)
        h = (self.init_h(z)
             .view(batch, self.num_layers, self.hidden_size)
             .permute(1, 0, 2).contiguous())   # (num_layers, batch, H)
        c = (self.init_c(z)
             .view(batch, self.num_layers, self.hidden_size)
             .permute(1, 0, 2).contiguous())
        return h, c

    def forward(self, z, target=None, teacher_forcing_ratio=0.5):
        """
        Parameters
        ----------
        z                    : (batch, latent_dim)
        target               : (batch, seq_len, output_size)  [training only]
        teacher_forcing_ratio: float in [0, 1]
            Fraction of timesteps that receive the real previous note as input.
            Annealed from 0.8 → 0.0 during training so the model gradually
            stops relying on ground-truth context and learns to generate freely.
        """
        batch   = z.size(0)
        h0, c0  = self._init_hidden(z)

        if target is not None and teacher_forcing_ratio > 0:
            # Build shifted input: [zeros | x_0 ... x_{T-2}]
            sos = torch.zeros(batch, 1, self.output_size, device=z.device)
            inp = torch.cat([sos, target[:, :-1, :]], dim=1)

            # Randomly zero out timesteps based on teacher forcing ratio
            # so the model still practises generating without ground truth
            if teacher_forcing_ratio < 1.0:
                mask = (torch.rand(batch, self.sequence_len, 1, device=z.device)
                        < teacher_forcing_ratio).float()
                inp  = inp * mask
        else:
            # Inference: all-zero input — model generates from z alone
            inp = torch.zeros(batch, self.sequence_len,
                              self.output_size, device=z.device)

        # Single batched LSTM call — no Python timestep loop
        out, _ = self.lstm(inp, (h0, c0))        # (batch, seq_len, H)
        out     = self.dropout(out)
        out     = self.output_layer(out)          # (batch, seq_len, 128)
        return self.sigmoid(out)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder combining Encoder and Decoder.

    Forward pass:
        z     = Encoder(X)          — compress
        X_hat = Decoder(z)          — reconstruct
        loss  = MSE(X, X_hat)       — computed in train_ae.py
    """

    def __init__(self, input_size=128, hidden_size=256,
                 latent_dim=128, sequence_len=64, num_layers=2):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_size, input_size,
                               sequence_len, num_layers)

    def forward(self, x, teacher_forcing_ratio=0.5):
        z     = self.encoder(x)
        x_hat = self.decoder(z, target=x,
                             teacher_forcing_ratio=teacher_forcing_ratio)
        return x_hat, z


# Quick shape verification
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x     = torch.randn(8, 64, 128).to(device)
    model = LSTMAutoencoder().to(device)

    # Training mode
    x_hat, z = model(x, teacher_forcing_ratio=0.5)
    print(f"Input  : {x.shape}")
    print(f"Latent : {z.shape}")
    print(f"Output : {x_hat.shape}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # Inference mode
    x_inf, _ = model(x, teacher_forcing_ratio=0.0)
    print(f"Inference output: {x_inf.shape}  — shapes match: {x_inf.shape == x.shape}")
