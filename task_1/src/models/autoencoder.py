# autoencoder.py
# LSTM Autoencoder model for music sequence reconstruction and generation
# Fixed: Decoder now uses autoregressive teacher-forced input instead of
#        repeating z, which prevented any temporal structure from forming.

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    LSTM Encoder: compresses input piano roll sequence into a latent vector z.

    Input shape : (batch_size, sequence_len, input_size)
                   sequence_len = 64 time steps
                   input_size   = 128 piano pitches
    Output shape: (batch_size, latent_dim)
    """
    def __init__(self, input_size=128, hidden_size=512, latent_dim=256, num_layers=1):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Project last hidden state → latent vector
        self.fc_mu  = nn.Linear(hidden_size, latent_dim)   # mean
        self.fc_log = nn.Linear(hidden_size, latent_dim)   # log-variance (for optional VAE use)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        h = hidden[-1]                       # (batch, hidden_size)
        z = self.fc_mu(h)                    # (batch, latent_dim)
        return z


class Decoder(nn.Module):
    """
    LSTM Decoder with proper temporal structure.

    KEY FIX: Instead of repeating z at every timestep (which kills temporal
    variation), we:
      1. Use z to initialise the LSTM hidden and cell states.
      2. Feed a learned start token + previous output at each step.
    This gives the LSTM a genuine sequence-generation signal.

    Input shape : (batch_size, latent_dim)
    Output shape: (batch_size, sequence_len, output_size)
    """
    def __init__(self, latent_dim=256, hidden_size=512, output_size=128,
                 sequence_len=64, num_layers=1):
        super(Decoder, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_size  = output_size

        # Project z → initial hidden state for ALL layers
        self.init_h = nn.Linear(latent_dim, num_layers * hidden_size)
        self.init_c = nn.Linear(latent_dim, num_layers * hidden_size)

        # Learnable "start of sequence" token (replaces the crude z-repeat)
        self.sos_token = nn.Parameter(torch.zeros(1, 1, output_size))

        # LSTM input = previous step's output (output_size features)
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Project LSTM output → piano roll logits
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid       = nn.Sigmoid()

    def _init_hidden(self, z):
        """Initialise LSTM h0 and c0 from latent vector z."""
        batch = z.size(0)
        h = self.init_h(z)                              # (batch, num_layers*hidden)
        c = self.init_c(z)
        h = h.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c = c.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        return h, c

    def forward(self, z, target=None, teacher_forcing_ratio=0.5):
        """
        z      : (batch, latent_dim)
        target : (batch, seq_len, output_size) — used for teacher forcing during training
        teacher_forcing_ratio: probability of using ground-truth previous step
        """
        batch = z.size(0)
        h, c  = self._init_hidden(z)

        # Start token: (batch, 1, output_size)
        inp = self.sos_token.expand(batch, 1, self.output_size)

        outputs = []
        for t in range(self.sequence_len):
            out, (h, c) = self.lstm(inp, (h, c))       # (batch, 1, hidden)
            step = self.sigmoid(self.output_layer(out)) # (batch, 1, output_size)
            outputs.append(step)

            # Teacher forcing: sometimes feed the real note, sometimes feed prediction
            use_tf = (target is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            inp = target[:, t:t+1, :] if use_tf else step

        return torch.cat(outputs, dim=1)   # (batch, seq_len, output_size)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder = Encoder + Decoder.
    """
    def __init__(self, input_size=128, hidden_size=512, latent_dim=256,
                 sequence_len=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=num_layers
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            output_size=input_size,
            sequence_len=sequence_len,
            num_layers=num_layers
        )

    def forward(self, x, teacher_forcing_ratio=0.5):
        z     = self.encoder(x)
        x_hat = self.decoder(z, target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return x_hat, z


# ── Quick shape test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dummy_input = torch.randn(8, 64, 128).to(device)
    model = LSTMAutoencoder().to(device)

    # Training mode (teacher forcing on)
    x_hat, z = model(dummy_input, teacher_forcing_ratio=0.5)
    print(f"Input shape    : {dummy_input.shape}")
    print(f"Latent z shape : {z.shape}")
    print(f"Output shape   : {x_hat.shape}")

    # Inference mode (no teacher forcing)
    x_hat_inf, _ = model(dummy_input, teacher_forcing_ratio=0.0)
    print(f"Inference shape: {x_hat_inf.shape}")
    print("\nModel architecture:")
    print(model)
