import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=88,
        hidden_dim=256,
        latent_dim=64,
        num_layers=2,
        seq_len=128,
        dropout=0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)

        # h_n shape: (num_layers, batch, hidden_dim)
        final_hidden = h_n[-1]

        z = self.to_latent(final_hidden)
        return z

    def decode(self, z):
        # z shape: (batch, latent_dim)
        batch_size = z.size(0)

        # Repeat z across all time steps
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoded, _ = self.decoder_lstm(z_repeated)

        # raw logits, no sigmoid here
        logits = self.output_layer(decoded)

        return logits

    def forward(self, x):
        z = self.encode(x)
        logits = self.decode(z)
        return logits