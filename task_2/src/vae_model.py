import torch
import torch.nn as nn

# This file defines the LSTMVAE class, which is a Variational Autoencoder (VAE) architecture 
# using LSTM layers for encoding and decoding sequences of piano rolls. 
# The encoder consists of an LSTM that processes the input sequence and produces a hidden state, 
# which is then transformed into the mean (mu) and log variance (logvar) of the latent space.

class LSTMVAE(nn.Module):
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

        # The encoder LSTM takes the input piano roll sequence and produces a hidden representation.
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    # The encode method processes the input sequence through the encoder LSTM and computes the mean and log variance of the latent space.
    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)

        final_hidden = h_n[-1]

        mu = self.to_mu(final_hidden)
        logvar = self.to_logvar(final_hidden)

        return mu, logvar

    # The reparameterize method implements the reparameterization trick to sample a latent vector z from the distribution defined by mu and logvar.
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + std * eps
        return z

    # The decode method takes a latent vector z and decodes it into a reconstructed piano roll sequence.
    def decode(self, z):
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoded, _ = self.decoder_lstm(z_repeated)

        logits = self.output_layer(decoded)

        return logits

    # The forward method defines the full forward pass of the VAE: 
    # it encodes the input, samples from the latent space, and decodes to produce the output logits.
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        return logits, mu, logvar