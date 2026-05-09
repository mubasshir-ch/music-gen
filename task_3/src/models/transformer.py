import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional context into the token embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Blocks (GPT-style uses Encoder blocks with a causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Aligns with our DataLoader output
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Projection to Vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates a causal mask to prevent attention to future tokens."""
        # Upper triangular matrix of -inf, with zeros on the diagonal
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, src_key_padding_mask=None):
        """
        src: [batch_size, seq_len]
        src_key_padding_mask: [batch_size, seq_len] (True for padding tokens)
        """
        seq_len = src.size(1)
        device = src.device
        
        # Create causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # Embed and scale
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # PyTorch PositionalEncoding expects [seq_len, batch_size, d_model] if batch_first=False
        # Since we use batch_first=True in Transformer, we transpose for PE, then transpose back
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Pass through Transformer
        # src_mask = causal mask (seq_len x seq_len)
        # src_key_padding_mask = padding mask (batch_size x seq_len)
        output = self.transformer(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Project to logits
        logits = self.fc_out(output)
        return logits
