import torch
import torch.nn as nn
import math
import pretty_midi
import numpy as np
import random
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    Args:
        d_model: Model dimension
        dropout: Dropout rate
        max_len: Max sequence length
    """
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
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MusicTransformer(nn.Module):
    """
    Decoder-only Music Transformer.
    
    h_t = Emb(x_t) + Emb(genre)
    - Causal attention
    - Positional encodings

    Args:
        vocab_size: Vocab size
        num_genres: Number of genres
        d_model: Model dimension
        nhead: Attention heads
        num_layers: Transformer layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
    """
    def __init__(self, vocab_size, num_genres=2, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Genre embedding
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """
        Causal mask.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, genre_ids, src_key_padding_mask=None):
        """
        Forward pass.
        
        h_t = Emb(x_t) + Emb(genre)
        
        Args:
            src: [batch_size, seq_len]
            genre_ids: [batch_size]
            src_key_padding_mask: padding mask
            
        Returns:
            Logits: [batch_size, seq_len, vocab_size]
        """
        seq_len = src.size(1)
        device = src.device
        
        # Causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # Embed tokens
        token_emb = self.embedding(src) * math.sqrt(self.d_model)
        genre_emb = self.genre_embedding(genre_ids).unsqueeze(1)
        
        # Apply genre conditioning: h_t = Emb(x_t) + Emb(genre)
        x = token_emb + genre_emb
        
        # Apply positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer pass
        output = self.transformer(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Logits
        logits = self.fc_out(output)
        return logits

class MarkovBaseline:
    """
    Markov baseline: P(pitch_t | pitch_{t-1})
    """
    def __init__(self, max_notes=200):
        self.max_notes = max_notes
        self.transition_probs = None
        self.durations = None

    def train(self, df, maestro_dir, sample_size=30):
        """
        Train transitions.
        """
        print(f"Building Markov Matrix from {sample_size} MAESTRO files...")
        transition_counts = np.zeros((128, 128))
        empirical_durations = []
        
        train_files = df[df['split'] == 'train']['midi_filename'].tolist()
        sample_files = random.sample(train_files, min(len(train_files), sample_size))
        
        for midi_file in tqdm(sample_files, desc="Analyzing transitions"):
            full_path = maestro_dir / midi_file
            try:
                midi_data = pretty_midi.PrettyMIDI(str(full_path))
                for inst in midi_data.instruments:
                    if not inst.is_drum:
                        notes = sorted(inst.notes, key=lambda n: n.start)
                        for i in range(len(notes) - 1):
                            current_pitch = notes[i].pitch
                            next_pitch = notes[i+1].pitch
                            transition_counts[current_pitch, next_pitch] += 1
                            
                            duration = notes[i].end - notes[i].start
                            empirical_durations.append(duration)
            except Exception:
                continue
        
        # Laplace smoothing
        self.transition_probs = (transition_counts + 1e-8) / (transition_counts.sum(axis=1, keepdims=True) + 128 * 1e-8)
        self.durations = empirical_durations

    def generate(self, filename):
        """
        Generate MIDI.
        """
        if self.transition_probs is None or self.durations is None:
            raise ValueError("Markov model must be trained before generation.")

        midi = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        current_pitch = 60 # Middle C
        current_time = 0.0
        
        for _ in range(self.max_notes):
            duration = random.choice(self.durations)
            note = pretty_midi.Note(
                velocity=80, 
                pitch=current_pitch, 
                start=current_time, 
                end=current_time + duration
            )
            piano.notes.append(note)
            current_time += duration
            
            # Sample next pitch
            row_probs = self.transition_probs[current_pitch]
            current_pitch = np.random.choice(np.arange(128), p=row_probs)
            
        midi.instruments.append(piano)
        midi.write(str(filename))

class RandomBaseline:
    """
    Random baseline: Lower bound baseline.
    """
    def __init__(self, window_seconds=30.0, notes_per_sample=150):
        self.window_seconds = window_seconds
        self.notes_per_sample = notes_per_sample

    def generate(self, filename):
        """
        Generate random MIDI.
        """
        midi = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        durations = [0.125, 0.25, 0.5, 1.0] # Durations
        
        for _ in range(self.notes_per_sample):
            pitch = random.randint(21, 108)
            start_time = random.uniform(0, self.window_seconds)
            duration = random.choice(durations)
            end_time = start_time + duration
            velocity = random.randint(60, 100)
            
            note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=start_time, end=end_time
            )
            piano.notes.append(note)
            
        midi.instruments.append(piano)
        midi.write(str(filename))
