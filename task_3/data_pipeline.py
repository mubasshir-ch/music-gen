import os
import pandas as pd
import torch
import itertools
from pathlib import Path
from miditok import REMI, TokenizerConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path("./datasets")
MAESTRO_DIR = BASE_DIR / "maestro-v3.0.0"
LAKH_DIR = BASE_DIR / "lmd_full"

# We'll grab just 50 Lakh files to satisfy the multi-genre requirement without bloating training time.
LAKH_SUBSET_SIZE = 50 
MAX_SEQ_LEN = 1024 # Standard context window for GPT-style Transformer

def build_tokenizer():
    """Builds the REMI tokenizer with our custom genre tokens."""
    # We add special tokens for padding and our proxy genres.
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_programs=False,
        use_tempos=False,
        special_tokens=["PAD", "Classical", "Modern"]
    )
    tokenizer = REMI(config)
    return tokenizer

def process_maestro(tokenizer, split_type='train'):
    """Parses the MAESTRO CSV and tokenizes the files for a specific split."""
    csv_path = MAESTRO_DIR / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Make sure it is extracted.")
        
    df = pd.read_csv(csv_path)
    # Strictly adhere to the provided splits to prevent data leakage
    df_split = df[df['split'] == split_type]
    
    tokenized_sequences = []
    genre_token_id = tokenizer["Classical_None"]
    
    print(f"Processing MAESTRO {split_type} split...")
    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        midi_path = MAESTRO_DIR / row['midi_filename']
        try:
            parsed = tokenizer(midi_path)
            
            # Robust extraction: handle both lists and direct objects
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    continue
                longest_track = max(parsed, key=lambda t: len(t.ids))
                tokens = longest_track.ids
            else:
                tokens = parsed.ids
                
            # Inject genre token at the start
            sequence = [genre_token_id] + tokens
            tokenized_sequences.append(sequence)
        except Exception:
            # Catching the occasional truly malformed MIDI
            continue
            
    return tokenized_sequences

def process_lakh_subset(tokenizer, num_files=LAKH_SUBSET_SIZE):
    """Safely extracts a small subset of Lakh files for genre diversity."""
    tokenized_sequences = []
    genre_token_id = tokenizer["Modern_None"]
    
    print(f"\nScanning for {num_files} clean files in the Lakh Dataset...")
    
    midi_generator = itertools.chain(LAKH_DIR.rglob("*.mid"), LAKH_DIR.rglob("*.midi"))
    
    processed_count = 0
    skipped_count = 0 # Track how much garbage we are hitting
    
    with tqdm(total=num_files, desc="Lakh Files Processed") as pbar:
        for midi_path in midi_generator:
            if processed_count >= num_files:
                break
                
            # Update UI *before* parsing so we see if it hangs
            pbar.set_postfix_str(f"Trying: {midi_path.name[:20]} | Skipped: {skipped_count}")
            
            try:
                parsed = tokenizer(midi_path)
                
                # Handle multitrack (Lakh) vs single-track (MAESTRO)
                if isinstance(parsed, list):
                    if len(parsed) == 0:
                        continue
                    # Grab the track with the most tokens (likely the main instrument)
                    longest_track = max(parsed, key=lambda t: len(t.ids))
                    tokens = longest_track.ids
                else:
                    tokens = parsed.ids
                
                sequence = [genre_token_id] + tokens
                tokenized_sequences.append(sequence)
                
                processed_count += 1
                pbar.update(1) 
            except Exception:
                # We solved the core issue, so we can safely return to silently 
                # skipping files that are genuinely corrupted.
                skipped_count += 1
                continue
            
    return tokenized_sequences

class MusicTokenDataset(Dataset):
    """PyTorch Dataset for Transformer training."""
    def __init__(self, sequences, max_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Truncate if too long (reserving 1 spot for the shifted target)
        if len(seq) > self.max_len + 1:
            seq = seq[:self.max_len + 1]
            
        return torch.tensor(seq, dtype=torch.long)

def collate_fn(batch, pad_id):
    """Pads variable length sequences and creates inputs/targets."""
    # Pad sequences to the max length in this specific batch
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=pad_id
    )
    
    # Input is sequence up to t-1, target is sequence shifted one position forward
    inputs = padded_batch[:, :-1]
    targets = padded_batch[:, 1:]
    
    # Create an attention mask (1 for real tokens, 0 for padding)
    attention_mask = (inputs != pad_id).long()
    
    return inputs, targets, attention_mask

# --- QUICK TEST / ENTRY POINT ---
if __name__ == "__main__":
    tokenizer = build_tokenizer()
    pad_idx = tokenizer["PAD_None"]
    
    # 1. Build Training Set (MAESTRO Train + Lakh Subset)
    train_seqs = process_maestro(tokenizer, split_type='train')
    lakh_seqs = process_lakh_subset(tokenizer)
    combined_train_seqs = train_seqs + lakh_seqs
    
    # 2. Build Validation Set (Strictly MAESTRO Val, no Lakh needed here to verify generalization)
    val_seqs = process_maestro(tokenizer, split_type='validation')
    
    # 3. Create DataLoaders
    train_dataset = MusicTokenDataset(combined_train_seqs)
    val_dataset = MusicTokenDataset(val_seqs)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, # Start small to test memory
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    print(f"\nPipeline Ready.")
    print(f"Total Training Sequences: {len(train_dataset)}")
    print(f"Total Validation Sequences: {len(val_dataset)}")
    print(f"Vocabulary Size: {len(tokenizer)}")
