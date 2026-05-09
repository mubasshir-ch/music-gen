import pandas as pd
import torch
import itertools
from pathlib import Path
from miditok import REMI, TokenizerConfig
from torch.utils.data import Dataset
from tqdm import tqdm

# Max sequence length
MAX_SEQ_LEN = 1024

def build_tokenizer():
    """
    Init REMI tokenizer.

    Returns:
        REMI tokenizer instance.
    """
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_programs=False,
        use_tempos=False,
        special_tokens=["PAD", "BOS"]
    )
    tokenizer = REMI(config)
    return tokenizer

def process_maestro(tokenizer, maestro_dir, split_type='train'):
    """
    Process MAESTRO (Classical).
    
    Args:
        tokenizer: REMI tokenizer.
        maestro_dir: Dataset root.
        split_type: 'train', 'validation', or 'test'.
        
    Returns:
        list: (token_ids, genre_id) where genre_id=0.
    """
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Ensure the dataset is present.")
        
    df = pd.read_csv(csv_path)
    df_split = df[df['split'] == split_type]
    
    tokenized_data = []
    genre_id = 0 # 0 -> Classical
    bos_token_id = tokenizer["BOS_None"]
    
    print(f"Processing MAESTRO {split_type} split...")
    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        midi_path = maestro_dir / row['midi_filename']
        try:
            parsed = tokenizer(midi_path)
            # Select longest track
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    continue
                longest_track = max(parsed, key=lambda t: len(t.ids))
                tokens = longest_track.ids
            else:
                tokens = parsed.ids
                
            # Prepend BOS
            sequence = [bos_token_id] + tokens
            tokenized_data.append((sequence, genre_id))
        except Exception:
            continue
    return tokenized_data

def process_lakh_subset(tokenizer, lakh_dir, num_files=50):
    """
    Process Lakh subset (Modern).
    
    Args:
        tokenizer: REMI tokenizer.
        lakh_dir: Dataset root.
        num_files: Max files to process.
        
    Returns:
        list: (token_ids, genre_id) where genre_id=1.
    """
    tokenized_data = []
    genre_id = 1 # 1 -> Modern
    bos_token_id = tokenizer["BOS_None"]
    
    print(f"\nScanning for {num_files} valid samples in the Lakh Dataset...")
    midi_generator = itertools.chain(lakh_dir.rglob("*.mid"), lakh_dir.rglob("*.midi"))
    
    processed_count = 0
    skipped_count = 0
    
    with tqdm(total=num_files, desc="Lakh Modern Subset") as pbar:
        for midi_path in midi_generator:
            if processed_count >= num_files:
                break
            pbar.set_postfix_str(f"Parsing: {midi_path.name[:20]}")
            try:
                parsed = tokenizer(midi_path)
                if isinstance(parsed, list):
                    if len(parsed) == 0:
                        continue
                    longest_track = max(parsed, key=lambda t: len(t.ids))
                    tokens = longest_track.ids
                else:
                    tokens = parsed.ids
                
                sequence = [bos_token_id] + tokens
                tokenized_data.append((sequence, genre_id))
                processed_count += 1
                pbar.update(1) 
            except Exception:
                skipped_count += 1
                continue
    return tokenized_data

class MusicTokenDataset(Dataset):
    """
    Music dataset wrapper.
    
    Args:
        data: List of (sequence, genre_id).
        max_len: Max length for truncation.
    """
    def __init__(self, data, max_len=MAX_SEQ_LEN):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, genre_id = self.data[idx]
        # Truncate to max_len
        if len(seq) > self.max_len + 1:
            seq = seq[:self.max_len + 1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(genre_id, dtype=torch.long)

def collate_fn(batch, pad_id):
    """
    Collate batch.
    
    Next-token prediction shifts.
    
    Args:
        batch: List from MusicTokenDataset.
        pad_id: Padding token ID.
        
    Returns:
        tuple: (inputs, targets, genre_ids, attention_mask)
    """
    sequences = [item[0] for item in batch]
    genre_ids = torch.stack([item[1] for item in batch])
    
    # Pad to max length in batch
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=pad_id
    )
    
    # Shift for NTP
    inputs = padded_batch[:, :-1]
    targets = padded_batch[:, 1:]
    
    # Padding mask
    attention_mask = (inputs != pad_id).long()
    return inputs, targets, genre_ids, attention_mask
