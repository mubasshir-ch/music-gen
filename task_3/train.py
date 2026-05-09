import torch
import torch.nn as nn
import torch.optim as optim
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data import build_tokenizer, process_maestro, process_lakh_subset, MusicTokenDataset, collate_fn
from src.models import MusicTransformer

# Training config
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("task_3/checkpoints")
MAESTRO_DIR = Path("task_3/datasets/maestro-v3.0.0")
LAKH_DIR = Path("task_3/datasets/lmd_full")

def train_epoch(model, dataloader, optimizer, criterion, clip=1.0):
    """
    One training epoch.
    
    Args:
        model: MusicTransformer
        dataloader: Training data
        optimizer: Adam/etc
        criterion: CrossEntropyLoss
        clip: Gradient clipping norm
    """
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training Epoch")
    
    for inputs, targets, genre_ids, mask in pbar:
        # Move to device
        inputs, targets, genre_ids = inputs.to(DEVICE), targets.to(DEVICE), genre_ids.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Padding mask (True = ignore)
        padding_mask = (mask == 0).to(DEVICE) 
        
        # Forward pass
        logits = model(inputs, genre_ids=genre_ids, src_key_padding_mask=padding_mask)
        
        # Flatten for loss: [Batch * SeqLen, VocabSize]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, genre_ids, mask in dataloader:
            inputs, targets, genre_ids = inputs.to(DEVICE), targets.to(DEVICE), genre_ids.to(DEVICE)
            padding_mask = (mask == 0).to(DEVICE)
            
            logits = model(inputs, genre_ids=genre_ids, src_key_padding_mask=padding_mask)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Executing training on: {DEVICE}")
    
    # Init tokenizer
    tokenizer = build_tokenizer()
    pad_idx = tokenizer["PAD_None"]
    vocab_size = len(tokenizer)
    
    # Load data
    # Genre mapping: Classical=0, Modern=1
    train_seqs = process_maestro(tokenizer, MAESTRO_DIR, 'train') + process_lakh_subset(tokenizer, LAKH_DIR)
    val_seqs = process_maestro(tokenizer, MAESTRO_DIR, 'validation')
    
    # Init dataloaders
    train_loader = DataLoader(MusicTokenDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader = DataLoader(MusicTokenDataset(val_seqs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))
    
    # Init model/opt/loss
    model = MusicTransformer(vocab_size=vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        
        # Calculate perplexity
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CHECKPOINT_DIR / "transformer_epoch_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved improved model to {save_path}")

if __name__ == "__main__":
    main()
