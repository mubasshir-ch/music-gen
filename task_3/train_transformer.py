import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

# Assuming you place this next to your other scripts or adjust imports accordingly
from data_pipeline import build_tokenizer, process_maestro, process_lakh_subset, MusicTokenDataset, collate_fn
from torch.utils.data import DataLoader

# Import your Transformer model (adjust path if needed)
from src.models.transformer import MusicTransformer

# --- CONFIGURATION ---
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "transformer_epoch_{}.pt"

def train_epoch(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets, mask in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass (src_key_padding_mask expects True for ignored padding tokens)
        padding_mask = (mask == 0).to(DEVICE) 
        logits = model(inputs, src_key_padding_mask=padding_mask)
        
        # Reshape for CrossEntropyLoss: 
        # logits: [batch_size * seq_len, vocab_size]
        # targets: [batch_size * seq_len]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping is essential for stable Transformer training
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss += loss.item()
        
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets, mask in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            padding_mask = (mask == 0).to(DEVICE)
            
            logits = model(inputs, src_key_padding_mask=padding_mask)
            
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    print(f"Training on: {DEVICE}")
    
    # 1. Setup Data Pipeline
    tokenizer = build_tokenizer()
    pad_idx = tokenizer["PAD_None"]
    vocab_size = len(tokenizer)
    
    train_seqs = process_maestro(tokenizer, 'train') + process_lakh_subset(tokenizer)
    val_seqs = process_maestro(tokenizer, 'validation')
    
    train_loader = DataLoader(MusicTokenDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader = DataLoader(MusicTokenDataset(val_seqs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))
    
    # 2. Initialize Model
    model = MusicTransformer(vocab_size=vocab_size).to(DEVICE)
    # (Placeholder if you haven't imported the class yet)
    
    # 3. Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ignore_index ensures padded zeros don't contribute to the cross-entropy calculation
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        
        # Calculate Perplexity: Perplexity = exp(CrossEntropyLoss)
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH.format("best"))
            print("--> Saved new best model")

if __name__ == "__main__":
    main()
    pass
