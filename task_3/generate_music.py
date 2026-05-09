import torch
import torch.nn.functional as F
import os
from pathlib import Path
from tqdm import tqdm
from miditok import TokSequence

from data_pipeline import build_tokenizer
from src.models.transformer import MusicTransformer

# --- CONFIGURATION ---
MODEL_WEIGHTS = "transformer_epoch_best.pt" # Change if your file is named differently
OUTPUT_DIR = Path("outputs/generated_midis")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generation Hyperparameters
MAX_TOKENS = 512       # Length of the generated sequence
TEMPERATURE = 1.0      # 1.0 = normal, 0.8 = conservative, 1.2 = creative
TOP_K = 20             # Restrict sampling to the top 20 most likely tokens

def top_k_logits(logits, k):
    """Masks everything but the top k logits with -infinity."""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), logits)

@torch.no_grad()
def generate_sequence(model, tokenizer, start_token_str, max_len=MAX_TOKENS, temp=TEMPERATURE, top_k=TOP_K):
    """Autoregressively generates a token sequence."""
    model.eval()
    
    # Initialize the sequence with the genre conditioning token
    start_token_id = tokenizer[f"{start_token_str}_None"]
    sequence = torch.tensor([[start_token_id]], dtype=torch.long).to(DEVICE)
    
    # We also want to prevent it from generating <PAD> tokens in the middle of a song
    pad_idx = tokenizer["PAD_None"]
    
    for _ in range(max_len):
        # 1. Forward pass (only need the very last token's logits)
        logits = model(sequence)
        next_token_logits = logits[:, -1, :] 
        
        # 2. Scale by temperature
        next_token_logits = next_token_logits / temp
        
        # 3. Mask out the PAD token so it never gets chosen
        next_token_logits[0, pad_idx] = -float('Inf')
        
        # 4. Apply Top-K filtering
        filtered_logits = top_k_logits(next_token_logits, top_k)
        
        # 5. Convert to probabilities and sample
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 6. Append to the sequence
        sequence = torch.cat([sequence, next_token], dim=1)
        
    return sequence[0].cpu().numpy().tolist()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Tokenizer
    tokenizer = build_tokenizer()
    vocab_size = len(tokenizer)
    
    # 2. Load Model Architecture & Weights
    print("Loading model...")
    model = MusicTransformer(vocab_size=vocab_size).to(DEVICE)
    
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Could not find {MODEL_WEIGHTS}. Did the training loop finish?")
        
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
    print("Model loaded successfully.")
    
    # Task 3 requires 10 long-sequence generated compositions[cite: 531].
    # Let's do 5 Classical and 5 Modern.
    tasks = [("Classical", 5), ("Modern", 5)]
    
    print(f"\nGenerating 10 compositions (Temp: {TEMPERATURE}, Top-K: {TOP_K})...")
    
    for genre, count in tasks:
        for i in tqdm(range(count), desc=f"Generating {genre}"):
            # Generate the raw integer tokens
            tokens = generate_sequence(model, tokenizer, start_token_str=genre)
            
            # Filter out ALL special tokens. This handles the starting genre token
            # AND guarantees no accidental special tokens crash the decoder.
            special_ids = {tokenizer["PAD_None"], tokenizer["Classical_None"], tokenizer["Modern_None"]}
            music_tokens = [tok for tok in tokens if tok not in special_ids]
            
            # Wrap in an outer list to explicitly tell miditok: "This is a 1-track song"
            # This satisfies the 2D requirement without any fancy objects.
            midi_obj = tokenizer.decode([music_tokens])
            
            # Save the file
            filename = OUTPUT_DIR / f"Task3_Transformer_{genre}_{i+1}.mid"
            midi_obj.dump_midi(filename)

    print(f"\nDone! Generated files saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
