import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.data import build_tokenizer
from src.models import MusicTransformer, MarkovBaseline, RandomBaseline

# Generation config
MODEL_WEIGHTS = Path("task_3/checkpoints/transformer_epoch_best.pt")
OUTPUT_DIR = Path("task_3/outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAESTRO_DIR = Path("task_3/datasets/maestro-v3.0.0")

# Generation Hyperparameters
MAX_TOKENS = 512
TEMPERATURE = 1.0
TOP_K = 20

def top_k_logits(logits, k):
    """
    Top-k filtering.
    
    Args:
        logits (Tensor): Model output logits.
        k (int): Top-k candidates to keep.
        
    Returns:
        Tensor: Masked logits.
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), logits)

@torch.no_grad()
def generate_sequence(model, tokenizer, genre_id, max_len=MAX_TOKENS, temp=TEMPERATURE, top_k=TOP_K):
    """
    Autoregressive generation.
    
    Args:
        model: MusicTransformer
        tokenizer: REMI tokenizer
        genre_id: 0 for Classical, 1 for Modern
        max_len: Max tokens
        temp: Sampling temperature
        top_k: Top-K filtering
        
    Returns:
        list: Generated token IDs.
    """
    model.eval()
    bos_token_id = tokenizer["BOS_None"]
    pad_idx = tokenizer["PAD_None"]
    
    # Init sequence with BOS
    sequence = torch.tensor([[bos_token_id]], dtype=torch.long).to(DEVICE)
    genre_ids = torch.tensor([genre_id], dtype=torch.long).to(DEVICE)
    
    for _ in range(max_len):
        # Forward pass (h_t = Emb(x_t) + Emb(genre))
        logits = model(sequence, genre_ids=genre_ids)
        
        # Scale logits by temp
        next_token_logits = logits[:, -1, :] 
        next_token_logits = next_token_logits / temp
        
        # Mask special tokens
        next_token_logits[0, pad_idx] = -float('Inf')
        next_token_logits[0, bos_token_id] = -float('Inf')
        
        # Apply Top-K and sample
        filtered_logits = top_k_logits(next_token_logits, top_k)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Update sequence
        sequence = torch.cat([sequence, next_token], dim=1)
        
    return sequence[0].cpu().numpy().tolist()

def main():
    
    # Transformer generation
    trans_out = OUTPUT_DIR / "generated_midis"
    trans_out.mkdir(parents=True, exist_ok=True)
    
    tokenizer = build_tokenizer()
    vocab_size = len(tokenizer)
    
    print("Initializing Transformer generation...")
    model = MusicTransformer(vocab_size=vocab_size).to(DEVICE)
    
    if MODEL_WEIGHTS.exists():
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
        print(f"Model weights loaded from {MODEL_WEIGHTS}")
        
        # Tasks by genre
        genre_map = {"Classical": 0, "Modern": 1}
        tasks = [("Classical", 5), ("Modern", 5)]
        
        for genre_name, count in tasks:
            genre_id = genre_map[genre_name]
            for i in tqdm(range(count), desc=f"Generating {genre_name}"):
                tokens = generate_sequence(model, tokenizer, genre_id=genre_id)
                
                # Remove special tokens
                special_ids = {tokenizer["PAD_None"], tokenizer["BOS_None"]}
                music_tokens = [tok for tok in tokens if tok not in special_ids]
                
                # Decode and save MIDI
                midi_obj = tokenizer.decode([music_tokens])
                filename = trans_out / f"Task3_Transformer_{genre_name}_{i+1}.mid"
                midi_obj.dump_midi(filename)
    else:
        print(f"WARNING: Transformer weights not found at {MODEL_WEIGHTS}")

    # Markov baseline
    markov_out = OUTPUT_DIR / "baselines/markov"
    markov_out.mkdir(parents=True, exist_ok=True)
    csv_path = MAESTRO_DIR / "maestro-v3.0.0.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        markov = MarkovBaseline()
        markov.train(df, MAESTRO_DIR)
        print(f"Generating Markov Baseline samples...")
        for i in tqdm(range(10)):
            save_path = markov_out / f"Markov_Baseline_{i+1}.mid"
            markov.generate(save_path)
    else:
        print(f"WARNING: MAESTRO metadata not found at {csv_path}. Skipping Markov.")

    # Random baseline
    random_out = OUTPUT_DIR / "baselines/random"
    random_out.mkdir(parents=True, exist_ok=True)
    random_gen = RandomBaseline()
    print(f"Generating Random Baseline samples...")
    for i in tqdm(range(10)):
        save_path = random_out / f"Random_Baseline_{i+1}.mid"
        random_gen.generate(save_path)

    print(f"\nGeneration complete. Samples available in: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
