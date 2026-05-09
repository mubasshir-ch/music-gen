import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data extracted directly from your successful training run terminal output
epochs = list(range(1, 11))
train_ppl = [54.99, 25.45, 22.55, 20.81, 19.63, 18.87, 18.30, 17.80, 17.35, 16.79]
val_ppl = [28.26, 24.09, 22.28, 21.14, 20.54, 20.10, 19.62, 19.27, 19.05, 18.32]

def main():
    plt.figure(figsize=(8, 5))
    
    # Plotting according to the specification: solid for train, dashed for val
    plt.plot(epochs, train_ppl, label='Training Perplexity', color='blue', linestyle='-')
    plt.plot(epochs, val_ppl, label='Validation Perplexity', color='orange', linestyle='--')
    
    plt.title('Task 3: Transformer Perplexity over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.xticks(epochs)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Save as high-res PNG as requested by the guide
    save_path = OUTPUT_DIR / "task3_perplexity_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path.absolute()}")
    
if __name__ == "__main__":
    main()
