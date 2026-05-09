import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIGURATION ---
MAESTRO_DIR = Path("datasets/maestro-v3.0.0")
CSV_PATH = MAESTRO_DIR / "maestro-v3.0.0.csv"
OUTPUT_DIR = Path("outputs/plots")

def plot_durations(df):
    """Generates the piece duration histogram."""
    plt.figure(figsize=(10, 6))
    
    # Convert seconds to minutes for readability
    durations_minutes = df['duration'] / 60.0
    
    plt.hist(durations_minutes, bins=50, color='teal', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Piece Durations in MAESTRO Dataset')
    plt.xlabel('Duration (Minutes)')
    plt.ylabel('Number of Pieces')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = OUTPUT_DIR / "eda_duration_histogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved duration histogram to: {save_path}")
    plt.close()

def plot_pitch_distribution(df, sample_size=50):
    """Generates the pitch distribution histogram from a sample of MIDI files."""
    print(f"Sampling {sample_size} files for pitch distribution...")
    
    # Randomly sample files to avoid processing all 1200+ files
    sample_files = df.sample(n=sample_size, random_state=42)['midi_filename'].tolist()
    
    pitch_counts = np.zeros(128)
    
    for midi_file in tqdm(sample_files, desc="Parsing MIDIs"):
        full_path = MAESTRO_DIR / midi_file
        try:
            midi_data = pretty_midi.PrettyMIDI(str(full_path))
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        pitch_counts[note.pitch] += 1
        except Exception:
            continue
            
    # MAESTRO is piano only, so we isolate the 88 piano keys (MIDI 21 to 108)
    piano_pitches = np.arange(21, 109)
    piano_counts = pitch_counts[21:109]
    
    plt.figure(figsize=(12, 6))
    plt.bar(piano_pitches, piano_counts, color='coral', edgecolor='black')
    
    plt.title('Pitch Distribution across 88 Piano Keys (MAESTRO Sample)')
    plt.xlabel('MIDI Pitch Number (21=A0, 60=Middle C, 108=C8)')
    plt.ylabel('Total Note Occurrences')
    plt.xticks(np.arange(21, 109, 5)) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = OUTPUT_DIR / "eda_pitch_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved pitch distribution to: {save_path}")
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.exists():
        print(f"Error: Could not find {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 1. Plot Durations (from CSV metadata)
    plot_durations(df)
    
    # 2. Plot Pitch Distribution (requires parsing actual MIDI files)
    plot_pitch_distribution(df)

if __name__ == "__main__":
    main()
