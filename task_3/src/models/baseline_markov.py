import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIGURATION ---
MAESTRO_DIR = Path("datasets/maestro-v3.0.0")
CSV_PATH = MAESTRO_DIR / "maestro-v3.0.0.csv"
OUTPUT_DIR = Path("outputs/baselines/markov")
NUM_SAMPLES = 10
MAX_NOTES = 200 # Length of generated sequence

def build_transition_matrix_and_durations(df, sample_size=30):
    """Parses a subset of MAESTRO to build the pitch transition matrix and collect durations."""
    print(f"Building Markov Matrix from {sample_size} MAESTRO files...")
    
    # 128x128 matrix to hold transition counts for all MIDI pitches
    transition_counts = np.zeros((128, 128))
    empirical_durations = []
    
    # We only need a small sample of the training set to build a solid transition matrix
    train_files = df[df['split'] == 'train']['midi_filename'].tolist()
    sample_files = random.sample(train_files, sample_size)
    
    for midi_file in tqdm(sample_files, desc="Analyzing transitions"):
        full_path = MAESTRO_DIR / midi_file
        try:
            midi_data = pretty_midi.PrettyMIDI(str(full_path))
            for inst in midi_data.instruments:
                if not inst.is_drum:
                    # Sort notes by start time to get sequential melodic transitions
                    notes = sorted(inst.notes, key=lambda n: n.start)
                    for i in range(len(notes) - 1):
                        current_pitch = notes[i].pitch
                        next_pitch = notes[i+1].pitch
                        transition_counts[current_pitch, next_pitch] += 1
                        
                        duration = notes[i].end - notes[i].start
                        empirical_durations.append(duration)
        except Exception:
            continue
            
    # Normalize the matrix to get probabilities (Laplace smoothing applied to avoid division by zero)
    transition_probs = (transition_counts + 1e-8) / (transition_counts.sum(axis=1, keepdims=True) + 128 * 1e-8)
    
    return transition_probs, empirical_durations

def generate_markov_midi(filename, transition_probs, durations):
    """Generates a MIDI sequence by sampling from the Markov transition matrix."""
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Start on Middle C (60)
    current_pitch = 60
    current_time = 0.0
    
    for _ in range(MAX_NOTES):
        # Sample duration from the empirical distribution
        duration = random.choice(durations)
        
        note = pretty_midi.Note(
            velocity=80, 
            pitch=current_pitch, 
            start=current_time, 
            end=current_time + duration
        )
        piano.notes.append(note)
        
        # Advance time. To make it sound like a melody, we assume the next note starts when this one ends.
        current_time += duration
        
        # Sample next pitch based on the probabilities in the current pitch's row
        row_probs = transition_probs[current_pitch]
        current_pitch = np.random.choice(np.arange(128), p=row_probs)
        
    midi.instruments.append(piano)
    midi.write(str(filename))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.exists():
        print(f"Error: Could not find MAESTRO CSV at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 1. Train the Markov Model
    transition_probs, empirical_durations = build_transition_matrix_and_durations(df)
    
    # 2. Generate outputs
    print(f"\nGenerating {NUM_SAMPLES} Markov Baseline samples...")
    for i in tqdm(range(NUM_SAMPLES)):
        save_path = OUTPUT_DIR / f"Markov_Baseline_{i+1}.mid"
        generate_markov_midi(save_path, transition_probs, empirical_durations)
        
    print(f"Done! Files saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
