import pretty_midi
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = Path("outputs/baselines/random")
NUM_SAMPLES = 10
WINDOW_SECONDS = 30.0  # Generate 30 seconds of "music"
NOTES_PER_SAMPLE = 150 # Approximate density

def generate_random_midi(filename):
    """Generates a MIDI file using purely uniform random distributions."""
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Allowed durations in seconds (e.g., 16th, 8th, quarter, half notes)
    durations = [0.125, 0.25, 0.5, 1.0]
    
    for _ in range(NOTES_PER_SAMPLE):
        # 1. Uniformly sample pitch from the 88-key range (MIDI 21 to 108)
        pitch = random.randint(21, 108)
        
        # 2. Random onset time within the window
        start_time = random.uniform(0, WINDOW_SECONDS)
        
        # 3. Random duration from the fixed set
        duration = random.choice(durations)
        end_time = start_time + duration
        
        # 4. Standard velocity
        velocity = random.randint(60, 100)
        
        note = pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start_time, end=end_time
        )
        piano.notes.append(note)
        
    midi.instruments.append(piano)
    midi.write(str(filename))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {NUM_SAMPLES} Random Baseline samples...")
    
    for i in tqdm(range(NUM_SAMPLES)):
        save_path = OUTPUT_DIR / f"Random_Baseline_{i+1}.mid"
        generate_random_midi(save_path)
        
    print(f"Done! Files saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
