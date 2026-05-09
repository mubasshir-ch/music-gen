import pretty_midi
import numpy as np
import glob
from pathlib import Path
from collections import Counter

# --- CONFIGURATION ---
GENERATED_DIR = Path("outputs/generated_midis")
# We need a reference file from MAESTRO to calculate Pitch Histogram Similarity.
# Make sure this path points to any valid file in your extracted dataset.
REFERENCE_MIDI = Path("datasets/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi") 

def get_pitch_distribution(midi_path):
    """Maps every note to its pitch class (0-11) and returns a normalized distribution."""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    counts = np.zeros(12)
    total_notes = 0
    
    for inst in midi_data.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                counts[note.pitch % 12] += 1
                total_notes += 1
                
    if total_notes == 0:
        return counts
    return counts / total_notes

def calc_pitch_histogram_similarity(gen_path, ref_dist):
    """Computes L1 distance between generated and reference pitch distributions."""
    gen_dist = get_pitch_distribution(gen_path)
    # L1 distance formula: sum(|p_i - q_i|)
    return np.sum(np.abs(ref_dist - gen_dist))

def calc_rhythm_diversity(midi_path):
    """Calculates distinct quantized durations / total notes."""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    durations = []
    
    for inst in midi_data.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                duration_sec = note.end - note.start
                # Quantize to nearest 50ms to avoid floating-point noise
                quantized = round(duration_sec / 0.05) * 0.05
                durations.append(quantized)
                
    total_notes = len(durations)
    if total_notes == 0:
        return 0.0
        
    unique_durations = len(set(durations))
    return unique_durations / total_notes

def calc_repetition_ratio(midi_path):
    """Calculates the ratio of overlapping 4-note patterns that appear more than once."""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    
    for inst in midi_data.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
            
    # Sort notes by start time (onset)
    notes.sort(key=lambda x: x.start)
    pitches = [n.pitch for n in notes]
    
    if len(pitches) < 4:
        return 0.0
        
    # Extract all overlapping 4-grams
    n_grams = [tuple(pitches[i:i+4]) for i in range(len(pitches)-3)]
    
    # Count how many n-grams appear > 1 time
    counts = Counter(n_grams)
    repeated_patterns = sum(1 for count in counts.values() if count > 1)
    
    total_patterns = len(n_grams)
    return repeated_patterns / total_patterns

def main():
    if not REFERENCE_MIDI.exists():
        print(f"ERROR: Could not find reference MIDI at {REFERENCE_MIDI}")
        print("Please update the path to point to a valid MAESTRO file.")
        return
        
    print("Computing reference pitch distribution...")
    ref_dist = get_pitch_distribution(REFERENCE_MIDI)
    
    midi_files = list(GENERATED_DIR.glob("*.mid"))
    if not midi_files:
        print(f"No MIDI files found in {GENERATED_DIR}")
        return
        
    print(f"Evaluating {len(midi_files)} generated files...\n")
    
    metrics = {
        "pitch_similarity": [],
        "rhythm_diversity": [],
        "repetition_ratio": []
    }
    
    for midi_file in midi_files:
        try:
            p_sim = calc_pitch_histogram_similarity(midi_file, ref_dist)
            r_div = calc_rhythm_diversity(midi_file)
            r_rep = calc_repetition_ratio(midi_file)
            
            metrics["pitch_similarity"].append(p_sim)
            metrics["rhythm_diversity"].append(r_div)
            metrics["repetition_ratio"].append(r_rep)
            
            print(f"File: {midi_file.name}")
            print(f"  Pitch Similarity: {p_sim:.3f} (Lower is better)")
            print(f"  Rhythm Diversity: {r_div:.3f} (Higher is better)")
            print(f"  Repetition Ratio: {r_rep:.3f} (Ideal: 0.1 - 0.5)")
            print("-" * 40)
        except Exception as e:
            print(f"Skipping {midi_file.name} due to error: {e}")
            
    print("\n--- FINAL AVERAGES (Task 3 Transformer) ---")
    print(f"Pitch Histogram Similarity: {np.mean(metrics['pitch_similarity']):.3f}")
    print(f"Rhythm Diversity Score:     {np.mean(metrics['rhythm_diversity']):.3f}")
    print(f"Repetition Ratio:           {np.mean(metrics['repetition_ratio']):.3f}")

if __name__ == "__main__":
    main()
