import numpy as np
from pathlib import Path
from src.utils import (
    get_pitch_distribution, 
    calc_pitch_histogram_similarity, 
    calc_rhythm_diversity, 
    calc_repetition_ratio, 
    verify_midi, 
    plot_perplexity
)

# Evaluation config
GENERATED_DIR = Path("task_3/outputs/generated_midis")
MARKOV_DIR = Path("task_3/outputs/baselines/markov")
RANDOM_DIR = Path("task_3/outputs/baselines/random")
PLOT_DIR = Path("task_3/outputs/plots")

# Reference MIDI for pitch distribution comparison
REFERENCE_MIDI = Path("task_3/datasets/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi") 

# Historic perplexity logs
EPOCHS = list(range(1, 11))
TRAIN_PPL = [54.99, 25.45, 22.55, 20.81, 19.63, 18.87, 18.30, 17.80, 17.35, 16.79]
VAL_PPL = [28.26, 24.09, 22.28, 21.14, 20.54, 20.10, 19.62, 19.27, 19.05, 18.32]

def evaluate_directory(dir_path, ref_dist, model_name):
    """
    Calculate metrics for directory.
    
    Args:
        dir_path: Directory with MIDI files.
        ref_dist: Reference pitch distribution.
        model_name: Model label.
        
    Returns:
        dict: Aggregate results.
    """
    midi_files = list(dir_path.glob("*.mid"))
    if not midi_files:
        print(f"No MIDI files found in {dir_path}")
        return None
        
    print(f"Evaluating {model_name} ({len(midi_files)} files)...")
    
    results = {
        "pitch_similarity": [],
        "rhythm_diversity": [],
        "repetition_ratio": []
    }
    
    for midi_file in midi_files:
        try:
            p_sim = calc_pitch_histogram_similarity(midi_file, ref_dist)
            r_div = calc_rhythm_diversity(midi_file)
            r_rep = calc_repetition_ratio(midi_file)
            
            results["pitch_similarity"].append(p_sim)
            results["rhythm_diversity"].append(r_div)
            results["repetition_ratio"].append(r_rep)
        except Exception:
            # Skip corrupted files
            continue
            
    if not results["pitch_similarity"]:
        return None

    return {
        "Pitch Similarity": np.mean(results["pitch_similarity"]),
        "Rhythm Diversity": np.mean(results["rhythm_diversity"]),
        "Repetition Ratio": np.mean(results["repetition_ratio"])
    }

def main():
    
    # Plot perplexity
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_perplexity(EPOCHS, TRAIN_PPL, VAL_PPL, PLOT_DIR / "task3_perplexity_curve.png")
    print(f"Performance plots generated in {PLOT_DIR}")

    # Verify MIDI files
    print("\n--- MIDI ARCHITECTURAL VERIFICATION ---")
    overall_success = True
    for directory in [GENERATED_DIR, MARKOV_DIR, RANDOM_DIR]:
        if not directory.exists():
            continue
        print(f"\nScanning: {directory.name}/")
        midi_files = list(directory.glob("*.mid"))
        for midi_file in midi_files:
            is_valid, reason = verify_midi(midi_file)
            status = "[PASS]" if is_valid else "[FAIL]"
            print(f"  {status} {midi_file.name:35} | Status: {reason}")
            if not is_valid:
                overall_success = False
    
    if overall_success:
        print("\nAll samples valid.")
    else:
        print("\nValidation failed.")

    # Compare metrics
    if not REFERENCE_MIDI.exists():
        print(f"ERROR: Reference MIDI not found at {REFERENCE_MIDI}")
        return
        
    print("\nGenerating comparative metric summary...")
    ref_dist = get_pitch_distribution(REFERENCE_MIDI)
    
    models = [
        (RANDOM_DIR, "Random Baseline"),
        (MARKOV_DIR, "Markov Baseline"),
        (GENERATED_DIR, "Transformer (Task 3)")
    ]
    
    summary = {}
    for dir_path, name in models:
        stats = evaluate_directory(dir_path, ref_dist, name)
        if stats:
            summary[name] = stats
            
    # Results summary
    print("\n" + "="*75)
    print(f"{'Model Architecture':<25} | {'Pitch Sim ↓':<12} | {'Rhythm Div ↑':<12} | {'Rep Ratio ↑':<12}")
    print("-" * 75)
    for name, stats in summary.items():
        print(f"{name:<25} | {stats['Pitch Similarity']:<12.3f} | {stats['Rhythm Diversity']:<12.3f} | {stats['Repetition Ratio']:<12.3f}")
    print("="*75)
    print("Note: Pitch Similarity (lower is better), Rhythm Diversity (higher is better), Repetition Ratio (higher is better).")

if __name__ == "__main__":
    main()
