import pretty_midi
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def get_pitch_distribution(midi_path):
    """
    Get pitch distribution (chroma).
    
    Octave-equivalent classes (0-11).

    Args:
        midi_path: Path to MIDI.

    Returns:
        np.ndarray: 12-dim vector of pitch class frequencies.
    """
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
    """
    Pitch histogram L1 distance.

    Args:
        gen_path: Path to generated MIDI.
        ref_dist: Reference pitch distribution.

    Returns:
        float: L1 distance.
    """
    gen_dist = get_pitch_distribution(gen_path)
    return np.sum(np.abs(ref_dist - gen_dist))

def calc_rhythm_diversity(midi_path):
    """
    Rhythm diversity ratio.

    Args:
        midi_path: Path to MIDI.

    Returns:
        float: unique_durations / total_notes.
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    durations = []
    
    for inst in midi_data.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                duration_sec = note.end - note.start
                # Quantize to 50ms
                quantized = round(duration_sec / 0.05) * 0.05
                durations.append(quantized)
                
    total_notes = len(durations)
    if total_notes == 0:
        return 0.0
        
    unique_durations = len(set(durations))
    return unique_durations / total_notes

def calc_repetition_ratio(midi_path):
    """
    Repetition ratio (4-grams).

    Args:
        midi_path: Path to MIDI.

    Returns:
        float: Repetition ratio.
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    
    for inst in midi_data.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
            
    notes.sort(key=lambda x: x.start)
    pitches = [n.pitch for n in notes]
    
    if len(pitches) < 4:
        return 0.0
        
    # Extract overlapping 4-note patterns
    n_grams = [tuple(pitches[i:i+4]) for i in range(len(pitches)-3)]
    counts = Counter(n_grams)
    repeated_patterns = sum(1 for count in counts.values() if count > 1)
    
    total_patterns = len(n_grams)
    return repeated_patterns / total_patterns

def verify_midi(midi_path):
    """
    Validate MIDI structure.

    Returns:
        tuple: (is_valid, reason).
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        if total_notes < 50:
            return False, f"Insufficient notes ({total_notes})"
            
        duration = midi_data.get_end_time()
        if duration < 5.0:
            return False, f"Clip too short ({duration:.2f}s)"
            
        return True, "Valid"
    except Exception as e:
        return False, f"Load Error: {e}"

def plot_perplexity(epochs, train_ppl, val_ppl, save_path):
    """
    Plot perplexity.

    Args:
        epochs: list of epochs.
        train_ppl: training perplexity.
        val_ppl: validation perplexity.
        save_path: save path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ppl, label='Training Perplexity', color='#1f77b4', linewidth=2, marker='o')
    plt.plot(epochs, val_ppl, label='Validation Perplexity', color='#ff7f0e', linewidth=2, linestyle='--', marker='s')
    
    plt.title('Sequence Modeling Performance: Perplexity Trends', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Perplexity (exp(Loss))', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(frameon=True, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
