# metrics.py
# Task 1 — Quantitative Evaluation
# Course: CSE425/EEE474 Neural Networks
#
# Implements the three evaluation metrics defined in the project spec,
# compares the LSTM Autoencoder against two baselines (Random Generator
# and Markov Chain), and saves publication-ready plots.
#
# Run AFTER generate_music.py has produced the 5 MIDI files.
#
# Metrics (from spec):
#   1. Pitch Histogram Similarity  H(p,q) = sum_i |pi - qi|
#   2. Rhythm Diversity Score      D = #unique_durations / #total_notes
#   3. Repetition Ratio            R = #repeated_patterns / #total_patterns

import os
import sys
import glob
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = r"D:\neural network\music-generation-unsupervised"
TEST_DATA = os.path.join(BASE_DIR, "data", "processed", "test.npy")
GEN_DIR   = os.path.join(BASE_DIR, "outputs", "generated_midis", "task1")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

FS = 16  # must match generate_music.py


# ══════════════════════════════════════════════════════════════════════════════
# Metric 1 — Pitch Histogram Similarity
# H(p, q) = sum_{i=1}^{12} |pi - qi|
# Measures how closely the generated pitch class distribution matches real data.
# Range: [0, 2].  Lower is better (0 = identical distribution).
# ══════════════════════════════════════════════════════════════════════════════

def pitch_class_histogram(midi_obj):
    """
    Compute a normalised 12-bin pitch class histogram over all notes.
    Pitch class collapses octaves: C4 and C5 both count as class 0.
    """
    counts = np.zeros(12)
    for inst in midi_obj.instruments:
        for note in inst.notes:
            counts[note.pitch % 12] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def pitch_histogram_similarity(midi_gen, midi_ref):
    """
    H(p, q) = sum_i |pi - qi|
    Lower = more similar pitch distribution to the reference.
    """
    p = pitch_class_histogram(midi_gen)
    q = pitch_class_histogram(midi_ref)
    return float(np.sum(np.abs(p - q)))


# ══════════════════════════════════════════════════════════════════════════════
# Metric 2 — Rhythm Diversity Score
# D_rhythm = #unique_durations / #total_notes
# Measures how varied the note durations are across a composition.
# Range: [0, 1].  Higher is better (1 = every note has a unique duration).
# ══════════════════════════════════════════════════════════════════════════════

def rhythm_diversity(midi_obj, resolution=0.05):
    """
    D_rhythm = #unique_durations / #total_notes

    Durations are rounded to `resolution` seconds so that near-identical
    note lengths (e.g. 0.312 s and 0.313 s) are treated as the same bucket.
    Without rounding, minor floating-point differences inflate the score.
    """
    durations = []
    for inst in midi_obj.instruments:
        for note in inst.notes:
            dur = round((note.end - note.start) / resolution) * resolution
            durations.append(dur)
    if not durations:
        return 0.0
    return len(set(durations)) / len(durations)


# ══════════════════════════════════════════════════════════════════════════════
# Metric 3 — Repetition Ratio
# R = #repeated_patterns / #total_patterns
# Measures how often the same melodic pattern reappears.
# Range: [0, 1].  Lower is better (0 = no repeated patterns).
# ══════════════════════════════════════════════════════════════════════════════

def repetition_ratio(midi_obj, pattern_len=4):
    """
    R = #repeated_patterns / #total_patterns

    A pattern is a tuple of `pattern_len` consecutive pitch values in
    chronological order. A pattern is counted as repeated the second (and
    subsequent) time it appears — the first occurrence is not penalised.
    """
    pitches = []
    for inst in midi_obj.instruments:
        for note in sorted(inst.notes, key=lambda n: n.start):
            pitches.append(note.pitch)

    if len(pitches) < pattern_len + 1:
        return 0.0

    patterns = [tuple(pitches[i:i + pattern_len])
                for i in range(len(pitches) - pattern_len + 1)]

    seen, repeated = set(), 0
    for p in patterns:
        if p in seen:
            repeated += 1
        seen.add(p)

    return repeated / len(patterns)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 1 — Random Note Generator
# Generates notes at uniform random pitches and durations with no structure.
# Serves as the lower bound for all three metrics.
# ══════════════════════════════════════════════════════════════════════════════

def make_random_midi(duration_sec=90, notes_per_sec=2.0,
                     pitch_min=48, pitch_max=83, tempo=90):
    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    t     = 0.0
    while t < duration_sec:
        pitch    = np.random.randint(pitch_min, pitch_max + 1)
        dur      = np.random.uniform(0.25, 1.5)
        end      = min(t + dur, duration_sec)
        vel      = np.random.randint(55, 90)
        piano.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=t, end=end))
        t += 1.0 / notes_per_sec + np.random.uniform(-0.1, 0.1)
        t  = max(t, 0.01)
    midi.instruments.append(piano)
    return midi


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 2 — Markov Chain Music Generator
# Learns pitch transition probabilities from real training data and samples
# from them to generate sequences. Captures local pitch statistics but has
# no notion of rhythm, harmony, or long-range structure.
# ══════════════════════════════════════════════════════════════════════════════

def build_markov_model(npy_path, order=2):
    """
    Build an order-2 Markov transition table from real piano roll data.
    State: last `order` active pitches.
    Transition: probability distribution over the next pitch.
    """
    data = np.transpose(np.load(npy_path), (0, 2, 1))  # (N, 64, 128)
    transitions = {}

    for segment in data:
        seq = []
        for t in range(segment.shape[0]):
            active = np.where(segment[t] > 0.5)[0]
            seq.append(int(active[np.argmax(segment[t, active])])
                       if len(active) > 0 else None)

        for i in range(len(seq) - order):
            state = tuple(seq[i:i + order])
            nxt   = seq[i + order]
            if None not in state and nxt is not None:
                transitions.setdefault(state, {})
                transitions[state][nxt] = transitions[state].get(nxt, 0) + 1

    # Convert counts to probabilities
    model = {}
    for state, nexts in transitions.items():
        total = sum(nexts.values())
        model[state] = {k: v / total for k, v in nexts.items()}
    return model


def make_markov_midi(model, duration_sec=90, tempo=90,
                     pitch_min=48, pitch_max=83,
                     note_dur=0.375, order=2):
    """Sample from the Markov model to produce a fixed-duration MIDI."""
    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    state = tuple(np.random.randint(pitch_min, pitch_max + 1, size=order))
    t     = 0.0

    while t < duration_sec:
        if state in model:
            choices = list(model[state].keys())
            probs   = list(model[state].values())
            pitch   = int(np.random.choice(choices, p=probs))
        else:
            pitch = int(np.random.randint(pitch_min, pitch_max + 1))

        pitch = int(np.clip(pitch, pitch_min, pitch_max))
        end   = min(t + note_dur, duration_sec)
        piano.notes.append(
            pretty_midi.Note(velocity=np.random.randint(60, 85),
                             pitch=pitch, start=t, end=end))
        state = state[1:] + (pitch,)
        t    += note_dur

    midi.instruments.append(piano)
    return midi


# ══════════════════════════════════════════════════════════════════════════════
# Reference MIDI from real test segments
# ══════════════════════════════════════════════════════════════════════════════

def build_reference_midi(npy_path, n_samples=20, tempo=90, fs=16):
    """
    Reconstruct a PrettyMIDI object from n_samples real test piano rolls.
    Used as the reference distribution for pitch histogram similarity.
    """
    data = np.transpose(np.load(npy_path), (0, 2, 1))  # (N, 64, 128)
    idx  = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    spb  = 1.0 / fs

    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    for seg_i, seg in enumerate(data[idx]):
        offset = seg_i * seg.shape[0] * spb
        for pitch in range(128):
            note_on = None
            for t in range(seg.shape[0]):
                if seg[t, pitch] > 0.5 and note_on is None:
                    note_on = t
                elif seg[t, pitch] <= 0.5 and note_on is not None:
                    piano.notes.append(pretty_midi.Note(
                        velocity=75, pitch=pitch,
                        start=offset + note_on * spb,
                        end=offset + t * spb))
                    note_on = None

    midi.instruments.append(piano)
    return midi


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

def evaluate():
    print("=" * 70)
    print("TASK 1  —  QUANTITATIVE EVALUATION")
    print("=" * 70)

    gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "task1_long_morph_*.mid")))
    if not gen_files:
        print(f"\nERROR: No generated MIDI files found in:\n  {GEN_DIR}")
        print("Run generate_music.py first, then re-run this script.")
        return None, None

    print(f"\nFound {len(gen_files)} generated MIDI files.")
    print("Building reference from real test data...")
    ref_midi = build_reference_midi(TEST_DATA, n_samples=20)

    print("Generating Random baseline...")
    rand_midi = make_random_midi(duration_sec=90)

    print("Training Markov model on test data...")
    markov_model = build_markov_model(TEST_DATA, order=2)
    markov_midi  = make_markov_midi(markov_model, duration_sec=90)

    # Compute metrics for each model
    results = {}

    results["Random Generator"] = {
        "pitch_sim"  : pitch_histogram_similarity(rand_midi,   ref_midi),
        "rhythm_div" : rhythm_diversity(rand_midi),
        "rep_ratio"  : repetition_ratio(rand_midi),
    }

    results["Markov Chain"] = {
        "pitch_sim"  : pitch_histogram_similarity(markov_midi, ref_midi),
        "rhythm_div" : rhythm_diversity(markov_midi),
        "rep_ratio"  : repetition_ratio(markov_midi),
    }

    per_track = {"pitch_sim": [], "rhythm_div": [], "rep_ratio": []}
    for f in gen_files:
        m = pretty_midi.PrettyMIDI(f)
        per_track["pitch_sim"].append(pitch_histogram_similarity(m, ref_midi))
        per_track["rhythm_div"].append(rhythm_diversity(m))
        per_track["rep_ratio"].append(repetition_ratio(m))

    results["LSTM Autoencoder"] = {
        "pitch_sim"  : float(np.mean(per_track["pitch_sim"])),
        "rhythm_div" : float(np.mean(per_track["rhythm_div"])),
        "rep_ratio"  : float(np.mean(per_track["rep_ratio"])),
    }

    # Print summary table
    print()
    print(f"{'Model':<25} {'Pitch Sim ↓':>12} {'Rhythm Div ↑':>14} {'Repetition ↓':>14}")
    print("-" * 67)
    for name, m in results.items():
        print(f"{name:<25} {m['pitch_sim']:>12.4f} "
              f"{m['rhythm_div']:>14.4f} {m['rep_ratio']:>14.4f}")
    print("-" * 67)

    print("\nNotes:")
    print("  Pitch Sim  ↓  Lower is better  (0 = matches real music exactly)")
    print("  Rhythm Div ↑  Higher is better (more varied note durations)")
    print("  Repetition ↓  Lower is better  (less repetitive patterns)\n")

    print("Per-track breakdown:")
    print(f"  {'File':<35} {'Pitch Sim':>10} {'Rhythm Div':>12} {'Repetition':>12}")
    for i, f in enumerate(gen_files):
        name = os.path.basename(f)
        print(f"  {name:<35} {per_track['pitch_sim'][i]:>10.4f} "
              f"{per_track['rhythm_div'][i]:>12.4f} "
              f"{per_track['rep_ratio'][i]:>12.4f}")

    return results, per_track


# ══════════════════════════════════════════════════════════════════════════════
# Plot generation
# ══════════════════════════════════════════════════════════════════════════════

def save_plots(results):
    model_names = list(results.keys())
    colors      = ["#e74c3c", "#f39c12", "#2ecc71"]

    # ── Bar chart: all three metrics side by side ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Task 1 — Evaluation Metrics: LSTM Autoencoder vs Baselines",
                 fontsize=13, fontweight="bold", y=1.01)

    metric_info = [
        ("pitch_sim",  "Pitch Histogram Similarity\n(↓ lower = more realistic)"),
        ("rhythm_div", "Rhythm Diversity Score\n(↑ higher = more varied)"),
        ("rep_ratio",  "Repetition Ratio\n(↓ lower = more creative)"),
    ]

    for ax, (key, title) in zip(axes, metric_info):
        vals = [results[n][key] for n in model_names]
        bars = ax.bar(range(len(model_names)), vals,
                      color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names],
                           fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "task1_evaluation_metrics.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Evaluation bar chart  → {p1}")

    # ── Pitch class histogram comparison ─────────────────────────────────────
    gen_files = sorted(glob.glob(os.path.join(GEN_DIR, "task1_long_morph_*.mid")))
    ref_midi  = build_reference_midi(TEST_DATA, n_samples=20)
    markov_m  = build_markov_model(TEST_DATA, order=2)
    markov_md = make_markov_midi(markov_m, duration_sec=90)

    ref_hist    = pitch_class_histogram(ref_midi)
    markov_hist = pitch_class_histogram(markov_md)
    gen_hists   = [pitch_class_histogram(pretty_midi.PrettyMIDI(f))
                   for f in gen_files]
    gen_avg     = np.mean(gen_hists, axis=0)

    pitch_names = ["C", "C#", "D", "D#", "E", "F",
                   "F#", "G", "G#", "A", "A#", "B"]
    x, w = np.arange(12), 0.25

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    ax2.bar(x - w, ref_hist,    w, label="Real Data",       color="#3498db", alpha=0.85)
    ax2.bar(x,     gen_avg,     w, label="LSTM AE (ours)",  color="#2ecc71", alpha=0.85)
    ax2.bar(x + w, markov_hist, w, label="Markov Chain",    color="#f39c12", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pitch_names)
    ax2.set_xlabel("Pitch Class", fontsize=11)
    ax2.set_ylabel("Relative Frequency", fontsize=11)
    ax2.set_title("Pitch Class Distribution — Real Data vs Generated vs Markov",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    p2 = os.path.join(PLOTS_DIR, "task1_pitch_histogram.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pitch class histogram → {p2}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, per_track = evaluate()
    if results:
        save_plots(results)
        print(f"\nAll plots saved to: {PLOTS_DIR}")
        print("Evaluation complete.")
