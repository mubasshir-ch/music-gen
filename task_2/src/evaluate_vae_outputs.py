from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated_vae"
REFERENCE_NPY = PROJECT_ROOT / "processed" / "validation.npy"

OUTPUT_CSV = PROJECT_ROOT / "outputs" / "vae_metrics.csv"

DURATION_QUANTIZATION = 0.05  # 50 ms


# =========================
# Pitch histogram
# =========================

def pitch_class_histogram_from_midi(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    counts = np.zeros(12, dtype=np.float64)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            pitch_class = note.pitch % 12
            counts[pitch_class] += 1

    total = counts.sum()

    if total == 0:
        return counts

    return counts / total


def pitch_class_histogram_from_pianoroll_npy(npy_path):
    data = np.load(npy_path)  # shape: (N, 128, 88)

    counts = np.zeros(12, dtype=np.float64)

    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21
        pitch_class = midi_pitch % 12

        counts[pitch_class] += data[:, :, pitch_idx].sum()

    total = counts.sum()

    if total == 0:
        return counts

    return counts / total


def pitch_histogram_similarity(reference_hist, generated_hist):
    return np.sum(np.abs(reference_hist - generated_hist))


# =========================
# Rhythm diversity
# =========================

def rhythm_diversity_score(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    durations = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            duration = note.end - note.start

            if duration > 0:
                quantized = round(duration / DURATION_QUANTIZATION) * DURATION_QUANTIZATION
                durations.append(quantized)

    if len(durations) == 0:
        return 0.0, 0, 0

    unique_durations = len(set(durations))
    total_notes = len(durations)

    score = unique_durations / total_notes

    return score, unique_durations, total_notes


# =========================
# Main
# =========================

def main():
    reference_hist = pitch_class_histogram_from_pianoroll_npy(REFERENCE_NPY)

    midi_files = sorted(
        list(GENERATED_DIR.glob("*.mid")) +
        list(GENERATED_DIR.glob("*.midi"))
    )

    if len(midi_files) == 0:
        raise RuntimeError(f"No MIDI files found in {GENERATED_DIR}")

    rows = []

    for midi_path in midi_files:
        generated_hist = pitch_class_histogram_from_midi(midi_path)

        pitch_similarity = pitch_histogram_similarity(
            reference_hist,
            generated_hist,
        )

        rhythm_score, unique_durations, total_notes = rhythm_diversity_score(midi_path)

        rows.append(
            {
                "file": midi_path.name,
                "pitch_histogram_similarity": pitch_similarity,
                "rhythm_diversity_score": rhythm_score,
                "unique_durations": unique_durations,
                "total_notes": total_notes,
            }
        )

    df = pd.DataFrame(rows)

    print("\nPer-file VAE metrics:")
    print(df)

    print("\nAverage VAE metrics:")
    print(df[["pitch_histogram_similarity", "rhythm_diversity_score", "total_notes"]].mean())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved metrics to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()