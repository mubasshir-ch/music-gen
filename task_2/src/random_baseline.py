from pathlib import Path

import numpy as np
import pretty_midi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_NPY = PROJECT_ROOT / "processed" / "train.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "random_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 5
NUM_NOTES = 200

PITCH_LOW = 21
PITCH_HIGH = 108

VELOCITY = 80

DURATIONS = [
    0.125,
    0.25,
    0.5,
    1.0,
]

MAX_GAP = 0.20

DURATION_QUANTIZATION = 0.05


# =========================
# Metrics
# =========================

def reference_pitch_histogram_from_training(data):
    counts = np.zeros(12)

    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21
        pitch_class = midi_pitch % 12

        counts[pitch_class] += data[:, :, pitch_idx].sum()

    return counts / counts.sum()


def pitch_histogram_similarity(midi_path, reference_hist):
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    counts = np.zeros(12)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            counts[note.pitch % 12] += 1

    if counts.sum() == 0:
        return None

    generated_hist = counts / counts.sum()

    return np.sum(np.abs(reference_hist - generated_hist))


def rhythm_diversity_score(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    durations = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            duration = note.end - note.start

            quantized = (
                round(duration / DURATION_QUANTIZATION)
                * DURATION_QUANTIZATION
            )

            durations.append(quantized)

    if len(durations) == 0:
        return 0.0

    return len(set(durations)) / len(durations)


# =========================
# Random MIDI generation
# =========================

def generate_random_midi(output_path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    current_time = 0.0

    for _ in range(NUM_NOTES):
        pitch = np.random.randint(PITCH_LOW, PITCH_HIGH + 1)

        duration = np.random.choice(DURATIONS)

        gap = np.random.uniform(0.0, MAX_GAP)

        start = current_time + gap
        end = start + duration

        note = pretty_midi.Note(
            velocity=VELOCITY,
            pitch=int(pitch),
            start=float(start),
            end=float(end),
        )

        piano.notes.append(note)

        current_time = start

    midi.instruments.append(piano)
    midi.write(str(output_path))


# =========================
# Main
# =========================

def main():
    print("Loading training reference distribution...")

    train_data = np.load(TRAIN_NPY)

    reference_hist = reference_pitch_histogram_from_training(train_data)

    rows = []

    print("\nGenerating random baseline MIDI files...")

    for i in range(NUM_SAMPLES):
        output_path = OUTPUT_DIR / f"random_sample_{i + 1}.mid"

        generate_random_midi(output_path)

        pitch_score = pitch_histogram_similarity(
            output_path,
            reference_hist,
        )

        rhythm_score = rhythm_diversity_score(output_path)

        rows.append((pitch_score, rhythm_score))

        print(f"\nSaved: {output_path}")
        print(f"Pitch histogram similarity: {pitch_score:.6f}")
        print(f"Rhythm diversity score:     {rhythm_score:.6f}")

    avg_pitch = np.mean([r[0] for r in rows])
    avg_rhythm = np.mean([r[1] for r in rows])

    print("\n========================")
    print("Average Random Baseline Metrics")
    print("========================")

    print(f"Average pitch histogram similarity: {avg_pitch:.6f}")
    print(f"Average rhythm diversity score:     {avg_rhythm:.6f}")


if __name__ == "__main__":
    main()