from pathlib import Path
import numpy as np
import pretty_midi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_NPY = PROJECT_ROOT / "processed" / "train.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "markov_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 16
NUM_SAMPLES = 5
NUM_NOTES = 200
VELOCITY = 80

PITCH_LOW = 21
NUM_PITCHES = 88

DURATIONS = [0.125, 0.25, 0.5, 1.0]


def extract_pitch_sequence(data):
    """
    data shape: (N, 128, 88)
    Converts piano-roll windows into a pitch sequence.
    At each timestep, if multiple notes exist, choose one randomly.
    """
    sequence = []

    for window in data:
        for t in range(window.shape[0]):
            active = np.where(window[t] == 1)[0]

            if len(active) > 0:
                pitch_idx = np.random.choice(active)
                midi_pitch = pitch_idx + PITCH_LOW
                sequence.append(midi_pitch)

    return sequence


def build_transition_matrix(pitch_sequence):
    """
    Builds first-order Markov transition matrix.
    matrix[i, j] = probability of pitch j after pitch i
    """
    counts = np.ones((NUM_PITCHES, NUM_PITCHES), dtype=np.float64)  # smoothing

    for a, b in zip(pitch_sequence[:-1], pitch_sequence[1:]):
        i = a - PITCH_LOW
        j = b - PITCH_LOW

        if 0 <= i < NUM_PITCHES and 0 <= j < NUM_PITCHES:
            counts[i, j] += 1

    probs = counts / counts.sum(axis=1, keepdims=True)

    return probs


def generate_pitch_sequence(transition_matrix, length=200):
    current = np.random.randint(PITCH_LOW, PITCH_LOW + NUM_PITCHES)
    generated = [current]

    for _ in range(length - 1):
        current_idx = current - PITCH_LOW

        next_idx = np.random.choice(
            np.arange(NUM_PITCHES),
            p=transition_matrix[current_idx],
        )

        current = next_idx + PITCH_LOW
        generated.append(current)

    return generated


def pitch_sequence_to_midi(pitch_sequence, output_path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    current_time = 0.0

    for pitch in pitch_sequence:
        duration = np.random.choice(DURATIONS)

        note = pretty_midi.Note(
            velocity=VELOCITY,
            pitch=int(pitch),
            start=current_time,
            end=current_time + duration,
        )

        piano.notes.append(note)
        current_time += duration

    midi.instruments.append(piano)
    midi.write(str(output_path))


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
            duration = round(duration / 0.05) * 0.05
            durations.append(duration)

    if len(durations) == 0:
        return 0.0

    return len(set(durations)) / len(durations)


def reference_pitch_histogram_from_training(data):
    counts = np.zeros(12)

    for pitch_idx in range(NUM_PITCHES):
        midi_pitch = pitch_idx + PITCH_LOW
        pitch_class = midi_pitch % 12

        counts[pitch_class] += data[:, :, pitch_idx].sum()

    return counts / counts.sum()


def main():
    print("Loading training data...")
    data = np.load(TRAIN_NPY)

    print("Training data shape:", data.shape)

    print("Extracting pitch sequence...")
    pitch_sequence = extract_pitch_sequence(data)

    print("Total extracted notes:", len(pitch_sequence))

    print("Building Markov transition matrix...")
    transition_matrix = build_transition_matrix(pitch_sequence)

    reference_hist = reference_pitch_histogram_from_training(data)

    print("\nGenerating Markov MIDI samples...")

    rows = []

    for i in range(NUM_SAMPLES):
        generated_sequence = generate_pitch_sequence(
            transition_matrix,
            length=NUM_NOTES,
        )

        output_path = OUTPUT_DIR / f"markov_sample_{i + 1}.mid"

        pitch_sequence_to_midi(generated_sequence, output_path)

        pitch_score = pitch_histogram_similarity(output_path, reference_hist)
        rhythm_score = rhythm_diversity_score(output_path)

        rows.append((output_path.name, pitch_score, rhythm_score))

        print(f"Saved: {output_path}")
        print(f"  Pitch histogram similarity: {pitch_score:.4f}")
        print(f"  Rhythm diversity score:     {rhythm_score:.4f}")

    print("\nAverage Markov metrics:")

    avg_pitch = np.mean([r[1] for r in rows])
    avg_rhythm = np.mean([r[2] for r in rows])

    print(f"Average pitch histogram similarity: {avg_pitch:.4f}")
    print(f"Average rhythm diversity score:     {avg_rhythm:.4f}")


if __name__ == "__main__":
    main()