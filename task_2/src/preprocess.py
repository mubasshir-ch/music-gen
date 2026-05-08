from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm


# =========================
# Config
# =========================

DATA_DIR = Path("data/maestro-v3.0.0")
CSV_PATH = DATA_DIR / "maestro-v3.0.0.csv"

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

FS = 16                  # frames per second
WINDOW_SIZE = 128        # 128 time steps = 8 seconds at fs=16
MIN_ACTIVE_RATIO = 0.02  # discard almost-silent windows

PIANO_LOW = 21           # A0
PIANO_HIGH = 109         # Python slicing excludes 109, so keeps 21..108


# =========================
# MIDI -> Piano Roll
# =========================

def midi_to_binary_pianoroll(midi_path: Path) -> np.ndarray:
    """
    Converts one MIDI file into a binary piano-roll of shape (T, 88).
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # pretty_midi gives shape (128, T), where 128 = MIDI pitches 0..127
    roll = midi.get_piano_roll(fs=FS)

    # Keep only piano keys 21..108
    roll = roll[PIANO_LOW:PIANO_HIGH, :]  # shape: (88, T)

    # Transpose so time comes first
    roll = roll.T  # shape: (T, 88)

    # Binarize: any active velocity becomes 1
    roll = (roll > 0).astype(np.float32)

    return roll


# =========================
# Piano Roll -> Windows
# =========================

def make_windows(roll: np.ndarray) -> list[np.ndarray]:
    """
    Splits a piano-roll of shape (T, 88) into windows of shape (128, 88).
    Filters out very sparse windows.
    """
    windows = []

    total_steps = roll.shape[0]
    usable_steps = total_steps - (total_steps % WINDOW_SIZE)

    for start in range(0, usable_steps, WINDOW_SIZE):
        window = roll[start:start + WINDOW_SIZE]

        active_ratio = window.mean()

        if active_ratio >= MIN_ACTIVE_RATIO:
            windows.append(window)

    return windows


# =========================
# Process one split
# =========================

def process_split(df: pd.DataFrame, split_name: str) -> np.ndarray:
    """
    Processes train/validation/test split separately.
    """
    split_df = df[df["split"] == split_name]

    all_windows = []
    skipped = 0

    print(f"\nProcessing split: {split_name}")
    print(f"Number of MIDI files: {len(split_df)}")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        midi_relative_path = row["midi_filename"]
        midi_path = DATA_DIR / midi_relative_path

        try:
            roll = midi_to_binary_pianoroll(midi_path)
            windows = make_windows(roll)
            all_windows.extend(windows)

        except Exception as e:
            skipped += 1
            print(f"Skipped: {midi_path}")
            print(f"Reason: {e}")

    if len(all_windows) == 0:
        raise RuntimeError(f"No windows created for split: {split_name}")

    data = np.stack(all_windows).astype(np.float32)

    print(f"Created windows: {data.shape}")
    print(f"Skipped files: {skipped}")

    return data


# =========================
# Main
# =========================

def main():
    df = pd.read_csv(CSV_PATH)

    print("CSV columns:")
    print(df.columns.tolist())

    for split_name in ["train", "validation", "test"]:
        data = process_split(df, split_name)

        output_path = OUTPUT_DIR / f"{split_name}.npy"
        np.save(output_path, data)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()