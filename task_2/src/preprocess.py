from pathlib import Path
import random

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

# this file is responsible for preprocessing the raw MIDI files from the MAESTRO and Lakh datasets into binary piano roll windows 
# that can be used for training the VAE model.
# It defines functions to convert MIDI files to piano rolls, split them into windows, and save


# =============== Config =========================

MAESTRO_DIR = Path("data/maestro-v3.0.0")
MAESTRO_CSV_PATH = MAESTRO_DIR / "maestro-v3.0.0.csv"

LAKH_DIR = Path("data/archive")

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

FS = 16                 # sampling frequency for the piano rolls (16 time steps per second, i.e., 64th notes). This means each time step in the piano roll corresponds to 1/16th of a second in the original MIDI file.
WINDOW_SIZE = 128       # number of time steps in each piano roll window (128 time steps = 8 seconds at FS=16). The piano rolls will be split into non-overlapping windows of this size for training the VAE model.

MIN_ACTIVE_RATIO = 0.02 # minimum active ratio for a piano roll window to be included in the dataset. 
MAX_ACTIVE_RATIO = 0.20

PIANO_LOW = 21  
PIANO_HIGH = 109

USE_LAKH = True
MAX_LAKH_FILES = 300
LAKH_TRAIN_RATIO = 0.85
RANDOM_SEED = 42

# Converts one MIDI file into binary piano-roll shape (T, 88).
# For Lakh MIDI, drum tracks are ignored.

def midi_to_binary_pianoroll(midi_path: Path) -> np.ndarray | None:
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    if midi.get_end_time() < 5:     # skip very short files
        return None

    rolls = []

    for instrument in midi.instruments:
        if instrument.is_drum:      # skip drum tracks
            continue

        if len(instrument.notes) == 0:
            continue

        roll = instrument.get_piano_roll(fs=FS)
        rolls.append(roll)

    if len(rolls) == 0:
        return None

    max_len = max(r.shape[1] for r in rolls)

    merged_roll = np.zeros((128, max_len), dtype=np.float32)    # merge all tracks by summing their piano rolls (overlapping notes will have higher values)

    for r in rolls:
        merged_roll[:, :r.shape[1]] += r

    roll = merged_roll 

    roll = roll[PIANO_LOW:PIANO_HIGH, :]    # keep only piano range (88 keys)
    roll = roll.T
    roll = (roll > 0).astype(np.float32)    # binarize: any active note becomes 1, rest are 0

    return roll

# Splits a piano roll into non-overlapping windows of shape (WINDOW_SIZE, 88).
# Windows that are too empty or too dense (based on active ratio) are discarded.

def make_windows(roll: np.ndarray) -> list[np.ndarray]:
    windows = []

    total_steps = roll.shape[0]
    usable_steps = total_steps - (total_steps % WINDOW_SIZE)

    for start in range(0, usable_steps, WINDOW_SIZE):
        window = roll[start:start + WINDOW_SIZE]    # shape (WINDOW_SIZE, 88)

        active_ratio = window.mean()                # calculate active ratio: proportion of active notes in the window

        if MIN_ACTIVE_RATIO <= active_ratio <= MAX_ACTIVE_RATIO:    # only keep windows that are not too empty or too dense
            windows.append(window)

    return windows

# Processes a specific split of the MAESTRO dataset.
# It loads the MIDI files, converts them to piano rolls, splits into windows, and collects all valid windows in a list.

def process_maestro_split(df: pd.DataFrame, split_name: str) -> list[np.ndarray]:
    split_df = df[df["split"] == split_name]

    all_windows = []
    skipped = 0

    print(f"\nProcessing MAESTRO split: {split_name}")
    print(f"MAESTRO MIDI files: {len(split_df)}")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):   # iterate over the rows of the DataFrame for the given split
        midi_relative_path = row["midi_filename"]
        midi_path = MAESTRO_DIR / midi_relative_path

        try:
            roll = midi_to_binary_pianoroll(midi_path)  # convert the MIDI file to a binary piano roll (shape (T, 88))

            if roll is None:    
                skipped += 1
                continue

            windows = make_windows(roll)    # split the piano roll into windows of shape (WINDOW_SIZE, 88) and filter by active ratio
            all_windows.extend(windows)     # add the valid windows from this MIDI file to the overall list of windows for this split

        except Exception as e:              # if any error occurs during processing (e.g., file not found, invalid MIDI), skip this file and count it as skipped
            skipped += 1
            print(f"Skipped MAESTRO: {midi_path}")
            print(f"Reason: {e}")

    print(f"MAESTRO windows: {len(all_windows)}")
    print(f"MAESTRO skipped files: {skipped}")

    return all_windows



# For the Lakh dataset, we first collect a list of MIDI files (up to MAX_LAKH_FILES) from the specified directory, 
# shuffle them, and then process each file similarly to MAESTRO.

def collect_lakh_files() -> list[Path]:
    midi_files = []

    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(LAKH_DIR.rglob(ext))

    midi_files = sorted(set(midi_files))        # remove duplicates and sort the list of MIDI files

    random.seed(RANDOM_SEED)
    random.shuffle(midi_files)                  # shuffle the list of MIDI files to get a random subset

    return midi_files[:MAX_LAKH_FILES]


# Processes the selected Lakh MIDI files and converts them into piano roll windows, similar to MAESTRO.

def process_lakh_files(lakh_files: list[Path]) -> list[np.ndarray]:
    all_windows = []
    skipped = 0

    print("\nProcessing Lakh MIDI subset")
    print(f"Selected Lakh files: {len(lakh_files)}")

    for midi_path in tqdm(lakh_files):
        try:
            roll = midi_to_binary_pianoroll(midi_path)

            if roll is None:
                skipped += 1
                continue

            windows = make_windows(roll)
            all_windows.extend(windows)

        except Exception as e:
            skipped += 1
            print(f"Skipped Lakh: {midi_path}")
            print(f"Reason: {e}")

    print(f"Lakh windows: {len(all_windows)}")
    print(f"Lakh skipped files: {skipped}")

    return all_windows

# After processing the Lakh MIDI files into windows, we split them into training and validation sets based on the specified LAKH_TRAIN_RATIO.

def split_lakh_windows(lakh_windows: list[np.ndarray]):
    random.seed(RANDOM_SEED)
    random.shuffle(lakh_windows)

    split_idx = int(len(lakh_windows) * LAKH_TRAIN_RATIO)

    lakh_train = lakh_windows[:split_idx]
    lakh_validation = lakh_windows[split_idx:]

    return lakh_train, lakh_validation


# Saves the list of piano roll windows as a single .npy file. 
# It also prints out the shape and active ratio of the saved data for verification.

def save_windows(windows: list[np.ndarray], output_path: Path):
    if len(windows) == 0:
        raise RuntimeError(f"No windows to save for {output_path}")

    data = np.stack(windows).astype(np.float32)     # stack the list of windows into a single numpy array of shape (num_windows, WINDOW_SIZE, 88) and convert to float32
    np.save(output_path, data)

    print(f"Saved: {output_path}")
    print(f"Shape: {data.shape}")
    print(f"Active ratio: {data.mean():.6f}")


# The main function orchestrates the entire preprocessing pipeline: 
# it processes the MAESTRO splits, 
# optionally adds Lakh data, 
# shuffles the windows, 
# and saves the final .npy files for training, validation, and testing.

def main():
    df = pd.read_csv(MAESTRO_CSV_PATH)

    print("MAESTRO CSV columns:")
    print(df.columns.tolist())

    train_windows = process_maestro_split(df, "train")
    validation_windows = process_maestro_split(df, "validation")
    test_windows = process_maestro_split(df, "test")

    if USE_LAKH:
        lakh_files = collect_lakh_files()

        if len(lakh_files) == 0:
            print(f"No Lakh MIDI files found in: {LAKH_DIR}")
        else:
            lakh_windows = process_lakh_files(lakh_files)
            lakh_train, lakh_validation = split_lakh_windows(lakh_windows)

            print(f"Adding Lakh train windows: {len(lakh_train)}")
            print(f"Adding Lakh validation windows: {len(lakh_validation)}")

            train_windows.extend(lakh_train)
            validation_windows.extend(lakh_validation)

    random.seed(RANDOM_SEED)
    random.shuffle(train_windows)
    random.shuffle(validation_windows)

    save_windows(train_windows, OUTPUT_DIR / "train.npy")
    save_windows(validation_windows, OUTPUT_DIR / "validation.npy")
    save_windows(test_windows, OUTPUT_DIR / "test.npy")

    print("\nDone.")
    print("Note: test.npy remains MAESTRO-only for cleaner evaluation.")


if __name__ == "__main__":
    main()