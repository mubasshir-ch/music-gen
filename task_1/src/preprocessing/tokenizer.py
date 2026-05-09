# tokenizer.py
# Full preprocessing pipeline: MIDI -> piano roll segments -> saved numpy files

import os
import numpy as np
from midi_parser import get_midi_files
from piano_roll import midi_to_piano_roll, segment_piano_roll
from sklearn.model_selection import train_test_split

def preprocess_dataset(raw_midi_dir, output_dir, segment_len=256, fs=16, max_files=500):
    
    midi_files = get_midi_files(raw_midi_dir)

    
    if max_files:
        midi_files = midi_files[:max_files]
        print(f"Using {len(midi_files)} files for preprocessing")

    all_segments = []

    for i, path in enumerate(midi_files):
        roll = midi_to_piano_roll(path, fs=fs)

        if roll is None:
            continue

        segments = segment_piano_roll(roll, segment_len=segment_len)
        all_segments.extend(segments)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(midi_files)} files | Total segments so far: {len(all_segments)}")

    print(f"\nTotal segments collected: {len(all_segments)}")

    # Stack into one numpy array: shape (N, 128, segment_len)
    data = np.array(all_segments)
    print(f"Final dataset shape: {data.shape}")

    # Train / test split (80/20)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train size: {train_data.shape} | Test size: {test_data.shape}")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train.npy"), train_data)
    np.save(os.path.join(output_dir, "test.npy"), test_data)
    print(f"\nSaved train.npy and test.npy to: {output_dir}")


# Run preprocessing
if __name__ == "__main__":
    # Using absolute path to avoid issues with working directory
    base_dir    = r"D:\neural network\music-generation-unsupervised"
    raw_midi_dir = os.path.join(base_dir, "data", "raw_midi")
    output_dir   = os.path.join(base_dir, "data", "processed")

    preprocess_dataset(
        raw_midi_dir=raw_midi_dir,
        output_dir=output_dir,
        segment_len=64,
        fs=16,
        max_files=500       # increase to None for full dataset later
    )