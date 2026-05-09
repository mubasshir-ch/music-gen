# piano_roll.py
# Converts MIDI files into piano roll numpy arrays

import pretty_midi
import numpy as np

def midi_to_piano_roll(midi_path, fs=16):
    """
    Converts a MIDI file to a binary piano roll.
    Args:
        midi_path : path to .mid file
        fs        : time steps per second (16 = 16 steps/second)
    Returns:
        piano_roll: numpy array of shape (128, time_steps)
                    128 = number of MIDI pitches
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi.get_piano_roll(fs=fs)  # shape: (128, T)

        # Binarize: 1 if note is active, 0 otherwise
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None


def segment_piano_roll(piano_roll, segment_len=64):
    """
    Splits a piano roll into fixed-length segments.
    Args:
        piano_roll  : numpy array (128, T)
        segment_len : number of time steps per segment (default 64)
    Returns:
        list of numpy arrays each of shape (128, segment_len)
    """
    segments = []
    T = piano_roll.shape[1]

    for start in range(0, T - segment_len + 1, segment_len):
        segment = piano_roll[:, start:start + segment_len]
        segments.append(segment)

    return segments


# Quick test
if __name__ == "__main__":
    import os
    test_file = None

    # Find first midi file to test
    for dirpath, _, filenames in os.walk(os.path.join("data", "raw_midi")):
        for f in filenames:
            if f.endswith('.mid'):
                test_file = os.path.join(dirpath, f)
                break
        if test_file:
            break

    if test_file:
        print(f"Testing with: {test_file}")
        roll = midi_to_piano_roll(test_file)
        if roll is not None:
            print(f"Piano roll shape: {roll.shape}")
            segs = segment_piano_roll(roll)
            print(f"Number of segments: {len(segs)}")
            print(f"Each segment shape: {segs[0].shape}")