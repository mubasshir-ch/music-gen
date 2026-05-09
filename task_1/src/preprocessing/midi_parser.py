# midi_parser.py
# Scans the raw_midi folder recursively and returns all .mid file paths

import os

def get_midi_files(root_dir):
    midi_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mid') or filename.endswith('.midi'):
                full_path = os.path.join(dirpath, filename)
                midi_files.append(full_path)

    print(f"Total MIDI files found: {len(midi_files)}")
    return midi_files


# Quick test
if __name__ == "__main__":
    root = os.path.join("data", "raw_midi")
    files = get_midi_files(root)
    print("First 5 files:")
    for f in files[:5]:
        print(f)