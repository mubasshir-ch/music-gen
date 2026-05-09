from pathlib import Path

import numpy as np
import pretty_midi
import torch

from vae_model import LSTMVAE

# This file loads a trained LSTM VAE model from a checkpoint, generates new piano roll samples by sampling from the latent space, applies a threshold to convert the output probabilities into binary piano rolls, and saves the generated piano rolls as MIDI files.
# The main steps are: 1) load the model, 2) sample from the latent space, 3) decode to get output probabilities, 4) apply threshold and cap to get binary piano rolls, and 5) convert the binary piano rolls to MIDI files and save them.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_lstm_vae.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "latest_lstm_vae.pt"
GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated_vae"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FS = 16
SEQ_LEN = 128

INPUT_DIM = 88
HIDDEN_DIM = 128        # reduced hidden dimension for faster generation
LATENT_DIM = 64
NUM_LAYERS = 2

NUM_SAMPLES = 8         # number of piano roll samples to generate from the VAE.
THRESHOLD = 0.5         # threshold for converting the output probabilities into binary piano rolls. Anything > threshold is considered active (1), and anything <= threshold is considered inactive (0).
MAX_NOTES_PER_FRAME = 6 # maximum number of active notes allowed per time frame. If the model outputs more than this number of active notes for a frame, we will keep only the top max_notes based on their probabilities and set the rest to inactive (0).
VELOCITY = 80           # velocity for the generated MIDI notes (0-127). This is a fixed value for all generated notes, but it could be randomized or made dynamic based on the model's output if desired.
MIN_NOTE_DURATION = 1.0 / FS


# Hepler function: Applies a threshold to the probability piano roll and caps the number of active notes per frame to max_notes.
def apply_threshold_and_cap(prob_roll, threshold, max_notes):
    binary = np.zeros_like(prob_roll, dtype=np.int32)

    for t in range(prob_roll.shape[0]):
        active = np.where(prob_roll[t] > threshold)[0]

        if len(active) > max_notes:
            top = active[np.argsort(prob_roll[t][active])[-max_notes:]]
            binary[t, top] = 1
        else:
            binary[t, active] = 1

    return binary

# Converts a binary piano roll into a MIDI file and saves it to the specified output path. 
# It iterates through each time frame of the piano roll, keeps track of active notes, and creates MIDI note events when notes start and end.
def pianoroll_to_midi(binary_roll, output_path):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    frame_duration = 1.0 / FS
    active_notes = {}

    for t in range(binary_roll.shape[0]):
        current_active = set(np.where(binary_roll[t] == 1)[0])

        for pitch_idx in current_active:
            if pitch_idx not in active_notes:
                active_notes[pitch_idx] = t

        ended = []

        for pitch_idx, start_t in active_notes.items():
            if pitch_idx not in current_active:
                pitch = pitch_idx + 21
                start = start_t * frame_duration
                end = max(t * frame_duration, start + MIN_NOTE_DURATION)

                piano.notes.append(
                    pretty_midi.Note(
                        velocity=VELOCITY,
                        pitch=pitch,
                        start=start,
                        end=end,
                    )
                )

                ended.append(pitch_idx)

        for pitch_idx in ended:
            del active_notes[pitch_idx]

    final_t = binary_roll.shape[0]

    for pitch_idx, start_t in active_notes.items():
        pitch = pitch_idx + 21
        start = start_t * frame_duration
        end = max(final_t * frame_duration, start + MIN_NOTE_DURATION)

        piano.notes.append(
            pretty_midi.Note(
                velocity=VELOCITY,
                pitch=pitch,
                start=start,
                end=end,
            )
        )

    midi.instruments.append(piano)
    midi.write(str(output_path))

    return len(piano.notes)

# Loads the trained VAE model from the checkpoint file, 
# reconstructs the model architecture based on the saved configuration, 
# and loads the model weights. The model is set to evaluation mode and returned along with its configuration.
def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    config = checkpoint.get("config", {})

    model = LSTMVAE(
        input_dim=config.get("input_dim", INPUT_DIM),
        hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
        latent_dim=config.get("latent_dim", LATENT_DIM),
        num_layers=config.get("num_layers", NUM_LAYERS),
        seq_len=config.get("seq_len", SEQ_LEN),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def main():
    print("Using device:", DEVICE)
    print("Loading:", CHECKPOINT_PATH)

    model, config = load_model()
    latent_dim = config.get("latent_dim", LATENT_DIM)       # get the latent dimension from the checkpoint config, or use the default if not available

    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            z = torch.randn(1, latent_dim).to(DEVICE)       # sample a random latent vector z from a standard normal distribution of shape (1, latent_dim)

            logits = model.decode(z)                        # decode the random latent vector to get output logits of shape (1, seq_len, input_dim)
            probs = torch.sigmoid(logits)                   # apply sigmoid to convert logits to probabilities in the range [0, 1]

            prob_roll = probs[0].cpu().numpy()              # convert the output probabilities to a numpy array of shape (seq_len, input_dim) for post-processing

            binary_roll = apply_threshold_and_cap(
                prob_roll,
                threshold=THRESHOLD,
                max_notes=MAX_NOTES_PER_FRAME,
            )

            output_path = GENERATED_DIR / f"vae_sample_{i + 1}.mid"

            note_count = pianoroll_to_midi(binary_roll, output_path)    # convert the binary piano roll to a MIDI file and save it to the output path. The function returns the number of notes in the generated MIDI file.

            print(f"Saved: {output_path}")
            print(f"  notes: {note_count}")
            print(f"  active ratio: {binary_roll.mean():.4f}")


if __name__ == "__main__":
    main()