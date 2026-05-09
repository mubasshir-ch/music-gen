from pathlib import Path

import numpy as np
import pretty_midi
import torch

from dataset import PianoRollDataset
from vae_model import LSTMVAE


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "latest_lstm_vae.pt"
DATA_PATH = PROJECT_ROOT / "processed" / "validation.npy"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vae_interpolation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FS = 16
VELOCITY = 80
THRESHOLD = 0.80

INPUT_DIM = 88
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 128


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
                end = t * frame_duration

                if end > start:
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
        end = final_t * frame_duration

        if end > start:
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

    return model


def main():
    print("Using device:", DEVICE)

    model = load_model()
    dataset = PianoRollDataset(DATA_PATH)

    x1 = dataset[0].unsqueeze(0).to(DEVICE)
    x2 = dataset[100].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        for i in range(8):
            alpha = i / 7

            z_alpha = (1 - alpha) * mu1 + alpha * mu2

            logits = model.decode(z_alpha)
            probs = torch.sigmoid(logits)

            binary_roll = (probs[0].cpu().numpy() > THRESHOLD).astype(np.int32)

            output_path = OUTPUT_DIR / f"interp_{i + 1}_alpha_{alpha:.3f}.mid"

            note_count = pianoroll_to_midi(binary_roll, output_path)

            print(f"Saved: {output_path}")
            print(f"alpha = {alpha:.3f}, notes = {note_count}")


if __name__ == "__main__":
    main()