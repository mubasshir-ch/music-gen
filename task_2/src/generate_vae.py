from pathlib import Path

import numpy as np
import pretty_midi
import torch

from vae_model import LSTMVAE


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_lstm_vae.pt"
GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated_vae"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FS = 16
SEQ_LEN = 128

INPUT_DIM = 88
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 2

NUM_SAMPLES = 8
THRESHOLD = 0.3
MAX_NOTES_PER_FRAME = 6
VELOCITY = 80
MIN_NOTE_DURATION = 1.0 / FS


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
    latent_dim = config.get("latent_dim", LATENT_DIM)

    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            z = torch.randn(1, latent_dim).to(DEVICE)

            logits = model.decode(z)
            probs = torch.sigmoid(logits)

            prob_roll = probs[0].cpu().numpy()

            binary_roll = apply_threshold_and_cap(
                prob_roll,
                threshold=THRESHOLD,
                max_notes=MAX_NOTES_PER_FRAME,
            )

            output_path = GENERATED_DIR / f"vae_sample_{i + 1}.mid"

            note_count = pianoroll_to_midi(binary_roll, output_path)

            print(f"Saved: {output_path}")
            print(f"  notes: {note_count}")
            print(f"  active ratio: {binary_roll.mean():.4f}")


if __name__ == "__main__":
    main()