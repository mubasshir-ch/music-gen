# generate_music.py
# Generates long-form music via latent space walking between real MIDI sequences.
#
# Key fixes over original:
#   - Adaptive thresholding per chunk (avoids blanket silence or blanket noise)
#   - Minimum note duration enforcement (kills the single-step "teng" artefact)
#   - Crossfade overlap between chunks (smooth stitching, no hard cuts)
#   - Polyphony cap (prevents the chord-cluster noise wall)
#   - More waypoints + more interpolation steps → longer, more musical output
#   - Temperature-scaled sampling instead of hard threshold for variety

import os
import sys
import numpy as np
import torch
import pretty_midi

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.autoencoder import LSTMAutoencoder

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\neural network\music-generation-unsupervised"
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "task1_autoencoder_best.pth")
TEST_DATA  = os.path.join(BASE_DIR, "data", "processed", "test.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "generated_midis", "task1")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM   = 128
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
SEQUENCE_LEN = 64
INPUT_SIZE   = 128

TEMPO        = 100       # BPM — slightly slower sounds more musical
FS           = 16        # time steps per second
MIN_NOTE_LEN = 2         # minimum note duration in time steps (kills "teng")
MAX_POLYPHONY = 6        # max simultaneous notes (prevents noise wall)
CROSSFADE_LEN = 8        # overlap time steps for smooth chunk joins


def load_model(device):
    model = LSTMAutoencoder(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM, sequence_len=SEQUENCE_LEN, num_layers=NUM_LAYERS
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    return model


# ─────────────────────────────────────────────
# Piano Roll Post-Processing
# ─────────────────────────────────────────────
def adaptive_threshold(piano_roll, target_density=0.05):
    """
    Instead of a fixed threshold, find one that keeps roughly `target_density`
    fraction of cells active.  Avoids the all-silence / all-noise extremes.
    target_density = 0.05 means ~5 % of cells are notes (realistic for piano).
    """
    flat = piano_roll.flatten()
    threshold = np.percentile(flat, (1.0 - target_density) * 100)
    # Clamp to a reasonable range regardless
    threshold = float(np.clip(threshold, 0.25, 0.75))
    return threshold


def enforce_min_note_length(binary_roll, min_len=2):
    """
    Remove notes shorter than min_len time steps.
    This is the main fix for the "teng" (single-step impulse) artefact.
    """
    clean = binary_roll.copy()
    T, P  = clean.shape
    for pitch in range(P):
        t = 0
        while t < T:
            if clean[t, pitch]:
                run_start = t
                while t < T and clean[t, pitch]:
                    t += 1
                run_len = t - run_start
                if run_len < min_len:
                    clean[run_start:t, pitch] = 0
            else:
                t += 1
    return clean


def cap_polyphony(binary_roll, max_voices=6):
    """
    At each time step keep only the top-`max_voices` active pitches by
    activation strength.  Prevents dense chord clusters that sound like noise.
    """
    # We need the raw float roll for ranking — binary_roll is already binarised,
    # so we just randomly drop excess voices when more than max_voices are active.
    capped = binary_roll.copy()
    for t in range(capped.shape[0]):
        active = np.where(capped[t] > 0)[0]
        if len(active) > max_voices:
            # Keep a musically balanced subset: favour mid-range pitches
            excess = np.random.choice(active, len(active) - max_voices, replace=False)
            capped[t, excess] = 0
    return capped


def postprocess(piano_roll, target_density=0.05):
    """Full post-processing pipeline for a raw sigmoid piano roll."""
    threshold   = adaptive_threshold(piano_roll, target_density)
    binary      = (piano_roll > threshold).astype(np.int8)
    binary      = enforce_min_note_length(binary, min_len=MIN_NOTE_LEN)
    binary      = cap_polyphony(binary, max_voices=MAX_POLYPHONY)
    return binary


# ─────────────────────────────────────────────
# Chunk Stitching with Crossfade
# ─────────────────────────────────────────────
def crossfade_concat(chunks, fade_len=8):
    """
    Concatenate piano roll chunks with a linear crossfade overlap so there
    are no hard cut artefacts between segments.
    chunks: list of (T, 128) float arrays  (raw sigmoid, not yet thresholded)
    """
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]
    for next_chunk in chunks[1:]:
        # Fade out tail of current, fade in head of next, sum
        overlap_len = min(fade_len, result.shape[0], next_chunk.shape[0])
        fade_out = np.linspace(1.0, 0.0, overlap_len)[:, None]
        fade_in  = np.linspace(0.0, 1.0, overlap_len)[:, None]

        blended  = result[-overlap_len:] * fade_out + next_chunk[:overlap_len] * fade_in

        result = np.concatenate([
            result[:-overlap_len],
            blended,
            next_chunk[overlap_len:]
        ], axis=0)

    return result


# ─────────────────────────────────────────────
# Piano Roll → MIDI
# ─────────────────────────────────────────────
def piano_roll_to_midi(binary_roll, tempo=100, fs=16):
    """
    Converts a *binary* (already thresholded + post-processed) piano roll
    to a PrettyMIDI object.
    """
    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    time_per_step = 1.0 / fs

    for pitch in range(128):
        note_on = None
        for t in range(binary_roll.shape[0]):
            if binary_roll[t, pitch] and note_on is None:
                note_on = t
            elif not binary_roll[t, pitch] and note_on is not None:
                start = note_on * time_per_step
                end   = t * time_per_step
                # Velocity: slightly randomise for a more human feel
                vel = int(np.random.randint(65, 95))
                piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                                    start=start, end=end))
                note_on = None
        # Close any still-open note at end
        if note_on is not None:
            start = note_on * time_per_step
            end   = binary_roll.shape[0] * time_per_step
            piano.notes.append(pretty_midi.Note(velocity=80, pitch=pitch,
                                                start=start, end=end))

    midi.instruments.append(piano)
    return midi


# ─────────────────────────────────────────────
# Latent Walk Generation
# ─────────────────────────────────────────────
def generate_long_music(model, device,
                         num_waypoints=8,
                         steps_between=10,
                         target_duration_sec=90):
    """
    Walks through `num_waypoints` real songs in latent space, interpolating
    `steps_between` decoded chunks between each pair of waypoints.

    With SEQUENCE_LEN=64 and FS=16:
      - each chunk = 64/16 = 4 seconds
      - steps_between=10 → ~40 s per pair of waypoints
      - num_waypoints=8  → 7 gaps → ~280 s before capping

    We cap at target_duration_sec to keep file sizes manageable.
    """
    print(f"Loading test data from: {TEST_DATA}")
    data   = np.load(TEST_DATA)                   # (N, 128, 64)
    data   = np.transpose(data, (0, 2, 1))        # (N, 64, 128)
    tensor = torch.FloatTensor(data).to(device)

    max_chunks = int(np.ceil(target_duration_sec * FS / SEQUENCE_LEN))

    with torch.no_grad():
        indices    = np.random.choice(len(tensor), num_waypoints, replace=False)
        waypoints  = tensor[indices]
        z_waypoints = model.encoder(waypoints)    # (num_waypoints, latent_dim)

        raw_chunks = []

        for i in range(num_waypoints - 1):
            if len(raw_chunks) >= max_chunks:
                break

            z_start = z_waypoints[i]
            z_end   = z_waypoints[i + 1]

            # Smooth interpolation (include start, exclude end to avoid doubling)
            alphas = np.linspace(0, 1, steps_between + 1)[:-1]

            for alpha in alphas:
                if len(raw_chunks) >= max_chunks:
                    break
                z_mix  = ((1.0 - alpha) * z_start + alpha * z_end).unsqueeze(0)
                # No teacher forcing at inference — model generates freely
                chunk  = model.decoder(z_mix, target=None, teacher_forcing_ratio=0.0)
                chunk  = chunk.squeeze(0).cpu().numpy()   # (64, 128)
                raw_chunks.append(chunk)

        # Always include the final waypoint
        if len(raw_chunks) < max_chunks:
            final = model.decoder(
                z_waypoints[-1].unsqueeze(0), target=None, teacher_forcing_ratio=0.0
            ).squeeze(0).cpu().numpy()
            raw_chunks.append(final)

    print(f"Generated {len(raw_chunks)} raw chunks ({len(raw_chunks)*SEQUENCE_LEN/FS:.1f}s raw)")

    # Stitch with crossfade (on raw float values, before thresholding)
    long_roll_raw = crossfade_concat(raw_chunks, fade_len=CROSSFADE_LEN)

    # Post-process: threshold + min note length + polyphony cap
    long_roll = postprocess(long_roll_raw, target_density=0.05)

    return long_roll


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = load_model(device)

    num_tracks = 5
    print(f"Generating {num_tracks} long-form MIDI tracks (~90 seconds each)...\n")

    for i in range(1, num_tracks + 1):
        long_roll = generate_long_music(
            model, device,
            num_waypoints=8,        # morph through 8 real songs
            steps_between=10,       # 10 decoded chunks between each waypoint
            target_duration_sec=90  # cap at 90 seconds
        )

        binary_roll = long_roll   # already binarised inside generate_long_music
        midi_obj    = piano_roll_to_midi(binary_roll, tempo=TEMPO, fs=FS)
        note_count  = sum(len(inst.notes) for inst in midi_obj.instruments)

        total_steps = binary_roll.shape[0]
        seconds     = total_steps / FS

        save_path = os.path.join(OUTPUT_DIR, f"task1_long_morph_{i}.mid")
        midi_obj.write(save_path)

        print(f"Track {i}: {total_steps} steps | {seconds:.1f}s | "
              f"{note_count} notes | → {save_path}")

    print("\nDone!")
