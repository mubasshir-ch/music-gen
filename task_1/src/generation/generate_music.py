import os
import sys
import numpy as np
import torch
import pretty_midi

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.autoencoder import LSTMAutoencoder

BASE_DIR   = r"D:\neural network\music-generation-unsupervised"
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "task1_autoencoder_best.pth")
TEST_DATA  = os.path.join(BASE_DIR, "data", "processed", "test.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "generated_midis", "task1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM     = 128
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
SEQUENCE_LEN   = 64
INPUT_SIZE     = 128

TEMPO          = 115
FS             = 16
TARGET_DENSITY = 0.04   # raised: more notes, fewer silences
MIN_NOTE_STEPS = 4      # raised: min 0.25 s — eliminates micro-clicks
MAX_POLYPHONY  = 8      # raised: richer chords
CROSSFADE_LEN  = 20     # longer blend window
PITCH_MIN      = 48
PITCH_MAX      = 83


def load_model(device):
    model = LSTMAutoencoder(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        sequence_len=SEQUENCE_LEN,
        num_layers=NUM_LAYERS,
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    return model


def adaptive_threshold(raw_roll, target_density=TARGET_DENSITY):
    percentile = (1.0 - target_density) * 100
    threshold  = float(np.percentile(raw_roll.flatten(), percentile))
    return float(np.clip(threshold, 0.05, 0.85))


def apply_pitch_range(roll):
    out = roll.copy()
    out[:, :PITCH_MIN]     = 0
    out[:, PITCH_MAX + 1:] = 0
    return out


def remove_short_notes(binary_roll, min_steps=MIN_NOTE_STEPS):
    out = binary_roll.copy().astype(np.int8)
    for p in range(out.shape[1]):
        col    = out[:, p]
        padded = np.concatenate([[0], col, [0]])
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < min_steps:
                out[s:e, p] = 0
    return out


def cap_polyphony(binary_roll, raw_roll, max_voices=MAX_POLYPHONY):
    out = binary_roll.copy()
    for t in range(out.shape[0]):
        active = np.where(out[t] > 0)[0]
        if len(active) > max_voices:
            strengths = raw_roll[t, active]
            keep      = np.argsort(strengths)[-max_voices:]
            drop_mask = np.ones(len(active), dtype=bool)
            drop_mask[keep] = False
            out[t, active[drop_mask]] = 0
    return out


def sustain_into_gaps(binary_roll, max_gap_steps=4):
    """
    Instead of inserting fake bridge notes, hold (sustain) the last active
    notes across short silent gaps. This sounds far more natural — the note
    simply rings a little longer rather than restarting mid-silence.
    Only gaps <= max_gap_steps (0.25 s at FS=16) are bridged this way.
    """
    out = binary_roll.copy()
    T   = out.shape[0]
    t   = 0
    while t < T:
        if out[t].sum() == 0:
            gap_start = t
            while t < T and out[t].sum() == 0:
                t += 1
            gap_end = t
            gap_len = gap_end - gap_start
            if gap_len <= max_gap_steps and gap_start > 0:
                # Sustain whichever pitches were playing just before the gap
                held = np.where(out[gap_start - 1] > 0)[0]
                for step in range(gap_start, gap_end):
                    out[step, held] = 1
        else:
            t += 1
    return out


def postprocess(raw_roll):
    thr    = adaptive_threshold(raw_roll)
    binary = (raw_roll > thr).astype(np.int8)
    binary = apply_pitch_range(binary)
    binary = remove_short_notes(binary, MIN_NOTE_STEPS)
    binary = cap_polyphony(binary, raw_roll, MAX_POLYPHONY)
    binary = sustain_into_gaps(binary, max_gap_steps=4)
    return binary


def crossfade_concat(chunks, fade_len=CROSSFADE_LEN):
    result = chunks[0]
    for nxt in chunks[1:]:
        olen     = min(fade_len, result.shape[0], nxt.shape[0])
        fade_out = np.linspace(1.0, 0.0, olen)[:, None]
        fade_in  = np.linspace(0.0, 1.0, olen)[:, None]
        blend    = result[-olen:] * fade_out + nxt[:olen] * fade_in
        result   = np.concatenate([result[:-olen], blend, nxt[olen:]], axis=0)
    return result


def binary_to_midi(binary_roll, tempo=TEMPO, fs=FS):
    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    spb   = 1.0 / fs

    for pitch in range(PITCH_MIN, PITCH_MAX + 1):
        note_on = None
        for t in range(binary_roll.shape[0]):
            if binary_roll[t, pitch] and note_on is None:
                note_on = t
            elif not binary_roll[t, pitch] and note_on is not None:
                start = note_on * spb
                end   = t * spb
                norm  = (pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
                vel   = int(np.clip(58 + norm * 22 + np.random.randint(-4, 5), 50, 90))
                piano.notes.append(
                    pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
                note_on = None
        if note_on is not None:
            piano.notes.append(
                pretty_midi.Note(velocity=70, pitch=pitch,
                                 start=note_on * spb,
                                 end=binary_roll.shape[0] * spb))

    midi.instruments.append(piano)
    return midi


def generate_track(model, device, num_waypoints=8, steps_between=14, target_sec=90):
    data   = np.transpose(np.load(TEST_DATA), (0, 2, 1))
    tensor = torch.FloatTensor(data).to(device)

    max_chunks   = int(np.ceil(target_sec * FS / SEQUENCE_LEN))
    active_counts = tensor.sum(dim=(1, 2)).cpu().numpy()
    valid_indices = np.where(active_counts > 15)[0]
    if len(valid_indices) < num_waypoints:
        valid_indices = np.arange(len(tensor))

    with torch.no_grad():
        idx         = np.random.choice(valid_indices, num_waypoints, replace=False)
        waypoints   = tensor[idx]
        z_waypoints = model.encoder(waypoints)

        raw_chunks = []
        for i in range(num_waypoints - 1):
            if len(raw_chunks) >= max_chunks:
                break
            z0     = z_waypoints[i]
            z1     = z_waypoints[i + 1]
            alphas = np.linspace(0, 1, steps_between + 1)[:-1]
            for alpha in alphas:
                if len(raw_chunks) >= max_chunks:
                    break
                z_mix = ((1.0 - alpha) * z0 + alpha * z1).unsqueeze(0)
                chunk = model.decoder(z_mix, target=None, teacher_forcing_ratio=0.0)
                raw_chunks.append(chunk.squeeze(0).cpu().numpy())

        if len(raw_chunks) < max_chunks:
            last = model.decoder(z_waypoints[-1].unsqueeze(0),
                                 target=None, teacher_forcing_ratio=0.0)
            raw_chunks.append(last.squeeze(0).cpu().numpy())

    print(f"  Decoded {len(raw_chunks)} chunks "
          f"({len(raw_chunks) * SEQUENCE_LEN / FS:.1f} s raw)")

    stitched = crossfade_concat(raw_chunks, CROSSFADE_LEN)
    binary   = postprocess(stitched)

    if binary.sum() == 0:
        print("  Warning: empty output — retrying at higher density")
        thr    = float(np.percentile(stitched.flatten(), 96.0))
        binary = (stitched > thr).astype(np.int8)
        binary = apply_pitch_range(binary)
        binary = remove_short_notes(binary, 2)
        binary = cap_polyphony(binary, stitched, MAX_POLYPHONY)
        binary = sustain_into_gaps(binary, max_gap_steps=4)

    return binary


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = load_model(device)
    print("\nGenerating 5 MIDI compositions (~90 s each)...\n")

    for i in range(1, 6):
        print(f"Track {i}:")
        binary   = generate_track(model, device,
                                  num_waypoints=8,
                                  steps_between=14,
                                  target_sec=90)
        midi_obj = binary_to_midi(binary)
        path     = os.path.join(OUTPUT_DIR, f"task1_long_morph_{i}.mid")
        midi_obj.write(path)

        notes = midi_obj.instruments[0].notes
        dur   = midi_obj.get_end_time()
        durs  = [n.end - n.start for n in notes]
        print(f"  {dur:.1f}s | {len(notes)} notes | "
              f"{len(notes)/dur:.2f} notes/s | avg_dur: {np.mean(durs):.2f}s")
        print(f"  Saved → {path}\n")

    print("All tracks generated.")