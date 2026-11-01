"""
Writes `tests/fixtures/parity.json`: what the numpy implementation computes for
a synthetic waveform, which `tests/parity.rs` compares the Rust port against.

    uv run --no-project --with numpy python rust/tests/make_fixtures.py jp.npz

Run it from the repository root against the `.npz` the `.safetensors` model
used by the Rust test was converted from.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from snfa import Aligner, trim_audio  # noqa: E402

SAMPLE_RATE = 16000
DURATION = 1.5
# Every STRIDE-th frame is stored to keep the fixture small.
STRIDE = 4
PHONEMES = "k o n n i ch i w a".split(" ")
OUTPUT = os.path.join(os.path.dirname(__file__), "fixtures", "parity.json")


def lcg(count, seed=1):
    """Uniform noise in [-0.5, 0.5), reproducible in Rust."""
    values = []
    state = seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) % 2**31
        values.append(state / 2**31 - 0.5)
    return values


def synthetic_waveform():
    """A gliding harmonic tone under a silence-tone-silence envelope, plus noise."""
    n = int(SAMPLE_RATE * DURATION)
    noise = lcg(n)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        phase = 2 * math.pi * (120 * t - 40 / (2 * math.pi * 1.3) * math.cos(2 * math.pi * 1.3 * t))
        tone = sum(math.sin(k * phase) / k for k in range(1, 11))
        envelope = math.sin(math.pi * min(max((t - 0.25) / 1.0, 0.0), 1.0)) ** 2
        samples.append(0.3 * envelope * tone + 0.01 * noise[i])
    return np.array(samples, dtype=np.float32)


def rounded(rows):
    return [[float(f"{v:.10g}") for v in row] for row in rows]


def main(npz):
    aligner = Aligner(npz)
    wav = synthetic_waveform()
    mel = aligner_mel(aligner, wav)
    logits = aligner.forward(mel)
    segments = aligner.align(wav, PHONEMES, pad_pause=True)
    _, (trim_start, trim_end) = trim_audio(wav, top_db=20, frame_length=1024, hop_length=256)
    fixture = {
        "model": os.path.basename(npz),
        "hop_size": aligner.hop_size,
        "stride": STRIDE,
        "frames": int(mel.shape[0]),
        "mel": rounded(mel[::STRIDE]),
        "logits": rounded(logits[::STRIDE]),
        "phonemes": PHONEMES,
        "segments": [[s.phoneme, s.start, s.end] for s in segments],
        "trim": [int(trim_start), int(trim_end)],
    }
    with open(OUTPUT, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
    print(f"wrote {OUTPUT}: {mel.shape[0]} frames, {len(segments)} segments")


def aligner_mel(aligner, wav):
    from snfa.stft import mel_spectrogram

    return mel_spectrogram(
        wav,
        sr=aligner.sr,
        n_fft=aligner.n_fft,
        hop_length=aligner.hop_size,
        n_mels=aligner.n_mels,
        p=2,
        log=True,
    ).T


if __name__ == "__main__":
    main(sys.argv[1])
