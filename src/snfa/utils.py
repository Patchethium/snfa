"""
ISC License

Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

from typing import Tuple

import numpy as np


def trim_audio(
    audio: np.ndarray, top_db: int = 20, frame_length: int = 2048, hop_length: int = 512
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading and trailing silence from an audio signal.

    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    top_db : float
        Threshold below reference to consider as silence (in dB)
    frame_length : int
        Length of the frame for energy calculation
    hop_length : int
        Number of samples between successive frames

    Returns:
    --------
    trimmed_audio : np.ndarray
        Audio signal with silence trimmed
    (start_idx, end_idx) : tuple
        Start and end indices of the trimmed audio
    """

    # Convert audio to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Calculate frame-wise energy using sliding window
    def frame_energy(signal, frame_len, hop_len):
        n_frames = 1 + (len(signal) - frame_len) // hop_len
        frames = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_len
            end = start + frame_len
            if end <= len(signal):
                frame = signal[start:end]
                # Calculate RMS energy
                frames[i] = np.sqrt(np.mean(frame**2))

        return frames

    # Calculate frame energies
    energies = frame_energy(audio, frame_length, hop_length)

    # Convert to dB using proper reference (similar to librosa.power_to_db)
    amin = 1e-10
    ref_value = np.max(energies)  # Use max energy as reference

    # Convert power (energy squared) to dB: 10 * log10(S / ref)
    log_spec = 10.0 * np.log10(np.maximum(amin, energies**2))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value**2))

    # Apply top_db threshold: max(dB) - top_db
    energies_db = np.maximum(log_spec, log_spec.max() - top_db)

    # Find frames above threshold (which is now 0 dB relative to max)
    threshold_db = energies_db.max() - top_db
    above_threshold = energies_db > threshold_db

    if not np.any(above_threshold):
        # If no frames above threshold, return empty audio
        return np.array([]), (0, 0)

    # Find start and end frame indices
    start_frame = np.argmax(above_threshold)
    end_frame = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])

    # Convert frame indices to sample indices
    start_idx = start_frame * hop_length
    end_idx = min((end_frame + 1) * hop_length + frame_length, len(audio))

    # Trim the audio
    trimmed_audio = audio[start_idx:end_idx]

    return trimmed_audio, (start_idx, end_idx)
