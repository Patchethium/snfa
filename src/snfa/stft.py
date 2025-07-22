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

import numpy as np


def hz_to_mel(f):
    """Linear to Mel-scale."""
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    """Mel to linear scale."""
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=None):
    """Mel Filterbank."""
    if fmax is None:
        fmax = sr // 2

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    # Compute the bin edges
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1 : i + 2]
        for j in range(left, center):
            filterbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filterbank[i - 1, j] = (right - j) / (right - center)
    return filterbank


def spectrogram(
    signal, n_fft=1024, win_length=1024, hop_length=160, window_fn=np.hanning, p=2
):
    """Wavform to its STFT Amplitude Spectrogram."""
    window = window_fn(win_length)
    frames = []
    for i in range(0, len(signal) - win_length + 1, hop_length):
        frame = signal[i : i + win_length] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        frames.append(np.abs(spectrum) ** p)  # power spectrum
    return np.array(frames).T  # shape: (freq_bins, time_frames)


def mel_spectrogram(
    signal,
    sr=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    p=2,
    fmin=0,
    fmax=None,
    log: bool = True,
):
    """Mel spectrogram."""
    spec = spectrogram(signal, n_fft=n_fft, hop_length=hop_length, p=p)
    mel_fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel_spec = np.dot(mel_fb, spec)
    if log:
        mel_spec = np.log(mel_spec + 1e-7)
    return mel_spec
