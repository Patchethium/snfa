"""
Adopted from librosa, ISC Licence, modified

**NOTICE**: it's not fully compatible with `librosa`
because I can't understand the pre-clipping in stft

https://github.com/librosa/librosa

## ISC License

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

from typing import Callable, Optional, Union
import warnings
import numpy as np


def power_to_db(
    S: np.ndarray,
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    S = np.asarray(S)

    if amin <= 0:
        print("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            print("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def stft(x: np.ndarray, n_fft=1024, hop_size=256, win_size=1024):
    """
    Short Time Fourier Transformation, rewritten very naively for my tiny brain to understand
    FIXME: This function has length mismatch from librosa.feature.melspectrogram
    for the prefix frame mechanism is not implemented
    """
    if np.max(x) > 1.0 or np.min(x) < -1.0:
        warnings.warn("input audio should be normalized to [-1,1]")

    window = np.hanning(n_fft)
    if win_size > n_fft:
        window = np.pad(window, win_size)
    num_windows = (x.shape[-1] - win_size) // hop_size + 1
    pad_length = num_windows * hop_size + win_size - x.shape[-1]

    # pad to window length
    x = np.pad(x, pad_length)

    res = []
    for win_idx in range(num_windows):
        windowed = x[..., win_idx * hop_size : win_idx * hop_size + win_size] * window
        res.append(np.fft.rfft(windowed, n=n_fft))

    return np.stack(res)


def mel_scale(n_mels=128, sr=16000, n_fft=1024):
    """
    algorithm copied from librosa, rewritten naively
    """
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    mel_bins = np.floor((n_fft + 1) * hz_points / sr)
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))

    for m in range(1, n_mels + 1):
        f_m_minus = int(mel_bins[m - 1])  # left
        f_m = int(mel_bins[m])  # center
        f_m_plus = int(mel_bins[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - mel_bins[m - 1]) / (mel_bins[m] - mel_bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (mel_bins[m + 1] - k) / (mel_bins[m + 1] - mel_bins[m])

    return fbank


def mel_spec(
    x: np.ndarray, n_fft=1024, hop_size=256, win_size=1024, n_mels=128, sr=16000
):
    stft_mat = stft(x, n_fft=n_fft, hop_size=hop_size, win_size=win_size)
    spec = np.abs(stft_mat) ** 2
    mel = mel_scale(n_mels=n_mels, sr=sr, n_fft=n_fft)
    mel_spec = np.matmul(spec, mel.T)
    return mel_spec


def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise Exception(f"threshold={threshold} must be strictly positive")

    if fill not in [None, False, True]:
        raise Exception(f"fill={fill} must be None or boolean")

    if not np.all(np.isfinite(S)):
        raise Exception("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S

    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise Exception("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    else:
        raise Exception(f"Unsupported norm: {repr(norm)}")

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def tiny(x: Union[float, np.ndarray]) -> float:
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny
