"""
copied from librosa, ISC Licence, modified
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
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::

            10 * log10(S / ref)

        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S/ref)) - top_db``

    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)

    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)

    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """
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
    Short Fourier Transformation, very naive
    """
    if np.max(x) > 1 or np.min(x) < -1:
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


def mel_scale(n_mel=128, sr=16000, n_fft=1024):
    """
    copied from librosa, rewritten very naively
    """
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mel + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    mel_bins = np.floor((n_fft + 1) * hz_points / sr)
    fbank = np.zeros((n_mel, int(np.floor(n_fft / 2 + 1))))

    for m in range(1, n_mel + 1):
        f_m_minus = int(mel_bins[m - 1])    # left
        f_m = int(mel_bins[m])              # center
        f_m_plus = int(mel_bins[m + 1])     # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - mel_bins[m - 1]) / (mel_bins[m] - mel_bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (mel_bins[m + 1] - k) / (mel_bins[m + 1] - mel_bins[m])

    return fbank


def mel_spec(
    x: np.ndarray, n_fft=1024, hop_size=256, win_size=1024, n_mel=128, sr=16000
):
    stft_mat = stft(x, n_fft=n_fft, hop_size=hop_size, win_size=win_size)
    spec = np.abs(stft_mat) ** 2
    mel = mel_scale(n_mel=n_mel, sr=sr, n_fft=n_fft)
    mel_spec = np.dot(spec, mel.T)
    return power_to_db(mel_spec, ref=np.max)
