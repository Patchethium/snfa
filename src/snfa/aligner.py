import importlib.resources as resources
from typing import List, Optional, Tuple

import numpy as np

from snfa.stft import mel_spectrogram
from snfa.viterbi import Segment, viterbi


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / np.sum(e_x, axis, keepdims=True)


def log_softmax(x, axis=-1):
    return np.log(softmax(x, axis))


class Linear:
    def __init__(self, weight, bias) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight.T) + self.bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class GRUCell:
    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
        self.ih = Linear(weight_ih, bias_ih)
        self.hh = Linear(weight_hh, bias_hh)

    def forward(self, x: np.ndarray, h: Optional[np.ndarray] = None) -> np.ndarray:
        """
        x: [D]
        h: [D]
        """
        if h is None:
            h = np.zeros(
                [
                    self.hh.weight.shape[-1],
                ]
            )
        rzn_ih = self.ih.forward(x)
        rzn_hh = self.hh.forward(h)

        rz_ih, n_ih = (
            rzn_ih[: rzn_ih.shape[-1] * 2 // 3],
            rzn_ih[rzn_ih.shape[-1] * 2 // 3 :],
        )
        rz_hh, n_hh = (
            rzn_hh[: rzn_hh.shape[-1] * 2 // 3],
            rzn_hh[rzn_hh.shape[-1] * 2 // 3 :],
        )

        rz = sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, axis=-1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h


class GRU:
    def __init__(self, cell: GRUCell, reverse: bool = False):
        self.cell = cell
        self.reverse = reverse

    def forward(
        self, x, h: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: [T, D], unbatched
        """
        if self.reverse:
            x = np.flip(x, axis=0)
        outputs = []
        for i in range(x.shape[0]):
            h = self.cell.forward(x[i], h)
            outputs.append(h)
        outputs = np.stack(outputs)
        if self.reverse:
            outputs = np.flip(outputs, axis=0)
        return outputs, h

    def __call__(
        self, x, h: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x, h)


def _get_asset_path(filename) -> str:
    return resources.files("snfa.models").joinpath(filename)


class Aligner:
    def __init__(self, filename: Optional[str] = None):
        if filename is None:
            filename = _get_asset_path("jp.npz")
        weights = np.load(filename, allow_pickle=True)

        new_weight = {}
        for k, v in weights.items():
            # convert them back to f32, it's faster on cpu
            if v.dtype == np.float16:
                new_weight[k] = v.astype(np.float32)
            else:
                new_weight[k] = v
        weights = new_weight

        meta_data = weights["meta_data"].item()

        self.n_mels = meta_data["n_mels"]
        self.sr = meta_data["sr"]
        self.dim = meta_data["dim"]
        self.hop_size = meta_data["hop_size"]
        self.win_size = meta_data["win_size"]
        self.n_fft = meta_data["n_fft"]
        self.phone_set = meta_data["phone_set"].split("\0")
        self.phone_dict = {p: i for i, p in enumerate(self.phone_set)}

        # Now load the model weights
        self.pre = Linear(weights["pre.weight"], weights["pre.bias"])
        self.rnn0 = GRU(
            GRUCell(
                weights["bi_rnn.weight_ih_l0"],
                weights["bi_rnn.weight_hh_l0"],
                weights["bi_rnn.bias_ih_l0"],
                weights["bi_rnn.bias_hh_l0"],
            ),
            False,
        )
        self.rnn0_rev = GRU(
            GRUCell(
                weights["bi_rnn.weight_ih_l0_reverse"],
                weights["bi_rnn.weight_hh_l0_reverse"],
                weights["bi_rnn.bias_ih_l0_reverse"],
                weights["bi_rnn.bias_hh_l0_reverse"],
            ),
            True,
        )
        self.rnn1 = GRU(
            GRUCell(
                weights["bi_rnn.weight_ih_l1"],
                weights["bi_rnn.weight_hh_l1"],
                weights["bi_rnn.bias_ih_l1"],
                weights["bi_rnn.bias_hh_l1"],
            ),
            False,
        )
        self.rnn1_rev = GRU(
            GRUCell(
                weights["bi_rnn.weight_ih_l1_reverse"],
                weights["bi_rnn.weight_hh_l1_reverse"],
                weights["bi_rnn.bias_ih_l1_reverse"],
                weights["bi_rnn.bias_hh_l1_reverse"],
            ),
            True,
        )
        self.fc = Linear(
            weights["fc.weight"],
            weights["fc.bias"],
        )

    def forward(self, mel: np.ndarray) -> np.ndarray:
        x = self.pre.forward(mel)
        f, _ = self.rnn0.forward(x)
        r, _ = self.rnn0_rev.forward(x)
        x = np.concatenate([f, r], axis=-1)
        f, _ = self.rnn1.forward(x)
        r, _ = self.rnn1_rev.forward(x)
        x = np.concatenate([f, r], axis=-1)
        x = self.fc.forward(x)
        logits = log_softmax(x, axis=-1)
        return logits

    def get_indices(self, ph):
        try:
            tokens = np.array([int(self.phone_dict[p]) for p in ph])
        except ValueError:
            print("WARN: phoneme not in phoneme set, check it with `Aligner.phone_set`")
        return tokens

    def align(
        self,
        wav: np.ndarray,
        ph: List[str],
        pad_pause: bool = True,
    ) -> List[Segment]:
        """
        Params
        ---
        x: audio signal, [T]
        ph: phoneme sequence, List[str]
        pad_pause: if True, pad start and end pause (the `pau` symbol); default: True

        Returns
        ---
        segments: List[Tuple[str, int, int, float]]
        List of phoneme, start time, end time, score
        """
        wav = wav.squeeze()
        assert wav.ndim == 1, "Audio data should be in shape (T,)"
        mel = mel_spectrogram(
            wav,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            p=2,
            log=True,
        )
        mel = mel.T
        if pad_pause:
            ph = ["pau"] + ph + ["pau"]
        tokens = self.get_indices(ph)
        logits = self.forward(mel)
        segments = viterbi(logits, tokens, phone_set=self.phone_set, blank_id=0)
        # TODO: optimize this, it looks nasty
        for seg in segments:
            seg.start = max(
                int((seg.start * self.hop_size - self.win_size // 2) / self.sr * 1000),
                0,
            )
            seg.end = max(
                int((seg.end * self.hop_size - self.win_size // 2) / self.sr * 1000),
                0,
            )
        return segments

    def __call__(self, x: np.ndarray, ph: List[str]):
        return self.align(x, ph)
