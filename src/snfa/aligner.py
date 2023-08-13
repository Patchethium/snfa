"""
one file inference with numpy
FIXME: far slower than PyTorch, needs optimization
"""

from io import BufferedReader
from typing import Iterable, List
import numpy as np
from . import stft
from . import viterbi


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / np.sum(e_x, axis, keepdims=True)

def l1_normalize(arr, axis=None):
    arr = arr - np.min(arr)
    norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
    normalized_arr = arr / norm
    return normalized_arr

def log_softmax(x, axis=-1):
    return np.log(softmax(x, axis))


def read_weight(f: BufferedReader, shape: Iterable[int]):
    length = 1
    for s in shape:
        length *= s
    bytes = f.read(length * 4)
    arr = np.frombuffer(bytes, dtype=np.float32, count=length)
    arr = arr.reshape(shape)
    return arr


class GRU:
    def __init__(self, hid_dim: int, input_dim: int, reverse: bool = False) -> None:
        self.h_dim = hid_dim
        self.i_dim = input_dim
        self.reverse = reverse

    def from_file(self, f: BufferedReader) -> None:
        self.wih = read_weight(f, [self.h_dim * 3, self.i_dim])
        self.whh = read_weight(f, [self.h_dim * 3, self.h_dim])
        self.bih = read_weight(f, [self.h_dim * 3])
        self.bhh = read_weight(f, [self.h_dim * 3])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.zeros([self.h_dim])
        start, end = (0, x.shape[0]) if not self.reverse else (x.shape[0] - 1, -1)
        step = 1 if not self.reverse else -1
        out = np.zeros([x.shape[0], self.h_dim])  # pre allocate
        for index in range(start, end, step):
            rzn_ih = np.matmul(x[index], self.wih.T) + self.bih
            rzn_hh = np.matmul(h, self.whh.T) + self.bhh

            i_r, i_z, i_n = np.split(rzn_ih, 3)
            h_r, h_z, h_n = np.split(rzn_hh, 3)

            r = sigmoid(i_r + h_r)
            z = sigmoid(i_z + h_z)

            n = np.tanh(i_n + r * h_n)

            h = (1.0 - z) * n + z * h
            out[index, :] = h
        return out


class Aligner:
    def __init__(self, filename: str = "model.bin"):
        f = open(filename, "rb")
        
        # Read metadata first, 8 is the amount of metadata entries
        # each entry is one int32 (4 bytes)
        meta_data: np.ndarray = np.frombuffer(
            f.read(8 * 4), np.int32, count=8
        )
        # the entry list
        [
            self.n_fft,
            self.hop_size,
            self.win_size,
            self.m,
            self.h,
            self.p,
            self.sr,
            phone_set_bytes_len,
        ] = meta_data

        self.phone_set = (
            f.read(phone_set_bytes_len).decode("ascii").split("\0")
        )  # one ascii character takes 1 byte

        # Now load the model weights
        self.gru1 = GRU(self.h, self.m)
        self.gru1_r = GRU(self.h, self.m, reverse=True)
        self.gru2 = GRU(self.h, self.h * 2)
        self.gru2_r = GRU(self.h, self.h * 2, reverse=True)

        self.gru1.from_file(f)
        self.gru1_r.from_file(f)
        self.gru2.from_file(f)
        self.gru2_r.from_file(f)

        self.wfc = read_weight(f, [self.p + 1, self.h * 2])
        self.bfc = read_weight(f, [self.p + 1])

        # check if we reach the EOF
        last_byte = f.read(1)
        if not len(last_byte) == 0:
            raise Warning("Model has more bytes than specified in meta-data")

        f.close()

    def model_forward(self, x) -> np.ndarray:
        """
        Params
        ---
        x: mel, [T,M]

        Returns
        ---
        labels: [T, P+1]
        """
        x_l0 = self.gru1(x)
        x_l0_r = self.gru1_r(x)
        x_l1_i = leaky_relu(np.concatenate([x_l0, x_l0_r], axis=1))

        x_l1 = self.gru2(x_l1_i)
        x_l1_r = self.gru2_r(x_l1_i)
        x_fc = leaky_relu(np.concatenate([x_l1, x_l1_r], axis=1))

        x = np.dot(x_fc, self.wfc.T) + self.bfc
        x = log_softmax(x)
        return x

    def mel(self, x: np.ndarray):
        mel = stft.mel_spec(
            x, self.n_fft, self.hop_size, self.win_size, self.m, self.sr
        )
        mel = stft.normalize(stft.power_to_db(mel, ref=np.max), axis=1) + 1.0
        mel = np.fliplr(mel)
        return mel

    def get_indices(self, ph):
        try:
            tokens = np.array([int(self.phone_set.index(p)) for p in ph])
        except ValueError:
            raise Exception("phoneme not in model's phoneme set")
        return tokens

    def align(self, x, ph):
        mel = self.mel(x)
        indices = self.get_indices(ph)

        labels = self.model_forward(mel)

        emission = l1_normalize(labels[:, 1:], axis=1)[:, indices]

        trellis = viterbi.get_trellis(emission)
        path = viterbi.backtrack(trellis)

        segments = viterbi.merge_repeats(path, indices)
        return segments, path, trellis, emission, labels

    def __call__(self, x: np.ndarray, ph: List[str]):
        return self.align(x, ph)

if __name__ == "__main__":
    alinger = Aligner("cv_jp.bin")
    print(alinger.hop_size)
