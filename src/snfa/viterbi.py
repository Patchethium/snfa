"""
Viterbi algorithm, adopted from `torchaudio` documentation

https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html

BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    return e_x / np.sum(e_x, axis, keepdims=True)


def get_trellis(logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    trellis = logits[:, tokens - 1]
    num_frame = trellis.shape[0]
    num_tokens = len(tokens)

    trellis[1:, 0] = np.cumsum(trellis[1:, 0], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] += np.maximum(
            # Score for staying at the same token
            trellis[t, 1:],
            # Score for changing to the next token
            trellis[t, :-1],
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int


def backtrack(trellis: np.ndarray) -> List[Point]:
    t, j = trellis.shape[0] - 1, trellis.shape[1] - 1

    path = [Point(j, t)]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j]
        changed = trellis[t - 1, j - 1]

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        path.append(Point(j, t))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        path.append(Point(j, t - 1))
        t -= 1

    return path[::-1]


@dataclass
class Segment:
    phoneme: str
    start: int
    end: int

    def __str__(self):
        return f"phoneme: {self.phoneme}, start: {self.start}, end: {self.end}"

    def __repr__(self):
        return (
            f"Segment(phoneme={self.phoneme!r}, start={self.start!r}, end={self.end!r})"
        )


def merge_repeats(
    path: List[Point], tokens: np.ndarray, phone_set: List[str]
) -> List[Segment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        segments.append(
            Segment(
                phone_set[tokens[path[i1].token_index]],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
            )
        )
        i1 = i2
    return segments


def viterbi(
    logits: np.ndarray, tokens: np.ndarray, phone_set: List[str], blank_id=0
) -> List[Segment]:
    emission = logits[:, 1:]
    emission = softmax(emission, axis=-1)
    emission = np.log(emission)
    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis)
    return merge_repeats(path, tokens, phone_set)
