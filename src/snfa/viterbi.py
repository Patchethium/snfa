from typing import List
import numpy as np
from dataclasses import dataclass


"""
Viterbi algorithm, adopted from `torchaudio` documentation, BSD 2-Clause Licence

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


@dataclass
class Point:
    token_index: int
    time_index: int


def get_trellis(emission: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """
    Get a cost matrix `trellis` from emission
    using Viterbi algorithm.
    """
    num_frames, num_tokens = emission.shape[0], tokens.shape[0]
    trellis = np.zeros((num_frames, num_tokens))
    trellis[1:, 0] = np.cumsum(emission[1:, tokens[0]], 0)
    trellis[0, 1:] = -np.inf
    trellis[-num_tokens + 1 :, 0] = np.inf

    for t in range(num_frames - 1):
        candidate = np.maximum(
            trellis[t, 1:] + emission[t + 1, tokens[1:]],
            trellis[t, :-1] + emission[t + 1, tokens[1:]],
        )
        trellis[t + 1, 1:] = candidate

    return trellis


def backtrack(trellis) -> List[Point]:
    """
    Get a path with maxium likelihood from cost matrix,
    basically just picks the better one in staying and transferring.
    """
    t, j = trellis.shape[0] - 1, trellis.shape[1] - 1

    path = [Point(j, t)]
    while j > 0:
        stayed = trellis[t - 1, j]
        changed = trellis[t - 1, j - 1]

        t -= 1
        if changed > stayed:
            j -= 1
        path.append(Point(j, t))

    while t > 0:
        path.append(Point(j, t - 1))
        t -= 1

    return path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, ph) -> List[Segment]:
    i1, i2 = 0, 0
    segments: List[Segment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        segments.append(
            Segment(
                ph[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
            )
        )
        i1 = i2
    return segments
