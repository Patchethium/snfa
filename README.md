# snfa

`snfa` (Simple Neural Forced Aligner) is a phoneme-to-audio forced aligner built for embedded usage in python programs, with its only inference dependency being `numpy` and python 3.7 or later.

> **Notice**: You still need `PyTorch` and some other libs if you want to do training.

> **Warning**: WIP, not functional

## Licence

`snfa` is released under `ISC Licence`, as shown [here](/LICENCE).

The file `stft.py` contains code copied from `librosa` which obeys the same `ISC` Licence but different copyright claim. A copy of `librosa`'s licence can be found [here](https://github.com/librosa/librosa/blob/main/LICENSE.md).

## Credit

The neural network used in `snfa` is basically a PyTorch implementation of `CTC*` structure described in [_Evaluating Speech—Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis_](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts).
