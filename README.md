# snfa

`snfa` (Simple Neural Forced Aligner) is a phoneme-to-audio forced aligner built for embedded usage in python programs, with its only inference dependency being `numpy` and python 3.7 or later.

- Tiny model size (2 MB)
- Numpy as the only dependency
- MFA comparable alignment quality

> [!NOTE]  
> You still need `PyTorch` and some other libs if you want to do training.

> [!WARNING]  
> WIP, not functional

## Inference

```bash
pip install snfa
```

```python
import snfa

aligner = snfa.aligner("cv_jp.bin")
transcript = "k o N n i ch i w a"

# you can also use `scipy` or `wavfile` as long as you normalize it to [-1,1]
x, _ = librosa.load("sample.wav", sr=aligner.sr)

segment = aligner(x, transcript)

print(segment)
```

## Training

First export python path, I'm tired of handling python module imports
```bash
# bash / zsh / fish
export PYTHONPATH=.
# No idea of the equivalent in Windows Powershell...
```

## Todos

- Rust crate
- multi-language

## Licence

`snfa` is released under `ISC Licence`, as shown [here](/LICENCE).

The file `aligner/stft.py` contains code adapted from `librosa` which obeys `ISC Licence` with different copyright claim. A copy of `librosa`'s licence can be found in [librosa's repo](https://github.com/librosa/librosa/blob/main/LICENSE.md).

The file `aligner/backtrack.py` contains code adapted from `torchaudio` which obeys `BSD 2-Clause "Simplified" License`. A copy of `torchaudio`'s licence can be found in [torchaudio's repo](https://github.com/pytorch/audio/blob/main/LICENSE).

## Credit

The neural network used in `snfa` is basically a PyTorch implementation of `CTC*` structure described in [_Evaluating Speech—Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis_](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts).
