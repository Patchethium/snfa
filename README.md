# snfa

`snfa` (Simple Neural Forced Aligner) is a phoneme-to-audio forced aligner built for embedded usage in python programs, with its only inference dependency being `numpy` and python 3.7 or later.

- Tiny model size (2 MB)
- Numpy as the only dependency
- MFA comparable alignment quality

**Note**: You still need `PyTorch` and some other libs if you want to do training.

## Inference

```bash
pip install snfa
```
Download the pretrained `cv_jp.bin` file from [release](https://github.com/Patchethium/snfa/releases/latest).

`cv_jp.bin` is weight file trained with Japanese Common Voice Corpus 14.0, 6/28/2023, the model weight is released into `Public Domain`.

```python
import snfa

aligner = snfa.Aligner("cv_jp.bin")
transcript = "k o N n i ch i w a".split(" ")

# you can also use `scipy` or `wavfile` as long as you normalize it to [-1,1]
x, _ = librosa.load("sample.wav", sr=aligner.sr)

segment, path, trellis, labels = aligner(x, transcript)

print(segment)
```

## Training

I'll cover this part if it's needed by anyone. Please let me know by creating an issue if you need.

## Todos

- Rust crate
- multi-language

## Licence

`snfa` is released under `ISC Licence`, as shown [here](/LICENCE).

The file `snfa/stft.py` contains code adapted from `librosa` which obeys `ISC Licence` with different copyright claim. A copy of `librosa`'s licence can be found in [librosa's repo](https://github.com/librosa/librosa/blob/main/LICENSE.md).

The file `snfa/backtrack.py` contains code adapted from `torchaudio` which obeys `BSD 2-Clause "Simplified" License`. A copy of `torchaudio`'s licence can be found in [torchaudio's repo](https://github.com/pytorch/audio/blob/main/LICENSE).

## Credit

The neural network used in `snfa` is basically a PyTorch implementation of `CTC*` structure described in [_Evaluating Speech—Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis_](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts).
