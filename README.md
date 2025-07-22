# snfa

`snfa` (Simple Neural Forced Aligner) is a phoneme-to-audio forced aligner built for embedded usage in python programs, with its only inference dependency being `numpy` and python 3.7 or later.

- Tiny model size (~1 MB)
- Numpy as the only dependency
- MFA comparable alignment quality

**Note**: You still need `PyTorch` and some other libs if you want to do training.

## Inference

```bash
pip install snfa
```

A pre-trained model weight `jp.npz` is included.

`jp.npz` is a weight file trained on Japanese Common Voice Corpus 14.0, 6/28/2023. The model weight is released into `Public Domain`.

```python
import snfa
import librosa # or soundfile, torchaudio, scipy, etc.


aligner = snfa.Aligner() # use custom model by passing its path to this function
# NOTE: the default model is uncased, it doesn't make difference between `U` and `u`
transcript = "k o N n i ch i w a".lower().split(" ") # remember to lower it here

# you can also use `scipy` or `wavfile` as long as it's 
# 1. mono channel numpy array with shape (T,), dtype=np.float32
# 2. normalized to [-1,1]
# 3. sample rate matches model's `sr`
x, sr = librosa.load("sample.wav", sr=aligner.sr)
# trim the audio for better performance
x, _ = librosa.effects.trim(x, top_db=20)
# we also provide a utility function to trim
# it's basically ripped off from librosa so you don't have to install it
x, _ = snfa.trim_audio(x, top_db=20)

segments = aligner(x, transcript)

print(segment)
# (phoneme label, start mili-sec, end mili-sec, score)
# [('pau', 0, 908, 0.9583546351318474),
#  ('k', 908, 928, 0.006900709283433312),
#  ('o', 928, 1088, 0.795996002234283),
# ...]

# NOTE: The timestamps are in mili-sec, you can convert them to the indices on wavform by
wav_index = int(timestamp * aligner.sr / 1000)
```

## Development

We use [`uv`](https://docs.astral.sh/uv/) to manage dependencies.

The following command will install them.

```bash
uv sync
```

### Training

Download [Common Voice Dataset](https://commonvoice.mozilla.org/en) and extract it somewhere.

We use the split from whole `validated.tsv`, while filtered out the `dev` and `test` split.

Filter the dataset:
```bash
uv run filter_dataset.py -d /path/to/common/voice/
```

Start training:
```bash
uv run -c config.yaml -d /path/to/common/voice/
```

Checkpoints and tensorboard logs will be saved to `logs/lightning_logs/`

Be noted that parameter `-d` should point to where the `*.tsv`s are. In Japanese CV dataset, it's sub directory `ja`.

## Bundle

When bundling app with `pyinstaller`, add

```python
from PyInstaller.utils.hooks import collect_data_files

data = collect_data_files('snfa')

# consume the data in Analyzer
```

To bundle the model weights properly. I'd appreciate it if you offer a better way.

## Todos

- Rust crate
- multi-language

## Licence

`snfa` is released under `ISC Licence`, as shown [here](/LICENCE).

The file `snfa/stft.py` and `snfa/util.py` contains code adapted from `librosa` which obeys `ISC Licence` with different copyright claim. A copy of `librosa`'s licence can be found in [librosa's repo](https://github.com/librosa/librosa/blob/main/LICENSE.md).

The file `snfa/viterbi.py` contains code adapted from `torchaudio` which obeys `BSD 2-Clause "Simplified" License`. A copy of `torchaudio`'s licence can be found in [torchaudio's repo](https://github.com/pytorch/audio/blob/main/LICENSE).

The testing audio file is ripped from Japanese Common Voice Corpus 14.0, 6/28/2023, Public Domain.

## Credit

The neural network used in `snfa` is basically a PyTorch implementation of `CTC*` structure described in [_Evaluating Speech—Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis_](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts).
