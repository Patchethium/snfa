# snfa

`snfa` (Simple Neural Forced Aligner) is a phoneme-to-audio forced aligner built for embedded usage in python programs, with its only inference dependency being `numpy` and python 3.7 or later.

- Tiny model size (~1 MB)
- Numpy as the only dependency
- MFA comparable alignment quality

> [!Note]
> You still need `PyTorch` and some other libs if you want to do training.

## Inference

```bash
pip install snfa
```

A pre-trained model weight `jp.npz` is included.

`jp.npz` is a weight file trained on Japanese Common Voice Corpus 22.0, 2025-06-20. The model weight is released into `Public Domain`.

```python
import snfa
from snfa import Segment
import librosa # or soundfile, torchaudio, scipy, etc.


aligner = snfa.Aligner() # use custom model by passing its path to this function
# NOTE: the default model is uncased, it doesn't make difference between `U` and `u`
transcript = "k o N n i ch i w a".lower().split(" ") # remember to lower it here

# you can also use `scipy` or `wavfile` as long as it's 
# 1. mono channel numpy array with shape (T,), dtype=np.float32
# 2. normalized to [-1,1]
# 3. sample rate matches model's `sr`
x, sr = librosa.load("sample.wav", sr=aligner.sr)
# trim the audio, this may improve alignment quality
x, _ = librosa.effects.trim(x, top_db=20)
# we also provide a utility function to trim
# it's basically ripped off from librosa so you don't have to install it
x, _ = snfa.trim_audio(x, top_db=20)

segments: list[Segment] = aligner(x, transcript)

print(segments)
# (phoneme label, start mili-sec, end mili-sec)
# [('pau', 0, 900),
#  ('k', 900, 920),
#  ('o', 920, 1080),
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

### Exporting

To use the model in `numpy`, export the checkpoint with

```bash
uv run export.py -c config.yaml --ckpt /path/to/checkpoint -o output.npz
```

### Publishing

Usually I am responsible for publishing the package to PyPI, this section serves as a reminder for myself.

1. copy the exported `jp.npz` to `src/snfa/models/`
2. uv build
3. uv publish

## Bundle

When bundling app with `pyinstaller`, add

```python
from PyInstaller.utils.hooks import collect_data_files

data = collect_data_files('snfa')

# consume `data` in Analyzer
```

To bundle the model weights properly. I'd appreciate it if you offer a better way.

## Todos

- Rust crate
- multi-language support

## Licence

`snfa` is released under `ISC Licence`, as shown [here](/LICENCE).

The file `snfa/stft.py` and `snfa/util.py` contains code adapted from `librosa` which obeys `ISC Licence` with different copyright claim. A copy of `librosa`'s licence can be found in [librosa's repo](https://github.com/librosa/librosa/blob/main/LICENSE.md).

The testing audio file is ripped from Japanese Common Voice Corpus 14.0, 6/28/2023, Public Domain.

## How it works

The model consists of a text encoder and a mel encoder. It uses Monotonic Alignment Search (MAS) for alignment, and Straight-Through Estimator to optimize both encoders. The pseudo code is as follows:

```python
z_t = text_encoder(phoneme)
z_m = mel_encoder(mel)
dist = -dist(z_t, z_m) # [T_m, T_t], negative pairwise distance matrix
alignment = MAS(dist)
z_m_ = matmul(alignment, z_m)
z_t_ = matmul(alignment^T, z_t)

logits_t = proj_t(z_m_)
z_t_ = proj_m(z_t_)

# optional, doesn't make much difference
# z_m_ = z_m + sg(z_m_ - z_m)
# z_t_ = z_t + sg(z_t_ - z_t)

L = (z_t_ - sg(z_m))**2 + cross_entropy(logits_t, phoneme)
```

Where `sg` is the stop-gradient operator.

> [!Note]
> As far as I know, this idea (using Straight-Through Estimator for Forced Alignment) is yet to be published before, but as trivial as it is, perhaps it's not a big thing after all.

### Silence Handling

We use g2p model to convert text to phoneme sequence, which doesn't include precise silences. To tackle this, we add a `_` placeholder to every inter-phoneme gap. Given a threshold, we consider any `_` longer than the threshold as silence, else it's appended to the previous phoneme. The threshold is set to 3 frames (30ms) by default.

## Cite

If you find `snfa` useful, it'd be a great pleasure if you cite it as follows:

```bibtex
@misc{snfa2023,
    author = {Patchethium},
    title = {snfa: Simple Neural Forced Aligner},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/patchethium/snfa}}
}
```