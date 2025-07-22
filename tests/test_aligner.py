"""Test module, what can I say?"""

import jpreprocess as jpp
import torchaudio as ta

from snfa import Aligner, trim_audio


def test_aligner():
    """Tests aligner."""
    aligner = Aligner()
    jp = jpp.jpreprocess()

    wav_file = "tests/common_voice_ja_19482480.mp3"
    text_file = "tests/common_voice_ja_19482480.txt"

    wav, sr = ta.load(wav_file)
    if sr != aligner.sr:
        wav = ta.functional.resample(wav, sr, aligner.sr)

    wav = wav.numpy()

    with open(text_file) as f:
        text = f.readline().rstrip()

    phoneme = jp.g2p(text).lower().split()
    _ = aligner.align(wav, phoneme, pad_pause=True)

def test_trim():
    """Tests trim audio."""
    wav_file = "tests/common_voice_ja_19482480.mp3"
    wav, _ = ta.load(wav_file)
    wav = wav.numpy()
    trimmed, _ = trim_audio(wav, top_db=20, frame_length=1024, hop_length=512)
    assert trimmed.shape != wav.shape # should be different after trimming
