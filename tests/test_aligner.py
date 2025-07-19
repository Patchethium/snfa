from snfa import Aligner
import jpreprocess as jpp
import torchaudio as ta


def test_aligner():
    aligner = Aligner()
    jp = jpp.jpreprocess()

    wav_file = "tests/common_voice_ja_19482480.mp3"
    text_file = "tests/common_voice_ja_19482480.txt"

    wav, sr = ta.load(wav_file)
    if sr != aligner.sr:
        wav = ta.functional.resample(wav, sr, aligner.sr)
    
    wav = wav.numpy()

    with open(text_file, "r") as f:
        text = f.readline().rstrip()
    
    phoneme = jp.g2p(text).lower().split()
    _ = aligner.align(wav, phoneme, pad_pause=True)
