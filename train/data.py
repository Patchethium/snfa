import torch
import torchaudio as ta
from torchaudio.transforms import MelSpectrogram, Resample
from torch.utils.data import Dataset
import jpreprocess as jpp
import csv
import os
from constants import jp_phones
import numpy as np
from hp import hp
from tqdm.auto import tqdm


class CommonVoiceDataset(Dataset):
    def __init__(self, path: str, o_sr: int=48000, sr: int=16000, hop_size: int=160, n_fft: int=1024, n_mels: int = 80):
        # it takes the path of train/dev/test.tsv file
        # format: client_id, path, sentence, up_votes, down_votes, age, gender, accents, variant, locale, segment
        assert os.path.exists(path)
        root = os.path.dirname(path)
        self.clip_root = os.path.join(root, "clips")
        self.files = []
        self.texts = []
        self.j = jpp.jpreprocess()
        with open(path, "r") as f:
            data = csv.reader(f, delimiter="\t")
            _ = next(data)
            for row in data:
                # filter out downvote > 3
                if int(row[4]) > 3:
                    continue
                self.files.append(row[1])
                self.texts.append(row[2])
        self.resample = Resample(o_sr, sr)
        self.mel = MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_size, n_mels=n_mels, window_fn=torch.hann_window, power=1)
        self.phone_dict = {p: i for i, p in enumerate(jp_phones)}
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wav, _ = ta.load(os.path.join(self.clip_root, self.files[index]))
        wav = wav.mean(0, keepdim=True)
        wav = self.resample(wav)
        mel = self.mel(wav)
        logmel = torch.log(mel + 1e-7).transpose(1,2).squeeze(0)
        phones = self.j.g2p(self.texts[index]).split(" ")
        phones = [self.phone_dict["pau"]] + [self.phone_dict[p] for p in phones] + [self.phone_dict["pau"]]
        phones = torch.tensor(phones, dtype=torch.long)
        return logmel, phones


class FileDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.len = len(os.listdir(path)) // 2
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mel = torch.tensor(np.load(os.path.join(self.path, f"mel_{index}.npy")))
        phone = torch.tensor(np.load(os.path.join(self.path, f"phone_{index}.npy")))
        return mel, phone


def main():
    corpus_path = hp["corpus_path"]
    train_ds = CommonVoiceDataset(
        os.path.join(corpus_path, "train.tsv"),
        sr=hp["sr"],
        n_fft=hp["n_fft"],
        hop_size=hp["hop_size"],
        n_mels=hp["n_mels"],
    )
    dev_ds = CommonVoiceDataset(
        os.path.join(corpus_path, "dev.tsv"),
        sr=hp["sr"],
        n_fft=hp["n_fft"],
        hop_size=hp["hop_size"],
        n_mels=hp["n_mels"],
    )
    test_ds = CommonVoiceDataset(
        os.path.join(corpus_path, "test.tsv"),
        sr=hp["sr"],
        n_fft=hp["n_fft"],
        hop_size=hp["hop_size"],
        n_mels=hp["n_mels"],
    )
    out_dir = hp["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "dev"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
    for name, ds in zip(["train", "dev", "test"], [train_ds, dev_ds, test_ds]):
        print(f"Processing {name} dataset")
        for i, (mel, phone) in tqdm(enumerate(ds)):
            np.save(os.path.join(out_dir, name, f"mel_{i}.npy"), mel)
            np.save(os.path.join(out_dir, name, f"phone_{i}.npy"), phone)
    print("Done!")

if __name__ == "__main__":
    main()