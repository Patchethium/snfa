import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jpreprocess as jpp
import lightning.pytorch as L
import torch
import torch.nn.functional as F
import torchaudio as ta
from lightning.pytorch import loggers
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

PHONE_SET = [
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
]

PHONE_SET = ["pad"] + PHONE_SET


@dataclass
class Config:
    dim: int

    n_mels: int
    sr: int
    n_fft: int
    win_size: int
    hop_size: int
    max_len: int
    cache_dir: str

    max_epoch: int
    lr: float
    decay: float
    batch_size: int
    num_workers: int = 4


class CVDataset(Dataset):
    def __init__(self, root: str, split: str | Iterable[str], cfg: Config):
        self.cfg = cfg
        self.root = root
        self.data = []
        if isinstance(split, str):
            split = [split]
        for s in split:
            with open(os.path.join(root, s), "r") as f:
                lines = f.readlines()
            reader = csv.reader(lines, delimiter="\t")
            _ = next(reader)  # skip header
            for data in reader:
                file, text, upvote, downvote = [data[i] for i in range(1, 5)]
                if upvote < downvote:
                    continue
                self.data.append((file, text))

        self.mel = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            win_length=cfg.win_size,
            hop_length=cfg.hop_size,
            n_mels=cfg.n_mels,
            f_min=0.0,
            f_max=None,
        )
        self.jpp = jpp.jpreprocess()
        self.phone_set = {phone: idx for idx, phone in enumerate(PHONE_SET)}

        if not os.path.exists(cfg.cache_dir):
            os.makedirs(cfg.cache_dir)

    def __len__(self):
        return len(self.data)

    def _g2p(self, text: str) -> Tensor:
        ph = self.jpp.g2p(text).lower()
        ph = ph.split()  # split by whitespace
        ph = ["pau"] + ph + ["pau"]  # add start and end pause
        ph_idx = [self.phone_set[p] for p in ph if p in self.phone_set]
        return torch.tensor(ph_idx, dtype=torch.long)  # pad is given in PHONE_SET

    def __getitem__(self, index):
        file, text = self.data[index]
        if os.path.exists(os.path.join(self.cfg.cache_dir, file + ".pt")):
            mel, phoneme = torch.load(os.path.join(self.cfg.cache_dir, file + ".pt"))
            return mel, phoneme
        path = os.path.join(self.root, "clips", file)
        wav, sr = ta.load(path)
        if sr != 16000:
            wav = ta.functional.resample(wav, sr, 16000)
        if len(wav.shape) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        mel = self.mel(wav)
        mel = torch.log(mel + 1e-7).transpose(0, 1)  # (T, C)

        phoneme = self._g2p(text)
        if mel.shape[1] > self.cfg.max_len:
            print(f"WARNING, {file} exceed max mel length")
            mel = mel[:, : self.cfg.max_len]

        torch.save((mel, phoneme), os.path.join(self.cfg.cache_dir, file + ".pt"))
        return mel, phoneme


def len2mask(lens: Tensor) -> Tensor:
    """Create a mask tensor from lengths."""
    max_len = lens.max().item()
    batch_size = lens.shape[0]
    mask = torch.arange(max_len, device=lens.device).expand(
        batch_size, max_len
    ) < lens.unsqueeze(1)
    return mask


def collate_fn(batch):
    mel, phoneme = zip(*batch)
    mel_lens = torch.LongTensor([m.shape[0] for m in mel])
    phoneme_lens = torch.LongTensor([p.shape[0] for p in phoneme])
    mel_mask = len2mask(mel_lens)
    mel = nn.utils.rnn.pad_sequence(mel, batch_first=True, padding_value=0.0)
    phoneme = nn.utils.rnn.pad_sequence(phoneme, batch_first=True, padding_value=0)
    return mel, phoneme, mel_lens, phoneme_lens, mel_mask


def get_dataloaders(root: str, cfg: Config) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CVDataset(root, "filtered_validated.tsv", cfg)
    val_dataset = CVDataset(root, "dev.tsv", cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


class Aligner(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=len(PHONE_SET)):
        super().__init__()
        self.n_mels = input_dim
        self.pre = nn.Linear(input_dim, hidden_dim)
        self.bi_rnn = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _forward(self, x, mask: Optional[Tensor]):  # x: (batch, time, mel)
        x = self.pre(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        r = x
        x, _ = self.bi_rnn(x)
        x = x + r
        o = self.fc(x)
        logits = torch.log_softmax(o, dim=-1)  # (batch, time, output_dim)
        return logits

    def forward(
        self,
        mel: Tensor,
        phoneme: Tensor,
        mel_lens: Tensor,
        phoneme_lens: Tensor,
        mel_mask: Tensor,
    ):
        logits = self._forward(mel, mel_mask)
        ctc_loss = F.ctc_loss(
            logits.transpose(0, 1),  # (time, batch, output_dim)
            phoneme,
            mel_lens,
            phoneme_lens,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        return ctc_loss

    def infer(self, mel: Tensor, mel_mask: Optional[Tensor] = None):
        logits = self._forward(mel, mel_mask)
        return logits


class Trainer(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = Aligner(cfg.n_mels, cfg.dim, len(PHONE_SET))

    def forward(self, mel):
        return self.model.infer(mel)

    def training_step(self, batch, batch_idx):
        mel, phoneme, mel_lens, phoneme_lens, mel_mask = batch
        ctc_loss = self.model(mel, phoneme, mel_lens, phoneme_lens, mel_mask)
        loss = ctc_loss
        losses = {
            "loss": loss,
            "ctc_loss": ctc_loss,
        }
        self.log_dict({"train/" + k: v for k, v in losses.items()})
        return losses

    def validation_step(self, batch, batch_idx):
        losses = self.training_step(batch, batch_idx)
        self.log_dict({"val/" + k: v for k, v in losses.items()})
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.cfg.decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    L.seed_everything(3407)
    parser = argparse.ArgumentParser(description="Train the SNFA model")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        default="data",
        help="Path to the dataset directory",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg = Config(**cfg)
    model = Trainer(cfg)
    train, val = get_dataloaders(args.data_dir, cfg)
    logger = loggers.TensorBoardLogger("logs")
    trainer = L.Trainer(
        max_epochs=cfg.max_epoch,
        accelerator="auto",
        log_every_n_steps=20,
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)


if __name__ == "__main__":
    main()
