import argparse
import csv
import datetime
import math
import os
from dataclasses import dataclass
from random import randint
from typing import List

import lightning.pytorch as L
import matplotlib.pyplot as plt
import torch
import torchaudio as ta
from jpreprocess import jpreprocess
from monotonic_align import maximum_path
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAUSE_TOKEN = "_"
CACHE_DIR = "./vendor"


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# config
@dataclass
class Config:
    lang: str
    # data
    dataset_path: str
    phone_set: List[str]
    sr: int
    n_mels: int
    n_fft: int
    hop_size: int
    win_size: int
    max_mel_len: int
    max_text_len: int

    # model
    dim: int
    dist_power: int
    mel_enc_layers: int
    text_enc_layers: int

    # training
    seed: int
    lr: float
    lr_decay: float
    batch_size: int
    epochs: int
    num_workers: int

    # inference
    pause_threshold: int


def get_config(path: str) -> Config:
    cfg_data = OmegaConf.load(path)
    cfg = Config(**cfg_data)
    assert PAUSE_TOKEN not in set(cfg.phone_set), (
        f"Pause token {PAUSE_TOKEN} should not be present in phone set"
    )
    cfg.phone_set = cfg.phone_set + [PAUSE_TOKEN]  # add pause symbol
    return cfg


# dataset
class CommonVoiceDataset(Dataset):
    def __init__(self, cfg: Config, split: str):
        # metadata is stored in a tsv file cfg.dataset_path/{split}.tsv
        
        self.cfg = cfg
        self.split = split
        self.data = []
        self.phone_dict = {p: i for i, p in enumerate(cfg.phone_set)}
        max_audio_len = cfg.max_mel_len * cfg.hop_size / cfg.sr * 1000  # in ms
        tsv_path = os.path.join(cfg.dataset_path, f"{split}.tsv")
        duration_tsv_path = os.path.join(cfg.dataset_path, "clip_durations.tsv")
        durations = {}
        with open(duration_tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                durations[row["clip"]] = int(row["duration[ms]"])
        filtered = 0
        filter_wavs = set(["common_voice_ja_39027804.mp3"]) # someone uploaded the whole chapter from `吾輩は猫である` as one clip
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["up_votes"] < row["down_votes"] or row["path"] in filter_wavs:  # skip low quality data
                    continue
                if durations[row["path"]] > max_audio_len or len(row["sentence"].strip()) > cfg.max_text_len:  # skip long audio
                    filtered += 1
                    continue
                self.data.append([row["sentence"].strip(), row["path"]])
        print(f"Filtered {filtered} samples longer than {max_audio_len} ms or text length {cfg.max_text_len}.")
        match cfg.lang:
            case "ja":
                self.jpp = jpreprocess()
                self.g2p = self._g2p_jpp
            case _:
                raise ValueError(f"Unsupported language: {cfg.lang}")
        self.mel_t = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_size,
            n_mels=cfg.n_mels,
            power=1.0,
            normalized=False,  # if True, the random gain would be meaningless
        )

    def insert_pause(self, phones: List[int]) -> List[int]:
        res = [PAUSE_TOKEN]
        for p in phones:
            res.append(p)
            res.append(PAUSE_TOKEN)
        return res

    def _add_noise(self, wav: Tensor) -> Tensor:
        noise_rate = randint(0, 5) / 1000  # 0 to 0.5%
        noise = torch.randn_like(wav) * noise_rate
        return wav + noise

    def _rand_gain(self, wav: Tensor) -> Tensor:
        gain_db = randint(-3, 3)  # -3 to +3 dB
        gain = 10 ** (gain_db / 20)
        return wav * gain

    def _g2p_jpp(self, text: str) -> List[int]:
        phones = self.jpp.g2p(text).split()
        phones = list(filter(lambda p: p in self.cfg.phone_set, phones))
        phones = self.insert_pause(phones)
        phone_indices = [self.phone_dict[p] for p in phones]
        return torch.LongTensor(phone_indices)

    def _mel_spec(self, wav_path: str) -> Tensor:
        wav, sr = ta.load(wav_path)
        if sr != self.cfg.sr:
            wav = ta.functional.resample(wav, sr, self.cfg.sr)
        wav = wav.mean(dim=0)  # convert to mono
        # data augmentation
        wav = self._add_noise(wav)
        wav = self._rand_gain(wav)
        mel_spec = self.mel_t.forward(wav).transpose(0, 1)  # (T, n_mels)
        if mel_spec.shape[0] > self.cfg.max_mel_len:
            mel_spec = mel_spec[: self.cfg.max_mel_len, :]
        return mel_spec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, wav_path = self.data[idx]
        cache_path = os.path.join(
            CACHE_DIR,
            f"{self.split}_{idx}.pt",
        )
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        phones = self.g2p(text)
        wav_path = os.path.join(self.cfg.dataset_path, "clips", wav_path)
        mel_spec = self._mel_spec(wav_path)
        torch.save((phones, mel_spec), cache_path)
        return phones, mel_spec


def len2mask(lengths: list[int]) -> Tensor:
    max_len = max(lengths)
    bs = len(lengths)
    mask = torch.zeros((bs, max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def collate_fn(batch):
    # returns padded phones&mels and mask for phones&mels
    phones, mels = zip(*batch)  # noqa: B905
    phone_lens = [p.shape[0] for p in phones]
    mel_lens = [m.shape[0] for m in mels]
    phones = pad_sequence(phones, batch_first=True, padding_value=0)
    mels = pad_sequence(mels, batch_first=True, padding_value=0.0)
    phone_mask = len2mask(phone_lens)
    mel_mask = len2mask(mel_lens)
    return phones, phone_mask, mels, mel_mask


# model
class Aligner(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.text_emb = nn.Embedding(len(cfg.phone_set), cfg.dim)
        self.text_enc = nn.ModuleList(
            [
                nn.GRU(cfg.dim, cfg.dim // 2, batch_first=True, bidirectional=True)
                for _ in range(cfg.text_enc_layers)
            ]
        )
        for layer in self.text_enc:
            layer.flatten_parameters()
        self.mel_head = nn.Linear(cfg.dim, cfg.n_mels)

    def forward(
        self,
        text: torch.LongTensor,
        text_mask: torch.BoolTensor,
        mel: torch.FloatTensor,
        mel_mask: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        x = self.text_emb(text)  # (B, T_t, dim)
        for layer in self.text_enc:
            r = x
            x, _ = layer(x)
            x = x + r  # residual
        x = self.mel_head.forward(x)  # (B, T_t, n_mels)
        attn_mask = text_mask.unsqueeze(-1) & mel_mask.unsqueeze(1)  # (B, T_t, T_m)
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.cfg.n_mels  # scalar
            factor = -0.5 * torch.ones(
                x.shape, dtype=x.dtype, device=x.device
            )  # [B, T_t, n_mels]
            y_square = torch.matmul(factor, mel.transpose(1, 2) ** 2)  # (B, T_t, T_m)
            y_mu_double = torch.matmul(
                2.0 * (factor * x), mel.transpose(1, 2)
            )  # (B, T_t, T_m)
            mu_square = torch.sum(factor * (x**2), dim=-1).unsqueeze(-1)  # (B, T_t, 1)
            log_prior = y_square - y_mu_double + mu_square + const
            path = maximum_path(log_prior, attn_mask)
            path = path.detach()
        mel_ = path.transpose(1, 2) @ x  # (B, T_t, n_mels)
        del path
        return mel_


# Lightning module
class SNFA(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = Aligner(cfg)

    def _forward(self, batch, prefix: str = "train"):
        phones, phone_mask, mels, mel_mask = batch
        recon_mel = self.model(phones, phone_mask, mels, mel_mask)
        loss = (
            torch.sum(
                0.5
                * ((mels - recon_mel) ** 2 + math.log(2 * math.pi))
                * mel_mask.unsqueeze(-1)
            )
            / mel_mask.sum()
            / self.cfg.n_mels
        )
        return loss

    def on_validation_epoch_start(self):
        self.sample_batch_idx = randint(0, self.val_len // self.cfg.batch_size - 1)

    def _plot_alignment(self, path: Tensor, prefix: str):
        plt.imshow(path[0].detach().cpu().numpy(), aspect="auto", origin="lower")
        self.logger.experiment.add_figure(
            f"{prefix}/alignment", plt.gcf(), global_step=self.global_step
        )
        plt.clf()
        plt.close()

    def training_step(self, batch, batch_idx):
        loss = self._forward(batch, "train")
        self.log("train/loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._forward(batch, "val")
            self.log("val/loss", loss.item())
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.cfg.lr_decay
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = CommonVoiceDataset(self.cfg, split="train")
        self.train_len = len(train_dataset)
        return DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataset = CommonVoiceDataset(self.cfg, split="dev")
        self.val_len = len(val_dataset)
        return DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )


# main
def main(cfg_path: str):
    ensure_dir(CACHE_DIR)
    cfg = get_config(cfg_path)
    model = SNFA(cfg)
    L.seed_everything(cfg.seed)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trainer = L.Trainer(
        log_every_n_steps=20,
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        logger=L.loggers.TensorBoardLogger("logs/"),
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor="val/loss",
                filename=date + "-{epoch:02d}-{val/loss:.4f}",
                save_top_k=3,
                mode="min",
            ),
            L.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/ja.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
