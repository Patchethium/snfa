from typing import List, Tuple
import librosa
import numpy as np
import torch
from hp import hp
from train.model import Decoder, Encoder
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn import CTCLoss, MSELoss
from os import path
from torch.nn.functional import one_hot
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils import clip_grad
from torch.optim.lr_scheduler import ExponentialLR


class SnfaDataset(Dataset):
    def __init__(self, mel_path: str, tsv_path: str, device: torch.device) -> None:
        super().__init__()
        self.mel_path = mel_path
        if not path.exists(mel_path):
            raise Exception("Mel path doesn't exist")
        if not path.exists(tsv_path):
            raise Exception("tsv path doesn't exist")
        self.df = pd.read_table(tsv_path)
        phoneme = self.df["phoneme"]
        self.paths: pd.Series[str] = self.df["path"]
        self.phone_set: List[str] = hp["phone_set"]
        self.phone_vec = []
        self.device = device
        for sentence in phoneme:
            phone_seq = sentence.split(" ")
            idx_seq = [
                self.phone_set.index(phone) + 1 for phone in phone_seq
            ]  # leave 0 for blank note
            idx_seq = torch.LongTensor(idx_seq).to(device)
            self.phone_vec.append(idx_seq)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        filename: str = self.paths[index]
        mel = np.load(path.join(self.mel_path, filename + ".npy"))
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = librosa.util.normalize(mel, axis=1) + 1.0
        mel_tensor = torch.from_numpy(mel).to(self.device)  # [N,S]
        mel_tensor = mel_tensor.transpose(0, 1)  # [S,N]
        ph = self.phone_vec[index]
        return mel_tensor, ph


def make_pad_mask(lens: List[int]):
    mask = [torch.ones(l) for l in [rows for rows in lens]]
    pad_mask = pad_sequence(mask, batch_first=True, padding_value=0)
    return pad_mask.transpose(0, 1).unsqueeze(-1)


def pad_collate(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (mels, phs) = zip(*batch)
    m_lens = [m.shape[0] for m in mels]
    p_lens = [p.shape[0] for p in phs]

    mel_pad = pad_sequence(mels, batch_first=False, padding_value=0)
    ph_pad = pad_sequence(phs, batch_first=False, padding_value=0)

    m_mask = make_pad_mask(m_lens).to(device)
    p_mask = make_pad_mask(p_lens).to(device)

    return (
        mel_pad,
        ph_pad,
        torch.LongTensor(m_lens).to(device),
        torch.LongTensor(p_lens).to(device),
        m_mask,
        p_mask,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        mel_dim=hp["n_mels"],
        hid_dim=hp["hid_dim"],
        phone_dim=hp["phone_dim"],
    ).to(device)
    decoder = Decoder(
        mel_dim=hp["n_mels"],
        hid_dim=hp["hid_dim"],
        phone_dim=hp["phone_dim"],
    ).to(device)
    dataset = SnfaDataset(
        mel_path=f"corpus/{hp['corpus_name']}/mels",
        tsv_path=f"corpus/{hp['corpus_name']}/train_clean.tsv",
        device=device,
    )

    train_ds, val_ds = random_split(dataset, [0.9, 0.1])

    train_dl, val_dl = [
        DataLoader(
            ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=pad_collate
        )
        for ds in (train_ds, val_ds)
    ]

    ctc = CTCLoss(blank=0, zero_infinity=True)
    mse = MSELoss()
    reconstr_weight: float = hp["reconstr_weight"]

    step = 0

    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=hp["learning_rate"]
    )
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    epochs = hp["epochs"]

    writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        for mel, phoneme, mel_lens, ph_lens, mel_mask, _ in train_dl:
            step += 1
            optimizer.zero_grad()
            label = encoder.forward(mel, mel_mask)

            ctc_loss = ctc.forward(label, phoneme.transpose(0, 1), mel_lens, ph_lens)
            reconstructed = decoder.forward(label, mel_mask)
            mse_loss = mse.forward(reconstructed, mel)
            loss = ctc_loss + reconstr_weight * mse_loss
            loss.backward()

            clip_grad.clip_grad_norm_(decoder.parameters(), 0.1)
            clip_grad.clip_grad_norm_(encoder.parameters(), 0.1)

            optimizer.step()

            writer.add_scalar("Train/CTC", ctc_loss, step)
            writer.add_scalar("Train/MSE", mse_loss, step)
            writer.add_scalar("Train/Total", loss, step)

            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                mel, phoneme, mel_lens, ph_lens, mel_mask, _ = next(iter(val_dl))
                label = encoder.forward(mel, mel_mask)
                ctc_loss = ctc.forward(
                    label, phoneme.transpose(0, 1), mel_lens, ph_lens
                )
                reconstructed = decoder.forward(label, mel_mask)
                mse_loss = mse.forward(reconstructed, mel)
                loss = ctc_loss + reconstr_weight * mse_loss
                writer.add_scalar("Val/CTC", ctc_loss, step)
                writer.add_scalar("Val/MSE", mse_loss, step)
                writer.add_scalar("Val/Total", loss, step)

                if step % hp["ckpt_step"] == 0:
                    torch.save(encoder.state_dict(), f"e-{step}.pth")
                    torch.save(decoder.state_dict(), f"d-{step}.pth")

                encoder.train()
                decoder.train()

        print(f"epoch: {epoch+1}")
        scheduler.step()


if __name__ == "__main__":
    main()
