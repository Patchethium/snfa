import torch
from torch import optim
from torch.utils.data import DataLoader
from hp import hp
from data import FileDataset
import os
from model import Aligner
from constants import jp_phones
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

def lens2mask(lens):
    max_len = max(lens)
    mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    return mask


def collate_fn(batch):
    mels, phones = zip(*batch)
    mel_lens = torch.tensor([len(m) for m in mels], dtype=torch.long)
    phone_lens = torch.tensor([len(p) for p in phones], dtype=torch.long)
    mel_mask = lens2mask(mel_lens)
    phone_mask = lens2mask(phone_lens)
    mels = pad_sequence(mels, batch_first=True)
    phones = pad_sequence(phones, batch_first=True)
    return mels, phones, mel_mask, phone_mask


def train():
    out_dir = hp["out_dir"]
    train_ds = FileDataset(os.path.join(out_dir, "train"))
    dev_ds = FileDataset(os.path.join(out_dir, "dev"))
    train_dl = DataLoader(
        train_ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=hp["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Aligner(len(jp_phones), hp["hid_dim"], hp["n_mels"])
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
    step = 0
    writer = SummaryWriter()
    for e in range(1, hp["epochs"] + 1):
        for mel, phones, mel_mask, phone_mask in train_dl:
            mel, phones, phone_mask, mel_mask = [
                t.to(device) for t in [mel, phones, phone_mask, mel_mask]
            ]
            optimizer.zero_grad()
            _, expanded, prior_loss = model.forward(
                phones, mel, phone_mask, mel_mask
            )
            prior_loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar("train/prior_loss", prior_loss.item(), step)
            if step % hp["plot_interval"] == 0:
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.imshow(mel[0].detach().T.cpu().numpy(), aspect="auto", origin="lower")
                ax2.imshow(expanded[0].detach().T.cpu().numpy(), aspect="auto", origin="lower")
                writer.add_figure("train/mel_vs_expanded", fig, step)
                plt.clf()
        if e % hp["save_interval_epoch"] == 0:
            torch.save(model.state_dict(), f"model_epoch_{e}.pth")
        print(f"Epoch {e} done!")
    writer.close()

if __name__ == "__main__":
    train()