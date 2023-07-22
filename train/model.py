import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, mel_dim: int, hid_dim: int, phone_dim: int) -> None:
        super().__init__()
        self.rnn1 = nn.LSTM(
            input_size=mel_dim, hidden_size=hid_dim, bidirectional=True
        )
        self.rnn2 = nn.LSTM(
            input_size=2 * hid_dim,
            hidden_size=hid_dim,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hid_dim, phone_dim + 1)
        self.softmax = nn.Softmax(2)

    def forward(self, x: Tensor):
        x, _ = self.rnn1(x)
        x = F.leaky_relu(x, 0.01)
        x, _ = self.rnn2(x)
        x = F.leaky_relu(x, 0.01)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class Decoder(nn.Module):
    def __init__(self, mel_dim: int, hid_dim: int, phone_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(phone_dim, hid_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hid_dim, mel_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):  # [S,B,P+1]
        x = x[:, :, 1:]  # [S,B,P], remove blank note
        x = F.normalize(x, p=1, dim=2)
        return self.layers(x)


if __name__ == "__main__":
    enc = Encoder(80, 128, 45)
    dec = Decoder(80, 128, 45)
    x = torch.rand(128, 32, 80)
    y = enc(x)
    z = dec(y)