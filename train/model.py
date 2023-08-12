from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self, mel_dim: int, hid_dim: int, phone_dim: int
    ) -> None:
        super(Encoder, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=mel_dim,
            hidden_size=hid_dim,
            bidirectional=True,
        )
        self.rnn2 = nn.GRU(
            input_size=2 * hid_dim,
            hidden_size=hid_dim,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hid_dim, phone_dim + 1)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x, _ = self.rnn1(x)
        x = F.leaky_relu(x, 0.01)
        x, _ = self.rnn2(x)
        x = F.leaky_relu(x, 0.01)
        x = self.fc(x)
        x = self.softmax(x)
        if mask is not None:
            x = x * mask
        return x


class Decoder(nn.Module):
    def __init__(self, mel_dim: int, hid_dim: int, phone_dim: int) -> None:
        super(Decoder, self).__init__()
        hid_dim *= 2
        self.layers = nn.Sequential(
            nn.Linear(phone_dim, hid_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hid_dim, mel_dim),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # [S,B,P+1]
        x = x[:, :, 1:]  # [S,B,P], remove blank note
        x = self.layers(x)
        if mask is not None:
            x = x * mask
        return x
