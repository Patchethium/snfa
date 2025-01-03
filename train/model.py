from torch import nn, Tensor
import torch
import math
from monotonic_align import maximum_path


class Aligner(nn.Module):
    """
    A very simple MLP with 3 hidden layers, ReLU activation and dropout.
    """
    def __init__(self, n_ph: int, dim: int, n_mels: int, dropout: float = 0.1):
        super().__init__()
        self.ph_emb = nn.Embedding(n_ph, dim)
        self.rnn = nn.GRU(dim, dim, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(dim * 2, n_mels)
        self.n_mels = n_mels

    def _forward(
        self, ph: Tensor, mel: Tensor, ph_mask: Tensor, mel_mask: Tensor
    ) -> Tensor:
        x = self.ph_emb(ph)
        x, _ = self.rnn(x)
        x = self.fc(x)
        with torch.no_grad():
            attn_mask = ph_mask.unsqueeze(2) & mel_mask.unsqueeze(1) # [B, Tp, Tm]
            const = -0.5 * math.log(2 * math.pi) * mel.shape[-1]  # scalar
            factor = -0.5 * torch.ones(
                x.shape, dtype=x.dtype, device=x.device
            )  # [B, Tp, Nm]
            y_square = torch.matmul(
                factor, (mel**2).transpose(1, 2)
            )  # [B,Tp,Nm] @ [B,Nm,Tm] = [B, Tp, Tm]
            y_mu_double = torch.matmul(
                2.0 * (factor * x), mel.transpose(1, 2)
            )  # [B, Tp, Nm] @ [B, Nm, Tm] = [B, Tp, Tm]
            mu_square = torch.sum(factor * (x**2), 2).unsqueeze(-1)  # [B,Tp,1]
            log_prior = y_square - y_mu_double + mu_square + const  # [B, Tp, Tm]
            log_prior = log_prior * attn_mask
            attn = maximum_path(log_prior, attn_mask)
            attn = attn.detach()  # [B, Tp, Tm]
        return attn, x

    def forward(self, ph: Tensor, mel: Tensor, ph_mask: Tensor, mel_mask: Tensor):
        attn, ph = self._forward(ph, mel, ph_mask, mel_mask)
        expanded = torch.matmul(attn.transpose(1,2), ph)
        prior_loss = ((expanded - mel).pow(2) * mel_mask.unsqueeze(-1)).sum() / mel_mask.sum() / self.n_mels
        return attn, expanded, prior_loss

    def inference(self, x: Tensor, mel:Tensor,ph_mask: Tensor, mel_mask: Tensor):
        attn, x = self._forward(x, mel, ph_mask, mel_mask)
        return attn
