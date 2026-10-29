"""Réseaux de prédiction de pannes à sortie de survie en temps discret (§ 5.2).

Chaque réseau encode la fenêtre de télémétrie, la combine aux attributs statiques et renvoie,
pour chaque composant et chaque jour de l'horizon, le logit du risque instantané : probabilité
de panne ce jour-là sachant qu'aucune panne n'est survenue avant [Gensheimer et Narasimhan].
"""

from __future__ import annotations

import torch
from torch import nn


class SurvivalHead(nn.Module):
    def __init__(self, in_features: int, n_components: int, horizon: int):
        super().__init__()
        self.n_components, self.horizon = n_components, horizon
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, n_components * horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.n_components, self.horizon)


class _Base(nn.Module):
    def __init__(self, static_size: int, n_components: int, horizon: int, encoded: int):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(static_size, 32), nn.ReLU())
        self.head = SurvivalHead(encoded + 32, n_components, horizon)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encode(seq), self.static(static)], dim=1))


class GRUNet(_Base):
    def __init__(self, seq_features, static_size, n_components, horizon, hidden=64):
        super().__init__(static_size, n_components, horizon, hidden)
        self.rnn = nn.GRU(seq_features, hidden, batch_first=True)

    def encode(self, seq):
        _, h = self.rnn(seq)
        return h[-1]


class LSTMNet(_Base):
    def __init__(self, seq_features, static_size, n_components, horizon, hidden=64):
        super().__init__(static_size, n_components, horizon, hidden)
        self.rnn = nn.LSTM(seq_features, hidden, batch_first=True)

    def encode(self, seq):
        _, (h, _) = self.rnn(seq)
        return h[-1]


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.act(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return self.act(x + y)


class TCNNet(_Base):
    """Convolutions causales dilatées [Bai et al.] : champ réceptif de 63 heures."""

    def __init__(self, seq_features, static_size, n_components, horizon, channels=32):
        super().__init__(static_size, n_components, horizon, channels)
        self.inp = nn.Conv1d(seq_features, channels, 1)
        self.blocks = nn.Sequential(*[_TemporalBlock(channels, d) for d in (1, 2, 4, 8, 16)])

    def encode(self, seq):
        x = self.blocks(self.inp(seq.transpose(1, 2)))
        return x[:, :, -1]


ARCHITECTURES = {"gru": GRUNet, "lstm": LSTMNet, "tcn": TCNNet}


def survival_loss(
    logits: torch.Tensor, event_bin: torch.Tensor, observed: torch.Tensor, event_weight: float = 1.0
) -> torch.Tensor:
    """Log-vraisemblance négative de la survie en temps discret, avec censure.

    Un composant tombé en panne le jour ``e`` contribue par la survie des jours 0 à e-1 et
    l'événement du jour e ; un composant censuré après ``c`` jours contribue par la survie des
    jours 0 à c-1. ``event_weight`` compense la rareté des pannes.
    """
    horizon = logits.shape[-1]
    days = torch.arange(horizon, device=logits.device).view(1, 1, -1)
    has_event = (event_bin >= 0).unsqueeze(-1)
    survived = days < observed.unsqueeze(-1)
    event = has_event & (days == event_bin.unsqueeze(-1))
    log_h = nn.functional.logsigmoid(logits)
    log_1mh = nn.functional.logsigmoid(-logits)
    ll = (survived * log_1mh).sum() + event_weight * (event * log_h).sum()
    return -ll / logits.shape[0]


def failure_probability(logits: torch.Tensor, days: int) -> torch.Tensor:
    """Probabilité de panne dans les ``days`` prochains jours : [N, K]."""
    log_surv = nn.functional.logsigmoid(-logits[..., :days]).sum(dim=-1)
    return 1.0 - torch.exp(log_surv)
