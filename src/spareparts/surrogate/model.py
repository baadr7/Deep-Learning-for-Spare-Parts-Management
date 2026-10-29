"""Ensemble de réseaux de neurones servant de modèle de substitution (§ 5.4).

Cinq perceptrons multicouches, initialisés différemment, apprennent la relation entre les
paramètres d'un scénario et ses indicateurs [Lakshminarayanan et al.]. La moyenne de leurs
prédictions sert d'estimation et leur dispersion de mesure d'incertitude.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from spareparts.surrogate.doe import OUTPUTS, PARAMS


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, n_out),
    )


class DeepEnsemble:
    def __init__(self, n_models: int = 5, hidden: int = 64):
        self.n_models, self.hidden = n_models, hidden
        self.lo = np.array([PARAMS[p][0] for p in PARAMS], dtype=np.float32)
        self.hi = np.array([PARAMS[p][1] for p in PARAMS], dtype=np.float32)
        self.models: list[nn.Module] = []
        self.y_mean = self.y_std = None

    def _x(self, X: pd.DataFrame) -> torch.Tensor:
        x = (X[list(PARAMS)].to_numpy(np.float32) - self.lo) / (self.hi - self.lo)
        return torch.from_numpy(x * 2.0 - 1.0)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, epochs: int = 600, lr: float = 3e-3) -> None:
        y = Y[OUTPUTS].to_numpy(np.float32)
        self.y_mean, self.y_std = y.mean(axis=0), y.std(axis=0) + 1e-8
        x = self._x(X)
        t = torch.from_numpy((y - self.y_mean) / self.y_std)
        self.models = []
        for i in range(self.n_models):
            torch.manual_seed(i)
            model = _mlp(x.shape[1], t.shape[1], self.hidden)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            for _ in range(epochs):
                perm = torch.randperm(len(x))
                for j in range(0, len(x), 128):
                    idx = perm[j : j + 128]
                    loss = nn.functional.mse_loss(model(x[idx]), t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            model.eval()
            self.models.append(model)

    def predict(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = self._x(X)
        with torch.no_grad():
            preds = np.stack([m(x).numpy() for m in self.models]) * self.y_std + self.y_mean
        mean = pd.DataFrame(preds.mean(axis=0), columns=OUTPUTS, index=X.index)
        std = pd.DataFrame(preds.std(axis=0), columns=OUTPUTS, index=X.index)
        return mean, std

    def save(self, path: Path, metrics: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden": self.hidden,
                "states": [m.state_dict() for m in self.models],
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "metrics": metrics or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple[DeepEnsemble, dict]:
        blob = torch.load(path, weights_only=False)
        ens = cls(n_models=len(blob["states"]), hidden=blob["hidden"])
        ens.y_mean, ens.y_std = blob["y_mean"], blob["y_std"]
        for state in blob["states"]:
            model = _mlp(len(PARAMS), len(OUTPUTS), ens.hidden)
            model.load_state_dict(state)
            model.eval()
            ens.models.append(model)
        return ens, blob["metrics"]


def score(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    """R² et erreur relative moyenne (MAPE) de chaque indicateur."""
    out = {}
    for k in OUTPUTS:
        t, p = Y_true[k].to_numpy(), Y_pred[k].to_numpy()
        r2 = 1.0 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
        nz = np.abs(t) > 1e-6
        mape = float(np.mean(np.abs((t[nz] - p[nz]) / t[nz]))) if nz.any() else float("nan")
        out[k] = {"r2": float(r2), "mape": mape}
    return out
