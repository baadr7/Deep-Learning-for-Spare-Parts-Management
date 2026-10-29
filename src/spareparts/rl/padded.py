"""Loi multi-catégorielle masquée calculée en une seule opération, pour MaskablePPO au niveau N2.

La loi de sb3-contrib crée un objet ``Categorical`` par dimension d'action à chaque appel. Au
niveau N2, avec 108 dimensions, cela coûte environ 80 ms par décision, quinze fois le temps de
simulation d'une journée. Ici, les logits sont rangés dans un tableau dimensions × options
complété par des options fictives toujours masquées, et toutes les opérations portent sur ce
tableau. Les valeurs calculées sont les mêmes que celles de sb3-contrib.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

MASKED_LOGIT = -1e8


class PaddedMaskableMultiCategorical(MaskableDistribution):
    def __init__(self, action_dims: list[int]):
        super().__init__()
        self.action_dims = [int(n) for n in action_dims]
        self.total = sum(self.action_dims)
        offsets = np.concatenate([[0], np.cumsum(self.action_dims)[:-1]])
        width = max(self.action_dims)
        index = np.zeros((len(self.action_dims), width), dtype=np.int64)
        valid = np.zeros((len(self.action_dims), width), dtype=bool)
        for d, size in enumerate(self.action_dims):
            index[d, :size] = offsets[d] + np.arange(size)
            valid[d, :size] = True
        self.index = torch.from_numpy(index)
        self.valid = torch.from_numpy(valid)
        self._raw: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self.log_p: torch.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.total)

    def proba_distribution(self, action_logits: torch.Tensor) -> PaddedMaskableMultiCategorical:
        self._raw = action_logits.view(-1, self.total)[:, self.index]
        self._mask = self.valid.expand_as(self._raw)
        self._normalize()
        return self

    def apply_masking(self, masks) -> None:
        mask = self.valid
        if masks is not None:
            flat = torch.as_tensor(masks, dtype=torch.bool).view(-1, self.total)
            mask = flat[:, self.index] & self.valid
        self._mask = mask.expand_as(self._raw)
        self._normalize()

    def _normalize(self) -> None:
        logits = self._raw.masked_fill(~self._mask, MASKED_LOGIT)
        self.log_p = logits - logits.logsumexp(dim=2, keepdim=True)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        a = actions.long().view(-1, len(self.action_dims), 1)
        return self.log_p.gather(2, a).squeeze(2).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        p = self.log_p.exp()
        return -(p * self.log_p).masked_fill(~self._mask, 0.0).sum(dim=(1, 2))

    def sample(self) -> torch.Tensor:
        # Astuce de Gumbel : l'argmax des log-probabilités bruitées suit la loi de chaque dimension.
        u = torch.rand_like(self.log_p).clamp_(1e-10, 1.0)
        noisy = (self.log_p - torch.log(-torch.log(u))).masked_fill(~self._mask, -torch.inf)
        return noisy.argmax(dim=2)

    def mode(self) -> torch.Tensor:
        return self.log_p.argmax(dim=2)

    def actions_from_params(
        self, action_logits: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


class PaddedMaskablePolicy(MaskableActorCriticPolicy):
    """Politique de MaskablePPO identique à celle de sb3-contrib, avec la loi ci-dessus."""

    def _build(self, lr_schedule) -> None:
        if isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_dist = PaddedMaskableMultiCategorical(list(self.action_space.nvec))
        super()._build(lr_schedule)
