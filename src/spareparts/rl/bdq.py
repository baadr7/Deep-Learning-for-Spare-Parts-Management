"""Branching Dueling Q-Network pour le niveau d'action N2 (§ 5.3) [Tavakoli et al., 2018].

Un DQN classique devrait évaluer chaque combinaison d'actions, soit des milliards au niveau N2.
L'architecture à branches partage un tronc commun puis attribue à chaque dimension de l'action
(quantité commandée par référence, commande urgente, maintenance de chaque machine) sa propre
tête d'avantages, combinée à une valeur d'état commune. La cible d'apprentissage est la moyenne,
sur les branches, des valeurs du réseau cible au choix glouton du réseau courant (double DQN).
Les options invalides sont masquées dans chaque branche.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from spareparts.envs.codecs import inaction_prior
from spareparts.policies.base import Policy
from spareparts.sim.engine import Decisions, FleetSimulation, Observation


class BranchingQNet(nn.Module):
    def __init__(self, obs_size: int, branch_sizes: np.ndarray, hidden: int = 256):
        super().__init__()
        self.sizes = [int(n) for n in branch_sizes]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.value = nn.Linear(hidden, 1)
        # Toutes les têtes d'avantages dans une seule couche. Les sorties sont rangées dans un
        # tableau branches × options complété par des options fictives, ce qui évite une boucle
        # Python sur la centaine de branches à chaque passage.
        self.advantage = nn.Linear(hidden, int(self.offsets[-1]))
        width = max(self.sizes)
        index = np.zeros((len(self.sizes), width), dtype=np.int64)
        valid = np.zeros((len(self.sizes), width), dtype=bool)
        for d, size in enumerate(self.sizes):
            index[d, :size] = self.offsets[d] + np.arange(size)
            valid[d, :size] = True
        self.register_buffer("index", torch.from_numpy(index), persistent=False)
        self.register_buffer("valid", torch.from_numpy(valid), persistent=False)

    def forward(self, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Valeurs Q de forme (lot, branches, options) ; -1e9 pour une option invalide."""
        h = self.trunk(obs)
        v = self.value(h)[:, :, None]
        adv = self.advantage(h)[:, self.index]
        m = masks[:, self.index] & self.valid
        mean = (adv * m).sum(dim=2, keepdim=True) / m.sum(dim=2, keepdim=True).clamp(min=1)
        return (v + adv - mean).masked_fill(~m, -1e9)


@dataclass
class BDQConfig:
    total_steps: int = 200_000
    buffer_size: int = 100_000
    batch_size: int = 64
    lr: float = 1e-4
    gamma: float = 0.99
    learning_starts: int = 2_000
    # Une mise à jour tous les quatre pas, comme le DQN de référence : chaque pas coûte une
    # journée de simulation, et une mise à jour par pas ralentissait l'apprentissage d'autant.
    train_every: int = 4
    target_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    seed: int = 0


class _Replay:
    def __init__(self, size: int, obs_size: int, n_branches: int, mask_size: int):
        self.obs = np.zeros((size, obs_size), np.float32)
        self.next_obs = np.zeros((size, obs_size), np.float32)
        self.actions = np.zeros((size, n_branches), np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.next_masks = np.zeros((size, mask_size), bool)
        self.size, self.pos, self.full = size, 0, False

    def add(self, o, a, r, o2, d, m2):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = o, a, r
        self.next_obs[i], self.done[i], self.next_masks[i] = o2, d, m2
        self.pos = (i + 1) % self.size
        self.full = self.full or self.pos == 0

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size if self.full else self.pos, n)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.done[idx],
            self.next_masks[idx],
        )


def greedy(q: torch.Tensor) -> np.ndarray:
    return q.argmax(dim=2).numpy()


def train_bdq(env, cfg: BDQConfig, log=print, history: list | None = None) -> BranchingQNet:
    """Entraîne le réseau ; ``history`` reçoit, si fourni, le pas et le coût de chaque épisode."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sizes = env.action_space.nvec
    obs_size = env.observation_space.shape[0]
    net = BranchingQNet(obs_size, sizes)
    # Même point de départ que P5 : l'option « ne rien faire » domine avant apprentissage, et
    # l'exploration aléatoire tire les options selon ce même biais plutôt qu'uniformément.
    prior = inaction_prior(env.codec)
    with torch.no_grad():
        net.advantage.bias.add_(torch.as_tensor(prior, dtype=torch.float32))
    target = BranchingQNet(obs_size, sizes)
    target.load_state_dict(net.state_dict())
    index, valid = net.index.numpy(), net.valid.numpy()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    replay = _Replay(cfg.buffer_size, obs_size, len(sizes), int(sizes.sum()))

    obs, _ = env.reset(seed=cfg.seed)
    masks = env.action_masks()
    episode_cost, episodes = 0.0, 0
    for step in range(cfg.total_steps):
        frac = min(1.0, step / (cfg.eps_fraction * cfg.total_steps))
        eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        with torch.no_grad():
            action = greedy(net(torch.from_numpy(obs)[None], torch.from_numpy(masks)[None]))[0]
        # Exploration branche par branche : tirage selon le biais initial, par l'astuce de
        # Gumbel (l'argmax du logit bruité suit la loi softmax des logits).
        allowed = masks[index] & valid
        logits = np.where(allowed, prior[index], -np.inf) + rng.gumbel(size=index.shape)
        explore = rng.random(len(sizes)) < eps
        action = np.where(explore, logits.argmax(axis=1), action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_masks = env.action_masks()
        replay.add(obs, action, reward, next_obs, float(terminated), next_masks)
        episode_cost += info["cost"]
        obs, masks = next_obs, next_masks
        if terminated or truncated:
            episodes += 1
            log(f"BDQ épisode {episodes:3d}  pas {step + 1:7d}  coût {episode_cost:9.0f}")
            if history is not None:
                history.append((step + 1, episode_cost))
            obs, _ = env.reset()
            masks = env.action_masks()
            episode_cost = 0.0

        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            o, a, r, o2, d, m2 = (torch.from_numpy(x) for x in replay.sample(cfg.batch_size, rng))
            ones = torch.ones(len(o), int(sizes.sum()), dtype=torch.bool)
            q_taken = net(o, ones).gather(2, a[:, :, None]).squeeze(2)
            with torch.no_grad():
                best = net(o2, m2).argmax(dim=2, keepdim=True)
                next_v = target(o2, m2).gather(2, best).squeeze(2).mean(dim=1)
                y = r + cfg.gamma * (1 - d) * next_v
            loss = ((q_taken - y[:, None]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            if step % cfg.target_every == 0:
                target.load_state_dict(net.state_dict())
    return net


def save_bdq(net: BranchingQNet, obs_size: int, path) -> None:
    torch.save({"state": net.state_dict(), "sizes": net.sizes, "obs": obs_size}, path)


def load_bdq(path) -> BranchingQNet:
    blob = torch.load(path, map_location="cpu")
    net = BranchingQNet(blob["obs"], np.array(blob["sizes"]))
    net.load_state_dict(blob["state"])
    return net


class BDQPolicy(Policy):
    """Exécute un réseau à branches entraîné dans le simulateur."""

    name = "BDQ"
    label = "Branching Dueling Q-Network, niveau N2"

    def __init__(self, net: BranchingQNet, horizon: int = 365):
        self.net, self.horizon = net.eval(), horizon

    def reset(self, sim: FleetSimulation) -> None:
        from spareparts.envs.codecs import Context, Level2Codec

        super().reset(sim)
        self.codec = Level2Codec(Context.from_sim(sim, self.horizon))

    def decide(self, obs: Observation) -> Decisions:
        o = torch.from_numpy(self.codec.observe(obs))[None]
        m = torch.from_numpy(self.codec.masks(obs))[None]
        with torch.no_grad():
            action = greedy(self.net(o, m))[0]
        return self.codec.decide(action, obs)
