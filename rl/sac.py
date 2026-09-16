"""
sac.py -- Soft Actor-Critic, written out rather than imported.

stable-baselines3 is not installed here and pulling it in would drag gymnasium
and a pinned torch with it. SAC is small enough to state plainly, and having it
in the repository means the exact algorithm behind a published number is
readable instead of being a version string.

Standard SAC with the usual pieces: twin Q critics with target networks to
damp overestimation, a squashed Gaussian actor with the tanh log-det
correction, and an automatically tuned temperature targeting an entropy of
-dim(A). Chosen over PPO because the episode is only 96 steps and days are a
scarce, fixed set -- off-policy replay reuses every transition many times,
which matters when the whole world is 374 training days.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden], out_act=nn.ReLU)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False, with_logp=True):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = mu if deterministic else dist.rsample()
        a = torch.tanh(z)
        if not with_logp:
            return a, None
        # tanh change of variables; the 2*(log2 - z - softplus(-2z)) form is the
        # numerically stable way to write log(1 - tanh(z)^2)
        logp = dist.log_prob(z).sum(-1)
        logp = logp - (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    def __init__(self, obs_dim, act_dim, size):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.ptr, self.full, self.size = 0, False, size

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n, device):
        idx = np.random.randint(0, len(self), size=n)
        t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class SAC:
    def __init__(self, obs_dim, act_dim, hidden=256, lr=3e-4, gamma=0.99,
                 tau=0.005, buffer=400_000, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.target.load_state_dict(self.critic.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)

        self.gamma, self.tau = gamma, tau
        self.buf = Replay(obs_dim, act_dim, buffer)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logp=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch_size=256):
        o, a, r, o2, d = self.buf.sample(batch_size, self.device)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1t, q2t = self.target(o2, a2)
            backup = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)

        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.q_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        api, logp = self.actor(o)
        q1p, q2p = self.critic(o, api)
        pi_loss = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.pi_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha.exp()
                       * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.a_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return dict(q_loss=q_loss.item(), pi_loss=pi_loss.item(),
                    alpha=self.log_alpha.exp().item())

    def save(self, path):
        torch.save(dict(actor=self.actor.state_dict(),
                        critic=self.critic.state_dict(),
                        log_alpha=self.log_alpha.detach().cpu()), path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))
