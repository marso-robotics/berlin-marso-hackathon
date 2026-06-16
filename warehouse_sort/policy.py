"""Policy contract + a simple PPO actor-critic baseline.

THE POLICY CONTRACT (this is all eval.py requires; you are NOT locked to this class or to RL):

    A policy is ANY object with
        act(obs, deterministic: bool = True) -> Tensor of shape (num_envs, action_dim) in [-1, 1]
    where ``obs`` is EXACTLY the environment's observation in the difficulty's locked obs mode
    (a flat state tensor for easy; a {"rgb", "state"} dict for medium/hard) and NOTHING else.
    The policy never receives the env, the ground-truth state, or evaluate() info, so it cannot
    read privileged information or game the scorer -- it only sees what a real robot would.

    eval.py builds your policy via a configurable entrypoint (config ``policy=<module>:<fn>``)
    where ``fn(checkpoint, sample_obs, action_space, device) -> policy`` returns any object that
    satisfies the contract above. RL nets, scripted controllers, classical CV + control,
    behaviour-cloning models, etc. all qualify. If ``policy`` is unset, the built-in ``Agent``
    below is loaded from the checkpoint (so a train.py checkpoint runs with no code changes).

The Agent below is a plain baseline that works for both observation modes:
* ``state`` -> obs is a flat tensor, encoded by an MLP.
* ``rgb``   -> obs is a dict {"rgb": (N,H,W,3) uint8, "state": (N,S)}; the image is encoded
  by a small NatureCNN and concatenated with the proprio ``state`` vector.
Participants are free to replace the encoder / architecture / algorithm entirely.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class NatureCNN(nn.Module):
    """Standard small conv stack; input (N,H,W,3) uint8 -> feature vector."""

    def __init__(self, h, w, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, 3, h, w)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n_flat, out_dim)), nn.ReLU())

    def forward(self, rgb_uint8):
        x = rgb_uint8.float().permute(0, 3, 1, 2) / 255.0
        return self.fc(self.cnn(x))


class Agent(nn.Module):
    def __init__(self, sample_obs, action_dim):
        super().__init__()
        self.is_dict = isinstance(sample_obs, dict)
        if self.is_dict:
            n, h, w, _ = sample_obs["rgb"].shape
            self.encoder = NatureCNN(h, w, out_dim=256)
            state_dim = sample_obs["state"].shape[1]
            feat_dim = 256 + state_dim
        else:
            self.encoder = None
            feat_dim = sample_obs.shape[1]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(feat_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feat_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def _features(self, obs):
        if self.is_dict:
            img = self.encoder(obs["rgb"])
            return torch.cat([img, obs["state"]], dim=1)
        return obs

    def get_value(self, obs):
        return self.critic(self._features(obs))

    def get_action_and_value(self, obs, action=None):
        feat = self._features(obs)
        mean = self.actor_mean(feat)
        # floor the std so exploration (esp. of the gripper open/close) doesn't collapse early
        logstd = self.actor_logstd.clamp(min=-1.6)
        std = torch.exp(logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(feat)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        feat = self._features(obs)
        mean = self.actor_mean(feat)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return Normal(mean, std).sample()
