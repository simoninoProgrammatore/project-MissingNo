"""Neural networks for the agent.

The agent sees a stack of grayscale frames and outputs two things:
- the *policy*: a probability for each button (the "actor");
- the *value*: how much future reward it expects from here (the "critic").
Both share the same convolutional "eyes".
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Orthogonal initialization: a standard trick that makes PPO train more reliably."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class CnnActorCritic(nn.Module):
    """A small CNN in the style of the classic Atari network (~1.2M parameters).

    Input: (batch, frames, 72, 80) uint8. Output: action logits and a value.
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int) -> None:
        super().__init__()
        frames = obs_shape[0]
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(frames, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_features = self.encoder(torch.zeros(1, *obs_shape)).shape[1]
        self.body = nn.Sequential(layer_init(nn.Linear(n_features, 512)), nn.ReLU())
        # Small std for the actor: start close to uniform over buttons.
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(self.encoder(obs.float() / 255.0))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor, action: torch.Tensor | None = None, greedy: bool = False):
        """Pick an action (or score a given one). Returns action, log-prob, entropy, value."""
        h = self.features(obs)
        dist = Categorical(logits=self.actor(h))
        if action is None:
            action = dist.probs.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1)
