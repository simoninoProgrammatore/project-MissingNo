"""Neural networks for the agent.

The agent sees a stack of grayscale frames and outputs two things:
- the *policy*: a probability for each button (the "actor");
- the *value*: how much future reward it expects from here (the "critic").
Both share the same convolutional "eyes".

Two networks:
- CnnActorCritic: no memory. It decides from the last 3 frames (about one
  second of game), so it cannot know things that are not on the screen, e.g.
  that it is carrying Oak's Parcel.
- RecurrentActorCritic: the same eyes plus a short-term memory (a GRU), carried
  from one step to the next and reset at the start of every episode. The network
  learns by itself what is worth remembering.
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
    """A small CNN in the style of the classic Atari network (~1.1M parameters).

    Input: (batch, frames, 72, 80) uint8. Output: action logits and a value.
    """

    recurrent = False

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


class RecurrentActorCritic(CnnActorCritic):
    """CnnActorCritic plus a short-term memory (GRU).

    At every step the memory reads the features of the screen and updates its
    state; actor and critic see both the screen features and the memory. With
    this skip connection the network can act like the memoryless one from the
    start, while it learns what to remember.

    The memory state is a (batch, memory_size) tensor kept by the caller: start
    from `initial_state`, pass it to `act`, keep the state it returns. `done`
    marks the steps where a new episode starts: the memory is wiped there.
    """

    recurrent = True

    def __init__(
        self, obs_shape: tuple[int, int, int], n_actions: int, memory_size: int = 256
    ) -> None:
        super().__init__(obs_shape, n_actions)
        self.memory_size = memory_size
        self.memory = nn.GRU(512, memory_size)
        for name, param in self.memory.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.constant_(param, 0.0)
        self.actor = layer_init(nn.Linear(512 + memory_size, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512 + memory_size, 1), std=1.0)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.memory_size, device=device)

    def _step(self, features, state, done):
        if done is not None:
            state = state * (1.0 - done).unsqueeze(-1)  # a new episode: forget
        _, state = self.memory(features.unsqueeze(0), state.unsqueeze(0))
        state = state.squeeze(0)
        return torch.cat([features, state], dim=-1), state

    def value(self, obs, state, done=None) -> torch.Tensor:
        h, _ = self._step(self.features(obs), state, done)
        return self.critic(h).squeeze(-1)

    def act(self, obs, state, done=None, action=None, greedy: bool = False):
        """One step. Returns action, log-prob, entropy, value and the new memory state."""
        h, state = self._step(self.features(obs), state, done)
        dist = Categorical(logits=self.actor(h))
        if action is None:
            action = dist.probs.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1), state

    def act_sequence(self, obs, state, done, action):
        """Score a whole sequence of steps, replaying the memory from `state`.

        obs (T, B, ...), done (T, B), action (T, B); state (B, memory_size) is the
        memory before the first step. Returns log-prob, entropy and value, (T, B).
        Used for learning: the gradient flows back through the whole sequence, so
        the network learns what to remember (up to T steps back).
        """
        steps, batch = done.shape
        features = self.features(obs.reshape(steps * batch, *obs.shape[2:]))
        features = features.reshape(steps, batch, -1)
        # Run the memory over whole stretches without a new episode in one call (much
        # faster than step by step), wiping it where an episode starts. Episodes are
        # long, so a rollout is usually one or two stretches.
        starts = [0] + [t for t in range(1, steps) if bool(done[t].any())] + [steps]
        outputs = []
        for a, b in zip(starts[:-1], starts[1:], strict=True):
            state = state * (1.0 - done[a]).unsqueeze(-1)
            out, last = self.memory(features[a:b], state.unsqueeze(0))
            state = last.squeeze(0)
            outputs.append(out)
        h = torch.cat([features, torch.cat(outputs)], dim=-1)
        dist = Categorical(logits=self.actor(h))
        return dist.log_prob(action), dist.entropy(), self.critic(h).squeeze(-1)
