"""Load a trained model and play with it, with or without memory.

Used by the scripts that watch or evaluate a model (scripts/watch.py,
scripts/evaluate.py): they do not need to know which network was trained.
"""

from __future__ import annotations

import numpy as np
import torch

from missingno_agents.networks import CnnActorCritic, RecurrentActorCritic


def build_network(obs_shape, n_actions: int, memory: str = "none", memory_size: int = 256):
    """The network for a `--memory` option: "none" or "gru"."""
    if memory == "none":
        return CnnActorCritic(obs_shape, n_actions)
    if memory == "gru":
        return RecurrentActorCritic(obs_shape, n_actions, memory_size)
    raise ValueError(f"Unknown memory {memory!r}: choose 'none' or 'gru'")


class Policy:
    """A trained agent, ready to play one episode after another.

    Call `reset()` at the start of every episode (it wipes the memory, if any),
    then `policy(obs)` at every step.
    """

    def __init__(self, checkpoint: str, greedy: bool = False) -> None:
        data = torch.load(checkpoint, map_location="cpu")
        config = data.get("config", {})
        self.memory = config.get("memory", "none")
        self.net = build_network(
            tuple(data["obs_shape"]),
            data["n_actions"],
            self.memory,
            config.get("memory_size", 256),
        )
        self.net.load_state_dict(data["model"])
        self.net.eval()
        self.config = config
        self.greedy = greedy
        self.reset()

    def reset(self) -> None:
        self.state = self.net.initial_state(1) if self.net.recurrent else None

    def __call__(self, obs: np.ndarray) -> int:
        x = torch.as_tensor(obs).unsqueeze(0)
        with torch.inference_mode():
            if self.net.recurrent:
                action, *_, self.state = self.net.act(x, self.state, greedy=self.greedy)
            else:
                action, *_ = self.net.act(x, greedy=self.greedy)
        return int(action.item())
