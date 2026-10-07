"""Highlights: GIFs of the episodes that went furthest, made while training runs.

- Every time an episode beats the record (more milestones, then more maps, then
  more tiles), its GIF is saved immediately as `record_*.gif` and kept forever.
- Every `every_steps` training steps, the best episode finished in that window
  (if any) is saved as `best_*.gif`; only the last `keep` of these are kept.

Only episodes from the start state are recorded (see PokemonEnv.record_every),
so a GIF always shows the agent's real progress from the bedroom.
GIFs are encoded in a separate process, so training never waits for them.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def frames_to_gif(npz_path: str, gif_path: str, stride: int = 1, duration_ms: int = 60) -> str:
    """Turn a recording into a GIF, then delete the recording."""
    from PIL import Image

    frames = np.load(npz_path)["frames"][::stride]
    images = [Image.fromarray(f, mode="L") for f in frames]
    if images:
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0
        )
    Path(npz_path).unlink(missing_ok=True)
    return gif_path


@dataclass(order=True)
class Candidate:
    score: tuple[int, int, int]
    path: str = field(compare=False)
    step: int = field(compare=False)


@dataclass
class Highlights:
    out_dir: Path
    every_steps: int = 10_000
    keep: int = 20
    best_score: tuple[int, int, int] = (-1, -1, -1)
    window: list[Candidate] = field(default_factory=list)
    next_flush: int = 0
    pool: ProcessPoolExecutor | None = None

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pool = ProcessPoolExecutor(max_workers=1)

    def add(self, path: str, milestones: int, maps: int, tiles: int, step: int) -> None:
        """A recorded episode just ended."""
        candidate = Candidate((milestones, maps, tiles), path, step)
        if candidate.score > self.best_score:
            self.best_score = candidate.score
            self._gif(candidate, "record")
            print(
                f"  record episode at step {step:,}: {milestones} milestones, {maps} maps, "
                f"{tiles} tiles -> GIF in {self.out_dir}"
            )
        else:
            self.window.append(candidate)

    def maybe_flush(self, global_step: int) -> None:
        """At the end of each window, keep the best episode of the window."""
        if global_step < self.next_flush:
            return
        self.next_flush = global_step + self.every_steps
        if self.window:
            best = max(self.window)
            for c in self.window:
                if c is not best:
                    Path(c.path).unlink(missing_ok=True)
            self._gif(best, "best")
            self.window = []
        self._rotate()

    def close(self) -> None:
        for c in self.window:
            Path(c.path).unlink(missing_ok=True)
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        self._rotate()

    def _gif(self, c: Candidate, kind: str) -> None:
        m, maps, tiles = c.score
        name = f"{kind}_step{c.step:010d}_m{m}_maps{maps}_tiles{tiles}.gif"
        self.pool.submit(frames_to_gif, c.path, str(self.out_dir / name))

    def _rotate(self) -> None:
        periodic = sorted(self.out_dir.glob("best_*.gif"))
        for old in periodic[: max(0, len(periodic) - self.keep)]:
            old.unlink(missing_ok=True)
