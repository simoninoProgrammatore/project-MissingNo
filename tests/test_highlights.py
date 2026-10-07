"""Highlights: record GIFs, best-of-window GIFs, rotation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from highlights import Highlights, frames_to_gif  # noqa: E402


def recording(tmp_path, name, n=4):
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(path, frames=np.zeros((n, 144, 160), dtype=np.uint8))
    return str(path)


def test_gif_is_written_and_recording_removed(tmp_path):
    src = recording(tmp_path, "a")
    gif = frames_to_gif(src, str(tmp_path / "a.gif"))
    assert Path(gif).exists() and not Path(src).exists()


def test_records_and_best_of_window(tmp_path):
    h = Highlights(tmp_path / "gifs", every_steps=100, keep=1)
    h.add(recording(tmp_path, "r1"), 2, 3, 50, step=10)  # first episode: a record
    h.add(recording(tmp_path, "w1"), 1, 2, 10, step=20)  # worse: waits for the window
    h.add(recording(tmp_path, "w2"), 2, 3, 40, step=30)  # better of the two, still no record
    h.maybe_flush(100)
    h.close()
    names = sorted(p.name for p in (tmp_path / "gifs").glob("*.gif"))
    assert names == [
        "best_step0000000030_m2_maps3_tiles40.gif",
        "record_step0000000010_m2_maps3_tiles50.gif",
    ]
    assert not list(tmp_path.glob("*.npz"))  # every recording consumed or deleted
