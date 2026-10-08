"""A minimal libretro frontend in pure Python, to run mGBA (Game Boy Advance).

libretro is the standard plug-in interface of emulators ("cores"): a core is one
shared library (mgba_libretro.dll / .so / .dylib) with a fixed set of C functions.
Loading it with ctypes gives a GBA emulator on Windows, Linux and macOS, with no
compilation: download the core once with scripts/get_mgba_core.py.

Where the core is looked for: the MISSINGNO_MGBA_CORE environment variable, then
cores/mgba_libretro.<dll|so|dylib> in the repository.

A libretro core keeps global state, so two emulators cannot share one loaded
library: each emulator loads its own private copy of the core file.
"""

from __future__ import annotations

import ctypes as C
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --- libretro constants (libretro.h) -------------------------------------------------
ENV_EXPERIMENTAL = 0x10000
ENV_GET_CAN_DUPE = 3
ENV_GET_SYSTEM_DIRECTORY = 9
ENV_SET_PIXEL_FORMAT = 10
ENV_GET_SAVE_DIRECTORY = 31
ENV_SET_MEMORY_MAPS = 36 | ENV_EXPERIMENTAL
PIXEL_0RGB1555, PIXEL_XRGB8888, PIXEL_RGB565 = 0, 1, 2
DEVICE_JOYPAD = 1
JOYPAD = {
    "b": 0,
    "select": 2,
    "start": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "a": 8,
    "l": 10,
    "r": 11,
}


class _GameInfo(C.Structure):
    _fields_ = [
        ("path", C.c_char_p),
        ("data", C.c_void_p),
        ("size", C.c_size_t),
        ("meta", C.c_char_p),
    ]


class _MemoryDescriptor(C.Structure):
    _fields_ = [
        ("flags", C.c_uint64),
        ("ptr", C.c_void_p),
        ("offset", C.c_size_t),
        ("start", C.c_size_t),
        ("select", C.c_size_t),
        ("disconnect", C.c_size_t),
        ("len", C.c_size_t),
        ("addrspace", C.c_char_p),
    ]


class _MemoryMap(C.Structure):
    _fields_ = [("descriptors", C.POINTER(_MemoryDescriptor)), ("num_descriptors", C.c_uint)]


_ENV = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
_VIDEO = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
_AUDIO = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
_AUDIO_BATCH = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
_POLL = C.CFUNCTYPE(None)
_INPUT = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


def find_core(name: str = "mgba") -> str:
    """Path of the libretro core, or a clear error explaining how to get it."""
    env = os.environ.get("MISSINGNO_MGBA_CORE")
    if env and Path(env).exists():
        return env
    ext = {"win32": "dll", "darwin": "dylib"}.get(sys.platform, "so")
    here = Path(__file__).resolve()
    for root in [Path.cwd(), *here.parents]:
        path = root / "cores" / f"{name}_libretro.{ext}"
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"The {name} libretro core (cores/{name}_libretro.{ext}) was not found. Get it with:\n"
        "    uv run python scripts/get_mgba_core.py\n"
        "or set MISSINGNO_MGBA_CORE to its path (see docs/SETUP.md, 'Game Boy Advance')."
    )


class Memory:
    """The console's memory, by address (only the regions the core maps: RAM, ROM, ...)."""

    def __init__(self) -> None:
        self.regions: list[tuple[int, int, np.ndarray]] = []  # (start, end, bytes)
        self._last = (0, 0, np.zeros(0, np.uint8))

    def add(self, start: int, length: int, pointer: int, offset: int = 0) -> None:
        raw = (C.c_uint8 * length).from_address(pointer + offset)
        self.regions.append((start, start + length, np.ctypeslib.as_array(raw)))

    def _region(self, address: int):
        start, end, data = self._last
        if start <= address < end:
            return start, data
        for region in self.regions:
            if region[0] <= address < region[1]:
                self._last = region
                return region[0], region[2]
        raise IndexError(f"Address {address:#010x} is not mapped")

    def __getitem__(self, address):
        if isinstance(address, slice):
            start, data = self._region(address.start)
            return bytes(data[address.start - start : address.stop - start])
        start, data = self._region(address)
        return int(data[address - start])


class LibretroEmulator:
    """A Game Boy Advance (or any libretro core) behind the Emulator interface.

    `window=True` opens a window (PySDL2, already installed with PyBoy).
    `interactive=True` also reads the keyboard, to play by hand (arrows, A = a,
    B = s, Start = Enter, Select = Backspace, L = q, R = w), as in PyBoy.
    """

    def __init__(
        self,
        rom_path: str,
        core_path: str | None = None,
        window: bool = False,
        interactive: bool = False,
    ) -> None:
        self._dir = tempfile.mkdtemp(prefix="missingno_core_")
        source = core_path or find_core()
        private = Path(self._dir) / Path(source).name  # one private copy per emulator
        shutil.copy(source, private)
        self.lib = C.CDLL(str(private))
        self._pixel_format = PIXEL_RGB565
        self._frame = np.zeros((160, 240, 3), np.uint8)
        self._want_frame = True
        self._held: dict[int, int] = {}  # joypad id -> frames left
        self._keyboard: set[int] = set()
        self._speed = 0
        self._window = None
        self._closed = False
        self.memory = Memory()
        self._sys_dir = C.create_string_buffer(self._dir.encode())
        self._callbacks = [
            _ENV(self._environment),
            _VIDEO(self._video),
            _AUDIO(lambda left, right: None),
            _AUDIO_BATCH(lambda data, frames: frames),
            _POLL(lambda: None),
            _INPUT(self._input),
        ]
        env, video, audio, audio_batch, poll, input_state = self._callbacks
        lib = self.lib
        lib.retro_set_environment(env)
        lib.retro_init()
        lib.retro_set_video_refresh(video)
        lib.retro_set_audio_sample(audio)
        lib.retro_set_audio_sample_batch(audio_batch)
        lib.retro_set_input_poll(poll)
        lib.retro_set_input_state(input_state)
        lib.retro_serialize_size.restype = C.c_size_t
        lib.retro_serialize.argtypes = [C.c_void_p, C.c_size_t]
        lib.retro_serialize.restype = C.c_bool
        lib.retro_unserialize.argtypes = [C.c_void_p, C.c_size_t]
        lib.retro_unserialize.restype = C.c_bool
        lib.retro_load_game.restype = C.c_bool

        rom = Path(rom_path).read_bytes()
        self._rom = C.create_string_buffer(rom, len(rom))
        info = _GameInfo(str(rom_path).encode(), C.cast(self._rom, C.c_void_p), len(rom), None)
        if not lib.retro_load_game(C.byref(info)):
            raise RuntimeError(f"The libretro core could not load {rom_path}")
        self.lib.retro_run()  # the core finishes its setup (and maps memory) on the first frame
        if not self.memory.regions:
            raise RuntimeError("The libretro core did not expose its memory map")
        if window:
            self._open_window(interactive)

    # ------------------------------------------------------------- Emulator interface

    def press(self, button: str, frames: int) -> None:
        self._held[JOYPAD[button]] = frames

    def tick(self, frames: int = 1, render: bool = True) -> bool:
        for i in range(frames):
            self._want_frame = render and i == frames - 1 or self._window is not None
            start = time.perf_counter()
            self.lib.retro_run()
            for key in list(self._held):
                self._held[key] -= 1
                if self._held[key] <= 0:
                    del self._held[key]
            if self._window is not None:
                if not self._show():
                    return False
                if self._speed > 0:
                    time.sleep(max(0.0, 1 / (60 * self._speed) - (time.perf_counter() - start)))
        return True

    def screen(self) -> np.ndarray:
        return self._frame

    def save_state(self) -> bytes:
        size = self.lib.retro_serialize_size()
        buffer = C.create_string_buffer(size)
        if not self.lib.retro_serialize(buffer, size):
            raise RuntimeError("The libretro core could not save its state")
        return buffer.raw

    def load_state(self, state: bytes) -> None:
        buffer = C.create_string_buffer(state, len(state))
        if not self.lib.retro_unserialize(buffer, len(state)):
            raise RuntimeError(
                "The libretro core could not load this state (made by another core version?)"
            )

    def set_speed(self, speed: int) -> None:
        self._speed = speed

    def stop(self) -> None:
        if self.lib is None:
            return
        self.lib.retro_unload_game()
        self.lib.retro_deinit()
        if self._window is not None:
            import sdl2

            sdl2.SDL_DestroyWindow(self._window)
            self._window = None
        self.lib = None
        shutil.rmtree(self._dir, ignore_errors=True)  # may fail on Windows: it is a temp dir

    @staticmethod
    def version() -> str:
        return "mGBA (libretro)"

    # -------------------------------------------------------------------- callbacks

    def _environment(self, cmd: int, data: int) -> bool:
        if cmd == ENV_GET_CAN_DUPE:
            C.cast(data, C.POINTER(C.c_bool))[0] = True
            return True
        if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY):
            C.cast(data, C.POINTER(C.c_char_p))[0] = C.cast(self._sys_dir, C.c_char_p).value
            return True
        if cmd == ENV_SET_PIXEL_FORMAT:
            self._pixel_format = C.cast(data, C.POINTER(C.c_int))[0]
            return self._pixel_format in (PIXEL_0RGB1555, PIXEL_XRGB8888, PIXEL_RGB565)
        if cmd == ENV_SET_MEMORY_MAPS:
            mmap = C.cast(data, C.POINTER(_MemoryMap))[0]
            self.memory.regions.clear()
            for i in range(mmap.num_descriptors):
                d = mmap.descriptors[i]
                if d.ptr and d.len:
                    self.memory.add(d.start, d.len, d.ptr, d.offset)
            return True
        return False  # everything else: defaults (no options, no rumble, no logging, ...)

    def _video(self, data: int, width: int, height: int, pitch: int) -> None:
        if not data or not self._want_frame:
            return  # a repeated frame, or one nobody looks at
        if self._pixel_format == PIXEL_XRGB8888:
            raw = np.ctypeslib.as_array((C.c_uint8 * (pitch * height)).from_address(data))
            pixels = raw.reshape(height, pitch // 4, 4)[:, :width]
            self._frame = pixels[:, :, 2::-1].copy()  # BGRX in memory -> RGB
            return
        raw = np.ctypeslib.as_array((C.c_uint16 * (pitch // 2 * height)).from_address(data))
        p = raw.reshape(height, pitch // 2)[:, :width].astype(np.uint32)
        if self._pixel_format == PIXEL_RGB565:
            r, g, b = (p >> 11) & 0x1F, (p >> 5) & 0x3F, p & 0x1F
            frame = np.stack([r * 255 // 31, g * 255 // 63, b * 255 // 31], axis=-1)
        else:  # 0RGB1555
            r, g, b = (p >> 10) & 0x1F, (p >> 5) & 0x1F, p & 0x1F
            frame = np.stack([r, g, b], axis=-1) * 255 // 31
        self._frame = frame.astype(np.uint8)

    def _input(self, port: int, device: int, index: int, button: int) -> int:
        if port != 0 or device != DEVICE_JOYPAD:
            return 0
        return int(button in self._held or button in self._keyboard)

    # ------------------------------------------------------------------- window

    def _open_window(self, interactive: bool) -> None:
        import sdl2

        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
        height, width = self._frame.shape[:2]
        self._window = sdl2.SDL_CreateWindow(
            b"MissingNo",
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            width * 3,
            height * 3,
            sdl2.SDL_WINDOW_SHOWN | sdl2.SDL_WINDOW_RESIZABLE,
        )
        self._renderer = sdl2.SDL_CreateRenderer(self._window, -1, 0)
        self._texture = sdl2.SDL_CreateTexture(
            self._renderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            width,
            height,
        )
        self._interactive = interactive
        self._keys = {
            sdl2.SDLK_UP: JOYPAD["up"],
            sdl2.SDLK_DOWN: JOYPAD["down"],
            sdl2.SDLK_LEFT: JOYPAD["left"],
            sdl2.SDLK_RIGHT: JOYPAD["right"],
            sdl2.SDLK_a: JOYPAD["a"],
            sdl2.SDLK_s: JOYPAD["b"],
            sdl2.SDLK_RETURN: JOYPAD["start"],
            sdl2.SDLK_BACKSPACE: JOYPAD["select"],
            sdl2.SDLK_q: JOYPAD["l"],
            sdl2.SDLK_w: JOYPAD["r"],
        }

    def _show(self) -> bool:
        import sdl2

        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(C.byref(event)):
            if event.type == sdl2.SDL_QUIT:
                self._closed = True
            elif self._interactive and event.type in (sdl2.SDL_KEYDOWN, sdl2.SDL_KEYUP):
                key = self._keys.get(event.key.keysym.sym)
                if key is not None:
                    if event.type == sdl2.SDL_KEYDOWN:
                        self._keyboard.add(key)
                    else:
                        self._keyboard.discard(key)
        if self._closed:
            return False
        frame = np.ascontiguousarray(self._frame)
        height, width = frame.shape[:2]
        sdl2.SDL_UpdateTexture(self._texture, None, frame.ctypes.data, width * 3)
        sdl2.SDL_RenderClear(self._renderer)
        sdl2.SDL_RenderCopy(self._renderer, self._texture, None, None)
        sdl2.SDL_RenderPresent(self._renderer)
        return True
