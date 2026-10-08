"""Get the mGBA libretro core, the Game Boy Advance emulator (for FireRed, ...).

It is one file, downloaded from the libretro build server into cores/:
cores/mgba_libretro.dll (Windows), .so (Linux) or .dylib (macOS).

Save states are tied to the emulator version: create your start states and train
with the SAME core file. For Kaggle, download the Linux core at the same time as
yours (--also-linux) and put cores/mgba_libretro.so in your private dataset, next
to the ROMs: the notebook uses it.

Usage:
    uv run python scripts/get_mgba_core.py                 # the core for this computer
    uv run python scripts/get_mgba_core.py --also-linux    # plus the Linux one, for Kaggle
    uv run python scripts/get_mgba_core.py --build         # compile mGBA 0.10.5 (Linux/macOS, needs git and cmake)
"""

import argparse
import io
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BUILDBOT = "https://buildbot.libretro.com/nightly"
CORES = Path("cores")


def url_for(system: str, machine: str) -> tuple[str, str]:
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine.lower(), machine.lower())
    if system == "Windows":
        return f"{BUILDBOT}/windows/x86_64/latest/mgba_libretro.dll.zip", "mgba_libretro.dll"
    if system == "Darwin":
        arch = "arm64" if machine == "arm64" else "x86_64"
        return f"{BUILDBOT}/apple/osx/{arch}/latest/mgba_libretro.dylib.zip", "mgba_libretro.dylib"
    return f"{BUILDBOT}/linux/x86_64/latest/mgba_libretro.so.zip", "mgba_libretro.so"


def download(url: str, name: str) -> Path:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        member = next(n for n in archive.namelist() if n.endswith(name))
        CORES.mkdir(exist_ok=True)
        out = CORES / name
        out.write_bytes(archive.read(member))
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def build(tag: str = "0.10.5") -> Path:
    """Compile the libretro core of mGBA from source (a few minutes)."""
    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            raise SystemExit(f"--build needs {tool}")
    work = Path(tempfile.mkdtemp(prefix="mgba_build_"))
    src = work / "mgba"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch", tag,
         "https://github.com/mgba-emu/mgba.git", str(src)],
        check=True,
    )  # fmt: skip
    off = [
        "BUILD_QT", "BUILD_SDL", "BUILD_SHARED", "BUILD_STATIC", "BUILD_PERF", "BUILD_TEST",
        "BUILD_SUITE", "USE_FFMPEG", "USE_ZLIB", "USE_PNG", "USE_LIBZIP", "USE_MINIZIP",
        "USE_SQLITE3", "USE_ELF", "USE_LUA", "USE_EPOXY", "USE_DEBUGGERS", "USE_EDITLINE",
        "USE_GDB_STUB", "ENABLE_SCRIPTING",
    ]  # fmt: skip
    flags = ["-DBUILD_LIBRETRO=ON", "-DCMAKE_BUILD_TYPE=Release", *[f"-D{o}=OFF" for o in off]]
    subprocess.run(["cmake", "-S", str(src), "-B", str(src / "build"), *flags], check=True)
    subprocess.run(
        ["cmake", "--build", str(src / "build"), "--target", "mgba_libretro", "-j"], check=True
    )
    built = next((src / "build").glob("mgba_libretro.*"))
    CORES.mkdir(exist_ok=True)
    out = CORES / built.name
    shutil.copy(built, out)
    shutil.rmtree(work, ignore_errors=True)
    print(f"Built {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--also-linux", action="store_true", help="also get the Linux core")
    parser.add_argument("--build", action="store_true", help="compile from source instead")
    args = parser.parse_args()
    if args.build:
        build()
        return
    download(*url_for(platform.system(), platform.machine()))
    if args.also_linux and platform.system() != "Linux":
        download(*url_for("Linux", "x86_64"))
    print("Done. Check it with: uv run pytest tests/test_gba.py")


if __name__ == "__main__":
    sys.exit(main())
