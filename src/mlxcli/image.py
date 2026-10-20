"""Local text-to-image generation through mflux (Z-Image Turbo by default)."""
from __future__ import annotations

import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config, prompts, server

EXECUTABLE = "mflux-generate-z-image-turbo"


def _find_executable() -> str | None:
    found = shutil.which(EXECUTABLE)
    if found:
        return found
    for folder in (Path.home() / ".local" / "bin", Path(sys.executable).parent):
        candidate = folder / EXECUTABLE
        if candidate.exists():
            return str(candidate)
    return None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_output(out: str | None, now: float | None = None) -> Path:
    """Decide where the image goes.

    - no value            -> <IMAGE_DIR>/mlx-<timestamp>.png
    - a folder            -> that folder / mlx-<timestamp>.png  (existing dir, trailing '/',
                             or any name without an image extension; created if missing)
    - bare file name      -> <IMAGE_DIR>/<name>   (no folder given -> default folder)
    - path with a folder  -> used as given, relative paths resolve from the current directory
    An existing file is never overwritten: a numeric suffix is added instead.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    default_name = f"mlx-{stamp}.png"

    if not out:
        target = config.IMAGE_DIR / default_name
    else:
        raw = Path(out).expanduser()
        is_folder = out.endswith("/") or raw.is_dir() or raw.suffix.lower() not in IMAGE_SUFFIXES
        if is_folder:
            target = raw / default_name
        elif raw.parent == Path("."):  # bare file name, no folder given
            target = config.IMAGE_DIR / raw.name
        else:
            target = raw

    target.parent.mkdir(parents=True, exist_ok=True)
    candidate, n = target, 2
    while candidate.exists():
        candidate = target.with_name(f"{target.stem}-{n}{target.suffix}")
        n += 1
    return candidate


def _default_image_alias() -> str:
    for alias, info in config.load_models().items():
        if info["kind"] == "image":
            return alias
    raise SystemExit("No image model registered. Add one: mlx add NAME org/repo --kind image")


def generate(words: list[str], out: str | None, model_name: str | None, width: int,
             height: int, steps: int, seed: int | None, style: str | None,
             open_after: bool, keep_server: bool) -> int:
    text = " ".join(words).strip()
    if not text:
        raise SystemExit('Usage: mlx image "what to draw" [--style textbook]')

    model = config.resolve_model(model_name or _default_image_alias())
    if model["kind"] != "image":
        raise SystemExit(f"'{model['alias']}' is not an image model.")

    exe = _find_executable()
    if exe is None:
        raise SystemExit(
            f"{EXECUTABLE} not found. Install mflux in its own environment "
            "(it pins different library versions than mlx-vlm):\n"
            "  pip install uv && uv tool install --upgrade mflux && uv tool update-shell"
        )

    if style:
        text = f"{text}. {prompts.load('style-' + style)}"

    # A 27B LLM and an image model do not fit together in 24 GB.
    if not keep_server and server.is_running():
        print("Stopping the LLM server to free memory (use --keep-server to skip)...")
        server.stop(quiet=True)

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    out_path = str(resolve_output(out))
    cmd = [exe, "--model", model["repo"], "--prompt", text, "--width", str(width),
           "--height", str(height), "--steps", str(steps), "--seed", str(seed),
           "--output", out_path]
    print(f"Generating {width}x{height}, {steps} steps, seed {seed} ...")
    code = subprocess.call(cmd)
    if code != 0:
        print(f"{EXECUTABLE} failed (exit {code}). Check: {EXECUTABLE} --help", file=sys.stderr)
        return code
    print(f"Saved: {out_path}")
    if open_after:
        subprocess.call(["open", out_path])
    return 0
