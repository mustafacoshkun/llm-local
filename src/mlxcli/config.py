"""Paths, ports, the model registry and the small state file."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HOME = Path(os.environ.get("MLX_HOME", Path.home() / "llm-local")).expanduser()
HOST = "127.0.0.1"
PORT = int(os.environ.get("MLX_PORT", "8080"))

# Plain-text state files. The LaTeX app and mdx read current_model, so keep this format.
CURRENT_MODEL_FILE = HOME / "current_model"
PID_FILE = HOME / "server.pid"
LOG_FILE = HOME / "server.log"
USER_PROMPTS_DIR = HOME / "prompts"
USER_MODELS_FILE = HOME / "models.json"
# Where generated images go when no folder is given (override with MLX_IMAGE_DIR).
IMAGE_DIR = Path(os.environ.get("MLX_IMAGE_DIR", Path.home() / "Downloads")).expanduser()

KINDS = ("llm", "image")
DEFAULT_VENV = ".venv"

# alias -> repo id, kind, optional venv (relative to HOME) and extra server args
BUILTIN_MODELS: dict[str, dict] = {
    "qwen": {"repo": "mlx-community/Qwen3.8-27B-4bit", "kind": "llm"},
    "gemma": {"repo": "mlx-community/gemma-4-26B-A4B-it-qat-4bit", "kind": "llm"},
    "paddle": {"repo": "OpenGryd/PaddleOCR-VL-1.6-MLX-16bit", "kind": "llm",
               "venv": ".venv-ocr", "args": ["--trust-remote-code"]},
    "zimage": {"repo": "filipstrand/Z-Image-Turbo-mflux-4bit", "kind": "image"},
}


def ensure_home() -> None:
    HOME.mkdir(parents=True, exist_ok=True)


def hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home.expanduser() / "hub"


def load_models() -> dict[str, dict[str, str]]:
    models = {alias: dict(info) for alias, info in BUILTIN_MODELS.items()}
    if USER_MODELS_FILE.exists():
        try:
            models.update(json.loads(USER_MODELS_FILE.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warning: could not read {USER_MODELS_FILE}: {exc}", file=sys.stderr)
    return models


def save_user_model(alias: str, repo: str, kind: str) -> None:
    ensure_home()
    user: dict = {}
    if USER_MODELS_FILE.exists():
        try:
            user = json.loads(USER_MODELS_FILE.read_text())
        except json.JSONDecodeError:
            user = {}
    user[alias] = {"repo": repo, "kind": kind}
    USER_MODELS_FILE.write_text(json.dumps(user, indent=2) + "\n")


def resolve_model(name: str) -> dict:
    """Return {"alias", "repo", "kind", "venv", "args"} for an alias or a raw 'org/name' repo id."""
    models = load_models()
    if name in models:
        info = {"alias": name, **models[name]}
    elif "/" in name:
        info = {"alias": name, "repo": name, "kind": "llm"}
    else:
        known = ", ".join(sorted(models))
        raise SystemExit(f"Unknown model '{name}'. Known aliases: {known}")
    info.setdefault("venv", DEFAULT_VENV)
    info.setdefault("args", [])
    return info


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def read_state() -> dict | None:
    """{"repo": str, "pid": int | None} from current_model / server.pid, or None."""
    repo = _read_text(CURRENT_MODEL_FILE)
    pid_text = _read_text(PID_FILE)
    pid = int(pid_text) if pid_text.isdigit() else None
    if not repo and pid is None:
        return None
    return {"repo": repo, "pid": pid}


def write_state(repo: str, pid: int) -> None:
    ensure_home()
    CURRENT_MODEL_FILE.write_text(repo + "\n")
    PID_FILE.write_text(f"{pid}\n")


def clear_pid() -> None:
    """Remove server.pid. current_model stays on purpose (consumers expect it, even if stale)."""
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass
