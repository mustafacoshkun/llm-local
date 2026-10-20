"""System prompts and style snippets live in plain .md files, separate from the code.

Lookup order: ~/llm-local/prompts/<name>.md (yours) -> packaged defaults.
The first line, if it starts with '#', is a short description and is not sent to the model.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

from . import config


def split_prompt(text: str) -> tuple[str, str]:
    """Return (summary, body). A leading '# ...' line is the summary."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("#"):
        return lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()
    return "", text.strip()


def _builtin_dir():
    return resources.files("mlxcli") / "prompts"


def list_prompts() -> list[tuple[str, str, str]]:
    """(name, source, summary), user prompts override packaged ones of the same name."""
    found: dict[str, tuple[str, str]] = {}
    for entry in _builtin_dir().iterdir():
        if entry.name.endswith(".md"):
            found[entry.name[:-3]] = ("built-in", split_prompt(entry.read_text())[0])
    if config.USER_PROMPTS_DIR.is_dir():
        for path in config.USER_PROMPTS_DIR.glob("*.md"):
            found[path.stem] = ("yours", split_prompt(path.read_text())[0])
    return sorted((name, src, summary) for name, (src, summary) in found.items())


def load(name: str) -> str:
    user_file = config.USER_PROMPTS_DIR / f"{name}.md"
    if user_file.is_file():
        return split_prompt(user_file.read_text())[1]
    builtin = _builtin_dir() / f"{name}.md"
    if builtin.is_file():
        return split_prompt(builtin.read_text())[1]
    names = ", ".join(n for n, _, _ in list_prompts())
    raise SystemExit(f"Unknown prompt '{name}'. Available: {names}")


def init_user_dir() -> Path:
    """Copy the packaged prompts to ~/llm-local/prompts so you can edit them."""
    config.USER_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in _builtin_dir().iterdir():
        target = config.USER_PROMPTS_DIR / entry.name
        if entry.name.endswith(".md") and not target.exists():
            target.write_text(entry.read_text())
    return config.USER_PROMPTS_DIR
