"""List, download and register models."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from . import config, server


def fmt_size(num_bytes: int) -> str:
    gb = num_bytes / 1024**3
    if gb >= 0.1:
        return f"{gb:.1f}G"
    return f"{num_bytes / 1024**2:.0f}M"


def scan_model_dir(model_dir: Path) -> tuple[int, float]:
    """(size in bytes, creation time). Each real file is counted once.

    Hugging Face stores files in blobs/ and exposes them through symlinks in
    snapshots/, so we follow symlinks and de-duplicate by inode.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for root, _dirs, files in os.walk(model_dir / "snapshots"):
        for name in files:
            try:
                st = os.stat(os.path.join(root, name))  # follows symlinks
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key)
                total += st.st_size
    st = model_dir.stat()
    return total, getattr(st, "st_birthtime", st.st_mtime)


def downloaded_models() -> list[dict]:
    hub = config.hub_dir()
    registry = config.load_models()
    by_repo = {info["repo"]: alias for alias, info in registry.items()}
    rows = []
    if hub.is_dir():
        for d in hub.glob("models--*"):
            if not d.is_dir():
                continue
            repo = d.name[len("models--"):].replace("--", "/")
            size, born = scan_model_dir(d)
            alias = by_repo.get(repo, "")
            kind = registry[alias]["kind"] if alias else "-"
            rows.append({"repo": repo, "alias": alias, "kind": kind,
                         "size": size, "born": born})
    rows.sort(key=lambda r: r["born"], reverse=True)  # newest first
    return rows


def list_models() -> int:
    rows = downloaded_models()
    state = config.read_state() or {}
    running_repo = state.get("repo") if server.is_running() else None

    print(f"Downloaded models ({config.hub_dir()}), newest first:")
    print(f"  {'MODEL':<52} {'ALIAS':<8} {'KIND':<6} {'SIZE':>6}  {'DOWNLOADED':<10}")
    for r in rows:
        date = time.strftime("%Y-%m-%d", time.localtime(r["born"]))
        mark = "  <- running" if r["repo"] == running_repo else ""
        print(f"  {r['repo']:<52} {r['alias']:<8} {r['kind']:<6} "
              f"{fmt_size(r['size']):>6}  {date:<10}{mark}")

    have = {r["repo"] for r in rows}
    missing = [(a, i["repo"]) for a, i in config.load_models().items() if i["repo"] not in have]
    if missing:
        print("\nRegistered but not downloaded:")
        for alias, repo in missing:
            print(f"  {alias:<8} {repo}   ->  mlx pull {alias}")

    total = sum(r["size"] for r in rows)
    print(f"\nTotal: {fmt_size(total)} in {len(rows)} models")
    return 0


def pull(name: str) -> int:
    from huggingface_hub import snapshot_download

    model = config.resolve_model(name)
    print(f"Downloading {model['repo']} ...")
    path = snapshot_download(repo_id=model["repo"])
    print(f"Done: {path}")
    return 0


def add(alias: str, repo: str, kind: str) -> int:
    if "/" not in repo:
        print("repo must look like 'org/name'", file=sys.stderr)
        return 1
    config.save_user_model(alias, repo, kind)
    print(f"Registered '{alias}' -> {repo} ({kind}). Download it with: mlx pull {alias}")
    return 0
