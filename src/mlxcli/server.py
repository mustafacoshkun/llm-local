"""Start, stop and inspect the local mlx_vlm model server."""
from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_server_command(command: str) -> bool:
    """True only for 'python ... -m mlx_vlm.server ...'.

    Exact token match on purpose: a plain substring match would also hit shells,
    editors or 'grep' lines that merely mention the module name. The interpreter
    name is compared case-insensitively because the macOS framework build of
    Python reports itself as 'Python'.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or not os.path.basename(tokens[0]).lower().startswith("python"):
        return False
    if "--help" in tokens or "-h" in tokens:  # our own '--help' probe is not a server
        return False
    return any(a == "-m" and b == "mlx_vlm.server" for a, b in zip(tokens, tokens[1:]))


def is_starter_command(command: str) -> bool:
    """True for a running 'mlx start ...' (console script or 'python -m mlxcli start')."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or not os.path.basename(tokens[0]).lower().startswith("python"):
        return False
    rest = tokens[1:]
    if rest[:2] == ["-m", "mlxcli"]:
        rest = rest[2:]
    elif rest and os.path.basename(rest[0]) == "mlx":
        rest = rest[1:]
    else:
        return False
    return bool(rest) and rest[0] == "start"


def _command_of(pid: int) -> str:
    out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                         capture_output=True, text=True).stdout
    return out.strip()


def _is_ours(pid: int) -> bool:
    """Guard against a stale pid file whose number was reused by an unrelated program."""
    command = _command_of(pid)
    return is_server_command(command) or is_starter_command(command)


def _find_server_pids() -> list[int]:
    """PIDs of any mlx_vlm server on this machine (not just ones we started)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    pids = []
    for line in out.splitlines():
        pid_str, _, command = line.strip().partition(" ")
        if pid_str.isdigit() and int(pid_str) != os.getpid() and is_server_command(command.strip()):
            pids.append(int(pid_str))
    return pids


def port_open() -> bool:
    try:
        with socket.create_connection((config.HOST, config.PORT), timeout=0.5):
            return True
    except OSError:
        return False


def running_pids() -> list[int]:
    pids = set(_find_server_pids())
    state = config.read_state()
    pid = state.get("pid") if state else None
    if pid and pid != os.getpid() and _alive(pid) and _is_ours(pid):
        pids.add(pid)  # the server, or a 'mlx start' that is still getting ready
    return sorted(pids)


def is_running() -> bool:
    return bool(running_pids())


def venv_python(model: dict) -> Path:
    return config.HOME / model["venv"] / "bin" / "python"


def _supports_flag(python: Path, flag: str) -> bool:
    """Check whether this mlx_vlm version accepts a CLI flag."""
    try:
        out = subprocess.run([str(python), "-m", "mlx_vlm.server", "--help"],
                             capture_output=True, text=True, timeout=120)
        return flag in (out.stdout + out.stderr)
    except (subprocess.SubprocessError, OSError):
        return False


def start(name: str, timeout: int = 300) -> int:
    model = config.resolve_model(name)
    if model["kind"] != "llm":
        print(f"'{name}' is an image model. Use: mlx image \"prompt\"", file=sys.stderr)
        return 1

    # Check the environment first, so a failed start does not kill a running server.
    python = venv_python(model)
    if not python.exists():
        print(f"Environment not found: {python.parent.parent}", file=sys.stderr)
        return 1

    stop(quiet=True)
    config.ensure_home()
    # Register this process first. The compatibility check below takes seconds; a 'mlx stop'
    # during that time must end this process instead of racing it and leaving a server behind.
    config.write_state(model["repo"], os.getpid())

    cmd = [str(python), "-m", "mlx_vlm.server", "--model", model["repo"],
           "--port", str(config.PORT), *model["args"]]
    # Bind to localhost only, so the model is not reachable from the local network.
    if _supports_flag(python, "--host"):
        cmd += ["--host", config.HOST]
    else:
        print("note: this mlx_vlm has no --host flag; the server may listen on all "
              "interfaces. Use a trusted network.", file=sys.stderr)

    with open(config.LOG_FILE, "wb") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    config.write_state(model["repo"], proc.pid)

    print("Loading", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open():
            print(f"\nReady: {model['repo']}")
            return 0
        if proc.poll() is not None:
            print(f"\nServer exited. Log: tail -30 {config.LOG_FILE}", file=sys.stderr)
            config.clear_pid()
            return 1
        print(".", end="", flush=True)
        time.sleep(1)
    print(f"\nTimed out after {timeout}s. Log: {config.LOG_FILE}", file=sys.stderr)
    return 1


def stop(quiet: bool = False) -> int:
    pids = running_pids()
    if not pids:
        config.clear_pid()
        if not quiet:
            print("Nothing is running.")
        return 0

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.2)
    for pid in pids:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    config.clear_pid()
    if not quiet:
        print("Stopped.")
    return 0


def status() -> int:
    """First line format is a contract: 'RUNNING: <repo-id>' or 'Nothing is running.'
    (mdx and the LaTeX app parse it)."""
    pids = running_pids()
    if not pids:
        print("Nothing is running.")
        return 0
    state = config.read_state() or {}
    print(f"RUNNING: {state.get('repo') or '(unknown model)'}")
    for pid in pids:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        mem = f"{int(rss) / 1024 / 1024:.1f} GB" if rss.isdigit() else "?"
        print(f"  pid {pid}  memory {mem}")
    ready = "yes" if port_open() else "no (still loading or failed)"
    print(f"  port {config.PORT} accepting connections: {ready}")
    return 0
