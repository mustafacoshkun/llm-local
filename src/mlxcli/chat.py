"""Talk to the running model server (OpenAI-compatible API)."""
from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path

from . import config, prompts, server


def _client():
    from openai import OpenAI

    return OpenAI(base_url=f"http://{config.HOST}:{config.PORT}/v1", api_key="none")


def _current_model() -> str:
    if not server.port_open():
        raise SystemExit("No model server is running. Start one with: mlx start gemma | qwen | paddle")
    state = config.read_state()
    if state and state.get("repo"):
        return state["repo"]
    try:  # server was started outside mlx; ask it
        ids = [m.id for m in _client().models.list().data]
        if ids:
            return ids[0]
    except Exception:
        pass
    raise SystemExit("The server is up but the model is unknown. Restart it with: mlx start <model>")


def _image_part(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"Image not found: {path}")
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic"}


def split_text_and_images(words: list[str]) -> tuple[str, list[str]]:
    """Old-style 'mlx ask "prompt" photo.png': extra words that are existing image files
    become images. The first word is always part of the prompt."""
    text: list[str] = []
    images: list[str] = []
    for i, word in enumerate(words):
        path = Path(word).expanduser()
        if i > 0 and path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            images.append(str(path))
        else:
            text.append(word)
    return " ".join(text).strip(), images


def strip_fences(text: str) -> str:
    """Remove a surrounding ```lang ... ``` block if the model added one."""
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip() + "\n"
    return text


def _complete(model: str, messages: list[dict], max_tokens: int,
              temperature: float | None, stream: bool, echo: bool = True) -> str:
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    client = _client()
    parts: list[str] = []
    if stream:
        for chunk in client.chat.completions.create(stream=True, **kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                parts.append(piece)
                if echo:
                    print(piece, end="", flush=True)
        if echo:
            print()
    else:
        text = client.chat.completions.create(**kwargs).choices[0].message.content or ""
        parts.append(text)
        if echo:
            print(text)
    return "".join(parts)


def ask(words: list[str], prompt_name: str | None, images: list[str], out: str | None,
        max_tokens: int, temperature: float | None, stream: bool, use_stdin: bool = False) -> int:
    text, found_images = split_text_and_images(words)
    images = [*images, *found_images]
    if use_stdin:  # cat rules.md | mlx ask --stdin "find unclear rules"
        piped = sys.stdin.read().strip()
        if piped:
            text = f"{text}\n\n---\n{piped}" if text else piped
    if not text and not images:
        raise SystemExit('Nothing to ask. Usage: mlx ask "question" [image.png] [--max-tokens N]')

    model = _current_model()
    content: list[dict] = [{"type": "text", "text": text}]
    content += [_image_part(p) for p in images]
    messages: list[dict] = []
    if prompt_name:  # no system message unless asked: OCR models such as paddle expect none
        messages.append({"role": "system", "content": prompts.load(prompt_name)})
    messages.append({"role": "user", "content": content})
    reply = _complete(model, messages, max_tokens, temperature, stream, echo=out is None)
    if out:
        Path(out).expanduser().write_text(strip_fences(reply))
        print(f"Saved: {out}")
    return 0


def repl(prompt_name: str | None, max_tokens: int, temperature: float | None) -> int:
    model = _current_model()
    system = [{"role": "system", "content": prompts.load(prompt_name)}] if prompt_name else []
    history: list[dict] = []
    print(f"Chatting with {model}.  /reset clears history, /exit quits.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        history.append({"role": "user", "content": line})
        reply = _complete(model, [*system, *history], max_tokens, temperature, stream=True)
        history.append({"role": "assistant", "content": reply})
