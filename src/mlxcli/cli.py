"""mlx command line entry point."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__, chat, config, image, models, prompts, server


def _cmd_prompts(args: argparse.Namespace) -> int:
    if args.path:
        print(config.USER_PROMPTS_DIR)
        return 0
    if args.init:
        print(f"Copied built-in prompts to {prompts.init_user_dir()} (edit them there).")
        return 0
    if args.name:
        print(prompts.load(args.name))
        return 0
    print(f"{'NAME':<22} {'SOURCE':<9} DESCRIPTION")
    for name, source, summary in prompts.list_prompts():
        print(f"{name:<22} {source:<9} {summary}")
    print(f"\nYour own prompts go in {config.USER_PROMPTS_DIR}/<name>.md")
    return 0


EXAMPLES = """examples:
  mlx start gemma                      daily model (qwen = heavier, paddle = OCR)
  mlx ask "Summarize MST in 2 sentences"
  mlx ask "Describe this image" photo.png
  mlx start paddle && mlx ask --max-tokens 4000 --temperature 0 "OCR:" page.png
  cat rules.md | mlx ask --stdin -p student-playtester "Which rules confuse you?"
  mlx image "a red fox" --style textbook
  mlx stop

one server at a time on port 8080; starting a model stops the previous one.
log: ~/llm-local/server.log
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlx",
        description="Serve, chat with and generate images from local MLX models on Apple Silicon.",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"mlx {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    s = sub.add_parser("start", help="start the model server (stops a running one first)")
    s.add_argument("model", help="gemma | qwen | paddle, another alias, or a Hugging Face repo id")
    s.add_argument("--timeout", type=int, default=300, help="seconds to wait for loading")
    s.set_defaults(func=lambda a: server.start(a.model, a.timeout))

    s = sub.add_parser("stop", help="stop the model server and free its memory")
    s.set_defaults(func=lambda a: server.stop())

    s = sub.add_parser("status", help="show what is running")
    s.set_defaults(func=lambda a: server.status())

    s = sub.add_parser("models", help="list downloaded models, newest first")
    s.set_defaults(func=lambda a: models.list_models())

    s = sub.add_parser("pull", help="download a model by alias or repo id")
    s.add_argument("model")
    s.set_defaults(func=lambda a: models.pull(a.model))

    s = sub.add_parser("add", help="register a new alias for a Hugging Face repo")
    s.add_argument("alias")
    s.add_argument("repo", help="e.g. mlx-community/SomeModel-4bit")
    s.add_argument("--kind", choices=config.KINDS, default="llm")
    s.set_defaults(func=lambda a: models.add(a.alias, a.repo, a.kind))

    s = sub.add_parser("ask", help="one question to the running model, optional image files after the text")
    s.add_argument("text", nargs="*", help="the question, then optional image files")
    s.add_argument("-p", "--prompt", help="system prompt name (default: none; see 'mlx prompts')")
    s.add_argument("-i", "--image", action="append", default=[], help="image file (repeatable)")
    s.add_argument("-o", "--out", help="save the answer to a file instead of printing")
    s.add_argument("--max-tokens", type=int, default=2000)
    s.add_argument("--temperature", type=float)
    s.add_argument("--no-stream", action="store_true")
    s.add_argument("--stdin", action="store_true", help="append piped stdin to the question")
    s.set_defaults(func=lambda a: chat.ask(
        a.text, a.prompt, a.image, a.out, a.max_tokens, a.temperature, not a.no_stream, a.stdin))

    s = sub.add_parser("chat", help="interactive chat with the running model")
    s.add_argument("-p", "--prompt", help="system prompt name")
    s.add_argument("--max-tokens", type=int, default=2000)
    s.add_argument("--temperature", type=float)
    s.set_defaults(func=lambda a: chat.repl(a.prompt, a.max_tokens, a.temperature))

    s = sub.add_parser("image", help="generate an image locally (stops the LLM server first)")
    s.add_argument("text", nargs="+", help="what to draw")
    s.add_argument("-o", "--out", help="output file or folder (default folder: ~/Downloads, file name: mlx-<timestamp>.png)")
    s.add_argument("-m", "--model", help="image model alias (default: first registered image model)")
    s.add_argument("--width", type=int, default=768)
    s.add_argument("--height", type=int, default=768)
    s.add_argument("--steps", type=int, default=9)
    s.add_argument("--seed", type=int)
    s.add_argument("--style", help="style snippet name, e.g. 'textbook' (prompts/style-textbook.md)")
    s.add_argument("--open", action="store_true", help="open the image when done")
    s.add_argument("--keep-server", action="store_true", help="do not stop the LLM server")
    s.set_defaults(func=lambda a: image.generate(
        a.text, a.out, a.model, a.width, a.height, a.steps, a.seed, a.style,
        a.open, a.keep_server))

    s = sub.add_parser("prompts", help="list, show or customise prompt files")
    s.add_argument("name", nargs="?", help="print this prompt")
    s.add_argument("--init", action="store_true", help="copy built-in prompts to ~/llm-local/prompts")
    s.add_argument("--path", action="store_true", help="print the folder for your own prompts")
    s.set_defaults(func=_cmd_prompts)

    s = sub.add_parser("help", help="show this help, or 'mlx help <command>' for one command")
    s.add_argument("topic", nargs="?", help="a command name, e.g. ask")

    def _help(args: argparse.Namespace) -> int:
        if not args.topic:
            p.print_help()
            return 0
        if args.topic not in sub.choices:
            print(f"Unknown command '{args.topic}'. Try: mlx help", file=sys.stderr)
            return 1
        sub.choices[args.topic].print_help()
        return 0

    s.set_defaults(func=_help)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    try:
        code = args.func(args) or 0
        sys.stdout.flush()
    except BrokenPipeError:  # e.g. `mlx status | head -1`
        os._exit(0)
    sys.exit(code)
