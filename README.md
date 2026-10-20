# llm-local

One command, `mlx`, to run local [MLX](https://github.com/ml-explore/mlx) models on Apple Silicon:
serve a model behind an OpenAI-compatible API, ask it questions (with images), chat, and
generate images. Tested on an M4 Pro with 24 GB RAM. Lives in `~/llm-local`.

```
mlx start gemma     # daily model     (qwen = heavier, paddle = OCR)
mlx ask "Summarize the main idea of MST in 2 sentences"
mlx ask "Describe this image" photo.png
mlx image "a red fox" --style textbook
mlx models          # downloaded models, size, date, newest first
mlx stop
```

## Commands

| Command | Description |
|---|---|
| `mlx start <gemma\|qwen\|paddle\|alias\|repo>` | Stop any running server, start the chosen model on port 8080 (localhost only) |
| `mlx stop` / `mlx status` | Stop the server / show what runs. `status` first line: `RUNNING: <repo-id>` or `Nothing is running.` |
| `mlx models` | Downloaded models with size and download date, newest first |
| `mlx pull <alias\|repo>` / `mlx add <alias> <repo> --kind llm\|image` | Download a model / register an alias |
| `mlx ask [--max-tokens N] [--temperature T] "prompt" [image ...]` | One question; image files go after the prompt. No system prompt unless `-p NAME` |
| `mlx chat [-p NAME]` | Interactive chat (`/reset`, `/exit`) |
| `mlx image "text" [-o file-or-folder] [--style NAME]` | Local image generation with Z-Image Turbo via mflux. Default folder `~/Downloads` (`MLX_IMAGE_DIR`) |
| `mlx prompts [name] [--init] [--path]` | List, show or customise prompt files |

## Setup

Main environment (Gemma, Qwen, the `mlx` command itself):

    python3 -m venv .venv
    .venv/bin/pip install -e .

OCR environment (PaddleOCR-VL, used only by `mlx start paddle`):

    python3 -m venv .venv-ocr
    .venv-ocr/bin/pip install -U mlx-vlm

Image generation (separate tool environment, it pins other library versions):

    uv tool install --upgrade mflux --with "pillow>=12.3.0,<13"

Put both on your PATH: `export PATH="$HOME/llm-local/.venv/bin:$HOME/llm-local/bin:$PATH"`.
`bin/mlx` is a launcher with absolute paths for GUI apps that call `/bin/bash ~/llm-local/bin/mlx ...`
with a minimal PATH. Models download from Hugging Face on first use.

## Files other tools rely on

- `~/llm-local/current_model`: repo id of the loaded model (can be stale when the server is down).
- `~/llm-local/server.pid`, `~/llm-local/server.log`.
- Server: `http://127.0.0.1:8080/v1` (`MLX_PORT` to change).
- `mlx stop` only ends `python -m mlx_vlm.server` processes, never other programs that hold the port.

## Prompts

Prompts are plain Markdown files, separate from the code. Yours go in `~/llm-local/prompts/<name>.md`
and override built-in ones. A first line starting with `#` is a description and is not sent to the model.
Built-in: `assistant`, `svg-diagram`, `student-playtester`, `style-textbook` (image style, `--style textbook`).

    mlx ask -p svg-diagram "capillary action in a tube" -o diagram.svg
    cat rules.md | mlx ask --stdin -p student-playtester "You are 9 and bore easily. Which rules confuse you?"

## Notes

- One server at a time; a 27B LLM and an image model should not run together in 24 GB.
- Loading takes 20-60 seconds. `mlx start` returns 1 on failure or timeout.
- Image models render long labels and chemical formulas unreliably; for exact text ask the LLM for SVG.
- Z-Image Turbo weights carry the Tongyi Qianwen license.

## Development

    .venv/bin/pip install -e ".[dev]" && .venv/bin/pytest
