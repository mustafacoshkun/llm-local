# llm-local

Small shell CLI around `mlx-vlm` for running local MLX models on Apple Silicon
(Gemma, Qwen, PaddleOCR-VL) behind one OpenAI-compatible server on port 8080.
Tested on an M4 Pro with 24 GB RAM. Expects to live in `~/llm-local`.

## Commands

| Command | Description |
|---|---|
| `mlx-start gemma\|qwen\|paddle` | Stop any running server, start the chosen model on port 8080 |
| `mlx-stop` | Stop the server |
| `mlx-status` | Show which model is running |
| `mlx-models` | List downloaded models and sizes |
| `mlx-ask [--max-tokens N] [--temperature T] "prompt" [image]` | Send a prompt, optionally with an image |
| `mlx-help` | Show usage |

## Setup

Main environment (Gemma, Qwen):

    python3 -m venv .venv
    .venv/bin/pip install -U mlx-vlm openai

OCR environment (PaddleOCR-VL):

    python3 -m venv .venv-ocr
    .venv-ocr/bin/pip install -U mlx-vlm

Add the commands to your PATH:

    export PATH="$HOME/llm-local/bin:$PATH"

Models are downloaded from Hugging Face on first start.

## Notes

- One server at a time. Starting a model stops the previous one.
- Loading takes 20-60 seconds.
- `server.log`, `server.pid` and `current_model` are local state and git-ignored.
- OCR example: `mlx-start paddle && mlx-ask --max-tokens 4000 --temperature 0 "OCR:" page.png`
