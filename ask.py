import sys, base64, mimetypes, pathlib, argparse
from openai import OpenAI

DIR = pathlib.Path.home() / "llm-local"
try:
    MODEL = (DIR / "current_model").read_text().strip()   # written by mlx-start
except FileNotFoundError:
    sys.exit("No model selected. Run: mlx-start qwen | gemma | paddle")
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")


def ask(prompt, image=None, max_tokens=800, temperature=None):
    content = [{"type": "text", "text": prompt}]
    if image:
        mime = mimetypes.guess_type(image)[0] or "image/jpeg"
        b64 = base64.b64encode(pathlib.Path(image).expanduser().read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    extra = {}
    if temperature is not None:
        extra["temperature"] = temperature
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        stream=True,
        **extra,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mlx-ask")
    parser.add_argument("prompt")
    parser.add_argument("image", nargs="?")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    ask(args.prompt, args.image, args.max_tokens, args.temperature)
