import os
import time

import pytest

from mlxcli import chat, config, image, models, prompts, server


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOME", tmp_path / "home")
    monkeypatch.setattr(config, "CURRENT_MODEL_FILE", tmp_path / "home" / "current_model")
    monkeypatch.setattr(config, "PID_FILE", tmp_path / "home" / "server.pid")
    monkeypatch.setattr(config, "USER_PROMPTS_DIR", tmp_path / "home" / "prompts")
    monkeypatch.setattr(config, "USER_MODELS_FILE", tmp_path / "home" / "models.json")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


def test_split_prompt_strips_title():
    summary, body = prompts.split_prompt("# Title\nbody line\n")
    assert summary == "Title" and body == "body line"


def test_builtin_prompts_load():
    names = [n for n, _, _ in prompts.list_prompts()]
    for expected in ("assistant", "svg-diagram", "student-playtester", "style-textbook"):
        assert expected in names
    assert not prompts.load("assistant").startswith("#")


def test_user_prompt_overrides_builtin(tmp_path):
    config.USER_PROMPTS_DIR.mkdir(parents=True)
    (config.USER_PROMPTS_DIR / "assistant.md").write_text("# mine\ncustom body")
    assert prompts.load("assistant") == "custom body"
    assert ("assistant", "yours", "mine") in prompts.list_prompts()


def test_unknown_prompt_exits():
    with pytest.raises(SystemExit):
        prompts.load("does-not-exist")


def test_strip_fences():
    assert chat.strip_fences("```svg\n<svg/>\n```") == "<svg/>\n"
    assert chat.strip_fences("<svg/>") == "<svg/>"


def test_resolve_model_alias_and_repo():
    assert config.resolve_model("qwen")["kind"] == "llm"
    assert config.resolve_model("zimage")["kind"] == "image"
    assert config.resolve_model("org/name")["repo"] == "org/name"
    with pytest.raises(SystemExit):
        config.resolve_model("nope")


def test_add_user_model():
    models.add("tiny", "org/Tiny-4bit", "llm")
    assert config.resolve_model("tiny")["repo"] == "org/Tiny-4bit"


def _fake_model(hub, repo, blob_bytes, age):
    d = hub / ("models--" + repo.replace("/", "--"))
    (d / "blobs").mkdir(parents=True)
    (d / "snapshots" / "abc").mkdir(parents=True)
    blob = d / "blobs" / "deadbeef"
    blob.write_bytes(b"x" * blob_bytes)
    link = d / "snapshots" / "abc" / "weights.safetensors"
    link.symlink_to(blob)
    t = time.time() - age
    os.utime(d, (t, t))


def test_models_sorted_newest_first_and_dedup(tmp_path):
    hub = tmp_path / "hub"
    _fake_model(hub, "old/Model", 1000, age=1000)
    _fake_model(hub, "mlx-community/Qwen3.8-27B-4bit", 5000, age=10)
    rows = models.downloaded_models()
    assert [r["repo"] for r in rows] == ["mlx-community/Qwen3.8-27B-4bit", "old/Model"]
    assert rows[0]["size"] == 5000          # symlink + blob counted once
    assert rows[0]["alias"] == "qwen" and rows[1]["alias"] == ""


def test_fmt_size():
    assert models.fmt_size(15 * 1024**3) == "15.0G"
    assert models.fmt_size(5 * 1024**2) == "5M"


def test_server_command_matching_is_exact():
    assert server.is_server_command("/usr/bin/python3 -m mlx_vlm.server --model x --port 8080")
    assert server.is_server_command("python3.12 -m mlx_vlm.server --model x")
    # things that only mention the name must NOT match
    assert not server.is_server_command("/bin/bash -c 'mkdir -p /tmp/fake/mlx_vlm && python3 -m mlx_vlm.server'")
    assert not server.is_server_command("grep mlx_vlm.server log.txt")
    assert not server.is_server_command("vim mlx_vlm.server.py")
    assert not server.is_server_command("")


@pytest.fixture
def img_dir(tmp_path, monkeypatch):
    d = tmp_path / "Downloads"
    monkeypatch.setattr(config, "IMAGE_DIR", d)
    return d


def test_output_default_is_downloads(img_dir):
    out = image.resolve_output(None, now=0)
    assert out.parent == img_dir and out.name.startswith("mlx-") and out.suffix == ".png"


def test_output_folder_gets_timestamped_name(tmp_path, img_dir):
    existing = tmp_path / "pics"
    existing.mkdir()
    assert image.resolve_output(str(existing)).parent == existing           # existing dir
    assert image.resolve_output(str(tmp_path / "new") + "/").parent == tmp_path / "new"
    assert image.resolve_output(str(tmp_path / "made")).parent == tmp_path / "made"  # no suffix = folder


def test_output_bare_filename_goes_to_default_folder(img_dir):
    assert image.resolve_output("fox.png") == img_dir / "fox.png"


def test_output_path_with_folder_is_kept(tmp_path, img_dir):
    assert image.resolve_output(str(tmp_path / "a" / "fox.png")) == tmp_path / "a" / "fox.png"


def test_output_never_overwrites(img_dir):
    first = image.resolve_output("fox.png")
    first.write_bytes(b"x")
    assert image.resolve_output("fox.png") == img_dir / "fox-2.png"


def test_paddle_profile_uses_ocr_venv_and_trust_remote_code():
    paddle = config.resolve_model("paddle")
    assert paddle["repo"] == "OpenGryd/PaddleOCR-VL-1.6-MLX-16bit"
    assert paddle["venv"] == ".venv-ocr" and "--trust-remote-code" in paddle["args"]
    gemma = config.resolve_model("gemma")
    assert gemma["venv"] == ".venv" and gemma["args"] == []


def test_state_files_roundtrip_and_stay_after_clear_pid():
    assert config.read_state() is None
    config.write_state("org/Model-4bit", 123)
    assert config.CURRENT_MODEL_FILE.read_text() == "org/Model-4bit\n"   # format other tools parse
    assert config.read_state() == {"repo": "org/Model-4bit", "pid": 123}
    config.clear_pid()
    assert config.read_state() == {"repo": "org/Model-4bit", "pid": None}  # current_model kept


def test_server_command_matches_macos_framework_python():
    cmd = ("/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/"
           "Contents/MacOS/Python -m mlx_vlm.server --model x --port 8080")
    assert server.is_server_command(cmd)


def test_status_first_line_contract(capsys, monkeypatch):
    monkeypatch.setattr(server, "running_pids", lambda: [])
    server.status()
    assert capsys.readouterr().out.strip() == "Nothing is running."
    config.write_state("org/Model-4bit", 1)
    monkeypatch.setattr(server, "running_pids", lambda: [1])
    monkeypatch.setattr(server, "port_open", lambda: True)
    server.status()
    assert capsys.readouterr().out.splitlines()[0] == "RUNNING: org/Model-4bit"


def test_start_checks_environment_before_stopping(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(server, "stop", lambda quiet=False: called.append("stop"))
    assert server.start("paddle", timeout=1) == 1          # .venv-ocr missing in the temp home
    assert "Environment not found" in capsys.readouterr().err
    assert called == []                                     # a running server must stay up


def test_old_style_image_argument(tmp_path):
    img = tmp_path / "page.png"
    img.write_bytes(b"x")
    text, images = chat.split_text_and_images(["OCR:", str(img)])
    assert text == "OCR:" and images == [str(img)]
    text, images = chat.split_text_and_images(["what", "is", "missing.png"])   # not a real file
    assert text == "what is missing.png" and images == []
    assert chat.split_text_and_images([str(img)]) == (str(img), [])          # first word = prompt


def test_starter_and_help_probe_detection():
    assert server.is_starter_command("/Users/x/llm-local/.venv/bin/python3.12 /Users/x/llm-local/.venv/bin/mlx start gemma")
    assert server.is_starter_command("python -m mlxcli start paddle")
    assert not server.is_starter_command("python /Users/x/llm-local/.venv/bin/mlx status")
    assert not server.is_starter_command("vim mlx start notes.txt")
    # our own compatibility probe must never count as a running server
    assert not server.is_server_command("python -m mlx_vlm.server --help")


def test_stale_pid_file_of_unrelated_process_is_ignored(monkeypatch):
    config.write_state("org/Model-4bit", 4242)
    monkeypatch.setattr(server, "_find_server_pids", lambda: [])
    monkeypatch.setattr(server, "_alive", lambda pid: True)
    monkeypatch.setattr(server, "_command_of", lambda pid: "/Applications/Safari.app/Contents/MacOS/Safari")
    assert server.running_pids() == []
    monkeypatch.setattr(server, "_command_of", lambda pid: "python /x/.venv/bin/mlx start gemma")
    assert server.running_pids() == [4242]
