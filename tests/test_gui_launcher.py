import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from pinf.gui import launcher


def test_app_path_points_to_packaged_app():
    assert launcher.APP_PATH == Path(launcher.__file__).resolve().parent / "app.py"
    assert launcher.APP_PATH.is_file()


def test_streamlit_command_runs_packaged_app_with_extra_args():
    cmd = launcher.streamlit_command(["--server.port", "8600"])
    assert cmd == [
        sys.executable, "-m", "streamlit", "run", str(launcher.APP_PATH),
        "--server.port", "8600",
    ]


def test_main_launches_streamlit_and_returns_its_exit_code(monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return type("Completed", (), {"returncode": 3})()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.main(["--server.headless", "true"]) == 3
    assert calls == [launcher.streamlit_command(["--server.headless", "true"])]


def test_main_explains_missing_streamlit(monkeypatch, capsys):
    monkeypatch.setattr(launcher.importlib.util, "find_spec", lambda name: None)
    assert launcher.main([]) == 1
    assert 'pip install "pinf[gui]"' in capsys.readouterr().err


def test_app_renders_without_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(launcher.APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Plausible Inference"
    assert (tmp_path / "inference_results").is_dir()
