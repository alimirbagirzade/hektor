"""MlxLLM birim testleri — çevrimdışı (gerçek MLX çalıştırmaz)."""

from __future__ import annotations

import subprocess
import unittest.mock as mock
from pathlib import Path

import pytest

from app.brain.mlx_llm import MlxLLM, MlxLLMUnavailable


@pytest.fixture
def mlx() -> MlxLLM:
    return MlxLLM(base_model="test-model/1b-4bit")


@pytest.fixture
def mlx_with_adapter(tmp_path: Path) -> MlxLLM:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").touch()
    return MlxLLM(base_model="test-model/1b-4bit", adapter_path=adapter)


def test_available_when_mlx_lm_importable(mlx: MlxLLM) -> None:
    with mock.patch("importlib.util.find_spec", return_value=object()):
        assert mlx.available() is True


def test_not_available_when_mlx_lm_missing(mlx: MlxLLM) -> None:
    with mock.patch("importlib.util.find_spec", return_value=None):
        assert mlx.available() is False


def test_generate_raises_when_unavailable(mlx: MlxLLM) -> None:
    with (
        mock.patch.object(mlx, "available", return_value=False),
        pytest.raises(MlxLLMUnavailable),
    ):
        mlx.generate("test prompt")


def test_generate_calls_subprocess(mlx: MlxLLM) -> None:
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stdout = "==========\ntest yanıt\n=========="
    fake.stderr = ""
    with (
        mock.patch.object(mlx, "available", return_value=True),
        mock.patch("subprocess.run", return_value=fake) as mock_run,
    ):
        result = mlx.generate("merhaba")
    assert result == "test yanıt"
    cmd = mock_run.call_args[0][0]
    assert "mlx_lm" in cmd
    assert "generate" in cmd
    assert "test-model/1b-4bit" in cmd


def test_generate_includes_adapter_path(mlx_with_adapter: MlxLLM) -> None:
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stdout = "==========\nyanıt\n=========="
    fake.stderr = ""
    with (
        mock.patch.object(mlx_with_adapter, "available", return_value=True),
        mock.patch("subprocess.run", return_value=fake) as mock_run,
    ):
        mlx_with_adapter.generate("prompt")
    cmd = mock_run.call_args[0][0]
    assert "--adapter-path" in cmd


def test_generate_no_adapter_if_path_not_exists(mlx: MlxLLM) -> None:
    mlx_no_adapter = MlxLLM("model", adapter_path="/nonexistent/path")
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stdout = "==========\nok\n=========="
    fake.stderr = ""
    with (
        mock.patch.object(mlx_no_adapter, "available", return_value=True),
        mock.patch("subprocess.run", return_value=fake) as mock_run,
    ):
        mlx_no_adapter.generate("hi")
    cmd = mock_run.call_args[0][0]
    assert "--adapter-path" not in cmd


def test_generate_raises_on_nonzero_exit(mlx: MlxLLM) -> None:
    fake = mock.MagicMock()
    fake.returncode = 1
    fake.stdout = ""
    fake.stderr = "some error"
    with (
        mock.patch.object(mlx, "available", return_value=True),
        mock.patch("subprocess.run", return_value=fake),
        pytest.raises(MlxLLMUnavailable),
    ):
        mlx.generate("prompt")


def test_generate_raises_on_timeout(mlx: MlxLLM) -> None:
    with (
        mock.patch.object(mlx, "available", return_value=True),
        mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=300)),
        pytest.raises(MlxLLMUnavailable, match="zaman aşımı"),
    ):
        mlx.generate("prompt")


def test_max_tokens_passed_to_command(mlx: MlxLLM) -> None:
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stdout = "==========\nok\n=========="
    fake.stderr = ""
    with (
        mock.patch.object(mlx, "available", return_value=True),
        mock.patch("subprocess.run", return_value=fake) as mock_run,
    ):
        mlx.generate("prompt", max_tokens=256)
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--max-tokens")
    assert cmd[idx + 1] == "256"
