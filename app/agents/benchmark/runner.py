from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

Status = Literal["excellent", "usable", "slow_but_usable", "unstable", "failed"]

_PROMPTS: list[dict] = [
    {
        "id": "json_001",
        "prompt": "Return a JSON object with fields: name, status, score. No markdown. Only JSON.",
        "expected_type": "json",
    },
    {
        "id": "code_001",
        "prompt": "Write a Python function: moving_average(values: list, n: int) -> list.",
        "expected_type": "code",
    },
    {
        "id": "reason_001",
        "prompt": "A machine makes 120 parts in 3 hours. How many in 8 hours? Give only a number.",
        "expected_type": "reasoning",
        "expected_number": "320",
    },
    {
        "id": "multilingual_001",
        "prompt": "Translate to Turkish: 'The local AI system is running correctly.'",
        "expected_type": "translation",
    },
]


@dataclass
class PromptResult:
    prompt_id: str
    status: str  # ok | failed | timeout
    response: str = ""
    latency_ms: float = 0.0
    quality: float = 0.0  # 0–1


@dataclass
class BenchmarkResult:
    model: str
    backend: str
    status: Status
    tokens_per_second: float = 0.0
    first_token_latency_ms: float = 0.0
    peak_ram_gb: float = 0.0
    quality_score: float = 0.0
    prompt_results: list[PromptResult] = field(default_factory=list)
    error: str = ""


def _call_ollama(model: str, prompt: str, timeout: int = 60) -> tuple[str, float, float]:
    """ollama run çağrısı → (yanıt, toplam_ms, tokens_per_sec)."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response = result.stdout.strip()
        # kaba tokens/sec tahmini (kelime ≈ 1.3 token)
        words = len(response.split())
        tps = (words * 1.3) / max(elapsed_ms / 1000, 0.001) if words > 0 else 0.0
        return response, elapsed_ms, round(tps, 1)
    except subprocess.TimeoutExpired:
        return "", (time.perf_counter() - start) * 1000, 0.0
    except Exception as e:
        return f"ERROR:{e}", 0.0, 0.0


def _evaluate_response(result: dict, response: str) -> float:
    """Basit heuristik kalite skoru (0–1)."""
    if not response or response.startswith("ERROR:"):
        return 0.0
    expected_type = result.get("expected_type", "")
    if expected_type == "json":
        try:
            json.loads(response)
            return 1.0
        except Exception:
            # JSON içeriyor mu?
            return 0.5 if "{" in response and "}" in response else 0.1
    elif expected_type == "code":
        return 1.0 if "def " in response else (0.5 if "function" in response else 0.2)
    elif expected_type == "reasoning":
        expected_num = result.get("expected_number", "")
        return 1.0 if expected_num and expected_num in response else 0.3
    elif expected_type == "translation":
        # Türkçe karakterler var mı?
        tr_chars = set("çğışöüÇĞİŞÖÜ")
        return 0.9 if any(c in response for c in tr_chars) else 0.4
    return 0.7 if len(response) > 10 else 0.2


def _ram_usage_gb() -> float:
    if _HAS_PSUTIL:
        try:
            return round(psutil.virtual_memory().used / 1024**3, 2)
        except Exception:
            pass
    return 0.0


def run(model_ollama_name: str, quick: bool = False) -> BenchmarkResult:
    """Modeli benchmark et. quick=True ise sadece ilk prompt."""
    import shutil

    if not shutil.which("ollama"):
        return BenchmarkResult(
            model=model_ollama_name,
            backend="ollama",
            status="failed",
            error="Ollama kurulu değil",
        )

    prompts = _PROMPTS[:1] if quick else _PROMPTS
    prompt_results: list[PromptResult] = []
    latencies: list[float] = []
    tps_list: list[float] = []
    quality_list: list[float] = []

    ram_before = _ram_usage_gb()

    for p in prompts:
        response, elapsed_ms, tps = _call_ollama(model_ollama_name, p["prompt"])
        quality = _evaluate_response(p, response)
        ok = not response.startswith("ERROR:") and len(response) > 0
        prompt_results.append(
            PromptResult(
                prompt_id=p["id"],
                status="ok" if ok else "failed",
                response=response[:300],
                latency_ms=round(elapsed_ms, 1),
                quality=quality,
            )
        )
        if ok:
            latencies.append(elapsed_ms)
            tps_list.append(tps)
            quality_list.append(quality)

    ram_after = _ram_usage_gb()
    peak_ram = max(ram_after - ram_before, 0.0)
    avg_tps = round(sum(tps_list) / max(len(tps_list), 1), 1)
    first_latency = round(latencies[0], 1) if latencies else 0.0
    avg_quality = round(sum(quality_list) / max(len(quality_list), 1), 2)

    failed_count = sum(1 for r in prompt_results if r.status == "failed")
    total = len(prompt_results)

    if failed_count == total:
        status: Status = "failed"
    elif avg_tps > 15 and avg_quality > 0.7:
        status = "excellent"
    elif avg_tps > 5 and avg_quality > 0.5:
        status = "usable"
    elif avg_tps > 2:
        status = "slow_but_usable"
    else:
        status = "unstable"

    return BenchmarkResult(
        model=model_ollama_name,
        backend="ollama",
        status=status,
        tokens_per_second=avg_tps,
        first_token_latency_ms=first_latency,
        peak_ram_gb=round(peak_ram, 2),
        quality_score=avg_quality,
        prompt_results=prompt_results,
    )
