"""Kod değerlendirici — model cevabındaki Python kodunu unit test'lerle YÜRÜTEREK ölçer.

Güvenlik (CLAUDE.md Kural 5 ruhu — model üretimi kod süreç içinde ``exec`` EDİLMEZ):
- Yürütme varsayılan KAPALI (``allow_execution=False``) → sonuç ``correct=None``
  ("ölçülemedi"), asla sessizce PASS değil.
- Açıkken: ayrı süreç (``python -I``: izole mod, kullanıcı site-packages/env yok), geçici
  dizin, zaman aşımı, en aza indirilmiş ortam değişkenleri.
- Statik ön-eleme: ağ/süreç/dosya-silme/dinamik-kod API'leri içeren kod ÇALIŞTIRILMAZ
  (``unsafe_code`` → yanlış).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.evals.profile.schema import EvalItem

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_UNSAFE = re.compile(
    r"\b(import\s+(os|subprocess|socket|shutil|requests|httpx|urllib|ctypes|multiprocessing)"
    r"|from\s+(os|subprocess|socket|shutil|requests|httpx|urllib|ctypes|multiprocessing)\b"
    r"|__import__|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|\bopen\s*\(|importlib|pickle)"
)
DEFAULT_TIMEOUT_S = 10.0


def extract_code(answer: str) -> str:
    blocks = _CODE_BLOCK.findall(answer)
    return max(blocks, key=len).strip() if blocks else ""


def run_code_tests(code: str, tests: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Kod + test'i izole alt süreçte çalıştır; returncode 0 → geçti."""
    with tempfile.TemporaryDirectory(prefix="hektor_code_eval_") as tmp:
        script = Path(tmp) / "solution_test.py"
        script.write_text(code + "\n\n# --- tests ---\n" + tests + "\n", encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        if os.name == "nt":  # Windows'ta python başlatmak için SystemRoot gerekir
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"zaman aşımı ({timeout_s}s)"}
        return {
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-400:],
        }


def evaluate_coding(
    item: EvalItem, answer: str, *, allow_execution: bool = False
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    code = extract_code(answer)
    checks["has_code"] = bool(code)
    if not item.requires_code_execution or not item.code_tests.strip():
        # Yürütme testi tanımlı değil → kabul edilen cevap/iddia kontrolüne düş.
        from app.evals.profile.math_eval import evaluate_numeric_or_exact

        return evaluate_numeric_or_exact(item, answer)
    if not code:
        return {"correct": False, "score": 0.0, "checks": checks | {"error": "kod bloğu yok"}}
    if _UNSAFE.search(code):
        checks["unsafe_code"] = True
        return {"correct": False, "score": 0.0, "checks": checks}
    if not allow_execution:
        checks["skipped"] = "kod yürütme kapalı (allow_code_execution=false)"
        return {"correct": None, "score": None, "checks": checks}
    run = run_code_tests(code, item.code_tests)
    checks["execution"] = run
    ok = bool(run.get("passed"))
    return {"correct": ok, "score": float(ok), "checks": checks}
