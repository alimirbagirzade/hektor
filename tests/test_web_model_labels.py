"""UI'daki model adları ayardan gelmeli — sabit yazılı model adı regresyon testi.

Olay (2026-09-27): LLM `qwen3:30b` yapıldığı hâlde web arayüzü "Qwen3 4B" gösteriyordu:
donanım panosu yalnız sabit registry'den gelen ÖNERİ listesini basıyor, LoRA metinleri
de "4B" sabitini içeriyordu. Artık aktif model `/api/status` + `/api/recommend.active`
ile ayardan okunur; statik dosyalarda model adı gömülü olmamalı.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.config import get_settings
from app.web.server import app

STATIC = Path(__file__).resolve().parents[1] / "app" / "web" / "static"
# Model kimliği kalıpları: ollama etiketi (qwen3:30b, llama3.1:8b) veya HF adı (Qwen3-4B).
_MODEL_ID = re.compile(
    r"(?i)\b(qwen\d(?:\.\d)?|llama\d(?:\.\d)?|mistral|deepseek-r\d):\d+(?:\.\d+)?b\b"
    r"|Qwen\d-\d+B"
    # Çıplak boyut etiketi model adı yerine kullanılmış: "(…, 4B)" / "4B brain".
    r"|\b\d+(?:\.\d+)?B(?=\)| brain)"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_status_model_alanlari_ayardan(client: TestClient) -> None:
    s = get_settings()
    body = client.get("/api/status").json()
    assert body["llm_model"] == s.llm_model
    assert body["peft_base_model"] == s.peft_base_model


def test_recommend_aktif_modeli_ayardan_dondurur(client: TestClient, monkeypatch) -> None:
    import httpx

    def _offline(*_a, **_k):
        raise httpx.ConnectError("ollama yok (test)")

    monkeypatch.setattr(httpx, "get", _offline)
    body = client.get("/api/recommend").json()
    active = body["active"]
    assert active["ollama"] == get_settings().llm_model
    assert active["installed"] is None  # Ollama erişilemezse "bilinmiyor", uydurma yok
    for r in body["recommended"]:
        assert r["active"] == (r["ollama"] == get_settings().llm_model)


@pytest.mark.parametrize("rel", ["index.html", "assets/app.js"])
def test_statik_arayuzde_sabit_model_adi_yok(rel: str) -> None:
    text = (STATIC / rel).read_text(encoding="utf-8")
    hits = sorted({m.group(0) for m in _MODEL_ID.finditer(text)})
    assert not hits, (
        f"{rel} içinde sabit model adı: {hits} — model adını /api/status veya "
        "/api/recommend'den oku (model değişince yazı da değişmeli)."
    )
