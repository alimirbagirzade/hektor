"""Kademe-2 (2026-09-28, 30B öncesi): başlatma ön-kontrolleri + durum kaydı işaretleri.

Çevrimdışı: gerçek storage/ yerine tmp kök kullanılır; eğitim başlatılmaz.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.training import detached_launch
from app.training.detached_launch import _adapter_dir_blocker, mark_detached_status


def test_adapter_dir_blocker(tmp_path: Path) -> None:
    """B4: aynı adla kalan checkpoint'ler/bitmiş koşu → sıfırdan başlatma reddedilir."""
    assert _adapter_dir_blocker(tmp_path / "yeni") is None
    d = tmp_path / "hektor_lora_v11_30b"
    d.mkdir()
    assert _adapter_dir_blocker(d) is None  # boş klasör sorun değil
    (d / "checkpoint-20").mkdir()
    msg = _adapter_dir_blocker(d)
    assert msg and "YENİ" in msg
    (d / "checkpoint-20").rmdir()
    (d / "run_complete.json").write_text("{}", encoding="utf-8")
    assert _adapter_dir_blocker(d)


def test_mark_detached_status_only_same_adapter(tmp_path: Path) -> None:
    st = tmp_path / "storage" / "train_status.json"
    st.parent.mkdir()
    st.write_text(json.dumps({"adapter": "a", "iterations": 600}), encoding="utf-8")
    assert not mark_detached_status("b", root=tmp_path, finished_at="x")
    assert "finished_at" not in json.loads(st.read_text(encoding="utf-8"))
    assert mark_detached_status("a", root=tmp_path, failed_at="t", error="RAM")
    data = json.loads(st.read_text(encoding="utf-8"))
    assert data["failed_at"] == "t" and data["error"] == "RAM" and data["iterations"] == 600
    assert not mark_detached_status("a", root=tmp_path / "yok", finished_at="x")


def test_recipe_blockers_bad_profile() -> None:
    out = detached_launch._recipe_blockers(None, "yok_boyle_profil")
    assert out and "Profil" in out[0]
