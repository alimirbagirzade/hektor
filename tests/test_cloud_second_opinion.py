"""Faz 3 · buluttan ikinci görüş — sahte sağlayıcıyla, AĞ ÇAĞRISI YOK.

Kabul şartları: varsayılan kapalı (üç ayar birden gerekir); önizleme hiçbir şey göndermez ve
gönderilecek metnin tamamını + özetini, sağlayıcı/şart/kota/maliyet/iptal bilgisini tek yerde
verir; gönderim önizlenen özetle eşleşmezse reddedilir; kaynak parçaları gönderilmez; yerel
cevap / bulut önerisi / bağımsız doğrulama ayrı saklanır; iptal yarım cevabı saklamaz; günlük
üst sınır uygulanır; bulut kökenli satır eğitim kapısında NO-GO; öğrenme/eğitim kodu bulut
modülünü kullanmaz; anahtar alanı yok.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from tests.chat_learning_helpers import iso, send  # noqa: F401

from app.cloud import policy
from app.cloud import second_opinion as so
from app.cloud.providers import FakeProvider, ProviderError, ProviderReply

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def turn(iso, monkeypatch):  # noqa: F811
    from app.feedback.chat_store import ChatStore

    store = ChatStore()
    conv = store.create_conversation("TEST")["conversation_id"]
    return send(store, conv, "Look-ahead bias nedir ve nasıl önlenir?")


def _enable(monkeypatch, provider="fake", daily_max=10):
    from app.config import settings as settings_mod

    monkeypatch.setenv("HEKTOR_CLOUD_SECOND_OPINION", "1")
    monkeypatch.setenv("HEKTOR_CLOUD_PROVIDER", provider)
    monkeypatch.setenv("HEKTOR_CLOUD_TERMS_ACK", "2026-10-06")
    monkeypatch.setenv("HEKTOR_CLOUD_DAILY_MAX", str(daily_max))
    settings_mod.get_settings.cache_clear()


def test_disabled_by_default_and_preview_sends_nothing(turn, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(so, "provider_factory", lambda n: calls.append(n) or FakeProvider())
    pv = so.preview(turn["turn_id"])
    assert pv["enabled"] is False and len(pv["blockers"]) == 3
    assert calls == []  # kapalıyken sağlayıcı oluşturulmaz bile
    with pytest.raises(so.CloudError, match="kapalı"):
        so.start(turn["turn_id"], pv["payload_sha256"])


def test_preview_shows_exact_payload_without_sources(turn, monkeypatch) -> None:
    _enable(monkeypatch)
    pv = so.preview(turn["turn_id"])
    assert pv["enabled"] and pv["payload"] == so.build_payload(turn)
    assert turn["question"] in pv["payload"] and turn["answer"] in pv["payload"]
    # Kaynak parçasının cevapta OLMAYAN kısmı gitmez (cevap kaynakla aynı cümleyi ve atıf
    # kimliğini içerebilir; o kısım cevabın kendisidir).
    assert "varyansı zamanla değişir" not in pv["payload"]
    for key in ("cost_note", "cancel_note", "not_sent", "quota", "policy", "est_input_tokens"):
        assert pv[key]
    assert pv["policy"]["training_use"] is False


def test_send_requires_previewed_hash_and_stores_three_separate_parts(turn, monkeypatch) -> None:
    _enable(monkeypatch)
    fake = FakeProvider("Bulut: shift(1) gerekir; 2 + 2 = 5.")
    monkeypatch.setattr(so, "provider_factory", lambda n: fake)
    with pytest.raises(so.CloudError, match="önizlemeden farklı"):
        so.start(turn["turn_id"], "0" * 64)
    pv = so.preview(turn["turn_id"])
    rec = so.start(turn["turn_id"], pv["payload_sha256"], wait=True)
    assert fake.prompts == [pv["payload"]]  # tam olarak önizlenen metin gitti
    assert rec["yerel"]["answer"] == turn["answer"]
    assert rec["bulut"]["status"] == "done" and "shift(1)" in rec["bulut"]["text"]
    ver = rec["bagimsiz_dogrulama"]
    assert ver["ran"] and ver["verdict"] == "dogrulama_degil"
    hesap = [c for c in ver["checks"] if c["kind"] == "hesap"]
    assert hesap and hesap[0]["status"] != "gecti"  # 2 + 2 = 5 yerel kontrolde yakalanır
    assert rec["training_use"] is False and "doğrulama DEĞİLDİR" in rec["note"]
    assert so.list_for_turn(turn["turn_id"])[0]["id"] == rec["id"]


def test_cancel_keeps_no_partial_answer(turn, monkeypatch) -> None:
    _enable(monkeypatch)
    gate = threading.Event()

    class Slow(FakeProvider):
        def ask(self, prompt, *, timeout_s):
            gate.wait(5)
            return ProviderReply(text="geç gelen cevap", model="slow")

    monkeypatch.setattr(so, "provider_factory", lambda n: Slow())
    pv = so.preview(turn["turn_id"])
    rec = so.start(turn["turn_id"], pv["payload_sha256"])
    assert rec["bulut"]["status"] == "pending"
    so.cancel(rec["id"])
    gate.set()
    import time

    time.sleep(0.5)
    got = so.get(rec["id"])
    assert got["bulut"]["status"] == "cancelled" and got["bulut"]["text"] == ""


def test_provider_failure_and_daily_cap(turn, monkeypatch) -> None:
    _enable(monkeypatch, daily_max=1)
    monkeypatch.setattr(so, "provider_factory", lambda n: FakeProvider(fail="zaman aşımı"))
    pv = so.preview(turn["turn_id"])
    rec = so.start(turn["turn_id"], pv["payload_sha256"], wait=True)
    assert rec["bulut"]["status"] == "failed" and rec["bagimsiz_dogrulama"] is None
    pv2 = so.preview(turn["turn_id"])
    assert not pv2["enabled"] and any("Günlük üst sınır" in b for b in pv2["blockers"])


def test_unknown_or_api_provider_is_not_enabled(turn, monkeypatch) -> None:
    _enable(monkeypatch, provider="anthropic_api")
    st = so.status()
    assert not st["enabled"] and any("Ticari Şartlar" in b for b in st["blockers"])
    _enable(monkeypatch, provider="bilinmeyen")
    assert not so.status()["enabled"]
    with pytest.raises(ProviderError):
        from app.cloud.providers import make_provider

        make_provider("anthropic_api")


def test_web_routes_human_scope(turn, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    _enable(monkeypatch)
    monkeypatch.setattr(so, "provider_factory", lambda n: FakeProvider("tamam"))
    client = TestClient(app)
    pv = client.get(f"/api/cloud/second-opinion/turn/{turn['turn_id']}/preview").json()
    r = client.post(
        f"/api/cloud/second-opinion/turn/{turn['turn_id']}",
        json={"payload_sha256": pv["payload_sha256"]},
    )
    assert r.status_code == 200 and r.json()["id"].startswith("so_")
    for rt in client.app.routes:
        path = getattr(rt, "path", "")
        if path.startswith("/api/cloud/") and "POST" in getattr(rt, "methods", set()):
            assert require_human in {d.call for d in rt.dependant.dependencies}


# ── B · öğretmen çıktısı: eğitim kapısı + statik ayrım ───────────────────────────


def test_no_provider_allows_training_use() -> None:
    for name in policy.POLICIES:
        assert policy.training_use_allowed(name)[0] is False
    assert policy.training_use_allowed("yok")[0] is False


def test_cloud_origin_lines_detected_local_teacher_not() -> None:
    lines = [
        json.dumps({"messages": [], "metadata": {"teacher": "qwen3:30b-a3b-instruct-2507"}}),
        json.dumps({"messages": [], "metadata": {"origin": "cloud_second_opinion"}}),
        json.dumps({"messages": [], "metadata": {"teacher": "claude-opus"}}),
        json.dumps({"messages": [], "metadata": {"source": "chat"}}),
    ]
    assert policy.cloud_origin_lines(lines) == [1, 2]


def test_pretrain_gate_blocks_cloud_rows(iso, monkeypatch) -> None:  # noqa: F811
    from app.config import get_settings
    from app.training import detached_launch as dl

    src = get_settings().root / "lora_sft.jsonl"
    row = {"messages": [{"role": "user", "content": "s"}, {"role": "assistant", "content": "c"}]}
    src.write_text(
        json.dumps(row) + "\n" + json.dumps({**row, "metadata": {"origin": "cloud"}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dl, "_combined_source", lambda s: src)
    blockers = dl._pretrain_gate_blockers(get_settings())
    assert any("bulut kökenli 1 satır" in b for b in blockers)


def test_learning_and_training_code_never_import_cloud_answers() -> None:
    """Bulut cevabı öğrenme havuzuna / eğitim verisine giden koda bağlanamaz (yalnız politika)."""
    offenders = []
    for sub in ("app/feedback", "app/lora", "app/training", "scripts"):
        for p in (REPO / sub).rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "app.cloud.second_opinion" in text or "app.cloud.providers" in text:
                offenders.append(str(p.relative_to(REPO)))
    assert offenders == []


def test_no_api_key_fields_in_cloud_settings() -> None:
    from app.config.settings import Settings

    names = [n for n in Settings.model_fields if n.startswith("cloud_")]
    assert names and not any("key" in n or "token" in n or "secret" in n for n in names)


# ── CLI sağlayıcı sertleştirmesi (ağ yok; sahte ikili) ───────────────────────────────


def test_cli_provider_refuses_batch_wrapper_and_missing_binary(monkeypatch, tmp_path) -> None:
    from app.cloud.providers import ClaudeCodeCLIProvider

    monkeypatch.setattr(
        "app.orchestration.executable.resolve_cli", lambda b: str(tmp_path / "claude.cmd")
    )
    ok, why = ClaudeCodeCLIProvider().available()
    assert not ok and "toplu dosya" in why
    with pytest.raises(ProviderError, match="toplu dosya"):
        ClaudeCodeCLIProvider().ask("x", timeout_s=5)
    monkeypatch.setattr("app.orchestration.executable.resolve_cli", lambda b: None)
    assert ClaudeCodeCLIProvider().available()[0] is False


def test_cli_provider_runs_in_empty_temp_dir_with_prompt_as_single_arg(
    monkeypatch, tmp_path
) -> None:
    """İstem tek argv öğesi; çalışma dizini boş geçici klasör (proje bağlamı gönderilmez)."""
    import sys

    from app.cloud.providers import ClaudeCodeCLIProvider

    out = tmp_path / "seen.json"
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "import json, os, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'cwd_files': os.listdir('.')}}, open(r'{out}', 'w'))\n"
        "print('ELEŞTİRİ: tamam')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.orchestration.executable.resolve_cli", lambda b: sys.executable)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")  # gerçek CLI (node) UTF-8 basar
    real_popen = __import__("subprocess").Popen

    def popen(argv, **kw):  # gerçek CLI yerine aynı argv ile sahte betik
        return real_popen([argv[0], str(script), *argv[1:]], **kw)

    monkeypatch.setattr("app.cloud.providers.subprocess.Popen", popen)
    prompt = 'SORU: "a" && del * ; $(x)'
    reply = ClaudeCodeCLIProvider().ask(prompt, timeout_s=30)
    seen = json.loads(out.read_text(encoding="utf-8"))
    assert "ELEŞTİRİ" in reply.text
    assert seen["argv"][:2] == ["-p", prompt] and "--safe-mode" in seen["argv"]
    assert seen["cwd_files"] == []  # CLAUDE.md vb. proje dosyası görünmez


def test_cli_provider_reports_not_logged_in_from_stdout(monkeypatch, tmp_path) -> None:
    """Gerçek CLI "Not logged in"i STDOUT'a, ilgisiz uyarıyı stderr'e yazar; sebep gizlenmemeli."""
    import sys

    from app.cloud.providers import NOT_LOGGED_IN_HINT, ClaudeCodeCLIProvider

    script = tmp_path / "fake_claude.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write('Denying Bash also turns off the PowerShell tool.\\n')\n"
        "print('Not logged in · Please run /login')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.orchestration.executable.resolve_cli", lambda b: sys.executable)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    real_popen = __import__("subprocess").Popen

    def popen(argv, **kw):
        return real_popen([argv[0], str(script)], **kw)

    monkeypatch.setattr("app.cloud.providers.subprocess.Popen", popen)
    with pytest.raises(ProviderError) as exc:
        ClaudeCodeCLIProvider().ask("x", timeout_s=30)
    assert str(exc.value) == NOT_LOGGED_IN_HINT and "/login" in str(exc.value)


def test_double_send_returns_pending_request(turn, monkeypatch) -> None:
    _enable(monkeypatch)
    gate = threading.Event()

    asked = []

    class Slow(FakeProvider):
        def ask(self, prompt, *, timeout_s):
            asked.append(prompt)
            gate.wait(5)
            return ProviderReply(text="ok", model="slow")

    monkeypatch.setattr(so, "provider_factory", lambda n: Slow())
    pv = so.preview(turn["turn_id"])
    a = so.start(turn["turn_id"], pv["payload_sha256"])
    b = so.start(turn["turn_id"], pv["payload_sha256"])
    gate.set()
    import time

    time.sleep(0.5)
    assert b["id"] == a["id"] and b["replayed"] and len(asked) == 1
    assert len(so.list_for_turn(turn["turn_id"])) == 1
