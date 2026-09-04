"""Script spawn sertleştirmesi — doğurulan ajanlar araç-seviyesinde kısıtlı mı?

Achilles üç yerde ajan doğurur; ikisi bu testlerin konusu (üçüncüsü AutoDriver,
`tests/test_scope_isolation.py`):

- `scripts/weekly-bug-scan.ps1` — SALT RAPOR. Eskiden kısıt yalnız PROMPT'taydı
  ("DO NOT edit code") → prompt talimatı güvenlik sınırı DEĞİLDİR. Artık teknik.
- `scripts/rag-research-loop.ps1` — yazma/commit/push İŞLEVSEL ŞART, bu yüzden
  araç kısıtı UYGULANAMAZ. Yalnız bedava olan (MCP kanalı) kapatılır. Test bu
  bilinçli farkı SABİTLER ki ileride sessizce gevşemesin.

Çevrimdışı: script METNİ okunur, çalıştırılmaz (PowerShell/claude gerekmez).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCAN = _ROOT / "scripts" / "weekly-bug-scan.ps1"
_LOOP = _ROOT / "scripts" / "rag-research-loop.ps1"


def _read(path: Path) -> str:
    if not path.exists():  # pragma: no cover - script silinirse test anlamsızlaşır
        pytest.skip(f"script yok: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _spawn_lines(path: Path) -> str:
    """Yalnız `claude` ÇAĞRI satırları (yorumlar hariç).

    Açıklama yorumları bayrak adlarını anlatmak için içerdiğinden, tüm dosya metnine
    bakmak yanlış eşleşme üretir — kısıt iddiası ÇAĞRIDAN doğrulanmalı.
    """
    lines = [ln for ln in _read(path).splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(ln for ln in lines if "claude" in ln and " -p " in ln)


# ── weekly-bug-scan: salt-rapor → tam araç kısıtı ────────────────────────────


def test_bug_scan_spawn_is_tool_restricted() -> None:
    """Rapor-only tarama Bash/Write olmadan koşar → auth'suz CLI'yi çağıramaz."""
    spawn = _spawn_lines(_SCAN)
    assert "--disallowedTools" in spawn
    # Deny-list script'te değişkene atanır; adları orada doğrula.
    denied = next(ln for ln in _read(_SCAN).splitlines() if ln.strip().startswith("$denied"))
    for tool in ("Bash", "Edit", "Write", "Task"):
        assert tool in denied, f"{tool} deny-list'te değil"


def test_bug_scan_spawn_disables_customization_channels() -> None:
    """Hook/plugin/MCP araç katmanının DIŞINDA çalışır → safe-mode şart."""
    spawn = _spawn_lines(_SCAN)
    assert "--safe-mode" in spawn
    assert "--strict-mcp-config" in spawn


def test_bug_scan_does_not_rely_on_prompt_as_boundary() -> None:
    """Prompt artık sınır olarak KULLANILMAZ — kısıt argv'de teknik olarak durur.

    Eski hâlde tek güvence prompt'taki "DO NOT edit code" cümlesiydi; bugün ajanın
    Bash'i hiç yok. Prompt gövdesinde talimat kalmamalı (yorumda anlatılması serbest).
    """
    body = "\n".join(ln for ln in _read(_SCAN).splitlines() if not ln.lstrip().startswith("#"))
    assert "DO NOT edit code" not in body


def test_bug_scan_disallowed_tools_is_last_flag() -> None:
    """`--disallowedTools` VARIADIC → argv'nin sonunda durmalı.

    Sonrasına bayrak eklenirse variadic onu bir "araç adı" sanıp yutar ve o bayrak
    SESSİZCE etkisiz kalır. Kısıtın tamamı buna dayandığı için sıra sabitlenir.
    """
    spawn = _spawn_lines(_SCAN)
    assert spawn, "claude çağrı satırı bulunamadı"
    # `2>&1 | Out-String` gibi PS yönlendirmeleri argv'ye girmez; bayrak taramasını
    # yalnız `--` ile başlayan öğeler üzerinde yap.
    flags = [tok for tok in spawn.split() if tok.startswith("--")]
    assert flags[-1] == "--disallowedTools", f"son bayrak {flags[-1]!r} (variadic yutabilir)"


def test_bug_scan_computes_diff_itself() -> None:
    """Bash yasak → ajan git çalıştıramaz; diff'i script hesaplayıp prompt'a gömer."""
    text = _read(_SCAN)
    assert "git diff --stat" in text


# ── rag-research-loop: yazma ŞART → yalnız MCP kanalı kapanır ────────────────


def test_research_loop_closes_mcp_channel() -> None:
    """Bedava olan kapatılır: MCP proxy 127.0.0.1:8765'e ayrı kanal açıyordu."""
    assert "--strict-mcp-config" in _spawn_lines(_LOOP)


def test_research_loop_does_not_claim_tool_restriction() -> None:
    """BİLİNÇLİ FARK: bu ajan kuşatılamaz — sahte bir kısıt eklenmiş olmamalı.

    `--disallowedTools` eklenirse script'in işi (kod entegre + test + push) biter.
    Sınırı kapatabilirmiş gibi yapmak, dokümanın reddettiği overclaim'dir.
    """
    assert "--disallowedTools" not in _spawn_lines(_LOOP)


def test_research_loop_warns_on_bypass_permissions() -> None:
    """bypassPermissions sessiz olmamalı — ajan kendi eğitimini onaylayabilir."""
    text = _read(_LOOP)
    assert "bypassPermissions" in text
    assert "Write-Warning" in text


def test_research_loop_blanks_human_token() -> None:
    """İnsan API token'ı çocuğa sızmaz (tam sınır değil, hijyen)."""
    text = _read(_LOOP)
    assert 'ACHILLES_API_TOKEN = ""' in text
