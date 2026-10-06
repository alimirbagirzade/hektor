"""İkinci görüş sağlayıcıları. Anahtar TAŞIMAZ; ağ çağrısını yalnız seçilen sağlayıcı yapar.

- ``fake``           : test sağlayıcısı (ağ yok, deterministik).
- ``claude_code_cli``: resmi, değiştirilmemiş ``claude -p`` ikilisi; kullanıcının KENDİ abonelik
  oturumu. Hektor kimlik bilgisi toplamaz/saklamaz. Araçsız ve izole bayraklarla koşar
  (``app.orchestration.engines`` ile aynı sertleştirme: --safe-mode, --strict-mcp-config,
  --disallowedTools). Tur başına insan tıklamasıyla çağrılır; toplu döngü yoktur.

Her sağlayıcı iptal edilebilir: ``cancel()`` alt süreci sonlandırır; yarım cevap döndürülmez.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ProviderError(RuntimeError):
    """Sağlayıcı çağrısı tamamlanamadı (iptal, zaman aşımı, araç yok)."""


@dataclass
class ProviderReply:
    text: str
    model: str
    detail: str = ""


class Provider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def ask(self, prompt: str, *, timeout_s: float) -> ProviderReply: ...

    def cancel(self) -> None: ...


class FakeProvider:
    """Ağsız test sağlayıcısı: istemin özetine dayalı sabit bir eleştiri döndürür."""

    name = "fake"

    def __init__(self, reply: str | None = None, *, fail: str = "") -> None:
        self._reply = reply
        self._fail = fail
        self.cancelled = False
        self.prompts: list[str] = []

    def available(self) -> tuple[bool, str]:
        return True, "test sağlayıcısı (ağ çağrısı yok)"

    def ask(self, prompt: str, *, timeout_s: float) -> ProviderReply:
        self.prompts.append(prompt)
        if self.cancelled:
            raise ProviderError("iptal edildi")
        if self._fail:
            raise ProviderError(self._fail)
        text = self._reply or (
            "İKİNCİ GÖRÜŞ (sahte sağlayıcı): Yerel cevaptaki iddiaları kaynakla karşılaştırın; "
            "maliyet ve look-ahead varsayımlarını açıkça belirtin."
        )
        return ProviderReply(text=text, model="fake-1")

    def cancel(self) -> None:
        self.cancelled = True


def _tools_off() -> str:
    """engines.py av profilinin yasak listesi + dosya OKUMA araçları (ikinci görüş yalnız
    gönderilen metni görür; depo/veri dosyalarını okuyamaz)."""
    from app.orchestration.engines import DISALLOWED_TOOLS

    return ",".join((*DISALLOWED_TOOLS, "Read", "Grep", "Glob"))


class ClaudeCodeCLIProvider:
    """Resmi ``claude`` CLI (kullanıcının kendi aboneliği). Anahtar/oturum Hektor'da tutulmaz."""

    name = "claude_code_cli"

    def __init__(self, binary: str = "claude") -> None:
        self._binary = binary
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._cancelled = False

    def _resolve(self) -> tuple[str | None, str]:
        """Mutlak yol (çalışma dizinindeki taklitçiye düşmeden). ``.cmd/.bat`` sarmalayıcı
        REDDEDİLİR: Windows toplu dosyaya geçen argümanı cmd.exe yeniden yorumlar — istem
        metni komut enjeksiyonuna dönüşebilir (shell=False bunu engellemez)."""
        from app.orchestration.executable import resolve_cli

        path = resolve_cli(self._binary)
        if not path:
            return None, "`claude` CLI güvenilir PATH dizinlerinde yok — Claude Code kurulu olmalı."
        if Path(path).suffix.lower() in (".cmd", ".bat"):
            return None, f"`{path}` bir toplu dosya sarmalayıcı; güvenli argüman geçişi yok."
        return path, f"`claude` bulundu ({path}); giriş durumu ancak çağrıda anlaşılır."

    def available(self) -> tuple[bool, str]:
        path, detail = self._resolve()
        return path is not None, detail

    def ask(self, prompt: str, *, timeout_s: float) -> ProviderReply:
        path, detail = self._resolve()
        if path is None:
            raise ProviderError(detail)
        argv = [
            path,
            "-p",
            prompt,
            "--safe-mode",
            "--strict-mcp-config",
            "--disallowedTools",  # VARIADIC → en sonda, tek virgüllü argüman
            _tools_off(),
        ]
        # Boş geçici çalışma dizini: CLI'nin proje bağlamı (CLAUDE.md vb.) okuyup isteme
        # eklememesi için — gönderilen yalnız önizlemede gösterilen metindir.
        with tempfile.TemporaryDirectory(prefix="hektor_cloud_") as cwd:
            with self._lock:
                if self._cancelled:
                    raise ProviderError("iptal edildi")
                self._proc = subprocess.Popen(  # shell=False: istem tek argv öğesi
                    argv,
                    cwd=cwd,
                    env=_child_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            try:
                out, err = self._proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                self.cancel()
                self._proc.communicate()
                raise ProviderError(
                    f"zaman aşımı ({timeout_s:.0f} sn) — yarım cevap saklanmadı"
                ) from exc
        if self._cancelled:
            raise ProviderError("iptal edildi — yarım cevap saklanmadı")
        if self._proc.returncode != 0:
            raise ProviderError(f"claude çıkış {self._proc.returncode}: {(err or '')[-300:]}")
        return ProviderReply(text=(out or "").strip(), model="claude-code-cli (abonelik)")

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()


def make_provider(name: str) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "claude_code_cli":
        return ClaudeCodeCLIProvider()
    raise ProviderError(
        f"Desteklenmeyen sağlayıcı: '{name}'. API anahtarlı sağlayıcılar bu dalda uygulanmadı "
        "(karar kullanıcıda; docs/TASARIM_FAZ3_BULUT.md §4)."
    )


# Kademe 2 (c0d6aea avı) C-4: `claude -p` alt sürecine geçmeyecek değişkenler.
# ANTHROPIC_API_KEY varsa CLI aboneliği değil ÜCRETLİ API'yi kullanır (politika: API anahtarlı
# istemci yok); HEKTOR_API_TOKEN insan sırrıdır; ayar-ezme yolları --safe-mode'u delebilir
# (aynı sertleştirme: app/orchestration/driver.py build_child_env).
_STRIP_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "HEKTOR_API_TOKEN",
    "CLAUDE_CODE_MANAGED_SETTINGS_PATH",
    "CLAUDE_CODE_REMOTE_SETTINGS_PATH",
    "CLAUDE_CODE_MOCK_REMOTE_SETTINGS",
)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in _STRIP_ENV:
        env.pop(key, None)
    return env
