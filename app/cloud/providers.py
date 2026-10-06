"""İkinci görüş sağlayıcıları. Anahtar TAŞIMAZ; ağ çağrısını yalnız seçilen sağlayıcı yapar.

- ``fake``           : test sağlayıcısı (ağ yok, deterministik).
- ``claude_code_cli``: resmi, değiştirilmemiş ``claude -p`` ikilisi; kullanıcının KENDİ abonelik
  oturumu. Hektor kimlik bilgisi toplamaz/saklamaz. Araçsız ve izole bayraklarla koşar
  (``app.orchestration.engines`` ile aynı sertleştirme: --safe-mode, --strict-mcp-config,
  --disallowedTools). Tur başına insan tıklamasıyla çağrılır; toplu döngü yoktur.

Her sağlayıcı iptal edilebilir: ``cancel()`` alt süreci sonlandırır; yarım cevap döndürülmez.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
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

    def available(self) -> tuple[bool, str]:
        path = shutil.which(self._binary)
        if not path:
            return False, "`claude` CLI PATH'te yok — Claude Code kurulu ve girişli olmalı."
        return True, f"`claude` bulundu ({path}); giriş durumu ancak çağrıda anlaşılır."

    def ask(self, prompt: str, *, timeout_s: float) -> ProviderReply:
        argv = [
            self._binary,
            "-p",
            prompt,
            "--safe-mode",
            "--strict-mcp-config",
            "--disallowedTools",  # VARIADIC → en sonda, tek virgüllü argüman
            _tools_off(),
        ]
        with self._lock:
            if self._cancelled:
                raise ProviderError("iptal edildi")
            self._proc = subprocess.Popen(  # shell=False: istem tek argv öğesi
                argv,
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
