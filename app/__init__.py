"""Achilles Trader AI — local-first trading research system."""

from __future__ import annotations

import contextlib
import sys

__version__ = "0.1.0"

# Windows konsolu varsayılan olarak cp1252/cp1254 kullanır; Rich/Türkçe çıktı
# (✓, kutu çizgileri, ş/ğ/ı) UnicodeEncodeError ile çöker ("charmap codec").
# Tüm entry point'ler (CLI + web) önce `app` paketini import ettiğinden,
# Rich Console oluşturulmadan stdout/stderr'i burada UTF-8'e sabitliyoruz.
# macOS/Linux zaten UTF-8 olduğu için yalnızca Windows'ta uygulanır.
if sys.platform == "win32":  # pragma: no cover - platforma özel
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                _reconfigure(encoding="utf-8", errors="replace")
