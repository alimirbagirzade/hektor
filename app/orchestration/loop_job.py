"""Sürekli döngünün arka plan işi sarmalayıcısı: hedef modülü çalıştırır, ÇIKIŞ KODUNU yazar.

``python -m app.orchestration.loop_job <çıkış_dosyası> <modül> [argümanlar…]``

Döngü (Kademe 2 2026-10-15 CL-2) ölen bir işin neden öldüğünü bilmeliydi: 9 = ağır iş kilidi /
sohbet meşgul (deneme sayılmaz, sonra yeniden denenir); diğer sıfır dışı kodlar ve yarım çıktı
deneme sayılır, sınırda aday başarısız / döngü duraklatılır. Kod dosyası atomik yazılır.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import runpy
import sys
import traceback
from pathlib import Path


def _code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    return exc.code if isinstance(exc.code, int) else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("kullanım: loop_job <çıkış_dosyası> <modül> [argümanlar…]", file=sys.stderr)
        return 2
    exit_file, module, args = Path(argv[0]), argv[1], argv[2:]
    if not module.startswith("app."):
        print(f"yalnız app.* modülü çalıştırılır: {module}", file=sys.stderr)
        return 2
    sys.argv = [module, *args]
    code = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = _code(exc)
    except BaseException:
        traceback.print_exc()
        code = 1
    tmp = exit_file.with_name(exit_file.name + ".tmp")
    tmp.write_text(
        json.dumps({"exit_code": code, "at": dt.datetime.now(dt.UTC).isoformat()}),
        encoding="utf-8",
    )
    os.replace(tmp, exit_file)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
