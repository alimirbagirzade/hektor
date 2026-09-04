#!/usr/bin/env python3
"""Korumalı yol bekçisi (Phase 4A) — GitHub automation CI guard'ı.

Bir değişiklik kümesindeki dosyalar KORUMALI yollara (data/, storage/, vector_db/,
models/, .env*, anahtar/db dosyaları) dokunuyorsa exit 2 ile durdurur. Claude Code
workflow'unda Claude'dan ÖNCE ve SONRA çalıştırılır → tehlikeli/sızıntı içeren
değişiklik PR'a giremez.

Saf eşleştirme mantığı (``is_protected`` / ``protected_changes``) testlerden
git'siz çağrılabilir; ``main`` ise CI'da ``git diff`` üzerinden çalışır.

Kullanım:
  python scripts/check_protected_paths.py --base origin/main
  python scripts/check_protected_paths.py --files app/x.py data/y.pdf   # elle/test
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys

# Dizin önekleri — bu ağaçların ALTINDAKİ her şey korumalı (çıktı/runtime/model/secret).
PROTECTED_DIRS: tuple[str, ...] = ("data/", "storage/", "vector_db/", "models/")
# Dosya-adı/yol glob'ları — secret ve veritabanı sızıntısı savunması.
PROTECTED_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.sqlite",
    "*.db",
)


def is_protected(path: str) -> bool:
    """Bu yol korumalı mı? (platformdan bağımsız; ileri-eğik bölü normalize edilir.)"""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p:
        return False
    for d in PROTECTED_DIRS:
        if p == d.rstrip("/") or p.startswith(d):
            return True
    base = p.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(base, g) or fnmatch.fnmatch(p, g) for g in PROTECTED_GLOBS)


def protected_changes(paths: list[str]) -> list[str]:
    """Verilen yollardan korumalı olanları (sıra korunur) döndür."""
    return [p for p in paths if is_protected(p)]


def _git_changed(base: str) -> list[str]:
    """base...HEAD üç-nokta diff: ortak atadan beri HEAD'de değişen dosyalar."""
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Korumalı yol bekçisi (CI guard).")
    ap.add_argument("--base", default="origin/main", help="karşılaştırma tabanı (git ref)")
    ap.add_argument("--files", nargs="*", help="elle dosya listesi (git yerine; test için)")
    args = ap.parse_args(argv)

    try:
        paths = args.files if args.files is not None else _git_changed(args.base)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"git diff başarısız: {exc}", file=sys.stderr)
        return 1

    hits = protected_changes(paths)
    if hits:
        print("KORUMALI YOL İHLALİ — aşağıdaki değişiklikler reddedildi:", file=sys.stderr)
        for h in hits:
            print(f"  - {h}", file=sys.stderr)
        print(
            "Bu yollar (data/storage/vector_db/models/.env/anahtar/db) otomasyonca "
            "DEĞİŞTİRİLEMEZ. PR açılmamalı.",
            file=sys.stderr,
        )
        return 2

    print(f"Korumalı yol ihlali yok ({len(paths)} dosya kontrol edildi).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
