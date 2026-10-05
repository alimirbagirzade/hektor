"""Sohbetten öğrenme (Faz 1) veritabanı değişikliğini GERİ AL — yalnız yeni tablolar.

Varsayılan KURU koşudur: hangi tabloların silineceğini ve satır sayılarını yazar, hiçbir şey
değiştirmez. ``--apply`` verilirse önce SQLite dosyasının yedeği alınır
(``<db>.bak-chat-<zaman>``), sonra YALNIZ ``app.feedback.chat_store`` tabloları silinir.
Echo (`feedback_corrections`) ve diğer tüm tablolara dokunulmaz. Veri sürümü klasörleri
(``data/learning/chat_datasets``) ve ``data/lora_sft/chat_selection.json`` SİLİNMEZ; elle
kaldırılabilir (eğitim verisini değiştirdiği için bilinçli olarak otomatik değildir).

Kullanım:
    uv run python scripts/chat_learning_rollback.py           # rapor
    uv run python scripts/chat_learning_rollback.py --apply   # yedekle + sil
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sqlite3

from app.config.settings import get_settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="Yedek al ve tabloları sil")
    args = ap.parse_args()

    from app.feedback.chat_store import CHAT_TABLES, drop_chat_tables

    db = get_settings().sqlite_file
    names = [t.name for t in CHAT_TABLES]
    if not db.exists():
        raise SystemExit(f"Veritabanı yok: {db}")
    with sqlite3.connect(db) as conn:
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for n in names:
            if n in present:
                count = conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                print(f"  {n}: {count} satır")
            else:
                print(f"  {n}: yok")
    if not args.apply:
        print("Kuru koşu — değişiklik yapılmadı. Silmek için --apply.")
        return
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = db.with_name(f"{db.name}.bak-chat-{stamp}")
    shutil.copy2(db, backup)
    print(f"Yedek: {backup}")
    dropped = drop_chat_tables(db)
    print("Silinen tablolar: " + ", ".join(dropped))


if __name__ == "__main__":
    main()
