"""Alt süreç bayrakları — Windows'ta konsol penceresi AÇILMASIN.

Neden (2026-10-06 olayı): konsolu olmayan bir süreç (``DETACHED_PROCESS`` ile başlatılmış
aday-işi çalıştırıcısı) bayraksız bir konsol programı (``python``, ``git``, ``powershell``)
başlatınca Windows o program için YENİ ve GÖRÜNÜR bir konsol penceresi açar. Paralel stres
testinde bu yüzlerce pencere açtı ve masaüstü kilitlendi.

Kurallar:
- Çıktısı yakalanan / dosyaya yönlenen her alt süreç ``creationflags=NO_WINDOW`` alır.
- Ayrık (arka plan) süreçler ``DETACHED_HIDDEN`` ile başlatılır: ``DETACHED_PROCESS``
  YERİNE ``CREATE_NO_WINDOW`` — süreç GİZLİ bir konsol edinir, torunları (betik içindeki
  ``git``/``ollama``/``llama.cpp`` dahil) bu gizli konsolu miras alır, pencere açamaz.
  ``DETACHED_PROCESS`` ile birlikte verilen ``CREATE_NO_WINDOW`` Windows'ta YOK SAYILIR.

POSIX'te bayraklar 0'dır (``Popen`` orada sıfır dışı ``creationflags`` kabul etmez).
``tests/test_no_console_windows.py`` bu kuralı ``app/`` genelinde zorlar.
"""

from __future__ import annotations

import os

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_NT = os.name == "nt"

#: Çıktısı yakalanan kısa ömürlü alt süreçler için.
NO_WINDOW: int = CREATE_NO_WINDOW if _NT else 0

#: Ebeveynden bağımsız, penceresiz arka plan süreçleri için (``DETACHED_PROCESS`` KULLANMA).
DETACHED_HIDDEN: int = (CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW) if _NT else 0
