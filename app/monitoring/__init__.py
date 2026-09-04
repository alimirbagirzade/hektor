"""app/monitoring — Sentinel (Nöbetçi): tüm ajan/altsistemleri izleyen sağlık monitörü.

"Birbirini denetleyen sistem" katmanının kapanış taşı (Layer 8 — Monitor & Alert):
Ollama, web, eğitim, orkestrasyon, STOP_ALL, disk, SQLite, feedback ve CPU-çekişmesini
(danışman Resource Negotiator) SALT-OKUMA yoklar; geçmişi SQLite'a yazar. Hiçbir şeyi
durdurmaz/başlatmaz/mutasyona uğratmaz — yalnız raporlar (öneri metniyle).
"""

from __future__ import annotations

from app.monitoring.sentinel import ProbeResult, Sentinel, SentinelReport
from app.monitoring.store import MonitoringStore

__all__ = ["MonitoringStore", "ProbeResult", "Sentinel", "SentinelReport"]
