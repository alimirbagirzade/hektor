"""app/feedback — Echo: kullanıcı düzeltmeleri → sentetik SFT adayı (feedback döngüsü).

Kullanıcı bir cevabın yanlış olduğunu bildirir → Echo düzeltmeyi kaydeder, Kural-1
güvenlik kontrolünden geçirir, onaylananları lora_sft FORMATINDA AYRI bir aday dosyaya
export eder. **Eğitim ASLA tetiklenmez** (Kural 8); export edilen aday yine pretrain-gate
+ dataset audit'ten geçmelidir (lora_sft.jsonl'e oto-merge YOK).
"""

from __future__ import annotations

from app.feedback.echo import VALID_CORRECTION_TYPES, EchoCollector
from app.feedback.store import FeedbackStore

__all__ = ["VALID_CORRECTION_TYPES", "EchoCollector", "FeedbackStore"]
