"""Doğrudan eğitici girişleri için zorunlu kapılar (Kademe 2 2026-10-11 F1-2).

`hektor train --run` / web / kolay akış kapıların TAMAMINI uygular. Ama `train()`'i doğrudan
çağıran araştırma betikleri (`scripts/train_v15_*.py --run`) hiçbir kapıdan geçmiyordu:
Kademe 2 kaydı, taze insan onayı (Kural 8), STOP_ALL ve ortak ağır iş kilidi yoktu. CLAUDE.md
"her eğitim yolu … ister" der. Bu modül o yollara asgari ortak kapıyı verir; sıra ucuzdan
pahalıya, onay EN SON tüketilir (önceki bir kapı düşerse onay yanmaz).

`peft_lora_train.py`'nin kendi `__main__ --run` yolu ise tamamen kapatıldı (bkz. orada):
yerine kapılı `hektor train --run` kullanılır.
"""

from __future__ import annotations

from pathlib import Path


def direct_run_blocker(description: str, *, data_path: Path | None = None) -> str | None:
    """Doğrudan eğitime engel varsa açıklaması (yoksa None — onay TÜKETİLMİŞTİR).

    ``data_path``: eğitilecek veri; Kademe 2 kaydı bu verinin özetini kapsamalı (yoksa
    kanonik `lora_sft.jsonl` özeti aranır).
    """
    from app.agents.runtime import supervisor
    from app.training import easy_train, resource_lock
    from app.training.detached_launch import is_running
    from app.training.unattended_policy import authorize_training_action

    if supervisor.is_stop_all_active():
        return "STOP_ALL aktif — gerçek eğitim bloklandı (`uv run hektor clear-stop-all`)."
    lock = resource_lock.blocker()
    if lock or is_running():
        return lock or "Zaten eğitim çalışıyor."
    data_sha: str | None = None
    if data_path is not None:
        from app.training.train_guard import sha256_file

        data_sha = sha256_file(data_path) or ""  # ham bayt özeti (kademe2-kayit --data-sha)
    k2 = easy_train.kademe2_check(data_sha=data_sha)
    if k2:
        return k2
    from app.orchestration import training_loop

    if data_sha is None:
        from app.config import get_settings
        from app.training.detached_launch import _combined_source
        from app.training.train_guard import sha256_file

        data_sha = sha256_file(_combined_source(get_settings())) or ""
    loop = training_loop.next_training_check(data_sha=data_sha)
    if loop:
        return "Eğitim döngüsü kapısı — " + " | ".join(loop)
    decision = authorize_training_action("train_run", description, agent_id="lora-trainer")
    if not decision.authorized:
        return (
            f"Gerçek eğitim TAZE insan onayı gerektirir. Onay isteği: {decision.approval_id} — "
            f"`uv run hektor approval-approve {decision.approval_id}`, sonra komutu tekrar "
            "çalıştır (Kural 8)."
        )
    return None
