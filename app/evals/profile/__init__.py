"""Profil eval altyapısı — LoRA karışım profillerinin güvenilir, deterministik değerlendirmesi.

Mevcut ``app.evals`` modülleri (discipline/lora-eval, retrieval, hipotez) DEĞİŞMEZ; bu alt
paket karışım profilleri için A/B/C/D karşılaştırması, sızıntı koruması, regression kapısı
ve insan denetimi ekler. Ayrıntı: docs/LORA_MIX_EVAL.md.
"""
