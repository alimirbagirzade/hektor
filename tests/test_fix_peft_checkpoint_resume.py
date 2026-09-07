"""PEFT LoRA: checkpoint'ten devam (resume) SESSİZ olmamalı — regresyon testleri.

Kademe-2 derin av bulgusu: ``train()`` çıktı klasöründeki son ``checkpoint-*``'tan
KOŞULSUZ devam ediyordu. Sonuçları:
(a) aynı adapter adıyla ikinci koşu eski adapter ağırlıklarını sessizce sürdürüyordu;
(b) eski ``global_step >= max_steps`` ise SIFIR adım eğitip ``ok=True`` dönüyordu —
    "eğitim yapıldı" denip hiçbir şey öğrenilmiyordu (CLAUDE.md Kural 2 ihlali).

Bu testler TAMAMEN ÇEVRİMDIŞI çalışır: gerçek model/tokenizer YÜKLENMEZ; ağır
bağımlılıklar (torch/peft/transformers) sahte modüllerle değiştirilir ve onlara
DOKUNULURSA test bilinçli olarak patlar (yani gerçek eğitim asla başlamaz).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.training.peft_lora_train import (
    PeftTrainConfig,
    build_command,
    dry_run,
    find_last_checkpoint,
    resolve_resume_checkpoint,
    resume_requested,
    train,
    zero_step_error,
)


class AgirBagimlilikHatasi(RuntimeError):
    """Test nöbetçisi: yükselirse ``train()`` ağır bağımlılıklara ULAŞMIŞ demektir."""


@pytest.fixture(autouse=True)
def _resume_env_temiz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ortamdaki HEKTOR_TRAIN_RESUME testleri kirletmesin."""
    monkeypatch.delenv("HEKTOR_TRAIN_RESUME", raising=False)


def _sahte_agir_modulleri_kur(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch/peft/transformers yerine 'dokunulursa patlayan' sahte modüller koy.

    ``_check_deps`` bunları import edebildiği için "eksik paket" yoluna sapılmaz; ama
    herhangi bir öznitelik erişimi (``torch.cuda``, ``from peft import ...``) hata verir
    → gerçek model indirme/yükleme MÜMKÜN DEĞİL.
    """
    for name in ("torch", "peft", "transformers"):
        mod = types.ModuleType(name)

        def _patla(attr: str, _n: str = name) -> None:
            raise AgirBagimlilikHatasi(f"{_n}.{attr} — train() ağır yola girdi")

        mod.__getattr__ = _patla  # type: ignore[method-assign]
        monkeypatch.setitem(sys.modules, name, mod)


def _checkpoint_yaz(output_dir: Path, step: int, *, state: bool = True) -> Path:
    """Sahte ``checkpoint-<step>`` klasörü (isteğe bağlı ``trainer_state.json``) üret."""
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"sahte")
    if state:
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
    return ckpt


def _cfg(output_dir: Path, **kw: object) -> PeftTrainConfig:
    return PeftTrainConfig(
        base_model="sahte/model-yok",
        train_jsonl=output_dir.parent / "train.jsonl",
        valid_jsonl=output_dir.parent / "valid.jsonl",
        adapter_output_path=output_dir,
        **kw,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------- find_last_checkpoint


def test_checkpoint_bulunamazsa_none_doner(tmp_path: Path) -> None:
    assert find_last_checkpoint(tmp_path) == (None, 0)
    assert find_last_checkpoint(tmp_path / "hic-yok") == (None, 0)


def test_en_yuksek_adimli_checkpoint_secilir(tmp_path: Path) -> None:
    _checkpoint_yaz(tmp_path, 25)
    _checkpoint_yaz(tmp_path, 300)
    (tmp_path / "checkpoint-bozuk").mkdir()  # regex'e uymayan klasör yok sayılır
    yol, adim = find_last_checkpoint(tmp_path)
    assert adim == 300
    assert yol is not None and yol.endswith("checkpoint-300")


def test_trainer_state_yoksa_klasor_adindan_adim_okunur(tmp_path: Path) -> None:
    _checkpoint_yaz(tmp_path, 120, state=False)
    assert find_last_checkpoint(tmp_path)[1] == 120


# ------------------------------------------------------- resolve_resume_checkpoint


def test_varsayilanda_mevcut_checkpoint_SESSIZCE_kullanilmaz(tmp_path: Path) -> None:
    """Asıl regresyon: devam KAPALI iken checkpoint kullanılmamalı ama uyarı verilmeli."""
    _checkpoint_yaz(tmp_path, 300)
    karar = resolve_resume_checkpoint(tmp_path, resume=False, max_steps=300)
    assert karar.checkpoint is None  # eski davranış: checkpoint-300'den devam ederdi
    assert karar.error is None
    assert karar.warnings and "checkpoint" in karar.warnings[0].lower()
    assert "resume_from_checkpoint=True" in karar.warnings[0]


def test_bos_klasorde_varsayilan_sessizdir(tmp_path: Path) -> None:
    karar = resolve_resume_checkpoint(tmp_path, resume=False, max_steps=300)
    assert karar == type(karar)()  # tamamen boş karar: checkpoint yok, uyarı yok


def test_devam_acikken_checkpoint_kullanilir(tmp_path: Path) -> None:
    _checkpoint_yaz(tmp_path, 100)
    karar = resolve_resume_checkpoint(tmp_path, resume=True, max_steps=300)
    assert karar.checkpoint is not None and karar.checkpoint.endswith("checkpoint-100")
    assert karar.last_step == 100
    assert karar.error is None
    assert karar.warnings and "SÜRDÜRÜLÜYOR" in karar.warnings[0]


def test_devam_acik_ama_checkpoint_yoksa_hata_degil_uyari(tmp_path: Path) -> None:
    karar = resolve_resume_checkpoint(tmp_path, resume=True, max_steps=300)
    assert karar.checkpoint is None
    assert karar.error is None
    assert karar.warnings and "sıfırdan" in karar.warnings[0]


@pytest.mark.parametrize("eski_adim", [300, 301])
def test_sifir_adim_durumu_HATA_dir(tmp_path: Path, eski_adim: int) -> None:
    """Eski adım hedefi karşılıyorsa: eğitim başlatılmaz, 'başarılı' DENMEZ (Kural 2)."""
    _checkpoint_yaz(tmp_path, eski_adim)
    karar = resolve_resume_checkpoint(tmp_path, resume=True, max_steps=300)
    assert karar.checkpoint is None
    assert karar.error is not None
    assert "SIFIR adım" in karar.error
    assert str(eski_adim) in karar.error


def test_hedef_bilinmiyorsa_sifir_adim_kiyasi_ertelenir(tmp_path: Path) -> None:
    """max_steps<=0 (iterations<=0) → hedef veriden gelecek; kıyas çağırana bırakılır."""
    _checkpoint_yaz(tmp_path, 300)
    karar = resolve_resume_checkpoint(tmp_path, resume=True, max_steps=0)
    assert karar.error is None
    assert karar.checkpoint is not None and karar.last_step == 300


def test_zero_step_error_metni_yol_ve_adimlari_icerir(tmp_path: Path) -> None:
    metin = zero_step_error(str(tmp_path / "checkpoint-300"), 300, 300)
    assert "checkpoint-300" in metin and "Kural 2" in metin


# ------------------------------------------------------------------ resume_requested


def test_resume_varsayilan_kapali(tmp_path: Path) -> None:
    assert resume_requested(_cfg(tmp_path)) is False
    assert dry_run(_cfg(tmp_path))["resume_from_checkpoint"] is False


def test_resume_config_alaniyla_acilir(tmp_path: Path) -> None:
    assert resume_requested(_cfg(tmp_path, resume_from_checkpoint=True)) is True


@pytest.mark.parametrize("deger", ["1", "true", "TRUE", "evet"])
def test_resume_ortam_degiskeniyle_acilir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deger: str
) -> None:
    monkeypatch.setenv("HEKTOR_TRAIN_RESUME", deger)
    assert resume_requested(_cfg(tmp_path)) is True


@pytest.mark.parametrize("deger", ["0", "false", "", "hayir"])
def test_resume_ortam_degiskeni_gecersiz_degerde_kapali(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deger: str
) -> None:
    monkeypatch.setenv("HEKTOR_TRAIN_RESUME", deger)
    assert resume_requested(_cfg(tmp_path)) is False


# ------------------------------------------------------------------- build_command


def test_build_command_varsayilanda_resume_gecmez(tmp_path: Path) -> None:
    assert "--resume" not in build_command(_cfg(tmp_path))


def test_build_command_acikken_resume_gecer(tmp_path: Path) -> None:
    assert "--resume" in build_command(_cfg(tmp_path, resume_from_checkpoint=True))


# --------------------------------------------------------------------------- train()


def test_train_sifir_adimda_ok_TRUE_DONMEZ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Senaryo: 300 adım tamamlanmış; aynı adapter adıyla yine 300 adım isteniyor.

    Eski davranış: checkpoint-300'den devam → 0 adım → ``ok=True`` (sessiz başarısızlık).
    Beklenen: eğitim HİÇ başlamaz, ``ok=False`` + açık hata (ağır modüllere bile
    dokunulmaz — nöbetçi hatası yükselmemeli).
    """
    _sahte_agir_modulleri_kur(monkeypatch)
    out = tmp_path / "hektor_lora"
    _checkpoint_yaz(out, 300)

    sonuc = train(_cfg(out, iterations=300, resume_from_checkpoint=True))

    assert sonuc["ok"] is False
    assert "SIFIR adım" in sonuc["error"]
    assert "checkpoint-300" in sonuc["error"]


def test_train_varsayilanda_checkpointi_yok_sayip_egitime_gider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Devam kapalıyken checkpoint-300 bloklamaz: sıfırdan eğitim yoluna devam edilir.

    Nöbetçi hatası, ``train()``'in resume kapısını GEÇİP gerçek eğitim hattına
    (torch/peft) ilerlediğini kanıtlar — gerçek yükleme yapılmadan.
    """
    _sahte_agir_modulleri_kur(monkeypatch)
    out = tmp_path / "hektor_lora"
    _checkpoint_yaz(out, 300)

    with pytest.raises(AgirBagimlilikHatasi):
        train(_cfg(out, iterations=300))
