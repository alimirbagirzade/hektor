"""Regresyon testleri: detached eğitim başlatıcısında adım hesabı + kaynak-gruplu bölme.

İki Kademe-2 bulgusunu kilitler (ikisi de `app/training/detached_launch.py`):

1. Adım sayısı TAM `train.jsonl` satır sayısından hesaplanıyordu; profil (`discipline_safe_local`)
   eğitim setini `max_examples` ile kırptığı için kırpılmış alt-küme üzerinde sessizce birkaç
   epoch koşuluyor, profilin `epochs: 1` vaadi ihlal ediliyordu (ezber/aşırı-uyum riski).
2. `ensure_train_split` satır-düzeyinde karıştırıyordu → aynı makalenin sentetik QA'ları hem
   train hem valid'e düşüyor, valid metrikleri sızıntı yüzünden iyimser görünüyordu.

Testler ÇEVRİMDIŞI: sentetik JSONL + geçici profil dosyası; eğitim BAŞLATILMAZ (Kural 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.training import detached_launch as dl

# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------


def _row(paper_id: str | None, i: int) -> str:
    """Tek bir SFT satırı (gerçek `lora_sft.jsonl` şemasıyla aynı: messages + metadata)."""
    meta: dict = {"synthetic": True}
    if paper_id is not None:
        # Gerçek veride source_id == paper_id; ikisini de yaz (öncelik sözleşmesi test edilir).
        meta |= {"paper_id": paper_id, "source_id": paper_id, "chunk_id": f"{paper_id}_c{i:04d}"}
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"soru {paper_id}-{i}"},
                {"role": "assistant", "content": f"cevap {paper_id}-{i}"},
            ],
            "metadata": meta,
        },
        ensure_ascii=False,
    )


def _corpus(n_papers: int = 20, per_paper: int = 10, n_orphan: int = 30) -> list[str]:
    """Sentetik korpus: makaleye bağlı satırlar + kaynak kimliği OLMAYAN satırlar."""
    lines = [_row(f"paper_{p:04d}", i) for p in range(n_papers) for i in range(per_paper)]
    lines += [_row(None, i) for i in range(n_orphan)]
    return lines


def _paper_ids(lines: list[str]) -> set[str]:
    out: set[str] = set()
    for ln in lines:
        pid = (json.loads(ln).get("metadata") or {}).get("paper_id")
        if pid:
            out.add(pid)
    return out


# --------------------------------------------------------------------------
# BULGU 1 — adım sayısı kırpılmış (fiilen eğitilen) örnek sayısından hesaplanmalı
# --------------------------------------------------------------------------


@pytest.fixture
def profil_dosyasi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Gerçek `discipline_safe_local` reçetesinin ilgili alanlarını taşıyan geçici profil."""
    path = tmp_path / "lora_profiles.yaml"
    path.write_text(
        "kirpan_profil:\n"
        "  r: 16\n"
        "  epochs: 1\n"
        "  max_examples: 300\n"
        "kirpan_iki_epoch:\n"
        "  r: 16\n"
        "  epochs: 2\n"
        "  max_examples: 300\n"
        "kirpmayan_profil:\n"
        "  r: 16\n"
        "  epochs: 1\n",
        encoding="utf-8",
    )

    from app.training import peft_lora_train

    gercek = peft_lora_train.load_lora_profile
    monkeypatch.setattr(
        peft_lora_train,
        "load_lora_profile",
        lambda name, profiles_path=None: gercek(name, profiles_path or path),
    )
    return path


def test_adim_sayisi_profil_kirpmasina_gore_hesaplanir(profil_dosyasi: Path) -> None:
    """1099 satırlık train + `max_examples: 300` → 1099 değil 300 adım (tam 1 epoch)."""
    iters, n_effective, epochs = dl.plan_iterations(1099, 0, "kirpan_profil")
    assert n_effective == 300
    assert epochs == 1
    assert iters == 300, "Adım sayısı kırpılmamış satır sayısından hesaplanıyor (BULGU 1 nüksü)"


def test_profil_epochs_degeri_karsilanir(profil_dosyasi: Path) -> None:
    """`epochs: 2` → adım sayısı kırpılmış küme üzerinde TAM 2 epoch olmalı."""
    iters, n_effective, epochs = dl.plan_iterations(1099, 0, "kirpan_iki_epoch")
    assert (n_effective, epochs, iters) == (300, 2, 600)


def test_cli_max_examples_profili_ezer(profil_dosyasi: Path) -> None:
    """CLI `--max-examples` profili ezer (app/main.py ile aynı öncelik) → adım da ona uyar."""
    assert dl.plan_iterations(1099, 50, "kirpan_profil") == (50, 50, 1)


def test_kirpma_yoksa_tum_train_kullanilir(profil_dosyasi: Path) -> None:
    """Profilde `max_examples` yoksa davranış eskisi gibi: 1 epoch = tüm train."""
    assert dl.plan_iterations(1099, 0, "kirpmayan_profil") == (1099, 1099, 1)


def test_profil_yoksa_kirpmasiz_tek_epoch() -> None:
    """Profil geçilmezse (veya bilinmezse) kırpma bilinmiyor → tüm train, 1 epoch."""
    assert dl.plan_iterations(500, 0, None) == (500, 500, 1)
    assert dl.plan_iterations(500, 0, "boyle_bir_profil_yok") == (500, 500, 1)


def test_kirpma_train_boyutundan_buyukse_train_boyutu_kullanilir(profil_dosyasi: Path) -> None:
    """`max_examples` mevcut veriden büyükse fiilen eğitilen sayı = train boyutu."""
    assert dl.plan_iterations(120, 0, "kirpan_profil") == (120, 120, 1)


def test_bos_train_en_az_bir_adim(profil_dosyasi: Path) -> None:
    """Sıfır örnek adım sayısını 0'a düşürüp trainer'ı bozmasın (en az 1 adım)."""
    iters, n_effective, _ = dl.plan_iterations(0, 0, "kirpan_profil")
    assert (iters, n_effective) == (1, 0)


# --------------------------------------------------------------------------
# BULGU 2 — kaynak-gruplu bölme (aynı paper_id tek tarafta) + determinizm
# --------------------------------------------------------------------------


def test_ayni_paper_id_train_ve_valid_arasinda_bolunmez() -> None:
    lines = _corpus()
    train, valid = dl.split_lines_by_source(lines)
    sizinti = _paper_ids(train) & _paper_ids(valid)
    assert not sizinti, f"Kaynak sızıntısı: {sorted(sizinti)} hem train hem valid'de"


def test_bolme_tum_satirlari_korur_ve_ikisi_de_dolu() -> None:
    lines = _corpus()
    train, valid = dl.split_lines_by_source(lines)
    assert sorted(train + valid) == sorted(lines)
    assert train and valid


def test_ayni_seed_ayni_bolmeyi_verir() -> None:
    """Kural 6: determinizm — aynı seed, aynı girdi → birebir aynı bölme."""
    lines = _corpus()
    assert dl.split_lines_by_source(lines, seed=42) == dl.split_lines_by_source(lines, seed=42)


def test_farkli_seed_farkli_bolme_verir() -> None:
    """Seed gerçekten etkili (sabit/görmezden gelinen seed regresyonunu yakalar)."""
    lines = _corpus()
    _, valid_a = dl.split_lines_by_source(lines, seed=42)
    _, valid_b = dl.split_lines_by_source(lines, seed=7)
    assert valid_a != valid_b


def test_girdi_sirasi_bolmeyi_degistirmez() -> None:
    """Gruplar kanonik (sorted) sıraya konduğu için satır sırası sonucu etkilemez."""
    lines = _corpus()
    a = dl.split_lines_by_source(lines, seed=42)
    b = dl.split_lines_by_source(list(reversed(lines)), seed=42)
    assert sorted(a[1]) == sorted(b[1])


def test_valid_orani_makul_kalir() -> None:
    """Grup bütünlüğü hedefi aşabilir ama valid küçük bir pay olarak kalmalı."""
    lines = _corpus()
    _train, valid = dl.split_lines_by_source(lines)
    hedef = int(len(lines) * dl._VALID_RATIO)
    assert hedef <= len(valid) <= hedef + 10  # aşım en çok bir grup (per_paper=10) kadar


def test_kaynaksiz_satirlar_tek_blok_olmaz() -> None:
    """Kaynak kimliği olmayan satırlar tek gruba yığılıp valid'i şişirmemeli."""
    lines = [_row(None, i) for i in range(200)]
    _train, valid = dl.split_lines_by_source(lines)
    assert 1 <= len(valid) <= 15  # ~%5 civarı; tek blok olsaydı 200 olurdu


def test_kaynaksiz_ayni_satir_ayni_tarafta_kalir() -> None:
    """Birebir aynı (kopya) satırlar içerik-hash'i aynı → aynı bölmede kalır."""
    kopya = _row(None, 0)
    lines = [kopya] * 4 + [_row(f"paper_{p:04d}", i) for p in range(10) for i in range(10)]
    train, valid = dl.split_lines_by_source(lines)
    assert train.count(kopya) in (0, 4)
    assert valid.count(kopya) in (0, 4)


def test_bozuk_satir_bolmeyi_cokertmez() -> None:
    """JSON olarak parse edilemeyen satır sessizce içerik-hash grubuna düşer."""
    lines = ["{bozuk json", *(_row(f"paper_{p:04d}", i) for p in range(10) for i in range(10))]
    train, valid = dl.split_lines_by_source(lines)
    assert sorted(train + valid) == sorted(lines)


def test_tek_grup_train_i_bosaltmaz() -> None:
    """Tüm satırlar tek makaleden geliyorsa valid boş kalır ama train ASLA boşalmaz."""
    lines = [_row("paper_tek", i) for i in range(20)]
    train, valid = dl.split_lines_by_source(lines)
    assert len(train) == 20
    assert valid == []


def test_bos_girdi() -> None:
    assert dl.split_lines_by_source([]) == ([], [])


# --------------------------------------------------------------------------
# Uçtan uca: ensure_train_split dosyaları kaynak-gruplu yazar
# --------------------------------------------------------------------------


def test_ensure_train_split_dosyalari_kaynak_gruplu_yazar(tmp_path: Path) -> None:
    src = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True)
    lines = _corpus()
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    settings = SimpleNamespace(root=tmp_path, jsonl_dir=tmp_path / "jsonl")
    n_train, n_valid = dl.ensure_train_split(settings)
    assert n_train + n_valid == len(lines)
    assert n_valid > 0

    train = [
        ln
        for ln in (tmp_path / "jsonl" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    valid = [
        ln
        for ln in (tmp_path / "jsonl" / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert (len(train), len(valid)) == (n_train, n_valid)
    assert not (_paper_ids(train) & _paper_ids(valid))
