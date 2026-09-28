"""Kademe 2 bug-avı (2026-09-28) — lora-audit ↔ pretrain-gate bağı ve olumsuzlama düzeltmeleri.

B1: pretrain-gate eğitilen dosyanın şu anki DB'den kanonik kurulumla aynı olduğunu (tazelik)
    doğrular ve HER satırın assistant cevabını lora-audit Gate 7 + Gate 5 tarayıcılarından
    geçirir (sentetik QA + disiplin satırları eskiden hiçbir kart kapısından geçmiyordu).
B3: kapı ayrıntıları 20'de kesilmez; rapor tam listeyi yazar.
B4: olumlu deyimler ("hiç şüphesiz", "no doubt", "can't lose") olumsuzlama değildir; Gate 7'ye
    yeni garanti vaadi kalıpları; Gate 6 "doğrudan uygulanamaz" tavsiye değildir.
B6: Gate 7 / Gate 5 çıplak alt-dize yanlış-pozitifleri (fail-closed) daraltıldı.
Tümü çevrimdışı; DB/LLM yok.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.lora.control_plane import LoRAControlPlane, PipelineReport
from app.lora.gates import gate_5_math, gate_6_philosophy, gate_7_safety
from app.lora.math_verifier import verify_math_content
from app.lora.negation import is_negated
from app.lora.safety_scanner import scan_for_secrets, tr_fold
from app.training import sft_assembly
from app.training.dataset_quality import audit_dataset
from app.training.sft_assembly import AssemblyResult, check_assembly_freshness


def _card(cid: str, text: str) -> dict:
    return {
        "card_id": cid,
        "paper_id": "p1",
        "review_status": "approved",
        "created_at": "2026-01-01",
        "difficulty": 0.3,
        "card_json": {"title": "Momentum", "summary": text},
    }


def _line(answer: str, **meta: object) -> str:
    obj: dict = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "soru"},
            {"role": "assistant", "content": answer},
        ]
    }
    if meta:
        obj["metadata"] = meta
    return json.dumps(obj, ensure_ascii=False)


def _negated(text: str, phrase: str) -> bool:
    folded = tr_fold(text)
    i = folded.index(phrase)
    return is_negated(folded, i, i + len(phrase))


# --------------------------------------------------------------------------- #
# B4 — olumlu deyimler olumsuzlama DEĞİL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Hiç şüphesiz garanti kâr sağlar.",
        "Hiç kuşkusuz garanti kâr sağlar.",
        "No doubt guaranteed profit.",
        "Without doubt guaranteed profit.",
        "Without a doubt guaranteed profit.",
        "Without question guaranteed profit.",
        "You can't lose - guaranteed profit.",
        "You cannot lose - guaranteed profit.",
        "It never fails: guaranteed profit.",
        "Not only guaranteed profit but more.",
        "Nothing but guaranteed profit.",
    ],
)
def test_affirmative_idiom_is_not_negation(text: str) -> None:
    assert not _negated(text, "garanti kar" if "garanti" in tr_fold(text) else "guaranteed")
    assert scan_for_secrets(text).passed is False, text  # Gate 7 BLOCKER yakalar
    assert verify_math_content(text).passed is False, text  # Gate 5 BLOCKER yakalar


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("There is no guaranteed profit in trading.", "guaranteed"),
        ("Do not buy now based on this signal.", "buy now"),
        ("Don't buy now based on this signal.", "buy now"),
        ("Profits are not necessarily guaranteed.", "guaranteed"),
        ("Getiri garanti değildir.", "garanti"),
    ],
)
def test_real_negation_still_recognized(text: str, phrase: str) -> None:
    assert _negated(text, phrase)


# --------------------------------------------------------------------------- #
# B4 — Gate 7 yeni garanti vaadi kalıpları
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Bu sistem garanti kazanç sağlar.",
        "BU SİSTEM GARANTİ KAZANÇ SAĞLAR.",
        "Garantili getiri elde edersiniz.",
        "Garanti getirisi yüksektir.",
        "The fund offers guaranteed returns.",
        "This setup guarantees profits every month.",
        "The strategy guarantees profit.",
        "Pure risk-free profit from this spread.",
        "A riskfree profit every day.",
        "Hiçbir risk yok, pozisyon aç.",
        "No risk at all with this trade.",
    ],
)
def test_gate7_new_guarantee_directives_block(text: str) -> None:
    result = scan_for_secrets(text)
    assert result.passed is False, text
    assert any("finansal" in v for v in result.violations)


# --------------------------------------------------------------------------- #
# B6 — Gate 7 / Gate 5 yanlış-pozitifleri
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "There is no guaranteed profit in trading.",
        "Assumes no risk-free arbitrage.",
        "The model has no risk premium for small caps.",
        "Firms with no risk management suffer larger drawdowns.",
        "There is no risk of look-ahead bias because positions are lagged.",
        "Do not buy now based on this signal.",
        "Banka garanti karşılığı kredi verir.",
        "Kâr garantisi yoktur; test edilmeli.",
    ],
)
def test_gate7_cautious_language_passes(text: str) -> None:
    assert scan_for_secrets(text).passed is True, scan_for_secrets(text).violations


@pytest.mark.parametrize(
    "text",
    [
        "The estimator is guaranteed to converge.",
        "The algorithm is guaranteed to return the optimal solution.",
        "Yakınsama garantisi sağlar.",
        "Mutlak toplamlanabilir dizi kovaryans-durağanlık garantisi sağlar.",
        "Assumes no risk-free arbitrage.",
        "The model has no risk premium for small caps.",
    ],
)
def test_gate5_math_guarantee_language_not_blocker(text: str) -> None:
    result = verify_math_content(text)
    assert result.passed is True, result.issues
    assert not any("aşırı emin" in i for i in result.issues)


@pytest.mark.parametrize(
    "text",
    [
        "Garanti kâr sağlar.",
        "Kâr garantilidir.",
        "This delivers guaranteed profit.",
        "Returns are guaranteed.",
        "You will never lose, guaranteed.",
        "It is not only guaranteed but risk-free.",
    ],
)
def test_gate5_financial_guarantee_still_blocks(text: str) -> None:
    assert verify_math_content(text).passed is False, text


# --------------------------------------------------------------------------- #
# B4 — Gate 6 "doğrudan uygulanamaz" tavsiye DEĞİL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Bu yöntem canlı piyasaya doğrudan uygulanamaz.",
        "Bu sonuç doğrudan uygulanmaz.",
        "Doğrudan uygulanamayan bir çerçevedir.",
        "Model doğrudan uygulanmamalı; önce test edilmeli.",
        "Bulgular doğrudan uygulanamıyor.",
    ],
)
def test_gate6_turkish_negative_verb_not_advice(text: str) -> None:
    assert gate_6_philosophy([_card("c1", text)]).review_count == 0


@pytest.mark.parametrize(
    "text",
    [
        "Bu yöntem doğrudan uygulanabilir.",
        "Modelin doğrudan uygulanması önerilir.",
    ],
)
def test_gate6_turkish_positive_still_flagged(text: str) -> None:
    assert gate_6_philosophy([_card("c1", text)]).review_count == 1


# --------------------------------------------------------------------------- #
# B3 — ayrıntılar kesilmez, rapor tam liste yazar
# --------------------------------------------------------------------------- #


def test_gate_details_not_truncated_and_report_lists_all(tmp_path) -> None:
    cards = [_card(f"c{i:02d}", "Traders should buy the breakout.") for i in range(25)]
    gate6 = gate_6_philosophy(cards, min_cards_for_block=1000)
    assert gate6.review_count == 25
    assert len(gate6.details) == 25  # eskiden details[:20]

    unsafe = [_card(f"s{i:02d}", "Buy now!") for i in range(23)]
    gate7 = gate_7_safety(unsafe)
    assert gate7.rejected_count == 23 and len(gate7.details) == 23

    plane = LoRAControlPlane(store=SimpleNamespace())  # type: ignore[arg-type]
    report = PipelineReport(stages=[gate6, gate7], total_review_needed=25)
    md = plane.generate_report(report, output_path=tmp_path / "r.md")
    assert "c24" in md and "s22" in md
    assert "İnceleme gereken: 25" in md


def test_gate5_details_not_truncated() -> None:
    cards = [_card(f"c{i:02d}", f"Look-ahead bias case {i}.") for i in range(22)]
    assert len(gate_5_math(cards).details) == 22


# --------------------------------------------------------------------------- #
# B1 — pretrain-gate satır taraması (Gate 7 + Gate 5)
# --------------------------------------------------------------------------- #


def _base_lines(n: int = 30) -> list[str]:
    return [
        _line(f"Cevap {i}: momentum hipotezi {i} için ayrı bir açıklama {i * 7}.") for i in range(n)
    ]


def test_pretrain_gate_blocks_gate7_hit_in_synthetic_line() -> None:
    lines = [*_base_lines(), _line("Sinyal geldi, buy now ve pozisyonu büyüt.")]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.gate7_line_hits == 1
    assert any("Gate 7" in b and "satır 31" in b for b in rep.blockers)


def test_pretrain_gate_blocks_unnegated_gate5_guarantee() -> None:
    lines = [*_base_lines(), _line("Hiç şüphesiz garantili getiri sağlar.")]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.guarantee_line_hits == 1


def test_pretrain_gate_negated_guarantee_and_certainty_are_not_blockers() -> None:
    lines = [
        *_base_lines(),
        _line("Piyasada garanti kâr yoktur; her hipotez maliyetlerle test edilmeli."),
        _line("There is no guaranteed profit in trading."),
        _line("Bu kesinlikle bir hipotezdir, sinyal değildir."),
    ]
    rep = audit_dataset(lines)
    assert rep.gate7_line_hits == 0
    assert rep.guarantee_line_hits == 0
    assert rep.certainty_line_hits == 1
    assert any("kesinlik dili" in w for w in rep.warnings)
    assert not any("Gate 7" in b or "Gate 5" in b for b in rep.blockers)


# --------------------------------------------------------------------------- #
# B1 — tazelik bağı (dosya ↔ şu anki DB kanonik birleştirmesi)
# --------------------------------------------------------------------------- #

_CARD_LINE = _line("Kart cevabı.", card_id="card_1", paper_id="p1")
_SYNTH = [_line(f"Sentetik {i}.", paper_id="p9") for i in range(3)]


def _assembler(*outputs: list[str], card_n: int = 1):
    calls = iter(outputs)
    seen: list[dict] = []

    def _build(settings, **kw) -> AssemblyResult:
        seen.append(kw)
        lines = next(calls)
        return AssemblyResult(lines=lines, synth_n=3, card_n=card_n, deduped=len(lines))

    return _build, seen


def test_freshness_identical_file_is_fresh_and_uses_canonical_params() -> None:
    expected = [*_SYNTH, _CARD_LINE]
    build, seen = _assembler(expected, expected)
    res = check_assembly_freshness(list(expected), SimpleNamespace(), assemble=build)
    assert res.fresh and res.deterministic and not res.blockers
    assert seen == [{"discipline": True, "discipline_ratio": 0.25, "seed": 0}] * 2


def test_freshness_stale_card_set_blocks() -> None:
    """Dosya, kartı sonradan reddedilmiş eski DB'den kurulmuş → NO-GO."""
    expected = list(_SYNTH)  # güncel DB'de kart yok (reddedildi)
    build, _ = _assembler(expected, expected, card_n=0)
    res = check_assembly_freshness([*_SYNTH, _CARD_LINE], SimpleNamespace(), assemble=build)
    assert res.status == "BAYAT"
    assert res.card_extra == 1 and res.extra == 1
    assert "assemble_sft.py" in res.blockers[0]


def test_freshness_order_only_difference_is_fresh_with_warning() -> None:
    expected = [*_SYNTH, _CARD_LINE]
    build, _ = _assembler(expected, expected)
    res = check_assembly_freshness(list(reversed(expected)), SimpleNamespace(), assemble=build)
    assert res.fresh and res.order_only and res.warnings


def test_freshness_nondeterministic_compares_card_subset_only() -> None:
    a = [*_SYNTH, _CARD_LINE]
    b = [*reversed(_SYNTH), _CARD_LINE, _line("başka disiplin")]
    build, _ = _assembler(a, b)
    res = check_assembly_freshness([*_SYNTH[:1], _CARD_LINE], SimpleNamespace(), assemble=build)
    assert not res.deterministic and res.scope == "kart satırları"
    assert res.fresh  # kart satırları aynı
    assert any("determinist değil" in w for w in res.warnings)

    build2, _ = _assembler(a, b)
    res2 = check_assembly_freshness(list(_SYNTH), SimpleNamespace(), assemble=build2)
    assert res2.status == "BAYAT" and res2.card_missing == 1


def test_freshness_assembly_failure_fails_closed() -> None:
    def _boom(settings, **kw) -> AssemblyResult:
        raise RuntimeError("db kilitli")

    res = check_assembly_freshness([_CARD_LINE], SimpleNamespace(), assemble=_boom)
    assert res.status == "doğrulanamadı" and res.blockers


def test_real_assembly_is_deterministic(tmp_path, monkeypatch) -> None:
    """Kanonik birleştirme aynı girdilerle iki kez aynı çıktıyı verir → tam karşılaştırma."""
    lora_dir = tmp_path / "data" / "lora_sft"
    lora_dir.mkdir(parents=True)
    (lora_dir / "synthetic_qa.jsonl").write_text(
        "\n".join(
            _line(f"Ayrık sentetik cevap numarası {i} yeterince farklıdır {i}.") for i in range(40)
        ),
        encoding="utf-8",
    )

    class _Store:
        def list_approved_cards(self) -> list:
            return []

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _Store)
    settings = SimpleNamespace(root=tmp_path)
    file_lines = sft_assembly.assemble_sft_lines(settings, seed=0).lines
    res = check_assembly_freshness(file_lines, settings)
    assert res.deterministic and res.fresh and not res.blockers


# --------------------------------------------------------------------------- #
# B1 — CLI: pretrain-gate tazelik NO-GO'su yalnız kanonik dosyada
# --------------------------------------------------------------------------- #


@pytest.fixture
def _cli_root(monkeypatch, tmp_path):
    settings = __import__("app.config.settings", fromlist=["get_settings"]).get_settings()
    monkeypatch.setattr(type(settings), "root", property(lambda _self: tmp_path))
    monkeypatch.setattr(
        "app.training.discipline_dataset.discipline_jsonl_lines", lambda *a, **k: []
    )
    canonical = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("\n".join(_base_lines()) + "\n", encoding="utf-8")
    return canonical


def _stale(lines, settings, **kw):
    return sft_assembly.FreshnessResult(status="BAYAT", blockers=["eğitim verisi BAYAT: test"])


def test_cli_pretrain_gate_stale_canonical_is_nogo(monkeypatch, _cli_root) -> None:
    from app.main import app

    monkeypatch.setattr("app.training.sft_assembly.check_assembly_freshness", _stale)
    monkeypatch.setenv("COLUMNS", "300")
    result = CliRunner().invoke(app, ["pretrain-gate", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["verdict"] == "NO-GO"
    assert data["freshness"] == "BAYAT"
    assert any("BAYAT" in b for b in data["blockers"])


def test_cli_pretrain_gate_non_canonical_file_skips_freshness(
    monkeypatch, _cli_root, tmp_path
) -> None:
    from app.main import app

    monkeypatch.setattr("app.training.sft_assembly.check_assembly_freshness", _stale)
    other = tmp_path / "aday.jsonl"
    other.write_text(_cli_root.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("COLUMNS", "300")
    result = CliRunner().invoke(app, ["pretrain-gate", "--json", "--jsonl", str(other)])
    data = json.loads(result.stdout)
    assert data["freshness"].startswith("atlandı")
    assert not any("BAYAT" in b for b in data["blockers"])
