"""Gate 5 kaynak sayı kontrolü — karttaki yüzde kendi makalesinde var mı?

Regresyon bağlamı (2026-09-27 lora-audit incelemesi): kart üreticisi makalede OLMAYAN
yüzdeler uyduruyordu ("could reduce VaR by 10-15%", "will reduce forgetting by at least
30%"); mevcut performans-iddiası kalıpları bunların yalnız bir kısmını yakalıyordu.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.lora.control_plane import LoRAControlPlane
from app.lora.gates import gate_5_math
from app.lora.math_verifier import source_numbers, verify_math_content

_SOURCE = (
    "We evaluate on ACL18. The model obtains accuracies of 63.707 and 56.879 on the two "
    "datasets and an annualized return of 31.24% after costs. Drawdown fell by 12%."
)


def _missing(text: str, source: str = _SOURCE) -> list[str]:
    return [i for i in verify_math_content(text, source_numbers(source)).issues if "kaynakta" in i]


def test_invented_percentage_flagged_for_review_not_blocked() -> None:
    result = verify_math_content(
        "The strategy could reduce VaR by 15% versus independence.", source_numbers(_SOURCE)
    )
    assert "kaynakta olmayan sayı: %15" in result.issues
    assert result.requires_review is True
    assert result.passed is True  # inceleme, BLOK değil


def test_percentage_present_in_source_not_flagged() -> None:
    assert _missing("It reports an annualized return of 31.24% and a 12% lower drawdown.") == []


def test_decimal_in_source_table_without_percent_sign_accepted() -> None:
    """Makale tabloda % işaretsiz yazmış ("accuracies of 63.707") → kart "63.707%" geçer."""
    assert _missing("STST achieves 63.707% accuracy on ACL18.") == []


def test_decimal_not_split_by_sentence_boundary() -> None:
    """Cümle bölme sayı içindeki noktayı kesmemeli ("63.707%" → "63" + "707%" DEĞİL)."""
    assert _missing("Accuracy was 63.707%. Return was 31.24%.") == []


def test_test_if_threshold_and_example_parameter_exempt() -> None:
    """Önerilen test eşiği ve açık örnek parametre iddia değildir."""
    text = (
        "Test if regime detection exceeds 70% accuracy out-of-sample. "
        "Place the stop-loss lower (e.g., 2% below current price)."
    )
    assert _missing(text) == []


def test_no_source_text_skips_check() -> None:
    """Kaynak metni yoksa doğrulanamaz → yanlışlıkla işaretlenmez."""
    assert not any("kaynakta" in i for i in verify_math_content("Gains of 15%.", None).issues)
    empty = source_numbers("")
    assert not any("kaynakta" in i for i in verify_math_content("Gains of 15%.", empty).issues)


def test_gate_5_uses_per_paper_source_numbers() -> None:
    card = {"card_id": "c1", "paper_id": "p1", "card_json": {"summary": "Gains of 15%."}}
    assert gate_5_math([card]).review_count == 0  # kaynak verilmedi → eski davranış
    assert gate_5_math([card], {"p1": source_numbers(_SOURCE)}).review_count == 1
    assert gate_5_math([card], {"p1": source_numbers("gains of 15% reported")}).review_count == 0


def test_control_plane_builds_source_numbers_from_chunks() -> None:
    class _Store:
        def list_chunks(self, pid: str) -> list[SimpleNamespace]:
            return [SimpleNamespace(text="return of 31.24%")] if pid == "p1" else []

    plane = LoRAControlPlane(store=_Store())  # type: ignore[arg-type]
    nums = plane._source_numbers([{"paper_id": "p1"}, {"paper_id": "p2"}, {"paper_id": ""}])
    assert nums == {"p1": frozenset({"31.24"}), "p2": frozenset()}


def test_control_plane_store_without_chunks_skips_check() -> None:
    plane = LoRAControlPlane(store=SimpleNamespace())  # type: ignore[arg-type]
    assert plane._source_numbers([{"paper_id": "p1"}]) == {}
