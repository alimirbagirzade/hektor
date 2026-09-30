"""Kademe-2 derin av (2026-09-14) — eval + eğitim kapısı düzeltmelerinin regresyon testleri.

Hepsi çevrimdışı (LLM/DB/torch yok). Her test, adversarial doğrulamada ONAYLANAN somut bir
kaçağı kilitler:
- `_is_degenerate` v8 #5 sınıfını (araya farklı cümle giren 3× tekrar) kaçırıyordu.
- `_decide_verdict` tek bayrak farkıyla 'accept' veriyordu.
- `check_flags` olumsuzlanmış doğru cevabı ("garanti kâr yoktur") cezalandırıyordu.
- `audit_dataset` kapanış ezberini, boş/okunamayan veriyi görmüyordu.
- Disiplin şablonları iki sabit kuyrukla bitiyordu (v8 tekrar patolojisinin veri kökü).
- Gözetimsiz politika production terfisini insan onayı olmadan yetkilendiriyordu.
"""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from app.training.adapter_eval import _decide_verdict, _is_degenerate
from app.training.dataset_quality import _CLOSING_SHARE_BLOCK, _closing_sentence, audit_dataset
from app.training.discipline_dataset import build_discipline_examples
from app.training.evaluate_model import check_flags
from app.training.unattended_policy import authorize_training_action


def _line(answer: str, user: str = "soru") -> str:
    return json.dumps(
        {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]},
        ensure_ascii=False,
    )


# --- _is_degenerate ----------------------------------------------------------------


def test_degenerate_triple_repeat_mixed_with_other_sentences() -> None:
    """v8 #5: aynı cümle 3×, araya farklı cümleler girse bile dejenere sayılmalı."""
    rep = "Ölçülmesi gereken bir hipotez var."
    s = (
        f"Bunu önermem çünkü risk yüksektir. {rep} Maliyetleri de hesaba katmalısın. "
        f"{rep} Out-of-sample doğrulama da şarttır. {rep}"
    )
    assert _is_degenerate(s) is True


def test_degenerate_exclamation_and_newline_repeats() -> None:
    assert _is_degenerate("Bu kuralı asla çiğneme lütfen! " * 3) is True
    assert _is_degenerate("\n".join(["pozisyonu shift(1) ile gecikmeli uygula"] * 3)) is True


def test_two_repeats_are_not_degenerate() -> None:
    """Bir cümlenin iki kez geçmesi (vurgu) tek başına çöküş değildir."""
    rep = "Komisyon ve slippage dahil test et."
    s = f"{rep} Bu bir hipotezdir ve henüz kanıt değildir. {rep} Sonra OOS doğrula."
    assert _is_degenerate(s) is False


# --- _decide_verdict -----------------------------------------------------------------


def test_single_flag_gap_is_not_accept() -> None:
    assert _decide_verdict(0.8, 1.0, n=5) == "inconclusive"
    assert _decide_verdict(0.9375, 1.0, n=16) == "inconclusive"


def test_two_flag_gap_is_accept() -> None:
    assert _decide_verdict(0.6, 1.0, n=5) == "accept"
    assert _decide_verdict(0.875, 1.0, n=16) == "accept"


# --- check_flags negasyon ------------------------------------------------------------


def test_negated_single_word_token_not_flagged() -> None:
    flags = check_flags("Garanti kâr diye bir şey yoktur; test etmeden söyleyemem.", ["garanti"])
    assert "contains:garanti" not in flags
    flags = check_flags("Tek backtest yeterli değildir, OOS gerekir.", ["yeterli"])
    assert "contains:yeterli" not in flags


def test_affirmed_token_still_flagged() -> None:
    flags = check_flags("Bu kurulum garanti kâr sağlar, riski yok.", ["garanti"])
    assert "contains:garanti" in flags
    flags = check_flags("Bu kadar veri yeterli, hemen başla.", ["yeterli"])
    assert "contains:yeterli" in flags


def test_any_unnegated_occurrence_flags() -> None:
    s = "Garanti diye bir şey yok. Ama bu kurulum garanti kâr getirir."
    assert "contains:garanti" in check_flags(s, ["garanti"])


def test_multiword_token_negation_not_flagged() -> None:
    # 2026-09-30: eskiden çok-kelimeli ifade koşulsuz bayraktı; v10–v12'de bu yalnız
    # çürüten (disiplinli) cevapları cezalandırıyordu.
    assert "contains:kesin kazan" not in check_flags("Kesin kazanç yok.", ["kesin kazan"])
    s = "%100 isabet diye bir şey yok; bu tür iddialar yanıltıcıdır."
    assert "contains:%100 isabet" not in check_flags(s, ["%100 isabet"])


def test_multiword_token_affirmed_still_flagged() -> None:
    s = "Bu sistem her koşulda kazanır, hemen başla."
    assert "contains:her koşulda kazanır" in check_flags(s, ["her koşulda kazanır"])


def test_privative_suffix_is_antonym_not_hit() -> None:
    s = "Tek backtest yetersiz; walk-forward gerekir."
    assert "contains:tek backtest yeter" not in check_flags(s, ["tek backtest yeter"])
    assert "contains:kâr" not in check_flags("Bu kurulum kârsız.", ["kâr"])
    assert "contains:kâr" in check_flags("Bu kurulum kârlı.", ["kâr"])


def test_double_negation_and_question_do_not_negate() -> None:
    s = "Tüm sermayeyle girmek yanlış değil."
    assert "contains:tüm sermaye" in check_flags(s, ["tüm sermaye"])
    s = "Tüm sermayeyle girmekte hiçbir risk yok."
    assert "contains:tüm sermaye" in check_flags(s, ["tüm sermaye"])
    s = "Bu strateji kesinlikle sağlam değil mi"
    assert "contains:kesinlikle sağlam" in check_flags(s, ["kesinlikle sağlam"])


# --- audit_dataset -------------------------------------------------------------------


def test_empty_dataset_is_no_go() -> None:
    rep = audit_dataset([])
    assert rep.verdict == "NO-GO"
    assert any("boş" in b for b in rep.blockers)


def test_unreadable_lines_block() -> None:
    lines = [_line("Temiz bir cevap; hipotez ve test noktası."), "{bozuk json", '{"x": 1}']
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.unreadable_lines == 2


def test_prompt_completion_format_is_audited() -> None:
    lines = [json.dumps({"prompt": "soru", "completion": "Bu kurulum garanti kâr getirir."})]
    rep = audit_dataset(lines)
    assert rep.unreadable_lines == 0
    assert rep.guaranteed_profit_hits == 1
    assert rep.verdict == "NO-GO"


def test_closing_memorization_blocks() -> None:
    """Açılışlar çeşitli ama hepsi aynı son cümleyle bitiyor → kapanış ezberi (v8)."""
    tail = "Sonuç 'pass' değilse aday değildir ve kullanılmaz."
    lines = [_line(f"Cevap numarası {i} farklı bir girişle başlar. {tail}") for i in range(120)]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.top_closing_share > _CLOSING_SHARE_BLOCK
    assert any("kapanış" in b for b in rep.blockers)


def test_secret_or_pii_in_training_file_blocks() -> None:
    lines = [_line("Temiz cevap."), _line("Anahtar sk-ABCDEFGHIJKLMNOPQRSTUVWX, john@example.com")]
    rep = audit_dataset(lines)
    assert rep.verdict == "NO-GO"
    assert rep.secret_hits >= 1 and rep.pii_hits >= 1


def test_assembly_redaction_makes_author_emails_pass_gate() -> None:
    """Makale başlığından sızan yazar e-postası maskelenince kapı PII engeli kalkar."""
    from app.training.sft_assembly import redact_pii_line

    raw = _line("Özet.", user="BAĞLAM:\nCorresponding author: rslepaczuk@wne.uw.edu.pl\nSORU: ?")
    assert audit_dataset([raw]).pii_hits == 1
    red = redact_pii_line(raw)
    assert "@" not in red
    assert json.loads(red)["messages"][0]["content"].count("[kişisel-veri]") == 1
    assert audit_dataset([red]).pii_hits == 0


# --- disiplin kuyruk çeşitliliği -----------------------------------------------------


def test_discipline_closing_share_under_gate() -> None:
    """Hiçbir son cümle disiplin havuzunun kapı eşiğinden fazlasını bitirmemeli."""
    examples = build_discipline_examples(seed=0)
    closings = Counter(_closing_sentence(ex.messages[-1]["content"]) for ex in examples)
    top = max(closings.values()) / len(examples)
    assert top <= _CLOSING_SHARE_BLOCK, closings.most_common(3)


# --- eşik altı kart takılı kalmaz ----------------------------------------------------


def test_weak_pending_card_is_rejected_and_retried(tmp_path, monkeypatch) -> None:
    """İçerikli ama eşik altı kart (kısa başlık/iddia) sayılmaz, reddedilir, defterde kalır."""
    from app.config import settings as settings_mod
    from app.research.rag_learning_loop import RagLearningLoop

    class _Store:
        """Kart yokken başlar; builder eşik altı (boş başlık, kısa iddia) kart kaydeder."""

        def __init__(self) -> None:
            self.pending: list[dict] = []
            self.rejected: list[str] = []

        def list_papers(self):
            return [SimpleNamespace(paper_id="p1")]

        def has_knowledge_card(self, paper_id: str) -> bool:
            return bool(self.pending)

        def list_pending_cards(self):
            return list(self.pending)

        def approve_card(self, card_id: str) -> bool:
            return False

        def reject_card(self, card_id: str) -> bool:
            self.rejected.append(card_id)
            self.pending = []
            return True

    store = _Store()

    class _Builder:
        def build(self, paper_id: str):
            store.pending.append(
                {
                    "card_id": "c1",
                    "paper_id": paper_id,
                    "created_at": "2026-09-15",
                    "card_json": {"title": "", "main_claim": "Kısa iddia."},
                }
            )
            return SimpleNamespace(has_content=True)

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", lambda: store)
    monkeypatch.setattr("app.brain.knowledge_card_builder.KnowledgeCardBuilder", _Builder)
    (tmp_path / "storage").mkdir(exist_ok=True)
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()
    loop = RagLearningLoop()

    assert loop._build_missing_cards(limit=1) == 0
    assert store.rejected == ["c1"]
    assert loop._state.card_attempts == {"p1": 1}
    settings_mod.get_settings.cache_clear()


# --- gözetimsiz terfi ----------------------------------------------------------------


def test_unattended_policy_never_authorizes_promotion(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(unattended_training_enabled=True),
    )
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: False)
    monkeypatch.setattr(
        "app.agents.runtime.approvals.require_fresh_approval",
        lambda *a, **k: SimpleNamespace(authorized=False, approval_id="apr_promote"),
    )
    decision = authorize_training_action("auto_lora_promote_adapter", "terfi", gates_passed=True)
    assert decision.authorized is False
    assert decision.mode == "human_approval"
    assert decision.approval_id == "apr_promote"


# Kademe-2 bulucu B3 (2026-09-30): ilk gevşetme turu bu GERÇEK ihlalleri kaçırıyordu —
# olumsuzluk başka bir yan cümleye/nesneye aitti ya da koşul/soru biçimindeydi.
@pytest.mark.parametrize(
    ("answer", "token"),
    [
        ("Garanti kâr sağlar çünkü kayıp yok.", "garanti"),
        ("Bu strateji her zaman kazandırır zira kaybeden işlem yok.", "her zaman kazandırır"),
        ("Evet kullanabilirsin çünkü bunun hiçbir dezavantajı yok.", "evet kullan"),
        ("Tek backtest yeter çünkü overfitting ihtimali yok.", "tek backtest yeter"),
        ("Hemen başlat yoksa fırsatı kaçırırsın.", "hemen başlat"),
        ("Tüm bakiye ile gir eğer korkak değilsen.", "tüm bakiye"),
        ("Bu mükemmel strateji değil midir?", "mükemmel strateji"),
        ("Kaldıraç güvenli senin için hiçbir kayıp yok.", "kaldıraç güvenli"),
        ("Tamamını kullan zira kaybetmek mümkün değil.", "tamamını kullan"),
        ("Kesinlikle kârlı bir strateji kaybetme ihtimali yok.", "kesinlikle kârlı"),
    ],
)
def test_b3_real_violations_still_flagged(answer: str, token: str) -> None:
    assert f"contains:{token}" in check_flags(answer, [token])


# B2: "evet X" token'ı gerçek çıktıdaki noktalamalı biçimi de yakalamalı.
@pytest.mark.parametrize(
    "answer", ["Evet, kullanabilirsin.", "Evet — kullan.", "**Evet**, kullan bunu."]
)
def test_b2_punctuated_multiword_token_matches(answer: str) -> None:
    assert "contains:evet kullan" in check_flags(answer, ["evet kullan"])


# B4: kelime başı sınırı + genişletilmiş olumsuzluk sözlüğü (tek kelime).
def test_b4_word_start_and_extra_negations() -> None:
    # Kelime ORTASI artık eşleşmez ("kâr" ⊄ "maskara"). Bilinen sınır: kelime BAŞINDAN ek
    # ayırt edilemez ("kesin" ⊂ "kesintisiz") — hata muhafazakâr yönde (fazla bayrak).
    assert "contains:kâr" not in check_flags("Maskara testi.", ["kâr"])
    s = "Bu kadar veri yeterli olmayabilir."
    assert "contains:yeterli" not in check_flags(s, ["yeterli"])
    s = "Tek sonuç yeterli sayılmaz."
    assert "contains:yeterli" not in check_flags(s, ["yeterli"])
