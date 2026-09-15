"""SFT birleştirmesinde PII maskeleme JSON'u bozmamalı (çevrimdışı; veritabanına dokunmaz).

Bulgu (2026-09-15, gerçek veride ölçüldü): maskeleme ham JSON metnine uygulanıyordu. Makale
başlık bloklarında yazar e-postaları satır sonundan hemen sonra gelir; ham metinde satır sonu
`\\n` olarak yazıldığından e-posta deseni kaçışın `n` harfini adrese kattı ve geriye geçersiz
`\\[` kaçışı kaldı → 1117 sentetik satırın 47'si bozuk JSON oldu, pretrain-gate NO-GO verdi.
"""

from __future__ import annotations

import json

from app.training.dataset_quality import audit_dataset
from app.training.sft_assembly import redact_pii_line


def _line(user: str, answer: str = "Özet cevap.") -> str:
    return json.dumps(
        {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]},
        ensure_ascii=False,
    )


# Gerçek bozulan satırın biçimi: satır sonunun HEMEN ardından gelen yazar e-postaları.
_AUTHOR_BLOCK = (
    "BAĞLAM:\nDepartment of Computer Science and Engineering, Bangalore, India\n"
    "dasashreeya@gmail.com\nshubhan.cs20@bmsce.ac.in\nAbstract—Cross-encoder\n\nSORU: ?"
)


def test_newline_adjacent_emails_keep_json_valid() -> None:
    raw = _line(_AUTHOR_BLOCK)
    red = redact_pii_line(raw)

    content = json.loads(red)["messages"][0]["content"]  # bozuk JSON olsaydı burada patlardı
    assert "@" not in content
    assert content.count("[kişisel-veri]") == 2
    # satır sonları korunur, kaçışın harfi adrese katılmaz
    assert "India\n[kişisel-veri]\n[kişisel-veri]\nAbstract" in content


def test_redacted_line_passes_gate_as_readable_and_pii_free() -> None:
    red = redact_pii_line(_line(_AUTHOR_BLOCK))
    rep = audit_dataset([red])
    assert rep.unreadable_lines == 0
    assert rep.pii_hits == 0


def test_line_without_pii_is_byte_identical() -> None:
    raw = _line("BAĞLAM:\nMomentum etkisi\nSORU: Nedir?")
    assert redact_pii_line(raw) is raw or redact_pii_line(raw) == raw


def test_invalid_json_is_returned_unchanged() -> None:
    broken = '{"messages": [bozuk john@example.com'
    assert redact_pii_line(broken) == broken


def test_line_separator_does_not_split_rewritten_line() -> None:
    raw = _line("Yazar " + chr(0x2028) + " john@example.com")
    red = redact_pii_line(raw)
    assert chr(0x2028) not in red
    assert len(red.splitlines()) == 1
    content = json.loads(red)["messages"][0]["content"]
    assert chr(0x2028) in content and "[kişisel-veri]" in content


def test_phone_after_newline_is_masked_and_json_valid() -> None:
    # Biçim `_PII_PATTERNS["phone_intl"]`in TAM kapsadığı biçim (+ülke alan 3-hane 2-4-hane);
    # desenin kapsamı bu testin konusu değil — konu, maskelemenin JSON'u bozmaması.
    red = redact_pii_line(_line("İletişim\n+90 532 123 4567\nSon"))
    content = json.loads(red)["messages"][0]["content"]
    assert "532" not in content
    assert "İletişim\n[kişisel-veri]\nSon" in content
