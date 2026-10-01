"""v14 öz-damıtma (self_distill) + bağlı değişiklikler — çevrimdışı (LLM/DB'siz, stub'lı)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.training import self_distill as sd
from app.training import sft_assembly
from app.training.dataset_quality import audit_dataset


@dataclass
class _Chunk:
    chunk_id: str
    paper_id: str
    text: str = "metin"
    title: str | None = None
    page_number: int | None = None
    section_name: str | None = None
    distance: float | None = 0.1

    @property
    def citation(self) -> str:
        return f"[{self.paper_id}:{self.chunk_id}]"


def _synth_line(q: str, paper: str = "paper_a", *, discipline: bool = False) -> str:
    meta = {"synthetic": True, "paper_id": paper, "chunk_id": f"{paper}_c1", "question": q}
    if discipline:
        meta["discipline"] = True
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"BAĞLAM:\nx\n\nSORU: {q}"},
                {"role": "assistant", "content": "cevap"},
            ],
            "metadata": meta,
        },
        ensure_ascii=False,
    )


_TR_LONG = (
    "Bu yöntem, oynaklık rejimleri ile ilgili bir değerlendirme sunar ve sınırlamaları "
    "açıkça ele alınır. Kalman filtresi gözlem gürültüsünü durum tahmininden ayırır. "
    "Momentum etkisi işlem maliyetleri düşüldükten sonra zayıflayabilir ve bu yüzden "
    "örneklem dışı test gerekir. Kelly oranı tahmin hatasına karşı duyarlıdır, bu nedenle "
    "kesirli uygulama tercih edilir. GARCH modelinde kalıcılık katsayıların toplamı ile "
    "ölçülür. Veri sızıntısı gecikmeli pozisyon ile önlenir. Hayatta kalma yanlılığı, "
    "listeden çıkan varlıklar veriden silinirse ortaya çıkar. Parametre taraması çoklu test "
    "sorununa yol açar ve düzeltme ister."
)


def _res(
    content: str = _TR_LONG, reason: str = "stop", p: int = 3000, o: int = 900
) -> sd.ChatResult:
    return sd.ChatResult(content=content, done_reason=reason, prompt_tokens=p, output_tokens=o)


# --------------------------------------------------------------------------- seçim


def test_question_filters_reject_passage_bound() -> None:
    assert not sd._question_ok("Pasajda belirtilen Sn toplamının davranışı nedir?", "rag")
    assert not sd._question_ok("Kaynak [250]deki çalışmanın konusu nedir, açıklayınız?", "rag")
    assert not sd._question_ok("Kelly 1956 makalesindeki ana sonuç nedir, neden?", "rag")
    assert sd._question_ok("GARCH(1,1) modelinde kalıcılık neyi ifade eder?", "rag")
    # Teorem/sayfa numarası RAG'sız sorulamaz.
    assert not sd._question_ok("Teorem 5.16 için gerekli koşul nedir, açıklayınız?", "plain")


def test_question_ok_allows_long_multipart() -> None:
    q = "Volatilite hedefleme hakkında:\na) Tanımı nedir?\nb) Bir sayısal örnek ver." + " x" * 200
    assert len(q) > 300
    assert sd._question_ok(q, "rag")


def test_select_jobs_deterministic_balanced_and_excludes() -> None:
    words = ["volatilite", "momentum", "kalman", "kelly", "garch"]
    lines = [
        _synth_line(f"Kavram {p} {w} nasıl çalışır ve neden önemlidir?", p)
        for p in "abc"
        for w in words
    ]
    lines.append(_synth_line("Disiplin sorusu hayatta kalma nedir?", "d", discipline=True))
    a = sd.select_jobs(lines, n_rag=4, n_plain=2, seed=1)
    b = sd.select_jobs(lines, n_rag=4, n_plain=2, seed=1)
    assert a == b
    assert sum(j.mode == "rag" for j in a) == 4 and sum(j.mode == "plain" for j in a) == 2
    assert all("Disiplin" not in j.question for j in a)
    # İlk katman her makaleden birer soru (round-robin).
    assert {j.origin_paper_id for j in a[:3]} == {"a", "b", "c"}
    banned = a[0].question
    c = sd.select_jobs(lines, n_rag=4, n_plain=2, seed=1, exclude=lambda qs: {banned})
    assert banned not in {j.question for j in c}


def test_select_jobs_dedups_questions_differing_only_by_digits() -> None:
    lines = [
        _synth_line("GARCH(1,1) modelinde kalıcılık neyi ifade eder?", "a"),
        _synth_line("GARCH(2,1) modelinde kalıcılık neyi ifade eder?", "b"),
    ]
    assert len(sd.select_jobs(lines, n_rag=5, n_plain=0)) == 1


def test_select_jobs_uses_generated_questions() -> None:
    extra = [
        {
            "question": "Kalman filtresi gürültüyü nasıl ayırır?",
            "paper_id": "p9",
            "chunk_id": "p9_c3",
        }
    ]
    jobs = sd.select_jobs([], n_rag=1, n_plain=0, extra=extra)
    assert jobs == [sd.DistillJob("rag", extra[0]["question"], "p9", "p9_c3")]


# --------------------------------------------------------------------------- kapılar


def test_validate_rejects_truncated_budget_short_cjk() -> None:
    assert sd.validate_distilled(_TR_LONG, "plain", _res(reason="length")) == "kesik"
    assert sd.validate_distilled(_TR_LONG, "plain", _res(p=5000, o=1200)) == "bütçe"
    assert sd.validate_distilled("kısa cevap bu", "plain", _res()) == "kısa"
    assert sd.validate_distilled(_TR_LONG + "期权", "plain", _res()) == "cjk"
    en = "This method evaluates volatility regimes and the limits are discussed. " * 12
    assert sd.validate_distilled(en, "plain", _res()) == "dil"


def test_validate_rejects_advice() -> None:
    ans = _TR_LONG + " Bu strateji garantili kâr sağlar."
    assert sd.validate_distilled(ans, "plain", _res()) == "tavsiye"
    # Metodoloji emri tavsiye değildir.
    ok = _TR_LONG + " Out-of-sample test yapılmalıdır."
    assert sd.validate_distilled(ok, "plain", _res()) is None


def test_validate_citations_rag_and_plain() -> None:
    chunks = [_Chunk("p1_c1", "p1"), _Chunk("p1_c2", "p1"), _Chunk("p2_c7", "p2")]
    good = _TR_LONG + " [p1:p1_c1] [p1:p1_c2] [p2:p2_c7]"
    assert sd.validate_distilled(good, "rag", _res(), chunks) is None
    assert sd.validate_distilled(_TR_LONG + " [p1:p1_c1]", "rag", _res(), chunks) == "atıf-az"
    fake = good + " [p9:p9_c1]"
    assert sd.validate_distilled(fake, "rag", _res(), chunks) == "uydurma-atıf"
    assert sd.validate_distilled(_TR_LONG + " [paper_x:c1]", "plain", _res()) == "uydurma-atıf"


# --------------------------------------------------------------------------- üretim


def test_rag_messages_match_live_prompt() -> None:
    from app.brain.rag_answerer import build_rag_prompt

    chunks = [_Chunk("p1_c1", "p1", "birinci"), _Chunk("p1_c2", "p1", "ikinci")]
    msgs = sd.build_messages(sd.DistillJob("rag", "Soru nedir?"), chunks)
    from app.config import get_settings

    system, user = build_rag_prompt(
        "Soru nedir?", chunks, reorder=get_settings().rag_reorder_context
    )
    assert msgs == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    assert user.startswith("SOURCES / KAYNAKLAR:") and user.endswith("QUESTION / SORU: Soru nedir?")


def _fake_chat(content: str):
    calls: list[int] = []

    def chat(messages, seed):
        calls.append(seed)
        return _res(content=content)

    return chat, calls


def test_run_distill_writes_accepted_logs_rejects_and_resumes(tmp_path) -> None:
    out = tmp_path / "distill_qa.jsonl"
    chunks = [_Chunk("p1_c1", "p1"), _Chunk("p1_c2", "p1"), _Chunk("p2_c7", "p2")]
    good = _TR_LONG + " [p1:p1_c1] [p1:p1_c2] [p2:p2_c7]"
    jobs = [sd.DistillJob("rag", f"Kavram {i} nasıl işler?", "p1", "p1_c1") for i in range(4)]
    chat, calls = _fake_chat(good)
    r1 = sd.run_distill(
        out, jobs, chat=chat, retrieve=lambda q: chunks, teacher="qwen3:base", seed=10, limit=2
    )
    assert r1["done"] == 2 and calls == [10, 11]
    r2 = sd.run_distill(
        out, jobs, chat=chat, retrieve=lambda q: chunks, teacher="qwen3:base", seed=10
    )
    assert r2["done"] == 4 and calls == [10, 11, 12, 13]  # kaldığı işten sürdü
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    meta = rows[0]["metadata"]
    assert meta["distilled"] and meta["mode"] == "rag" and meta["teacher"] == "qwen3:base"
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": good}

    # Retrieval boş → satır yok, red günlüğüne düşer.
    out2 = tmp_path / "b.jsonl"
    sd.run_distill(out2, jobs[:1], chat=chat, retrieve=lambda q: [], teacher="qwen3:base")
    assert not out2.read_text(encoding="utf-8").strip()
    rej = [json.loads(x) for x in sd.rejects_path_for(out2).read_text("utf-8").splitlines()]
    assert rej[0]["status"] == "retrieval-boş"


def test_run_distill_refuses_adapter_teacher_and_mismatch(tmp_path) -> None:
    out = tmp_path / "d.jsonl"
    jobs = [sd.DistillJob("plain", "Kavram nasıl işler, neden?")]
    chat, _ = _fake_chat(_TR_LONG)
    with pytest.raises(ValueError, match="adapter"):
        sd.run_distill(out, jobs, chat=chat, retrieve=lambda q: [], teacher="hektor-v13-30b")
    sd.run_distill(out, jobs, chat=chat, retrieve=lambda q: [], teacher="qwen3:base")
    other = [sd.DistillJob("plain", "Başka kavram nasıl işler?")]
    with pytest.raises(ValueError, match="meta uyuşmuyor"):
        sd.run_distill(out, other, chat=chat, retrieve=lambda q: [], teacher="qwen3:base")
    with pytest.raises(ValueError, match="meta uyuşmuyor"):
        sd.run_distill(out, jobs, chat=chat, retrieve=lambda q: [], teacher="qwen3:other")


# --------------------------------------------------------------------------- soru üretimi


def test_sample_chunks_filters_junk_and_balances() -> None:
    good = "Volatilite kümelenmesi finansal getirilerde gözlenen bir olgudur. " * 15
    chunks = [_Chunk(f"a_c{i}", "a", good) for i in range(5)]
    chunks += [_Chunk("b_c1", "b", "References\n" + good), _Chunk("b_c2", "b", "kısa")]
    chunks += [_Chunk("c_c1", "c", good)]
    out = sd.sample_chunks(chunks, per_paper=2, seed=0, exclude_papers=frozenset({"c"}))
    assert sorted({c.paper_id for c in out}) == ["a"] and len(out) == 2
    assert out == sd.sample_chunks(chunks, per_paper=2, seed=0, exclude_papers=frozenset({"c"}))


def test_validate_question_multipart_and_dependent() -> None:
    mp = "Kalman filtresi hakkında:\na) Kazanç nedir?\nb) Basit bir sayısal örnek ver."
    assert sd.validate_question(mp, multipart=True) is None
    assert sd.validate_question("Kalman nedir?", multipart=True) == "altmadde"
    dep = "Bu metinde:\na) Kazanç nedir, açıkla?\nb) Örnek ver, hesapla."
    assert sd.validate_question(dep, multipart=True) == "bağımlı"
    assert sd.validate_question("Kalman filtresi gürültüyü nasıl ayırır?", multipart=False) is None


def test_generate_questions_resumes_and_loads_accepted(tmp_path) -> None:
    good = "Volatilite kümelenmesi finansal getirilerde gözlenen bir olgudur. " * 15
    chunks = [_Chunk(f"a_c{i}", "a", good) for i in range(3)]
    out = tmp_path / "q.jsonl"
    seen: list[int] = []

    def gen(prompt: str, seed: int) -> str:
        seen.append(seed)
        if "alt maddeli" in prompt:
            return json.dumps({"question": "Kümelenme:\na) Nedir, açıkla?\nb) Örnek hesapla."})
        return json.dumps({"question": "Volatilite kümelenmesi neden oluşur, açıklayınız?"})

    sd.generate_questions(out, chunks[:2], generate=gen, seed=5)
    sd.generate_questions(out, chunks, generate=gen, seed=5)
    assert seen == [5, 6, 7]  # işlenen parçalar atlandı
    rows = sd.load_generated_questions(out)
    assert len(rows) == 3 and rows[0]["multipart"] is True


# --------------------------------------------------------------------------- birleştirme


class _EmptyStore:
    def list_approved_cards(self) -> list:
        return []


def test_assembly_caps_synth_prefers_enriched_and_adds_distill(tmp_path, monkeypatch) -> None:
    lora = tmp_path / "data" / "lora_sft"
    lora.mkdir(parents=True)
    rows = []
    for i in range(10):
        meta = {"synthetic": True, "enriched": i < 3}
        rows.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"soru {i} farklı"},
                        {
                            "role": "assistant",
                            "content": f"Bu {i}. ayrık cevap kelimeleri {i} {i * 7}.",
                        },
                    ],
                    "metadata": meta,
                },
                ensure_ascii=False,
            )
        )
    (lora / "synthetic_qa.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    distill = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "damıtma sorusu"},
                {"role": "assistant", "content": "Uzun damıtma cevabı burada yer alıyor."},
            ],
            "metadata": {"distilled": True, "mode": "plain", "question": "Kalman nasıl işler?"},
        },
        ensure_ascii=False,
    )
    (lora / "distill_qa.jsonl").write_text(distill + "\n", encoding="utf-8")
    # Önceki tur dosyası da okunur; kapıdan geçemeyen satır (uydurma kaynakça) elenir.
    bad = distill.replace("burada yer alıyor.", "burada. Kaynaklar: Shazeer et al. (2017)")
    (lora / "distill_qa.r1.jsonl").write_text(bad + "\n", encoding="utf-8")
    (lora / "distill_qa.rejects.jsonl").write_text(bad + "\n", encoding="utf-8")
    monkeypatch.setattr(sft_assembly, "build_dataset", lambda cards: [])
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _EmptyStore)

    res = sft_assembly.assemble_sft_lines(
        SimpleNamespace(root=tmp_path), discipline=False, synth_cap=5
    )
    assert res.distill_n == 1 and distill in res.lines
    synth_kept = [ln for ln in res.lines if ln != distill]
    assert len(synth_kept) == 5
    assert sum(json.loads(ln)["metadata"]["enriched"] for ln in synth_kept) == 3
    # Orijinal sıra korunur.
    assert synth_kept == [r for r in rows if r in synth_kept]
    again = sft_assembly.assemble_sft_lines(
        SimpleNamespace(root=tmp_path), discipline=False, synth_cap=5
    )
    assert again.lines == res.lines
    no_d = sft_assembly.assemble_sft_lines(
        SimpleNamespace(root=tmp_path), discipline=False, synth_cap=5, distill=False
    )
    assert distill not in no_d.lines


# --------------------------------------------------------------------------- şablon kapısı


def _row(system: str, answer: str, user: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ]
        },
        ensure_ascii=False,
    )


def test_template_gate_exempts_phrases_from_own_system_prompt() -> None:
    mandated = "Bu bulgu doğrudan trading kuralına çevrilemez ve ayrıca test edilmelidir."
    system = f"Format: 1. Short Answer / Kısa Cevap. Uygun değilse: {mandated}"
    import random

    vocab = [f"kavram{chr(97 + a)}{chr(97 + b)}" for a in range(26) for b in range(26)]

    def body(i: int) -> str:
        # Her cevap kendine özgü kelimeler: istem dışı ortak 8-gram yok.
        return " ".join(random.Random(i).sample(vocab, 12))

    lines = [_row(system, f"{body(i)}. {mandated}", f"q{i}") for i in range(60)]
    rep = audit_dataset(lines)
    assert not any("şablon tekrarı" in b for b in rep.blockers)

    # Aynı ifade sistem isteminde YOKSA hâlâ engellenir (v8 iskelet ezberi).
    bare = [_row("Genel asistan.", f"{body(i)}. {mandated}", f"q{i}") for i in range(60)]
    rep2 = audit_dataset(bare)
    assert any("şablon tekrarı" in b for b in rep2.blockers)


# --------------------------------------------------------------------------- Kademe 2 (v14)


def test_citation_parse_strict_unique_and_malformed() -> None:
    from app.brain.answer_quality import parse_citations, verify_citations

    cites, bad = parse_citations("[pa:pa_c1, pb:pb_c9] [pa: pa_c2; s.4] [0:1] [09:30] x[t:T]")
    assert cites == [("pa", "pa_c1"), ("pb", "pb_c9"), ("pa", "pa_c2")] and not bad
    assert parse_citations("[paper_zz_c0009] [bkz. paper_x:c1]")[1] == [
        "[paper_zz_c0009]",
        "[bkz. paper_x:c1]",
    ]
    chunks = [_Chunk("pa_c1", "pa"), _Chunk("pa_c2", "pa")]
    loose = verify_citations("[pa:pa_c9] [zz:pa_c1]", chunks)
    assert not loose.has_unsupported  # canlı uyarı: gevşek (makale ya da parça eşleşmesi)
    strict = verify_citations("[pa:pa_c9] [zz:pa_c1] [pa:pa_c1]", chunks, strict=True)
    assert strict.unsupported == ["pa:pa_c9", "zz:pa_c1"] and strict.n_unique == 3
    rep = verify_citations("[pa:pa_c1] [pa:pa_c1] [pa:pa_c1]", chunks, strict=True)
    assert rep.n_cited == 3 and rep.n_unique == 1


def test_rag_gate_requires_two_unique_sources() -> None:
    chunks = [_Chunk("p1_c1", "p1"), _Chunk("p1_c2", "p1")]
    one = _TR_LONG + " [p1:p1_c1] [p1:p1_c1] [p1:p1_c1]"
    assert sd.validate_distilled(one, "rag", _res(), chunks) == "atıf-az"
    two = _TR_LONG + " [p1:p1_c1] [p1:p1_c2]"
    assert sd.validate_distilled(two, "rag", _res(), chunks) is None
    swapped = _TR_LONG + " [p1:p1_c1] [p9:p1_c2]"
    assert sd.validate_distilled(swapped, "rag", _res(), chunks) == "uydurma-atıf"


@pytest.mark.parametrize(
    "tail",
    [
        "\n\n### Kaynaklar\n- Shazeer ve ark.",
        " Ayrıntı için https://arxiv.org/abs/1701.06538 bakılabilir.",
        " Bu yaklaşım Fedus et al. tarafından önerildi.",
        " Bu cevap RAG bağlamına dayanır.",
        " Engle (1982) bu modeli tanımladı.",
    ],
)
def test_plain_gate_rejects_fabricated_sources(tail: str) -> None:
    assert sd.validate_distilled(_TR_LONG + tail, "plain", _res()) == "uydurma-kaynak"


def test_plain_messages_carry_no_source_note() -> None:
    msgs = sd.build_messages(sd.DistillJob("plain", "Kalman nasıl işler?"), None)
    assert msgs[0]["content"].endswith(sd.PLAIN_NO_SOURCE_NOTE)


def test_repetition_gate_ignores_scaffold_but_catches_content_loops() -> None:
    mandated = "Bu bulgu doğrudan trading kuralına çevrilemez."
    system = f"6. Trading Hypothesis / Trading Hipotezi\nDeğilse: {mandated}"
    bodies = [
        "Oynaklık kümelenmesi kısa ufukta belirgindir [p1:p1_c1, s.9].",
        "İşlem maliyeti küçük kenarları siler [p1:p1_c1, s.9].",
        "Örneklem dışı dönem ayrı tutulmalıdır [p1:p1_c1, s.9].",
        "Rejim değişimi parametre kararlılığını bozar [p1:p1_c1, s.9].",
    ]
    scaffold = "\n".join(
        f"{i}. Başlık {i}\n{mandated}\n{b}" for i, b in zip(range(6, 10), bodies, strict=True)
    )
    ans = _TR_LONG + "\n" + scaffold
    # Zorunlu cümle 4×, başlıklar ve aynı kaynağa tekrarlı atıf → dejenerasyon DEĞİL.
    assert sd.content_gate(ans, "plain", None, system) != "tekrar"
    loop = _TR_LONG + " Momentum etkisi kalıcıdır ve her rejimde sürer." * 6
    assert sd.content_gate(loop, "plain", None, system) == "tekrar"


def test_audit_gate_blocks_financial_direction() -> None:
    assert sd.content_gate(_TR_LONG + " Bu stratejide risk yok.", "plain", None, "") == "gate7"


def test_revalidate_line_uses_prompt_pairs_and_question_dependency() -> None:
    from app.brain.rag_answerer import build_rag_prompt

    chunks = [_Chunk("p1_c1", "p1", "a"), _Chunk("p2_c7", "p2", "b")]
    system, user = build_rag_prompt("Kalman nasıl işler?", chunks, reorder=False)

    def row(answer: str, q: str = "Kalman nasıl işler?") -> str:
        return json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {"mode": "rag", "question": q},
            },
            ensure_ascii=False,
        )

    assert sd.revalidate_line(row(_TR_LONG + " [p1:p1_c1] [p2:p2_c7]")) is None
    assert sd.revalidate_line(row(_TR_LONG + " [p1:p1_c1] [p1:p1_c9]")) == "uydurma-atıf"
    q = "Example 7.3.1'deki tahminler neden konservatif kabul edilir?"
    assert sd.revalidate_line(row(_TR_LONG + " [p1:p1_c1] [p2:p2_c7]", q)) == "bağımlı-soru"
    assert sd.question_dependent("Bu calismada onerilen yontem nedir?", "rag")


def test_dumps_escapes_line_separators() -> None:
    line = sd._dumps({"a": "x y\x85z"})
    assert len(line.splitlines()) == 1 and json.loads(line)["a"] == "x y\x85z"


def test_distill_one_survives_retrieval_error_and_records_provenance() -> None:
    def boom(q: str) -> list:
        raise PermissionError("cache busy")

    chat, _ = _fake_chat(_TR_LONG)
    job = sd.DistillJob("rag", "Kalman nasıl işler?")
    st, line, _ = sd.distill_one(job, chat=chat, retrieve=boom, seed=0, teacher="qwen3:b")
    assert st == "retrieval-hata" and line is None

    chunks = [_Chunk("p1_c1", "p1"), _Chunk("p2_c7", "p2")]
    good = _TR_LONG + " [p1:p1_c1] [p2:p2_c7]"
    chat2, _ = _fake_chat(good)
    st, line, _ = sd.distill_one(
        job,
        chat=chat2,
        retrieve=lambda q: chunks,
        seed=0,
        teacher="qwen3:b",
        provenance=lambda: {"query": "How does Kalman work?", "translation": "çevrildi"},
    )
    assert st == "kabul" and line is not None
    assert json.loads(line)["metadata"]["retrieval"]["translation"] == "çevrildi"


def test_run_distill_refuses_config_change(tmp_path) -> None:
    out = tmp_path / "d.jsonl"
    jobs = [sd.DistillJob("plain", "Kavram nasıl işler, neden?")] * 2
    chat, _ = _fake_chat(_TR_LONG)
    kw = {"chat": chat, "retrieve": lambda q: [], "teacher": "qwen3:base"}
    sd.run_distill(out, jobs, limit=1, config={"rag_version": "a"}, **kw)
    with pytest.raises(ValueError, match="yapılandırma"):
        sd.run_distill(out, jobs, config={"rag_version": "b"}, **kw)
    sd.run_distill(out, jobs, config={"rag_version": "a"}, **kw)


def test_used_questions_skips_own_output(tmp_path) -> None:
    def w(name: str, q: str) -> None:
        (tmp_path / name).write_text(
            json.dumps({"messages": [], "metadata": {"question": q}}) + "\n", encoding="utf-8"
        )

    w("distill_qa.jsonl", "Kalman nasil isler")
    w("distill_qa.r1.jsonl", "Momentum nedir")
    w("distill_qa.rejects.jsonl", "Red sorusu")
    used = sd.used_questions(tmp_path, skip=tmp_path / "distill_qa.jsonl")
    assert used == {sd.norm_question("Momentum nedir")}


def test_template_gate_exempts_prompt_heading_plus_mandated_line() -> None:
    """Kademe 2 F1-3: başlık satırı + zorunlu cümle ALT ALTA → aradaki 8-gram istemde yok;
    satır düzeyinde istem çıkarması olmadan kanonik v14 seti %14 ile NO-GO veriyordu."""
    import random

    from app.brain.rag_answerer import build_rag_prompt

    system, _ = build_rag_prompt("x", [])
    vocab = [f"kavram{chr(97 + a)}{chr(97 + b)}" for a in range(26) for b in range(26)]

    def body(i: int) -> str:
        return " ".join(random.Random(i).sample(vocab, 12))

    answer = (
        "1. Short Answer / Kısa Cevap\n{b}.\n"
        "6. Trading Hypothesis / Trading Hipotezi\n"
        "Bu bulgu doğrudan trading kuralına çevrilemez.\n"
        "7. Test Plan / Test Planı\n{b} plan."
    )
    lines = [_row(system, answer.format(b=body(i)), f"q{i}") for i in range(60)]
    assert not any("şablon tekrarı" in b for b in audit_dataset(lines).blockers)
    # İstemde OLMAYAN ortak içerik satırı hâlâ engellenir.
    memorized = "Bu yöntem her piyasada aynı sinyali üretir ve sonuç değişmez gibi görünür."
    lines2 = [_row(system, f"{body(i)}.\n{memorized}", f"q{i}") for i in range(60)]
    assert any("şablon tekrarı" in b for b in audit_dataset(lines2).blockers)


def test_thin_template_rows_drops_only_formulaic_distilled_rows() -> None:
    """v14: kalıplaşmış Test Planı cümlesi pretrain-gate şablon engelini aşıyordu; yalnız
    damıtma satırları, %1.8 sınırına kadar inceltilir (deterministik)."""
    import random

    vocab = [f"kavram{chr(97 + a)}{chr(97 + b)}" for a in range(26) for b in range(26)]
    formula = "Zaman dilimi dakikalık veri ile in sample ve out of sample ayrımı yapılır."

    def row(i: int, *, distilled: bool, with_formula: bool) -> str:
        body = " ".join(random.Random(i).sample(vocab, 12))
        ans = f"{body}.\n{formula}" if with_formula else f"{body}."
        meta = {"distilled": True} if distilled else {"discipline": True}
        return json.dumps(
            {"messages": [{"role": "assistant", "content": ans}], "metadata": meta},
            ensure_ascii=False,
        )

    lines = [row(i, distilled=True, with_formula=True) for i in range(10)]
    lines += [row(100 + i, distilled=False, with_formula=True) for i in range(5)]
    lines += [row(200 + i, distilled=True, with_formula=False) for i in range(285)]
    kept, dropped = sft_assembly.thin_template_rows(lines, share=0.018)
    cap = int(0.018 * len(lines))  # 5
    # İlk 5 damıtma satırı sınırı doldurur, sonraki 5'i atılır; disiplin satırları sınır
    # dolmuş olsa da asla atılmaz.
    assert dropped == 10 - cap
    assert kept[:cap] == lines[:cap]
    assert all(ln in kept for ln in lines[10:15])
    assert len([ln for ln in kept if "Zaman dilimi" in ln]) == cap + 5
    assert kept == sft_assembly.thin_template_rows(lines, share=0.018)[0]
