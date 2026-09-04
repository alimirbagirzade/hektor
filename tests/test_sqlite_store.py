from sqlalchemy import text

from app.memory.sqlite_store import Base


def test_engine_opens_wal_and_busy_timeout(store):
    """Paylaşılan SQLite dosyasında eşzamanlı yazım için WAL + busy_timeout açık olmalı.

    SqliteStore aynı dosyayı OrchestrationStore/MasteryStore/RlmStore ile paylaşır;
    busy_timeout per-bağlantı olduğundan SqliteStore'un da her bağlantıda açması gerekir.
    """
    with store.engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        busy_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) == 30000


def test_schema_has_expected_tables(store):
    expected = {
        "papers",
        "chunks",
        "summaries",
        "knowledge_cards",
        "training_examples",
        "strategies",
        "backtests",
        "model_evaluations",
        "adapters",
    }
    assert expected.issubset(set(Base.metadata.tables))


def test_upsert_and_get_paper(store):
    store.upsert_paper(
        paper_id="paper_test1",
        file_hash="hashtest1",
        source_path="/tmp/x.pdf",
        title="T",
        year="2020",
    )
    got = store.get_paper_by_hash("hashtest1")
    assert got is not None and got.paper_id == "paper_test1"


def test_add_chunks(store):
    store.upsert_paper(paper_id="paper_c", file_hash="h_c", source_path="x")
    n = store.add_chunks(
        [
            {
                "chunk_id": "paper_c_c0000",
                "paper_id": "paper_c",
                "chunk_index": 0,
                "text": "hello",
                "char_count": 5,
                "token_estimate": 1,
                "embedded": 1,
            },
        ]
    )
    assert n == 1
    assert len(store.list_chunks("paper_c")) == 1


def test_delete_chunks_for_paper(store) -> None:
    # force re-index temizliği: bir makalenin tüm chunk'ları silinmeli, diğeri kalmalı.
    store.upsert_paper(paper_id="paper_d", file_hash="h_d", source_path="x")
    store.upsert_paper(paper_id="paper_e", file_hash="h_e", source_path="y")
    store.add_chunks(
        [
            {"chunk_id": f"paper_d_c{i:04d}", "paper_id": "paper_d", "chunk_index": i, "text": "d"}
            for i in range(3)
        ]
        + [{"chunk_id": "paper_e_c0000", "paper_id": "paper_e", "chunk_index": 0, "text": "e"}]
    )
    removed = store.delete_chunks_for_paper("paper_d")
    assert removed == 3
    assert store.list_chunks("paper_d") == []
    assert len(store.list_chunks("paper_e")) == 1  # diğer makale etkilenmedi


def test_has_embedded_chunks(store) -> None:
    # Yarım ingest tespiti (Kademe-2 av MEDIUM-2): paper satırı + embedded=0 chunk varsa
    # has_embedded_chunks False döner (Chroma'ya yazılmamış → onarım gerekir); başarılı
    # mark_chunks_embedded sonrası True.
    store.upsert_paper(paper_id="paper_emb", file_hash="h_emb", source_path="x")
    assert store.has_embedded_chunks("paper_emb") is False  # hiç chunk yok
    store.add_chunks(
        [
            {
                "chunk_id": "paper_emb_c0000",
                "paper_id": "paper_emb",
                "chunk_index": 0,
                "text": "abc",
                "embedded": 0,
            }
        ]
    )
    assert store.has_embedded_chunks("paper_emb") is False  # chunk var ama embedded=0 (yarım)
    store.mark_chunks_embedded(["paper_emb_c0000"])
    assert store.has_embedded_chunks("paper_emb") is True  # gömüldü → tam


def test_risk_reports_table_exists(store) -> None:
    assert "risk_reports" in Base.metadata.tables


def test_save_and_list_risk_reports(store) -> None:
    sample = {
        "strategy_name": "TestSMA",
        "n_trades": 42,
        "kelly": {
            "win_rate": 0.55,
            "avg_win": 0.02,
            "avg_loss": 0.01,
            "odds": 2.0,
            "full_kelly": 0.275,
            "half_kelly": 0.1375,
            "quarter_kelly": 0.06875,
            "capped_kelly": 0.25,
        },
        "drawdown_scale": {
            "current_drawdown_pct": -5.0,
            "max_allowed_pct": -20.0,
            "scale_factor": 1.0,
            "in_drawdown_zone": False,
        },
        "fixed_risk": {
            "equity": 10000.0,
            "risk_per_trade_pct": 1.0,
            "stop_distance_pct": 2.0,
            "position_size_pct": 50.0,
            "position_size_usd": 5000.0,
        },
        "warnings": [],
        "recommendation": "normal pozisyon",
    }
    store.save_risk_report("rr_bt_test1", "bt_test1", sample)
    rows = store.list_risk_reports()
    assert len(rows) >= 1
    r = next(x for x in rows if x["report_id"] == "rr_bt_test1")
    assert r["backtest_id"] == "bt_test1"
    assert r["strategy_name"] == "TestSMA"
    assert r["n_trades"] == 42
    assert abs(r["win_rate"] - 0.55) < 1e-6
    assert abs(r["half_kelly"] - 0.1375) < 1e-6
    assert abs(r["position_size_usd"] - 5000.0) < 1e-6


def test_save_risk_report_upsert(store) -> None:
    sample = {
        "strategy_name": "EMA",
        "n_trades": 10,
        "kelly": {
            "win_rate": 0.6,
            "avg_win": 0.03,
            "avg_loss": 0.02,
            "odds": 1.5,
            "full_kelly": 0.1,
            "half_kelly": 0.05,
            "quarter_kelly": 0.025,
            "capped_kelly": 0.1,
        },
        "drawdown_scale": {
            "current_drawdown_pct": 0.0,
            "max_allowed_pct": -20.0,
            "scale_factor": 1.0,
            "in_drawdown_zone": False,
        },
        "fixed_risk": {
            "equity": 5000.0,
            "risk_per_trade_pct": 1.0,
            "stop_distance_pct": 2.0,
            "position_size_pct": 30.0,
            "position_size_usd": 1500.0,
        },
        "warnings": [],
        "recommendation": "iyi",
    }
    store.save_risk_report("rr_bt_dup", "bt_dup", sample)
    store.save_risk_report("rr_bt_dup", "bt_dup", sample)  # upsert — no duplicate
    rows = store.list_risk_reports()
    assert sum(1 for r in rows if r["report_id"] == "rr_bt_dup") == 1


def test_get_risk_report_full_json(store) -> None:
    sample = {
        "strategy_name": "RSI",
        "n_trades": 5,
        "kelly": {
            "win_rate": 0.4,
            "avg_win": 0.01,
            "avg_loss": 0.01,
            "odds": 1.0,
            "full_kelly": 0.0,
            "half_kelly": 0.0,
            "quarter_kelly": 0.0,
            "capped_kelly": 0.0,
        },
        "drawdown_scale": {
            "current_drawdown_pct": -25.0,
            "max_allowed_pct": -20.0,
            "scale_factor": 0.5,
            "in_drawdown_zone": True,
        },
        "fixed_risk": {
            "equity": 10000.0,
            "risk_per_trade_pct": 1.0,
            "stop_distance_pct": 2.0,
            "position_size_pct": 25.0,
            "position_size_usd": 2500.0,
        },
        "warnings": ["DD yüksek"],
        "recommendation": "küçült",
    }
    store.save_risk_report("rr_bt_rsi", "bt_rsi", sample)
    full = store.get_risk_report("rr_bt_rsi")
    assert full is not None
    assert full["strategy_name"] == "RSI"
    assert full["drawdown_scale"]["in_drawdown_zone"] is True
    assert store.get_risk_report("rr_nonexistent") is None
