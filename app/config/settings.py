"""Central configuration via pydantic-settings.

All settings can be overridden by environment variables prefixed with ``HEKTOR_``
or by a local ``.env`` file. Paths are resolved relative to the project root.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

# Project root = two levels up from this file (app/config/settings.py -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PREFIX = "HEKTOR_"
# Proje Achilles → Hektor olarak yeniden adlandırıldı. Mevcut makinelerdeki .env ve
# kabuk değişkenleri sessizce ETKİSİZ kalmasın diye eski önek okunmaya devam eder.
LEGACY_ENV_PREFIX = "ACHILLES_"

# Yeniden adlandırma öncesi/sonrası SQLite dosya adları (bkz. Settings.sqlite_file).
_DEFAULT_SQLITE_PATH = Path("storage/sqlite/hektor_trader_ai.db")
_LEGACY_SQLITE_PATH = Path("storage/sqlite/achilles_trader_ai.db")


def _promote_legacy_env(env_file: Path | None) -> list[str]:
    """Eski ``ACHILLES_*`` ayarlarını ``HEKTOR_*`` karşılığına taşı (yalnız boşsa).

    Hem süreç ortamını hem ``.env`` dosyasını tarar. Yeni önek zaten tanımlıysa ASLA
    ezilmez — açık ayar her zaman kazanır. Taşınan her anahtar bir kez uyarı loglar;
    böylece geçiş sessiz değil, görünür olur.

    ``env_file=None`` → dosya hiç okunmaz. Bu, ``Settings`` ``.env``'i devre dışı
    bıraktığında (test oturumu) bu kancanın onu ARKA KAPIDAN ``os.environ``'a
    taşımasını engeller; iki yol da aynı yapılandırmaya uyar.
    """
    legacy: dict[str, str] = {
        k: v for k, v in os.environ.items() if k.startswith(LEGACY_ENV_PREFIX)
    }
    if env_file is not None and env_file.is_file():
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key.startswith(LEGACY_ENV_PREFIX):
                    # Süreç ortamı .env'i ezer (pydantic-settings ile aynı öncelik).
                    legacy.setdefault(key, value.strip().strip("\"'"))
        except OSError:  # pragma: no cover - okunamayan .env ayarları engellememeli
            pass

    promoted: list[str] = []
    for key, value in sorted(legacy.items()):
        new_key = ENV_PREFIX + key[len(LEGACY_ENV_PREFIX) :]
        if os.environ.get(new_key) is None:
            os.environ[new_key] = value
            promoted.append(key)
    if promoted:
        log.warning(
            "Eski %s* ortam değişkenleri kullanılıyor (%s). Proje Hektor olarak "
            "yeniden adlandırıldı; lütfen %s* önekine geçin — eski önek desteği "
            "geçicidir.",
            LEGACY_ENV_PREFIX,
            ", ".join(promoted),
            ENV_PREFIX,
        )
    return promoted


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama (yerel — TEK LLM backend) ---
    # Pay-per-token bulut API'si (OpenAI/Anthropic/Google) bu projede KULLANILMAZ;
    # istemci kodu ve ayarları bilinçli olarak yoktur (kalıcı proje kısıtı).
    ollama_host: str = "http://127.0.0.1:11434"  # localhost yerine IP — Windows IPv6 sorununu önler
    llm_model: str = "qwen3:4b"
    # Modeli sorgu sonrası ne kadar yüklü tutsun. RAM darsa (ör. aynı anda LoRA eğitimi)
    # "0" → hemen boşalt (eğitimle ~7GB çakışmayı önler). Varsayılan "30s"; büyük
    # makinede ".env: HEKTOR_OLLAMA_KEEP_ALIVE=5m" hızlı ardışık sorgu için.
    ollama_keep_alive: str = "30s"
    embed_model: str = "nomic-embed-text"

    # mlx-lm LoRA eğitimi için HuggingFace model ID (Ollama formatı geçersiz)
    mlx_base_model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

    # PEFT (Windows/Linux) LoRA eğitimi için HuggingFace base model.
    # MLX 4-bit formatı transformers ile yüklenemez; bu yüzden ayrı HF model gerekir.
    # DİKKAT: Ollama'daki `qwen3:4b` tag'i Instruct-2507 checkpoint'idir (256K ctx);
    # adapter'ın Ollama'da çalışması için eğitim base'i BİREBİR aynı olmalı.
    peft_base_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    # --- Storage ---
    sqlite_path: Path = Field(default=_DEFAULT_SQLITE_PATH)
    chroma_path: Path = Field(default=Path("vector_db/chroma"))

    # --- RAG ---
    rag_top_k: int = 6
    chunk_size: int = 1200
    chunk_overlap: int = 200
    # Retrieval robustluğu (eğitimsiz kalite artışı — yazılı ama bağlanmamış
    # bileşenleri canlı yola alır). Hepsi LLM-free; çevrimdışı testlerle uyumlu.
    rag_rerank: bool = True  # over-fetch + heuristik reranker (semantik+kw+bölüm+formül)
    rag_overfetch: int = 4  # dense'ten top_k * overfetch aday çek, rerank et, top_k'ya kes
    # BM25 + dense hibrit (Faz A3): keyword adaylarını ekler. Korpus Chroma'dan lazy
    # kurulur; boşsa sessizce dense-only kalır → çevrimdışı testlerde davranış değişmez.
    rag_hybrid: bool = True
    # Cross-encoder reranker (Faz A8): en yüksek etkili sıralayıcı ama ağır (model
    # indirme + CPU latency). OPT-IN. Açmak için: HEKTOR_RAG_CROSS_ENCODER=true +
    # `uv pip install sentence-transformers`. Model yoksa heuristik reranker'a düşer.
    rag_cross_encoder: bool = False
    # Varsayılan hafif baz model (~280MB, ağırlıklı zh/en). Gerçek çok-dillilik (TR dahil
    # 100+ dil) için `BAAI/bge-reranker-v2-m3` önerilir (daha ağır ~2GB; modest CPU'da
    # latency artar). Modeli HEKTOR_RAG_CROSS_ENCODER_MODEL ile değiştir.
    rag_cross_encoder_model: str = "BAAI/bge-reranker-base"
    # FlashRank reranker (ONNX-int8 cross-encoder, torch GEREKMEZ): bge-reranker CPU'da
    # >15s/sorgu (kullanılamaz) iken FlashRank ~30-100ms (web-araştırma; bkz. roadmap Zincir 3).
    # OPT-IN, cross_encoder'a göre ÖNCELİKLİ. Açmak için HEKTOR_RAG_FLASHRANK=true +
    # `uv pip install flashrank`. Model yoksa heuristiğe düşer. A/B ile doğrulanmalı.
    rag_flashrank: bool = False
    rag_flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    # Reciprocal Rank Fusion (RRF) füzyon modu (opt-in): dense + BM25 sıralı listelerini
    # skor normalize etmeden sıra-tabanlı birleştirir (heuristik rerank yerine). Skor
    # kalibrasyonu gerektirmez → karşılaştırılamaz skorlu kaynaklarda sağlam. LLM-free,
    # deterministik. Varsayılan kapalı (alpha/rerank davranışı değişmez); açmak için
    # HEKTOR_RAG_RRF=true. RRF sabiti `rag_rrf_k` (yaygın varsayılan 60).
    rag_rrf: bool = False
    rag_rrf_k: int = 60
    # Graf-tabanlı retrieval (SPRIG-lite, opt-in): term–chunk bipartite graf üzerinde
    # dense-hit'lerden tohumlanmış Personalized PageRank ile çok-hop ilgili chunk'ları
    # yüzeye çıkarır; sonucu dense ile RRF ile füzyonlar. LLM-free, deterministik, CPU-only.
    # Dense'in kaçırdığı (paylaşılan terimle bağlı) chunk'ları getirebilir. Varsayılan kapalı
    # → mevcut retrieval davranışı değişmez. Açmak için HEKTOR_RAG_GRAPH=true.
    rag_graph: bool = False
    rag_graph_damping: float = 0.85  # PageRank yayılma katsayısı (1-damping = restart)
    rag_graph_iters: int = 20  # sabit iterasyon (determinizm)
    # Sorgu yönlendirici (opt-in, Zincir 2 — derin araştırma): sorgu-tipine göre yol.
    # KISA keyword/entity (ticker/kısaltma/sayı/tırnak) → konveks-füzyon hibrit (BM25 katkısı);
    # UZUN semantik → saf dense (ölçülen en iyi+hızlı). Hibridin yalnız işe yaradığı yerde
    # kullanılmasını sağlar (literatür: BM25 exact-match'te kazanır). Varsayılan kapalı.
    rag_router: bool = False
    rag_router_alpha: float = 0.7  # konveks füzyon dense ağırlığı (0.7-0.9 dense-favori önerilir)
    # Contextual Retrieval (Faz P2): chunk'ı embed etmeden önce "başlık / bölüm:" ön-eki
    # ekler (orijinal metin Chroma document'ında korunur). Tutarlılık için TÜM korpus
    # aynı ayarla embed edilmeli → açmadan önce `hektor reindex-contextual` çalıştır.
    # Varsayılan kapalı (yarı-prefix'li korpus tutarsızlık yaratırdı).
    rag_contextual_embed: bool = False

    # --- RLM Controller (Recursive/Reasoning LM orkestrasyonu) ---
    # Mevcut RAG retrieval + doğrulama modüllerini çok-adımlı, kaynaklı bir cevap
    # akışında orkestre eden kontrol katmanı (app/rlm). Yeni bilgi deposu değildir;
    # LLM-free skorlayıcılar + mevcut verifier'lar üzerine kuruludur (çevrimdışı uyumlu).
    rlm_max_retrieval_rounds: int = 3  # çok-turlu retrieval üst sınırı
    # Kanıt eşikleri (0-100). Değişmez: retry ≤ answer ≤ skip_retry — score()
    # içinde normalize edilir (çelişkili env/değer karar bandlarını bozmaz).
    rlm_min_evidence_to_retry: int = 40  # bu-üstü → tekrar retrieval; altı → yetersiz/abstain
    rlm_min_evidence_to_answer: int = 60  # bu-üstü → cevap (altı retry/yetersiz)
    rlm_min_evidence_to_skip_retry: int = 80  # bu-üstü → ek tur gereksiz
    rlm_enable_query_reformulation: bool = True  # yetersiz turda sorguyu bölüm-odaklı genişlet
    # NOT: iddia doğrulama + çelişki taraması KONFİG'LE KAPATILAMAZ (güvenlik, kural 4/7) —
    # bilerek ayar değil. Eskiden rlm_enable_claim_verification/_contradiction_check ölü
    # bayraklardı (hiç okunmuyordu = sahte-guard); kaldırıldı (doğrulama her zaman çalışır).
    rlm_allow_live_trading_signal: bool = False  # MUTLAK kural 1 — asla True olmaz (yalnız hipotez)
    rlm_seed: int = 42  # determinizm (kural 6) — tüm LLM çağrıları bu seed ile
    # Taslak LLM çağrısı SINIRLI olmalı: sınırsızken qwen3:4b CPU'da tek cevap >9dk sürüp
    # süreç kill'ine takılıyor ve run 'running' asılı kalıyordu (gerçek smoke'ta gözlendi).
    # 600s: CPU-only i7'de qwen3:4b draft+doğrulama ~5-6dk sürer; 300s sistematik timeout'a
    # düşürüyordu (tüm sorular no_llm). GPU varsa env ile düşürülebilir.
    rlm_draft_max_tokens: int = 900  # üretim uzunluğu tavanı (yapısal cevap için yeterli)
    rlm_draft_timeout_s: int = 600  # süre tavanı; aşılırsa LLMUnavailable → no_llm

    # --- RLM çalışma-zamanı ---
    # Motor TEK ve yereldir (RlmController + Ollama). Dış "recursive reasoning" motor
    # adapter'ı YOKTUR: API anahtarı + docker isterdi ve hiç koşturulmadı.
    rlm_production_mode: bool = True  # üretim: local-exec/shell/network/fs-write YASAK

    # --- Cevap-kalitesi (deterministik, LLM'siz; derin araştırma yol haritası) ---
    # "Lost in the middle" (arXiv 2307.03172): LLM'ler bağlamın ortasını unutur → en
    # alakalı chunk'ları başa/sona koy. Bedava, chunk EKLEMEZ/ÇIKARMAZ → varsayılan AÇIK.
    rag_reorder_context: bool = True
    # Citation-id doğrulama: cevaptaki [paper_id:chunk_id] atıfları getirilen kaynaklarda
    # yoksa UYDURMA uyarısı ekle (deterministik, LLM'siz; citation-forcing yine uydurabilir).
    # Yalnız dayanaksız atıf VARKEN uyarı ekler → düşük risk, varsayılan AÇIK (Kural 7).
    rag_verify_citations: bool = True
    # CRAG-lite güven kapısı: retrieval ZAYIFSA (alakasız/belirsiz) cevap üretmeden ABSTAIN
    # (Kural 7 — uydurma yok). Eşikler korpusa göre KALİBRE edilmeli → varsayılan KAPALI
    # (over-abstain riskini önlemek için opt-in; aç: HEKTOR_RAG_ABSTAIN=true).
    rag_abstain: bool = False
    # cosine benzerlik tabanı (1−distance); en iyi chunk bunun altındaysa alakasız sayılır.
    # KALİBRE EDİLDİ (2026-06-20, temiz ortam, 2 örnek): nomic benzerlikleri sıkışık —
    # alakalı sorgu en-iyi ~0.80, ALAKASIZ sorgu en-iyi ~0.51. Ayırıcı eşik ~0.55-0.6.
    # 0.55 → belirgin alakasızı yakalar, meşruyu geçirir. NOT: 2-örnek tahmini, golden-set
    # ile rafine et (abstain-oranı vs hata-oranı). Kapı zaten opt-in (rag_abstain=False).
    rag_abstain_min_similarity: float = 0.55
    rag_abstain_min_margin: float = 0.02  # top-1↔top-2 marjı bunun altında → belirsiz

    # --- Trading ---
    default_market: str = "XAUUSD"
    default_timeframe: str = "15m"

    # --- Behavior ---
    allow_fake_embeddings: bool = True
    log_level: str = "INFO"

    # --- Sentez aynalama (synthesis mirror) ---
    # Üretilen her sentez makalesi (reports/synthesis/sentez_*.md) bu dizine de
    # kopyalanır. Boşsa aynalama kapalıdır (varsayılan → test/CI davranışı değişmez).
    # Makineye özel yol .env içinde verilir: HEKTOR_SYNTHESIS_MIRROR_DIR=...
    synthesis_mirror_dir: str = ""

    # --- Literatür keşif ajanı (literature scout) ---
    # Bulunan makalelerin PDF'lerinin indirileceği "gelen kutusu" kökü. Boşsa repo-içi
    # `data/literature_inbox/` kullanılır (test/CI davranışı değişmez, Desktop'a yazmaz).
    # Makineye özel yol .env içinde: HEKTOR_SCOUT_INBOX_DIR=C:\...\Gerekli kaynaklar\_yeni
    scout_inbox_dir: str = ""

    # --- Auto-LoRA Pipeline ---
    # Gözetimsiz eğitim yetkisinin TEK anahtarı. Varsayılan KAPALI: her gerçek eğitim
    # ve terfi tek-kullanımlık taze insan onayı ister (CLAUDE.md Kural 8). Kullanıcı
    # bilinçli olarak açarsa (`.env`) kapılar geçtiğinde hat gözetimsiz ilerleyebilir.
    unattended_training_enabled: bool = False
    auto_lora_min_cards: int = 20  # eğitim başlamadan gereken minimum kart
    auto_lora_check_interval_min: int = 60  # kaç dakikada bir kontrol
    auto_lora_eval_threshold: float = 0.5  # eval pass_rate eşiği
    auto_lora_eval_sample_n: int = 8  # eval'de kaç soru (min_n altı 'accept' bloklanır; v5 dersi)

    # --- Web (FastAPI) ---
    # Güvenlik: varsayılan olarak SADECE localhost'a bağlanır (dışarı açılmaz).
    web_host: str = "127.0.0.1"
    web_port: int = 8765
    # Boşsa kimlik doğrulama kapalıdır (yalnız localhost güvenli sayılır).
    # Sunucuyu ağa açacaksan MUTLAKA güçlü bir token ata.
    api_token: str = ""
    # CORS: yalnız bu kökenlere izin verilir (frontend aynı origin'den sunulur).
    cors_origins: str = "http://127.0.0.1:8765,http://localhost:8765"
    # PDF upload üst sınırı (MB).
    max_upload_mb: int = 500
    # Basit hız sınırı: IP başına dakikadaki istek.
    rate_limit_per_min: int = 120
    # Yükleme uçlarına (PDF/CSV) ek, daha sıkı limit — ağ DoS / disk doldurma.
    # Toplu kütüphane içe-aktarımı için 60/dk (eski 20/dk normal sürükle-bırak'ta
    # bir kısım dosyayı 429'a düşürüyordu). Üst sınır aşılırsa frontend bekleyip
    # yeniden dener; gerekirse HEKTOR_UPLOAD_RATE_LIMIT_PER_MIN ile değiştir.
    upload_rate_limit_per_min: int = 60
    # Host-header saldırısı: boş = kısıt yok (lokal). Ağa açarken "alanadi.com,1.2.3.4" ver.
    trusted_hosts: str = ""
    # TLS (reverse proxy/HTTPS) arkasındaysan true → Strict-Transport-Security başlığı.
    hsts_enabled: bool = False

    # --- Kök dizin (veri/rapor/durum yollarının tabanı) ---
    # Boş → PROJECT_ROOT. Testler tmp'ye yönlendirir; böylece hiçbir test GERÇEK
    # data/ · storage/ · models/ ağacına yazamaz.
    root_path: Path | None = None

    # --- Arka plan döngüleri (web lifespan) ---
    # False → auto-lora / rag-loop / self-heal / unattended supervisor HİÇ başlamaz.
    # Testler bunu kapatır: aksi halde TestClient lifespan'i tetikleyip unattended
    # supervisor'ı çalıştırıyor ve makinede kurulu bir abonelik motorunu (codex/claude)
    # GERÇEKTEN doğurabiliyordu (kota + Kural 8 ihlali).
    background_loops_enabled: bool = True
    # Web'den PDF yüklenince ingest ardından OTOMATİK bilgi kartı + anlama skoru üret.
    # Varsayılan KAPALI: her yükleme 1-2 dk yerel LLM işi demektir (GPU'suz makinede
    # ~40 sn/çağrı) ve ingest/synth-qa ile aynı Ollama'yı paylaşır. Boş kart artık
    # kaydedilmediği için açmak güvenlidir; yalnız CPU maliyeti bilinçli seçilmeli.
    auto_card_on_upload: bool = False
    # Bilgi kartı üreticisine verilen makale metni tavanı (karakter). Prompt işleme yerel
    # CPU'da ~30 token/sn: 6000 krk ≈ 1700 token ≈ 55 sn; 3000 ≈ yarısı. Ölçüldü
    # (2026-09-06): retry'larla kart başına ~6 dk → 145 makale ~14 saat. Hız gerekince
    # 3000'e in (kart biraz daha yüzeysel olabilir); builder orta-kesit denemelerini de
    # bu tavana göre yapar.
    card_max_chars: int = 6000
    # Kart üreticisinin tek LLM çağrısı için zaman aşımı (sn). Ollama istekleri SIRAYA koyar
    # (varsayılan tek slot): synth-qa gibi başka bir iş koşarken kart isteği önce ~2-3 dk
    # kuyrukta bekler, sonra ~2 dk üretir. 180 sn'lik tavan bu durumda isteği ÜRETİM SÜRERKEN
    # düşürür → sunucu işi bitirir ama istemci `{}` alır, retry aynı kaderi paylaşır (ölçüldü
    # 2026-09-06: 10. kart 20+ dk boşa döndü). Paylaşımlı Ollama'da 420+ sn kullan.
    card_llm_timeout_s: int = 180

    # --- Derived dirs ---
    # TÜM veri/rapor/durum yolları `root`'tan türer. `root` env ile değiştirilebilir
    # (HEKTOR_ROOT_PATH) → test oturumu tmp'ye yönlendirir ve GERÇEK ağaca yazamaz.
    # Eskiden yollar modül-seviyesi PROJECT_ROOT'a sabitti; testler her tam koşuda
    # data/lora_sft/lora_sft.jsonl + data/training/jsonl/*.jsonl üretip bir sonraki
    # koşuyu (data-gate GO → train handoff) kirletiyordu.
    @property
    def root(self) -> Path:
        """VERİ kökü: data/ · reports/ · storage/ · models/ · logs/ buranın altındadır."""
        return Path(self.root_path).resolve() if self.root_path else PROJECT_ROOT

    @property
    def source_root(self) -> Path:
        """KAYNAK kökü — depo ağacı; `root` değişse bile SABİT kalır.

        Depoyla birlikte gelen okunur dosyalar buradadır: `automation_manifest.yaml`,
        `evals/*.jsonl`, `configs/`, `app/prompts/`. Bunlar veri değildir; testler
        veri kökünü tmp'ye alsa da bu dosyalar depo ağacından okunmalıdır.
        """
        return PROJECT_ROOT

    def _under_root(self, p: str | Path) -> Path:
        """Mutlak değilse `root` altına çöz."""
        path = Path(p)
        return path if path.is_absolute() else (self.root / path)

    @property
    def sqlite_file(self) -> Path:
        """SQLite dosyası — yeniden adlandırma öncesi veritabanını öksüz bırakmaz.

        Achilles → Hektor geçişinde varsayılan dosya adı ``achilles_trader_ai.db``'den
        ``hektor_trader_ai.db``'ye döndü. Mevcut kurulumda YALNIZ eski dosya varsa ona
        düşülür (yeni dosya oluşup korpus/kart geçmişi kaybolmuş gibi görünmesin diye).
        Açık ``HEKTOR_SQLITE_PATH`` ayarı her zaman kazanır.
        """
        path = self._under_root(self.sqlite_path)
        if path.exists() or self.sqlite_path != _DEFAULT_SQLITE_PATH:
            return path
        legacy = self._under_root(_LEGACY_SQLITE_PATH)
        if legacy.exists():
            log.warning(
                "Eski veritabanı kullanılıyor: %s. Hektor'a geçiş için dosyayı "
                "%s olarak yeniden adlandırabilirsiniz (WAL/SHM dosyalarıyla birlikte).",
                legacy,
                path.name,
            )
            return legacy
        return path

    @property
    def chroma_dir(self) -> Path:
        return self._under_root(self.chroma_path)

    @property
    def raw_pdf_dir(self) -> Path:
        return self.root / "data" / "papers" / "raw_pdf"

    @property
    def extracted_text_dir(self) -> Path:
        return self.root / "data" / "papers" / "extracted_text"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "data" / "papers" / "metadata"

    @property
    def jsonl_dir(self) -> Path:
        return self.root / "data" / "training" / "jsonl"

    @property
    def market_raw_dir(self) -> Path:
        return self.root / "data" / "market" / "raw"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def state_dir(self) -> Path:
        """Çalışma-zamanı durum dosyaları (`storage/`): ajan state JSON'ları, STOP_ALL…

        CWD'ye bağlı `Path("storage")` KULLANMA: süreç başka dizinden başlatılırsa
        (Windows servisi, detached eğitim) durum sessizce yanlış yere yazılır/okunur.
        """
        return self.root / "storage"

    @property
    def adapters_dir(self) -> Path:
        return self.root / "models" / "adapters"

    @property
    def agent_runs_dir(self) -> Path:
        """Agent runtime gözlemcisi (Phase 1) — koşu başına JSONL günlükleri."""
        return self.root / "reports" / "agent_runs"

    @property
    def prompts_dir(self) -> Path:
        return self.source_root / "app" / "prompts"

    @property
    def eval_sets_dir(self) -> Path:
        """Davranışsal eval setleri (`evals/*.jsonl`) — depoyla gelir, veri değildir."""
        return self.source_root / "evals"

    @property
    def manifest_file(self) -> Path:
        """Ajan manifesti — depoyla gelir, veri değildir."""
        return self.source_root / "automation_manifest.yaml"

    def ensure_dirs(self) -> None:
        """Create all runtime directories that must exist."""
        for d in (
            self.sqlite_file.parent,
            self.chroma_dir,
            self.raw_pdf_dir,
            self.extracted_text_dir,
            self.metadata_dir,
            self.jsonl_dir,
            self.market_raw_dir,
            self.adapters_dir,
            self.reports_dir / "papers",
            self.reports_dir / "training",
            self.reports_dir / "backtests",
            self.reports_dir / "evals",
            self.reports_dir / "agent_runs",
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    # Eski ACHILLES_* önekini Settings kurulmadan ÖNCE taşı; aksi halde
    # yeniden adlandırma sonrası mevcut .env sessizce yok sayılırdı. Hangi dosyanın
    # okunacağı Settings'in KENDİ yapılandırmasından gelir — böylece `.env` kapalıyken
    # (test oturumu) bu kanca da onu okumaz.
    env_file = Settings.model_config.get("env_file")
    _promote_legacy_env(Path.cwd() / str(env_file) if env_file else None)
    return Settings()


def configure_logging(level: str | None = None) -> None:
    settings = get_settings()
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
