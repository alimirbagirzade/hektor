"""Auto-LoRA Pipeline — otomatik denetim + kullanıcı onaylı eğitim döngüsü.

Akış:
  1. [OTOMATİK] Her check_interval_min dakikada onaylı kart sayısını kontrol eder.
  2. MIN_ELIGIBLE_CARDS eşiğini geçince Gate 0-8 otomatik çalışır.
  3. [KULLANICI ONAYI] Eğitimi başlatmak için web UI onayı gerekir (CLAUDE.md kural 8).
  4. [OTOMATİK] Eğitim bitince eval çalıştırır, adapter'ı EVAL_PASSED olarak kaydeder.
  5. [KULLANICI ONAYI] Production'a terfi için web UI onayı gerekir.

State dosyası: storage/auto_lora_state.json
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.agents.runtime import log_step, tracked

log = logging.getLogger(__name__)


class PipelineStage(StrEnum):
    IDLE = "idle"
    CHECKING = "checking"
    GATE_FAILED = "gate_failed"
    READY_TO_TRAIN = "ready_to_train"  # kullanıcı onayı bekliyor
    TRAINING = "training"
    TRAIN_FAILED = "train_failed"
    EVALUATING = "evaluating"
    EVAL_FAILED = "eval_failed"
    EVAL_SKIPPED = "eval_skipped"  # eval seti yok — TERFİ EDİLEMEZ (Anayasa II/VI)
    EVAL_PASSED = "eval_passed"  # production onayı bekliyor
    PROMOTED = "promoted"


@dataclass
class AutoPipelineState:
    stage: PipelineStage = PipelineStage.IDLE
    last_check: str = ""
    last_gate_result: str = ""
    gate_summary: str = ""
    approved_cards_at_last_check: int = 0
    adapter_id: str = ""
    adapter_path: str = ""
    eval_scores: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    last_run: str = ""

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "last_check": self.last_check,
            "last_gate_result": self.last_gate_result,
            "gate_summary": self.gate_summary,
            "approved_cards_at_last_check": self.approved_cards_at_last_check,
            "adapter_id": self.adapter_id,
            "adapter_path": self.adapter_path,
            "eval_scores": self.eval_scores,
            "last_error": self.last_error,
            "last_run": self.last_run,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AutoPipelineState:
        s = cls()
        s.stage = PipelineStage(d.get("stage", PipelineStage.IDLE.value))
        s.last_check = d.get("last_check", "")
        s.last_gate_result = d.get("last_gate_result", "")
        s.gate_summary = d.get("gate_summary", "")
        s.approved_cards_at_last_check = d.get("approved_cards_at_last_check", 0)
        s.adapter_id = d.get("adapter_id", "")
        s.adapter_path = d.get("adapter_path", "")
        s.eval_scores = d.get("eval_scores", {})
        s.last_error = d.get("last_error", "")
        s.last_run = d.get("last_run", "")
        return s


def _state_file() -> Path:
    """Durum dosyası — CWD'ye değil `settings.state_dir`'e bağlı."""
    from app.config.settings import get_settings

    return get_settings().state_dir / "auto_lora_state.json"


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


class AutoLoRAPipeline:
    """Otomatik LoRA denetim ve eğitim döngüsü yöneticisi."""

    def __init__(
        self,
        min_eligible_cards: int = 20,
        check_interval_min: int = 60,
        eval_pass_threshold: float = 0.5,
        auto_enabled: bool = False,
        eval_sample_n: int = 8,
    ) -> None:
        self.min_eligible_cards = min_eligible_cards
        self.check_interval_min = check_interval_min
        self.eval_pass_threshold = eval_pass_threshold
        self.auto_enabled = auto_enabled
        # CPU'da 4B inference ağır — eval başına soru sayısını sınırla (None=tümü).
        self.eval_sample_n = eval_sample_n
        self._state = self._load_state()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ state
    # NOT: durum yolu CWD'ye DEĞİL `settings.state_dir`'e bağlıdır (bkz. _state_file).

    def _load_state(self) -> AutoPipelineState:
        path = _state_file()
        if path.exists():
            try:
                return AutoPipelineState.from_dict(json.loads(path.read_text()))
            except Exception:
                pass
        return AutoPipelineState()

    def _save_state(self) -> None:
        path = _state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state.to_dict(), indent=2))

    def get_status(self) -> dict:
        return self._state.to_dict() | {
            "auto_enabled": self.auto_enabled,
            "min_eligible_cards": self.min_eligible_cards,
            "check_interval_min": self.check_interval_min,
            "eval_pass_threshold": self.eval_pass_threshold,
        }

    def set_enabled(self, enabled: bool) -> None:
        self.auto_enabled = enabled

    # ------------------------------------------------------------------ gates

    @tracked("auto-lora-pipeline", trigger_type="background_loop")
    async def check_and_prepare(self) -> dict:
        """Gate 0-8 çalıştır. Geçerse READY_TO_TRAIN'e geçer (kullanıcı onayı bekler)."""
        async with self._lock:
            self._state.stage = PipelineStage.CHECKING
            self._state.last_check = _utcnow()
            self._state.last_error = ""
            self._save_state()

        try:
            from app.lora.control_plane import LoRAControlPlane
            from app.memory.sqlite_store import SqliteStore

            store = SqliteStore()
            n_approved = len(store.list_approved_cards())
            log_step(f"{n_approved} onaylı kart ile Gate 0-8 kontrolü")

            async with self._lock:
                self._state.approved_cards_at_last_check = n_approved

            if n_approved < self.min_eligible_cards:
                async with self._lock:
                    self._state.stage = PipelineStage.IDLE
                    self._state.gate_summary = (
                        f"Yetersiz kart: {n_approved}/{self.min_eligible_cards}"
                    )
                    self._save_state()
                return {"ok": False, "reason": self._state.gate_summary}

            log.info("Auto-LoRA: Gate 0-8 başlatılıyor (%d kart)", n_approved)

            # SqliteStore thread-affine'dir: kontrol düzlemini çalışacağı worker
            # thread'inin içinde kur. Event-loop thread'indeki `store`u taşıma.
            def _run_gates():
                return LoRAControlPlane().run_full()

            report = await asyncio.to_thread(_run_gates)

            async with self._lock:
                if report.passed:
                    self._state.stage = PipelineStage.READY_TO_TRAIN
                    self._state.last_gate_result = "passed"
                    self._state.gate_summary = (
                        f"{report.total_approved} kart onaylandı, "
                        f"{report.total_rejected} reddedildi"
                    )
                    log.info("Auto-LoRA: Gate geçti → kullanıcı onayı bekleniyor")
                else:
                    failed = [s.name for s in report.stages if not s.passed]
                    self._state.stage = PipelineStage.GATE_FAILED
                    self._state.last_gate_result = "failed"
                    self._state.gate_summary = f"Başarısız gate'ler: {failed}"
                    log.warning("Auto-LoRA: Gate başarısız: %s", failed)
                self._save_state()

            return {"ok": report.passed, "summary": self._state.gate_summary}

        except Exception as exc:
            async with self._lock:
                self._state.stage = PipelineStage.IDLE
                self._state.last_error = str(exc)
                self._save_state()
            log.exception("Auto-LoRA: check_and_prepare hatası")
            return {"ok": False, "reason": str(exc)}

    # ------------------------------------------------------------------ train

    async def start_training(self, adapter_name: str, iters: int = 300) -> dict:
        """Kullanıcı onayıyla eğitimi DETACHED başlatır. READY_TO_TRAIN gerekir.

        Eskiden in-process `training_manager` kullanıyordu → web çökünce/yeniden
        başlayınca eğitim ölüyordu (state'teki "Eğitim COMPLETED olmadı" bundandı).
        Artık manuel buton ile aynı detached yolu kullanır (clobber-proof split dahil).
        """
        async with self._lock:
            if self._state.stage != PipelineStage.READY_TO_TRAIN:
                return {
                    "ok": False,
                    "reason": (
                        f"Beklenen durum READY_TO_TRAIN, mevcut: {self._state.stage}. "
                        "Önce 'Gate Kontrolü Başlat' çalıştırın."
                    ),
                }

        from app.training.unattended_policy import authorize_training_action

        decision = authorize_training_action(
            "auto_lora_start_training",
            f"Auto-LoRA gerçek eğitimi: {adapter_name} ({iters} adım)",
            gates_passed=True,
        )
        if decision.mode == "stop_all":
            return {"ok": False, "reason": decision.reason, "blocked_by": "stop_all"}
        if not decision.authorized:
            return {
                "ok": False,
                "needs_approval": True,
                "approval_id": decision.approval_id,
                "reason": (
                    "Gerçek eğitim TAZE onay gerektirir. Onayla: "
                    f"achilles approval-approve {decision.approval_id}, sonra tekrar başlat."
                ),
            }

        from app.training.detached_launch import launch

        res = await asyncio.to_thread(launch, adapter_name, iters)
        if not res.get("ok"):
            return {"ok": False, "reason": res.get("message", "Eğitim başlatılamadı")}

        async with self._lock:
            self._state.stage = PipelineStage.TRAINING
            self._state.adapter_path = f"models/adapters/{adapter_name}"
            self._state.last_run = _utcnow()
            self._state.last_error = ""
            self._save_state()

        task = asyncio.create_task(self._watch_training(adapter_name))
        task.add_done_callback(lambda _t: None)
        return {"ok": True, "adapter_name": adapter_name}

    async def _watch_training(self, adapter_name: str) -> None:
        """Detached eğitimi yokla; bitince (adapter çıktısı varsa) eval çalıştır."""
        from app.config.settings import get_settings
        from app.training.detached_launch import training_status

        settings = get_settings()
        try:
            # 1) Sürecin başlamasını bekle (ilk log birkaç sn; en çok ~3 dk).
            started = False
            for _ in range(36):
                await asyncio.sleep(5)
                if training_status().get("running"):
                    started = True
                    break

            # 2) Bitene kadar periyodik yokla (detached; manager pipe'ı yok).
            if started:
                while training_status().get("running"):
                    await asyncio.sleep(30)

            # 3) Sonuç: adapter çıktısı oluştuysa tamam → eval; yoksa başarısız.
            adapter_dir = settings.adapters_dir / adapter_name
            if adapter_dir.exists() and any(adapter_dir.iterdir()):
                log.info("Auto-LoRA: Eğitim tamamlandı (detached) → eval başlatılıyor")
                await self._run_eval(adapter_name)
            else:
                async with self._lock:
                    self._state.stage = PipelineStage.TRAIN_FAILED
                    self._state.last_error = "Eğitim tamamlanmadı (adapter çıktısı yok)"
                    self._save_state()
        except Exception as exc:
            async with self._lock:
                self._state.stage = PipelineStage.TRAIN_FAILED
                self._state.last_error = str(exc)
                self._save_state()
            log.exception("Auto-LoRA: _watch_training hatası")

    # ------------------------------------------------------------------ eval

    async def _run_eval(self, adapter_name: str) -> None:
        async with self._lock:
            self._state.stage = PipelineStage.EVALUATING
            self._save_state()

        try:
            from app.config.settings import get_settings

            # Eval setleri depo ağacındadır; CWD'ye bağlı `Path("evals")` detached
            # koşuda yanlış dizine bakıp sessizce EVAL_SKIPPED üretiyordu.
            eval_dir = get_settings().eval_sets_dir
            eval_sets = list(eval_dir.glob("*.jsonl")) if eval_dir.exists() else []

            if not eval_sets:
                # Anayasa II/VI: test edilmeden "geçti" deme. Eval seti yoksa
                # EVAL_PASSED VARSAYMA — EVAL_SKIPPED yap (promote_to_production
                # yalnız EVAL_PASSED'i terfi ettirir, bu durum terfiyi bloklar).
                log.warning("Auto-LoRA: eval seti yok → EVAL_SKIPPED (terfi edilemez)")
                async with self._lock:
                    self._state.stage = PipelineStage.EVAL_SKIPPED
                    self._save_state()
                await self._register_adapter(adapter_name)
                return

            # GERÇEK adapter eval (Kural 2): base Ollama'yı değil, eğitilen PEFT
            # adapter'ını yükleyip base ile KARŞILAŞTIR. Eski ModelEvaluator base
            # modeli ölçüp körlemesine 'EVAL_PASSED' damgalıyordu → v5 adapter
            # regresyonunun kök sebebi (adapter base'den kötüydü ama terfi adayı oldu).
            from app.config import get_settings
            from app.training.adapter_eval import evaluate_adapter

            adapter_dir = get_settings().adapters_dir / adapter_name
            scores: dict[str, Any] = {}
            regression_any = False
            improved_any = False
            try:
                for es in eval_sets:
                    res = await asyncio.to_thread(
                        evaluate_adapter, adapter_dir, es, n=self.eval_sample_n
                    )
                    scores[es.stem] = res.to_dict()
                    regression_any = regression_any or res.regression
                    improved_any = improved_any or (res.verdict == "accept")
            except ImportError as exc:
                # torch/transformers/peft kurulu değil → adapter GERÇEKTEN ölçülemez.
                # Anayasa II/VI: ölçmeden 'geçti' deme → EVAL_SKIPPED (terfi edilemez).
                log.warning("Auto-LoRA: adapter eval bağımlılıkları yok (%s) → EVAL_SKIPPED", exc)
                async with self._lock:
                    self._state.stage = PipelineStage.EVAL_SKIPPED
                    self._state.last_error = f"Adapter eval bağımlılıkları eksik: {exc}"
                    self._save_state()
                await self._register_adapter(adapter_name)
                return

            # Terfi yalnız adapter base'den İYİ ise: hiç regresyon yok + en az bir gelişme
            # + en iyi eval skoru kullanıcı eşiğini (eval_pass_threshold) geçiyor.
            # BUG-M9 fix: eval_pass_threshold parametre tanımlı ama hiç kullanılmıyordu.
            best_score = max(
                (max(0.0, float(s.get("adapter_score", 0.0))) for s in scores.values()),
                default=0.0,
            )
            passed = improved_any and not regression_any and best_score >= self.eval_pass_threshold

            # Eval sonuçlarını kalıcı DB'ye kaydet
            try:
                from app.memory.sqlite_store import SqliteStore

                store = SqliteStore()
                for es in eval_sets:
                    r = scores.get(es.stem, {})
                    n_items = int(r.get("n", 0))
                    # adapter_score [0,1] dışına (negatif) çıkabilir (çok-bayraklı cevap);
                    # DB'ye anlamsız negatif pass_rate/passed_items yazmamak için kelepçele.
                    adapter_score = max(0.0, float(r.get("adapter_score", 0.0)))
                    store.save_eval_history(
                        adapter_name=adapter_name,
                        eval_set=es.stem,
                        pass_rate=adapter_score,
                        total_items=n_items,
                        passed_items=max(0, round(adapter_score * n_items)),
                    )
            except Exception:
                log.warning("Auto-LoRA: eval_history kaydedilemedi")

            # Anlama-merdiveni kıyası (v5-savunması dikişi — adapter vs base L3/L4).
            # PEFT bağımlılıkları yoksa veya adapter dizini yoksa sessizce atlanır.
            try:
                from app.lora.peft_llm_shim import PeftAdapterLLMShim
                from app.memory.sqlite_store import SqliteStore as _Store
                from app.verification.exams.understanding_score import score_full_ladder

                # Base bacağı adapter'ın KENDİ base'ini (adapter'sız) kullanmalı — llm=None
                # verseydik base=Ollama qwen3:4b olur ve 4B'yi 1.5B+adapter ile kıyaslardık
                # (elmayla-armut → 1.5B sayısal sınavlarda hep kaybeder → sahte-gerileme).
                # Bellek: CPU'da iki 1.5B modeli aynı anda tutmamak için base bacağını
                # yükle→koş→SERBEST BIRAK, sonra adapter bacağını yükle (adapter_eval deseni).
                base_shim = await asyncio.to_thread(PeftAdapterLLMShim.load_base_of, adapter_dir)
                adapter_ladder = None
                base_ladder = None
                if base_shim is not None:
                    _store = _Store()
                    base_ladder = await asyncio.to_thread(
                        score_full_ladder, 0, llm=base_shim, store=_store, use_sessions_l5=False
                    )
                    del base_shim  # ~1.5B modeli boşalt (adapter yüklenmeden önce)
                    shim = await asyncio.to_thread(PeftAdapterLLMShim.load, adapter_dir)
                    if shim is not None:
                        adapter_ladder = await asyncio.to_thread(
                            score_full_ladder, 0, llm=shim, store=_store, use_sessions_l5=False
                        )
                        del shim
                if base_ladder is not None and adapter_ladder is not None:
                    base_rate = base_ladder.pass_rate or 0.0
                    adapter_rate = adapter_ladder.pass_rate or 0.0
                    scores["_ladder"] = {
                        "base_pass_rate": round(base_rate, 4),
                        "adapter_pass_rate": round(adapter_rate, 4),
                    }
                    # Anlama regresyonu: adapter base'in %5'ten fazla altındaysa ek-red.
                    if base_rate > 0 and adapter_rate < base_rate - 0.05:
                        regression_any = True
                        passed = False
                        log.warning(
                            "Auto-LoRA: anlama-merdiveni GERİLEMESİ — base %.0f%% > adapter %.0f%%",
                            base_rate * 100,
                            adapter_rate * 100,
                        )
                    else:
                        log.info(
                            "Auto-LoRA: anlama-merdiveni OK — base %.0f%% / adapter %.0f%%",
                            base_rate * 100,
                            adapter_rate * 100,
                        )
            except Exception as _ladder_exc:
                log.debug("Auto-LoRA: anlama-merdiveni kıyası atlandı — %s", _ladder_exc)

            async with self._lock:
                self._state.eval_scores = scores
                if passed:
                    self._state.stage = PipelineStage.EVAL_PASSED
                    log.info("Auto-LoRA: Adapter base'den İYİ → EVAL_PASSED (terfi onayı bekliyor)")
                elif regression_any:
                    self._state.stage = PipelineStage.EVAL_FAILED
                    self._state.last_error = "Adapter base'e göre GERİLEDİ (regresyon) — terfi YOK"
                    log.warning("Auto-LoRA: Adapter regresyon → EVAL_FAILED")
                else:
                    # eşit/inconclusive → terfi etme ama adapter'ı SMOKE_PASSED olarak kaydet.
                    self._state.stage = PipelineStage.EVAL_SKIPPED
                    self._state.last_error = "Adapter base ile eşit (inconclusive) — terfi YOK"
                    log.info("Auto-LoRA: Adapter base ile eşit → EVAL_SKIPPED")
                self._save_state()

            if self._state.stage in (PipelineStage.EVAL_PASSED, PipelineStage.EVAL_SKIPPED):
                await self._register_adapter(adapter_name)

        except Exception as exc:
            async with self._lock:
                self._state.stage = PipelineStage.EVAL_FAILED
                self._state.last_error = str(exc)
                self._save_state()
            log.exception("Auto-LoRA: eval hatası")

    async def _register_adapter(self, adapter_name: str) -> None:
        try:
            from app.config.settings import get_settings
            from app.lora.adapter_registry import AdapterRecord, AdapterRegistry, AdapterStatus

            settings = get_settings()
            registry = AdapterRegistry()
            # evaluate_adapter to_dict() 'adapter_score' üretir (eski 'pass_rate' DEĞİL) ve
            # anahtar eval-set adına bağlı (sabit 'discipline_core' değil) → ilk geçerli skoru al.
            adapter_score = next(
                (
                    s["adapter_score"]
                    for s in self._state.eval_scores.values()
                    if isinstance(s, dict) and s.get("adapter_score") is not None
                ),
                None,
            )
            record = AdapterRecord(
                base_model=settings.peft_base_model,
                adapter_name=adapter_name,
                eval_score=adapter_score,
            )
            # Eval gerçekten geçtiyse EVAL_PASSED; eval atlandıysa yalnız SMOKE_PASSED
            # (registry de yanlış "eval geçti" damgası vurmasın — Anayasa II/VI).
            record.status = (
                AdapterStatus.EVAL_PASSED
                if self._state.stage == PipelineStage.EVAL_PASSED
                else AdapterStatus.SMOKE_PASSED
            )
            adapter_id = await asyncio.to_thread(registry.register, record)
            async with self._lock:
                self._state.adapter_id = adapter_id
                self._save_state()
            log.info("Auto-LoRA: Adapter kaydedildi: %s", adapter_id)
        except Exception as exc:
            log.exception("Auto-LoRA: adapter kayıt hatası")
            async with self._lock:
                self._state.last_error = str(exc)
                self._save_state()

    # ------------------------------------------------------------------ promote

    async def promote_to_production(self) -> dict:
        """Kullanıcı onayıyla production'a terfi et."""
        async with self._lock:
            if self._state.stage != PipelineStage.EVAL_PASSED:
                return {
                    "ok": False,
                    "reason": f"Beklenen EVAL_PASSED, mevcut: {self._state.stage}",
                }
            adapter_id = self._state.adapter_id

        if not adapter_id:
            return {"ok": False, "reason": "Kayıtlı adapter ID bulunamadı"}

        from app.training.unattended_policy import authorize_training_action

        decision = authorize_training_action(
            "auto_lora_promote_adapter",
            f"Adapter production terfisi: {adapter_id}",
            gates_passed=True,
        )
        if decision.mode == "stop_all":
            return {"ok": False, "reason": decision.reason, "blocked_by": "stop_all"}
        if not decision.authorized:
            return {
                "ok": False,
                "needs_approval": True,
                "approval_id": decision.approval_id,
                "reason": (
                    "Terfi TAZE onay gerektirir. Onayla: "
                    f"achilles approval-approve {decision.approval_id}, sonra tekrar dene."
                ),
            }

        try:
            from app.lora.adapter_registry import AdapterRegistry

            registry = AdapterRegistry()
            ok = await asyncio.to_thread(registry.promote, adapter_id, True)
            if ok:
                async with self._lock:
                    self._state.stage = PipelineStage.PROMOTED
                    self._save_state()
                return {"ok": True, "adapter_id": adapter_id}
            return {"ok": False, "reason": "Registry terfi başarısız"}
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    # ------------------------------------------------------------------ reset

    async def reset(self) -> None:
        """Pipeline'ı IDLE'a sıfırla."""
        async with self._lock:
            self._state = AutoPipelineState()
            self._save_state()

    # ------------------------------------------------------------------ background loop

    async def background_loop(self) -> None:
        log.info(
            "Auto-LoRA arka plan döngüsü başladı (enabled=%s, interval=%d dk, min_cards=%d)",
            self.auto_enabled,
            self.check_interval_min,
            self.min_eligible_cards,
        )
        while True:
            if self.auto_enabled:
                if self._state.stage in (PipelineStage.IDLE, PipelineStage.GATE_FAILED):
                    log.info("Auto-LoRA: Periyodik kontrol başlatılıyor")
                    await self.check_and_prepare()
                if self._state.stage == PipelineStage.READY_TO_TRAIN:
                    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
                    await self.start_training(f"achilles_auto_{stamp}")
                elif self._state.stage == PipelineStage.EVAL_PASSED:
                    await self.promote_to_production()
            await asyncio.sleep(self.check_interval_min * 60)


# ---------- singleton ----------

_pipeline: AutoLoRAPipeline | None = None


def get_auto_pipeline() -> AutoLoRAPipeline:
    global _pipeline
    if _pipeline is None:
        from app.config.settings import get_settings

        s = get_settings()
        _pipeline = AutoLoRAPipeline(
            min_eligible_cards=getattr(s, "auto_lora_min_cards", 20),
            check_interval_min=getattr(s, "auto_lora_check_interval_min", 60),
            eval_pass_threshold=getattr(s, "auto_lora_eval_threshold", 0.5),
            eval_sample_n=getattr(s, "auto_lora_eval_sample_n", 8),
            auto_enabled=s.unattended_training_enabled,
        )
    return _pipeline
