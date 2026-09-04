// ACHILLES — bağımsız canlılık izleme. Aynı-köken /api/status yoklar.
// CSP uyumlu (eval/inline yok). Sunucu ölünce fetch başarısız → KAPALI'ya döner.
"use strict";

const POLL_MS = 3000;
const $ = (id) => document.getElementById(id);

let failCount = 0;
let lastOkAt = null; // Date
const openedAt = Date.now();

function setState(name) {
  // name: "live" | "down" | "init"
  document.body.classList.remove("state-live", "state-down", "state-init");
  document.body.classList.add("state-" + name);
}

function fmtTime(d) {
  const p = (n) => String(n).padStart(2, "0");
  return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}

function fmtDur(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + "sn";
  const m = Math.floor(s / 60);
  if (m < 60) return m + "dk " + (s % 60) + "sn";
  const h = Math.floor(m / 60);
  return h + "sa " + (m % 60) + "dk";
}

function num(n) {
  return typeof n === "number" ? n.toLocaleString("tr-TR") : "—";
}

function setCell(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.classList.remove("ok", "bad");
  if (cls) el.classList.add(cls);
}

function onOk(data, latencyMs) {
  failCount = 0;
  lastOkAt = new Date();
  setState("live");
  $("heroLabel").textContent = "CANLI";
  $("heroDetail").textContent = "sunucu yanıt veriyor — http://127.0.0.1:8765";
  $("latency").textContent = latencyMs;

  const ollamaOk = !!data.ollama_available;
  setCell("mOllama", ollamaOk ? "açık" : "KAPALI", ollamaOk ? "ok" : "bad");
  setCell("mBackend", data.active_backend || data.llm_backend || "—");
  setCell("mModel", data.llm_model || "—");
  setCell("mEmbed", data.embedding_mode || "—");
  setCell("mPapers", num(data.n_papers));
  setCell("mChunks", num(data.n_chunks));
  setCell("mLastOk", fmtTime(lastOkAt));
  setCell("mFails", "0");

  $("lastErr").textContent = "";
  document.title = "🟢 CANLI · ACHILLES";
}

function onFail(err) {
  failCount += 1;
  setState("down");
  $("heroLabel").textContent = "BAĞLANTI YOK";
  const since = lastOkAt ? "son yanıt " + fmtTime(lastOkAt) : "hiç yanıt alınamadı";
  $("heroDetail").textContent = "sunucu yoklamaya cevap vermiyor (" + since + ")";
  $("latency").textContent = "—";
  setCell("mLastOk", lastOkAt ? fmtTime(lastOkAt) : "—");
  setCell("mFails", String(failCount), "bad");
  $("lastErr").textContent = "hata: " + (err && err.message ? err.message : err);
  document.title = "🔴 KAPALI · ACHILLES";
}

async function poll() {
  $("pollState").textContent = "yokluyor…";
  const t0 = performance.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 6000);
    const res = await fetch("/api/status", { cache: "no-store", signal: ctrl.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    const latency = Math.round(performance.now() - t0);
    onOk(data, latency);
  } catch (e) {
    onFail(e);
  } finally {
    $("pollState").textContent = "yenilendi " + fmtTime(new Date());
  }
}

function tickUptime() {
  $("uptime").textContent = fmtDur(Date.now() - openedAt);
}

function init() {
  $("intervalSec").textContent = String(Math.round(POLL_MS / 1000));
  $("refreshBtn").addEventListener("click", poll);
  setState("init");
  poll();
  setInterval(poll, POLL_MS);
  setInterval(tickUptime, 1000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
