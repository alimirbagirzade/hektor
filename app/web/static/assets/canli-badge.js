// Ana arayüz header'ındaki canlılık rozeti. app.js'ten BAĞIMSIZ (çakışmasın).
// /api/status'u birkaç saniyede bir yoklar; sunucu ölünce KAPALI'ya döner.
(function () {
  "use strict";
  var POLL_MS = 4000;

  function el(id) {
    return document.getElementById(id);
  }
  function pad(n) {
    return String(n).padStart(2, "0");
  }
  function now() {
    var d = new Date();
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }
  function setState(cls, text, title) {
    var b = el("liveBadge");
    if (!b) return;
    b.className = "live-badge " + cls;
    b.title = title;
    var t = el("liveBadgeText");
    if (t) t.textContent = text;
  }

  function poll() {
    var ctrl = new AbortController();
    var timer = setTimeout(function () {
      ctrl.abort();
    }, 6000);
    fetch("/api/status", { cache: "no-store", signal: ctrl.signal })
      .then(function (r) {
        clearTimeout(timer);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (s) {
        var ollamaWarn = s && s.ollama_available === false ? " · ollama yok" : "";
        setState("live-ok", "CANLI", "Web sunucusu CANLI · son yoklama " + now() + ollamaWarn);
      })
      .catch(function () {
        clearTimeout(timer);
        setState("live-down", "KAPALI", "Web sunucusuna ulaşılamadı · son deneme " + now());
      });
  }

  function init() {
    poll();
    setInterval(poll, POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
