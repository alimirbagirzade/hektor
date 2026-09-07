/* HEKTOR terminal — frontend logic (CSP-safe, harici dosya). */
(function () {
  "use strict";

  var TOKEN_KEY = "hektor_api_token";
  var MAX_UPLOAD_MB = 500; // backend varsayılanı (settings.max_upload_mb); /api/status'tan güncellenir
  // localStorage artifact ortamında engelli olabilir; güvenli sarmalayıcı:
  function getToken() {
    try {
      return window.localStorage.getItem(TOKEN_KEY) || "";
    } catch (e) {
      return window.__hektorToken || "";
    }
  }
  function setToken(t) {
    try {
      window.localStorage.setItem(TOKEN_KEY, t);
    } catch (e) {
      window.__hektorToken = t;
    }
  }

  function authHeaders(extra) {
    var h = extra || {};
    var t = getToken();
    if (t) h["Authorization"] = "Bearer " + t;
    return h;
  }

  // Ham HTTP kodlarını eylem-odaklı Türkçe mesajlara çevirir.
  function humanError(status, detail) {
    if (detail && typeof detail === "string" && !/^HTTP \d+$/i.test(detail)) {
      return detail; // backend zaten anlamlı bir mesaj verdiyse onu koru
    }
    switch (status) {
      case 400:
        return "Geçersiz istek — girdiyi kontrol edin.";
      case 401:
        return "Yetkisiz — SİSTEM sekmesinden API token girin.";
      case 403:
        return "Erişim reddedildi.";
      case 404:
        return "İstenen uç bulunamadı (sunucu güncel mi?).";
      case 413:
        return "Dosya çok büyük — boyut limitini aşıyor.";
      case 422:
        return "Girdi doğrulanamadı — alanları kontrol edin.";
      case 429:
        return "Çok hızlı istek — birkaç saniye bekleyip tekrar deneyin.";
      case 500:
      case 502:
      case 503:
        return "Sunucu/servis hatası — Ollama kapalı olabilir; SİSTEM sekmesinden durumu kontrol edin.";
      default:
        return "Beklenmeyen hata" + (status ? " (kod " + status + ")" : "") + ".";
    }
  }

  function api(path, opts) {
    opts = opts || {};
    opts.headers = authHeaders(opts.headers);
    return fetch("/api" + path, opts).then(
      function (r) {
        if (r.status === 401) {
          toast(humanError(401), true);
          throw new Error(humanError(401));
        }
        if (r.status === 429) {
          toast(humanError(429), true);
          throw new Error(humanError(429));
        }
        return r.json().then(
          function (body) {
            if (!r.ok) throw new Error(humanError(r.status, body && body.detail));
            return body;
          },
          function () {
            // gövde JSON değil (ör. ham hata sayfası)
            throw new Error(humanError(r.status));
          }
        );
      },
      function () {
        // ağ hatası — sunucuya hiç ulaşılamadı
        throw new Error(
          "Hektor backend'e ulaşılamadı. PowerShell: .\\scripts\\run-web-service.ps1"
        );
      }
    );
  }

  // ---------- toast ----------
  var toastEl = document.getElementById("toast");
  var toastTimer = null;
  function toast(msg, isErr) {
    toastEl.textContent = msg;
    toastEl.className = "toast" + (isErr ? " err" : "");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.className = "toast hidden";
    }, 3800);
  }

  function esc(s) {
    if (s === null || s === undefined) return "";
    // Tırnaklar da kaçırılır: data-* / title= gibi attribute-context'lerde kullanıcı/LLM
    // kaynaklı metin (arXiv sorgusu, formül açıklaması, strateji adı) attribute'tan
    // taşıp yeni attribute enjekte edemesin (stored attribute-injection savunması).
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ---------- tabs + gruplar (navigasyon) ----------
  // 11 sekme 5 mantıklı gruba toplanır. data-tab değerleri ve panel-<name>
  // ID'leri DEĞİŞMEZ; bu yalnız üst-navigasyon + görünürlük katmanıdır.
  var TAB_GROUPS = [
    { key: "kesfet", tabs: ["research", "rlm"] },
    { key: "kutuphane", tabs: ["papers"] },
    { key: "trader", tabs: ["trader", "backtest"] },
    { key: "egitim", tabs: ["review", "training", "eval", "orchestration", "feedback"] },
    { key: "izleme", tabs: ["learning", "agents", "sentinel", "about", "agentmap"] },
  ];
  var TAB_TO_GROUP = {};
  TAB_GROUPS.forEach(function (g) {
    g.tabs.forEach(function (name, i) {
      TAB_TO_GROUP[name] = { key: g.key, order: i };
    });
  });
  var GROUP_LABELS = {
    kesfet: "Keşfet & sor",
    kutuphane: "Kütüphane",
    trader: "Trader & backtest",
    egitim: "Eğitim hattı",
    izleme: "İzleme & sağlık",
  };
  var NEXT_STEPS = {
    research: "Sonuç gelmiyorsa önce Kütüphane'den makale ekle, sonra burada soru sor.",
    rlm: "Daha titiz, iddia-düzeyi kaynak-doğrulamalı cevap için bir koşuya tıkla.",
    papers: "PDF sürükle-bırak ya da arXiv'den çek; sonra Keşfet & sor'da soru sor.",
    trader: "Formül çıkar → agentic araştırma; öneriler otomatik backtest edilir.",
    backtest: "Sentetik veya CSV ile çalıştır; sonucu PASS/FAIL/INCONCLUSIVE oku.",
    review: "Bekleyen kartları onayla/reddet; onaylananlar Eğitim dataset'ine girer.",
    training: "(İleri) Önce Onay'da kart onayla, dataset oluştur, sonra eğit.",
    eval: "Adapter + eval seti seç; kural ihlali (uydurma/garanti kâr) taranır.",
    about: "Canlı durum, donanım profili ve disiplin kuralları burada.",
    learning: "RAG öğrenme döngüsü, loss eğrisi ve objektif anlama geçmişi.",
    agents: "Görünürlük + onay + STOP_ALL; yeni tehlikeli yetenek yok.",
    orchestration:
      "Tek tık başlat; salt-okuma aşamaları çalışır, insan kapısında durur. " +
      "Derin av + onay sonrası 'Sürdür' ile kaldığı yerden devam eder.",
    feedback:
      "Yanlış cevabın doğrusunu yaz → aday eğitim örneği. Onayla → export. " +
      "Eğitim otomatik başlamaz; aday yine veri kapısından geçer.",
    sentinel:
      "Tüm ajanların sağlığını tek bakışta gör. Uyarı/kritik varsa öneri " +
      "metnindeki adımı uygula; nöbetçi hiçbir şeyi kendisi durdurmaz.",
    agentmap:
      "Işıklı yol: hangi ajan hangisini devreye sokar. ⚡ ile ana ajan (claude -p) " +
      "otonom sürüşü başlatır; gerçek eğitim yine onay kapısında durur (Kural 8).",
  };
  function updateNextStep(name) {
    var box = document.getElementById("nextStep");
    var txt = document.getElementById("nextStepText");
    if (!box || !txt) return;
    var hint = NEXT_STEPS[name];
    if (!hint) {
      box.style.display = "none";
      return;
    }
    var info = TAB_TO_GROUP[name];
    var label = info ? GROUP_LABELS[info.key] : "";
    txt.innerHTML =
      "<strong>Şu an:</strong> " + esc(label) +
      " &middot; <strong>Sonraki adım:</strong> " + esc(hint);
    box.style.display = "flex";
  }
  var ACTIVE_TAB_KEY = "hektor_active_tab";
  var tabs = document.querySelectorAll(".tab");
  var groupBtns = document.querySelectorAll(".group-btn");

  function validTab(name) {
    return !!(name && TAB_TO_GROUP[name] && document.getElementById("panel-" + name));
  }

  function runTabLoader(name) {
    if (name === "papers") loadPapers();
    if (name === "trader") {
      loadTraderBrain();
      loadLoraAdapters();
    }
    if (name === "backtest") loadBacktestHistory();
    if (name === "training") loadTrainingStatus();
    if (name === "review") loadPendingCards();
    if (name === "eval") loadEvalSets();
    if (name === "about") loadSystemStatus();
    if (name === "agents") loadAgentsDashboard();
    if (name === "rlm") loadRlmDashboard();
    if (name === "orchestration") loadOrchestration();
    if (name === "feedback") loadFeedback();
    if (name === "sentinel") loadSentinel();
    if (name === "agentmap") loadAgentMap();
  }

  function showGroupTabs(groupKey) {
    tabs.forEach(function (t) {
      var info = TAB_TO_GROUP[t.getAttribute("data-tab")];
      if (info && info.key === groupKey) {
        t.style.display = "";
        t.style.order = String(info.order);
      } else {
        t.style.display = "none";
      }
    });
    groupBtns.forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-group") === groupKey);
    });
  }

  function setActiveTab(name, opts) {
    if (!validTab(name)) name = "research";
    opts = opts || {};
    showGroupTabs(TAB_TO_GROUP[name].key);
    tabs.forEach(function (t) {
      t.classList.toggle("active", t.getAttribute("data-tab") === name);
    });
    document.querySelectorAll(".panel").forEach(function (p) {
      p.classList.remove("active");
    });
    document.getElementById("panel-" + name).classList.add("active");
    if (opts.persist !== false) {
      try {
        window.localStorage.setItem(ACTIVE_TAB_KEY, name);
      } catch (e) {}
      var target = "#sekme=" + name;
      if (target !== window.location.hash) {
        try {
          window.history.replaceState(null, "", target);
        } catch (e) {
          window.location.hash = "sekme=" + name;
        }
      }
    }
    updateNextStep(name);
    if (opts.runLoader !== false) runTabLoader(name);
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      setActiveTab(tab.getAttribute("data-tab"));
    });
  });

  groupBtns.forEach(function (gb) {
    gb.addEventListener("click", function () {
      var groupKey = gb.getAttribute("data-group");
      var active = document.querySelector(".tab.active");
      var info = active ? TAB_TO_GROUP[active.getAttribute("data-tab")] : null;
      if (info && info.key === groupKey) {
        showGroupTabs(groupKey); // zaten bu gruptayız; yalnız şeridi göster
      } else {
        var grp = TAB_GROUPS.filter(function (g) {
          return g.key === groupKey;
        })[0];
        setActiveTab(grp.tabs[0]);
      }
    });
  });

  function tabFromHash() {
    var h = window.location.hash || "";
    var m = h.match(/sekme=([a-z_]+)/i);
    if (m && validTab(m[1])) return m[1];
    if (h.length > 1 && validTab(h.slice(1))) return h.slice(1);
    return null;
  }

  // ---------- gelişmiş/gözlem grubu görünürlüğü ----------
  // "İzleme & sağlık" (öğrenme/ajanlar/nöbetçi/sistem/ajan haritası) saf gözlem
  // panellerini içerir — varsayılan gizli, "⋯ gelişmiş" ile açılır. Tercih
  // localStorage'da saklanır. GİZLEME yalnız NAVİGASYON katmanıdır: paneller,
  // API uçları ve arka plan işleri (öğrenme döngüsü, nöbetçi, orkestrasyon)
  // her hâlükârda çalışmaya devam eder.
  var ADVANCED_GROUPS = ["izleme"];
  var ADVANCED_KEY = "hektor_advanced_on";
  var groupNav = document.getElementById("groupNav");
  var advancedToggle = document.getElementById("advancedToggle");

  function isAdvancedGroup(key) {
    return ADVANCED_GROUPS.indexOf(key) !== -1;
  }
  function applyAdvanced(on) {
    if (groupNav) groupNav.classList.toggle("show-advanced", on);
    if (advancedToggle)
      advancedToggle.setAttribute("aria-expanded", on ? "true" : "false");
    try {
      window.localStorage.setItem(ADVANCED_KEY, on ? "1" : "0");
    } catch (e) {}
  }
  if (advancedToggle) {
    advancedToggle.addEventListener("click", function () {
      var willOpen = !(groupNav && groupNav.classList.contains("show-advanced"));
      applyAdvanced(willOpen);
      // Kapatırken aktif sekme gizlenen grupta kaldıysa çekirdeğe dön (research);
      // yoksa görünmeyen grupta asılı kalır.
      if (!willOpen) {
        var active = document.querySelector(".tab.active");
        var g = active ? TAB_TO_GROUP[active.getAttribute("data-tab")] : null;
        if (g && isAdvancedGroup(g.key)) setActiveTab("research");
      }
    });
  }

  function restoreActiveTab() {
    var name = tabFromHash();
    if (!name) {
      try {
        var saved = window.localStorage.getItem(ACTIVE_TAB_KEY);
        if (validTab(saved)) name = saved;
      } catch (e) {}
    }
    if (!name) name = "research";
    // Gelişmiş toggle: saklı tercih VEYA restore edilen sekme gelişmiş gruptaysa aç
    // (aksi hâlde gizli grupta aktif sekmede kalıp toggle'ı kapalı göstermek çelişir).
    var savedAdvanced = false;
    try {
      savedAdvanced = window.localStorage.getItem(ADVANCED_KEY) === "1";
    } catch (e) {}
    var restoredGroup = TAB_TO_GROUP[name] ? TAB_TO_GROUP[name].key : null;
    applyAdvanced(savedAdvanced || isAdvancedGroup(restoredGroup));
    // 'research' varsayılan zaten aktif; ilk yüklemede onun loader'ı yok.
    setActiveTab(name, { runLoader: name !== "research" });
  }

  window.addEventListener("hashchange", function () {
    var name = tabFromHash();
    if (name) setActiveTab(name);
  });
  setTimeout(restoreActiveTab, 0);

  // ---------- durum çubuğu: ekstra göstergeleri aç/kapat ----------
  var statusMoreBtn = document.getElementById("statusMoreBtn");
  var statusExtra = document.getElementById("statusExtra");
  if (statusMoreBtn && statusExtra) {
    statusMoreBtn.addEventListener("click", function () {
      var collapsed = statusExtra.classList.toggle("collapsed");
      statusMoreBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      statusMoreBtn.textContent = collapsed ? "⋯ tümü" : "× kapat";
    });
  }

  // ---------- adapter listesi (global, status'dan yüklenir) ----------
  var _adapters = [];

  function populateAdapterSelects(adapters) {
    _adapters = adapters || [];
    ["adapterSelect", "evalAdapterSelect"].forEach(function (id) {
      var sel = document.getElementById(id);
      if (!sel) return;
      var current = sel.value;
      while (sel.options.length > 1) sel.remove(1); // ilk "Ollama" seçeneği kalsın
      _adapters.forEach(function (a) {
        var opt = document.createElement("option");
        opt.value = a.version;
        opt.textContent = a.version + " (" + a.base_model.split("/").pop() + ")";
        sel.appendChild(opt);
      });
      sel.value = current;
    });
  }

  // ---------- status ----------
  function refreshStatus() {
    var dot = document.getElementById("connDot");
    var txt = document.getElementById("connText");
    api("/status", { method: "GET" })
      .then(function (s) {
        dot.className = "dot " + (s.ollama_available ? "dot-ok" : "dot-warn");
        txt.className = s.ollama_available ? "conn-ok" : "conn-warn";
        txt.textContent = s.ollama_available ? "ollama bağlı" : "ollama yok (RAG sınırlı)";
        document.getElementById("embedMode").textContent = "gömme: " + s.embedding_mode;
        document.getElementById("paperCount").textContent = "makale: " + s.n_papers;
        if (s.max_upload_mb) {
          MAX_UPLOAD_MB = s.max_upload_mb;
          var pdfH = document.getElementById("pdfHint");
          var csvH = document.getElementById("csvHint");
          if (pdfH)
            pdfH.innerHTML =
              "yalnız .pdf · maks " +
              MAX_UPLOAD_MB +
              "&nbsp;MB · birden çok seçebilirsin · içerik doğrulanır";
          if (csvH)
            csvH.innerHTML =
              "kolonlar: time, open, high, low, close[, volume] · maks " +
              MAX_UPLOAD_MB +
              "&nbsp;MB · içerik doğrulanır";
        }
      })
      .catch(function () {
        dot.className = "dot dot-err";
        txt.className = "conn-err";
        txt.textContent = "sunucuya ulaşılamadı";
      });
    // adapter listesini de yükle
    api("/training/status", { method: "GET" })
      .then(function (d) { populateAdapterSelects(d.adapters || []); })
      .catch(function () {});
    // RAG ustalık % — kaç makaleyi içerik kartına dönüştürüp "anladı" + eğitim hazırlığı
    api("/rag-mastery", { method: "GET" })
      .then(function (m) {
        var el = document.getElementById("ragMastery");
        if (el) {
          var txt =
            "RAG anladı: %" + m.coverage_percent +
            " (" + m.papers_with_real + "/" + m.n_papers + ")";
          if (m.comprehension_percent != null) {
            // Kaba öz-değerlendirme — objektif değil; objektif skor için "obj. anlama" rozeti.
            txt += " · öz-değ. %" + m.comprehension_percent +
                   " (" + m.papers_scored + " makale)";
          }
          txt += " · eğitim %" + m.train_readiness_percent;
          el.textContent = txt;
        }
      })
      .catch(function () {});
    // Son KALICI objektif anlama skorunu rozette pasif göster (DB okuması, LLM çağırmaz)
    api("/understanding-score/history?limit=1", { method: "GET" })
      .then(function (h) {
        var el = document.getElementById("objUnderstanding");
        var r = h && h.history && h.history[0];
        if (!el || !r || r.pass_rate == null) return;
        el.textContent = "obj. anlama %" + Math.round(r.pass_rate * 100);
        el.title =
          "Son KALICI sınav skoru (" + String(r.created_at || "").slice(0, 10) +
          ") — tıkla: tazele + kaydet\nSeviyeler: " + levelBreakdown(r.by_level);
      })
      .catch(function () {});
    // Sürüm/sapma rozeti — bu makine GitHub main ile güncel mi? (sessiz drift'i görünür kıl)
    api("/version", { method: "GET" })
      .then(function (v) {
        var el = document.getElementById("versionBadge");
        if (!el) return;
        var cmd = "update.ps1 -Force (Win) · ./update.sh --force (mac/Linux)";
        if (!v.git) {
          el.className = "muted";
          el.textContent = "sürüm: git yok";
          el.title = "git deposu bulunamadı";
        } else if (v.converged) {
          el.className = "conn-ok";
          el.textContent = "sürüm: güncel ✓";
          el.title = "main · " + (v.head || "") + " — GitHub origin/main ile aynı" +
            (v.last_update ? "\nson güncelleme: " + v.last_update : "");
        } else if (!v.on_main) {
          el.className = "conn-warn";
          el.textContent = "sürüm: ⚠ dal " + (v.branch || "?") + " (main değil)";
          el.title = "Bu makine 'main' dalında değil → güncellemeler oturmaz.\nÇözüm: " + cmd;
        } else if (v.behind > 0) {
          el.className = "conn-warn";
          el.textContent = "sürüm: ⚠ " + v.behind + " commit geride — güncelle";
          el.title = "Bu makine GitHub main'in " + v.behind +
            " commit gerisinde.\nÇözüm: " + cmd +
            (v.last_update ? "\nson güncelleme: " + v.last_update : "");
        } else {
          el.className = "muted";
          el.textContent = "sürüm: main +" + v.ahead + " (yerel)";
          el.title = "Yerel main, origin/main'in " + v.ahead +
            " commit önünde (henüz push edilmemiş).";
        }
      })
      .catch(function () {});
  }

  // ---------- ask (RAG) ----------
  var askForm = document.getElementById("askForm");
  askForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = document.getElementById("question").value.trim();
    var topk = parseInt(document.getElementById("topk").value, 10) || null;
    if (q.length < 3) {
      toast("Soru çok kısa.", true);
      return;
    }
    var btn = document.getElementById("askBtn");
    var res = document.getElementById("askResult");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>SORGULANIYOR';
    res.className = "result";
    res.innerHTML =
      '<div class="result-section"><span class="spinner"></span> indeksten getiriliyor…</div>';

    var adapterSel = document.getElementById("adapterSelect");
    var adapterVersion = adapterSel ? (adapterSel.value || null) : null;
    api("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, top_k: topk, adapter_version: adapterVersion }),
    })
      .then(function (data) {
        renderAsk(res, data);
      })
      .catch(function (err) {
        res.innerHTML =
          '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
      })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "SORGULA →";
      });
  });

  // ---------- LoRA sohbet (eğitilen adapter ile lokal, PEFT) ----------
  function loadLoraAdapters() {
    var sel = document.getElementById("loraChatAdapter");
    if (!sel) return;
    api("/lora-adapters")
      .then(function (data) {
        var cur = sel.value;
        var adapters = data.adapters || [];
        var opts = '<option value="">(base model — eğitimsiz, 4B)</option>';
        adapters.forEach(function (a) {
          opts += '<option value="' + esc(a) + '">' + esc(a) + "</option>";
        });
        sel.innerHTML = opts;
        // Varsayılan: EĞİTİLMİŞ 1.5B adapter (kullanıcı kazara base=4B'ye düşmesin).
        if (cur) {
          sel.value = cur;
        } else if (adapters.indexOf("hektor_lora_qwen15b") >= 0) {
          sel.value = "hektor_lora_qwen15b";
        } else if (adapters.length) {
          sel.value = adapters[0];
        }
      })
      .catch(function () {});
  }

  var loraChatForm = document.getElementById("loraChatForm");
  if (loraChatForm) {
    loraChatForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var q = document.getElementById("loraChatQuestion").value.trim();
      if (q.length < 2) {
        toast("Soru çok kısa.", true);
        return;
      }
      var adapter = document.getElementById("loraChatAdapter").value || null;
      var maxTokens =
        parseInt(document.getElementById("loraChatMaxTokens").value, 10) || 256;
      var btn = document.getElementById("loraChatBtn");
      var res = document.getElementById("loraChatResult");
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>ÜRETİLİYOR';
      res.className = "result";
      res.innerHTML =
        '<div class="result-section"><span class="spinner"></span> model çalışıyor (CPU\'da dakikalar sürebilir)…</div>';
      api("/lora-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, adapter: adapter, max_tokens: maxTokens }),
      })
        .then(function (data) {
          var badge =
            '<span class="badge badge-adapter">' +
            esc(data.adapter) +
            '</span> <span class="muted small">base: ' +
            esc(data.base_model) +
            "</span>";
          res.innerHTML =
            '<div class="result-section">' +
            badge +
            '</div><div class="result-section result-body">' +
            esc(data.answer).replace(/\n/g, "<br>") +
            "</div>";
        })
        .catch(function (err) {
          res.innerHTML =
            '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          btn.disabled = false;
          btn.textContent = "💬 SOR →";
        });
    });
  }

  function renderAsk(res, data) {
    var badge = data.adapter_used
      ? '<span class="badge badge-adapter">LoRA: ' + esc(data.adapter_used) + "</span>"
      : data.llm_used
      ? '<span class="badge badge-llm">LLM cevabı</span>'
      : '<span class="badge badge-rag">yalnız kaynaklar (LLM yok)</span>';

    var srcHtml = "";
    if (data.sources && data.sources.length) {
      srcHtml = data.sources
        .map(function (s) {
          var page = s.page ? ", s." + s.page : "";
          var dist = s.distance != null ? "d=" + s.distance.toFixed(3) : "";
          return (
            '<div class="source-chip"><span class="cite">[' +
            esc(s.paper_id) +
            ":" +
            esc(s.chunk_id) +
            page +
            "]" +
            (s.title ? " — " + esc(s.title) : "") +
            '</span><span class="dist">' +
            dist +
            "</span></div>"
          );
        })
        .join("");
    } else {
      srcHtml = '<div class="muted small">Kaynak bulunamadı. Önce MAKALELER sekmesinden PDF ekle.</div>';
    }

    res.innerHTML =
      '<div class="result-section"><div class="result-label">durum</div>' +
      badge +
      ' <span class="muted small">gömme: ' +
      esc(data.embedding_mode) +
      "</span></div>" +
      '<div class="result-section"><div class="result-label">cevap</div>' +
      '<div class="result-body">' +
      esc(data.answer) +
      "</div></div>" +
      '<div class="result-section"><div class="result-label">kaynaklar</div>' +
      '<div class="sources">' +
      srcHtml +
      "</div></div>";
  }

  // ---------- arXiv ----------
  var arxivSearchBtn = document.getElementById("arxivSearchBtn");
  var arxivFetchBtn  = document.getElementById("arxivFetchBtn");

  function getArxivParams() {
    var q = (document.getElementById("arxivQuery").value || "").trim();
    var max = parseInt(document.getElementById("arxivMax").value, 10) || 5;
    return { q: q, max: Math.min(Math.max(max, 1), 20) };
  }

  if (arxivSearchBtn) {
    arxivSearchBtn.addEventListener("click", function () {
      var p = getArxivParams();
      if (p.q.length < 3) { toast("Sorgu çok kısa.", true); return; }
      var res = document.getElementById("arxivResult");
      arxivSearchBtn.disabled = true;
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> arXiv aranıyor…</div>';
      api("/arxiv/search?q=" + encodeURIComponent(p.q) + "&max_results=" + p.max, { method: "GET" })
        .then(function (d) {
          if (!d.results || !d.results.length) {
            res.innerHTML = '<div class="result-section muted">Sonuç bulunamadı.</div>';
            return;
          }
          var rows = d.results.map(function (e) {
            return (
              '<div class="arxiv-row">' +
              '<div class="arxiv-title">' + esc(e.title) + '</div>' +
              '<div class="muted small">' + esc(e.arxiv_id) + ' · ' + esc(e.published) +
              (e.authors.length ? ' · ' + esc(e.authors.slice(0, 2).join(", ")) + (e.authors.length > 2 ? ' et al.' : '') : '') +
              '</div>' +
              '<div class="arxiv-abstract muted small">' + esc((e.abstract || "").slice(0, 180)) + '…</div>' +
              '</div>'
            );
          }).join("");
          res.innerHTML =
            '<div class="result-section"><div class="result-label">' + d.total + ' sonuç — "' + esc(d.query) + '"</div>' +
            rows + '</div>';
        })
        .catch(function (err) {
          res.innerHTML = '<div class="result-section">Hata: ' + esc(err.message) + '</div>';
        })
        .finally(function () { arxivSearchBtn.disabled = false; });
    });
  }

  if (arxivFetchBtn) {
    arxivFetchBtn.addEventListener("click", function () {
      var p = getArxivParams();
      if (p.q.length < 3) { toast("Sorgu çok kısa.", true); return; }
      var res = document.getElementById("arxivResult");
      arxivFetchBtn.disabled = true;
      arxivFetchBtn.innerHTML = '<span class="spinner"></span>İNDİRİLİYOR';
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> PDF\'ler indiriliyor + indeksleniyor…</div>';
      api("/arxiv/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: p.q, max_results: p.max, auto_ingest: true }),
      })
        .then(function (d) {
          var rows = (d.results || []).map(function (r) {
            var badge = r.skipped
              ? '<span class="badge badge-warn">zaten var</span>'
              : '<span class="badge badge-success">indirildi</span>';
            return '<div class="arxiv-row">' + badge + ' ' + esc(r.arxiv_id) + ' — ' + esc(r.title) + '</div>';
          }).join("");
          res.innerHTML =
            '<div class="result-section"><div class="result-label">' + esc(d.message) + '</div>' +
            rows + '</div>';
          if (d.fetched > 0 || d.ingested > 0) {
            toast("arXiv: " + d.fetched + " indirildi, " + d.ingested + " indekslendi ✓");
            loadPapers(); // makale listesini yenile
          }
        })
        .catch(function (err) {
          res.innerHTML = '<div class="result-section">Hata: ' + esc(err.message) + '</div>';
        })
        .finally(function () {
          arxivFetchBtn.disabled = false;
          arxivFetchBtn.textContent = "⬇ İNDİR + İNDEKSLE";
        });
    });
  }

  // ---------- arXiv kayıtlı sorgular ----------
  function loadArxivQueries() {
    var el = document.getElementById("arxivQueriesList");
    if (!el) return;
    el.innerHTML = '<span class="spinner"></span>';
    api("/arxiv/queries", { method: "GET" })
      .then(function (d) {
        if (!d.queries || !d.queries.length) {
          el.innerHTML = '<p class="muted small">Henüz kayıtlı sorgu yok. Bir sorgu gir ve 📌 Sorguyu Kaydet\'e tıkla.</p>';
          return;
        }
        el.innerHTML = d.queries.map(function (q) {
          return '<div class="bt-card" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
            '<span style="flex:1"><b>' + esc(q.query) + '</b>' +
            '<span class="muted small"> · maks ' + q.max_results + ' · ' + q.run_count + ' çalıştırma' +
            (q.last_run_at ? ' · ' + new Date(q.last_run_at).toLocaleString("tr-TR") : '') + '</span></span>' +
            '<button class="btn btn-primary btn-run-query" data-qid="' + esc(q.query_id) + '" data-query="' + esc(q.query) + '">▶ Çalıştır</button>' +
            '<button class="btn btn-del-query" data-qid="' + esc(q.query_id) + '" style="color:#e55">✕</button>' +
            '</div>';
        }).join("");
        el.querySelectorAll(".btn-run-query").forEach(function (b) {
          b.addEventListener("click", function () { runArxivQuery(b.getAttribute("data-qid"), b.getAttribute("data-query"), b); });
        });
        el.querySelectorAll(".btn-del-query").forEach(function (b) {
          b.addEventListener("click", function () { deleteArxivQuery(b.getAttribute("data-qid")); });
        });
      })
      .catch(function () { el.innerHTML = '<p class="muted small">Sorgular yüklenemedi.</p>'; });
  }

  function runArxivQuery(qid, queryText, btn) {
    if (btn) { btn.disabled = true; btn.textContent = "…"; }
    api("/arxiv/queries/" + qid + "/run", { method: "POST" })
      .then(function (d) { toast("✓ " + d.message); loadArxivQueries(); })
      .catch(function (e) { toast("Hata: " + esc(e.message), true); })
      .finally(function () { if (btn) { btn.disabled = false; btn.textContent = "▶ Çalıştır"; } });
  }

  function deleteArxivQuery(qid) {
    api("/arxiv/queries/" + qid, { method: "DELETE" })
      .then(function () { loadArxivQueries(); })
      .catch(function (e) { toast("Silinemedi: " + esc(e.message), true); });
  }

  var arxivSaveBtn = document.getElementById("arxivSaveBtn");
  if (arxivSaveBtn) {
    arxivSaveBtn.addEventListener("click", function () {
      var q = (document.getElementById("arxivQuery").value || "").trim();
      var max = parseInt(document.getElementById("arxivMax").value, 10) || 5;
      if (!q) { toast("Önce bir sorgu gir.", true); return; }
      api("/arxiv/queries", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, max_results: max, auto_ingest: true }) })
        .then(function () { toast("Sorgu kaydedildi ✓"); loadArxivQueries(); })
        .catch(function (e) { toast("Kaydedilemedi: " + esc(e.message), true); });
    });
  }

  var refreshArxivQueries = document.getElementById("refreshArxivQueries");
  if (refreshArxivQueries) refreshArxivQueries.addEventListener("click", loadArxivQueries);
  loadArxivQueries();

  // ---------- papers ----------
  var _allPapers = [];
  var _comprehensionCache = {};

  function loadPapers() {
    var list = document.getElementById("papersList");
    list.innerHTML = '<div class="empty"><span class="spinner"></span> yükleniyor…</div>';
    api("/papers", { method: "GET" })
      .then(function (papers) {
        _allPapers = papers || [];
        renderPapers();
        loadComprehensionScores();
      })
      .catch(function (err) {
        document.getElementById("papersList").innerHTML =
          '<div class="empty">Hata: ' + esc(err.message) + "</div>";
      });
  }

  function renderPapers() {
    var list = document.getElementById("papersList");
    var query = (document.getElementById("paperSearch").value || "").toLowerCase();
    var filter = (document.querySelector(".filter-btn.active") || {}).getAttribute("data-filter") || "all";
    var sort = (document.getElementById("paperSort") || {}).value || "default";

    var visible = _allPapers.filter(function (p) {
      if (filter === "card" && !p.has_card) return false;
      if (filter === "nocard" && p.has_card) return false;
      if (query) {
        var title = (p.title || "").toLowerCase();
        var id = (p.paper_id || "").toLowerCase();
        if (title.indexOf(query) < 0 && id.indexOf(query) < 0) return false;
      }
      return true;
    });

    if (sort === "title-asc") {
      visible.sort(function (a, b) { return (a.title || "").localeCompare(b.title || ""); });
    } else if (sort === "title-desc") {
      visible.sort(function (a, b) { return (b.title || "").localeCompare(a.title || ""); });
    } else if (sort === "card-first") {
      visible.sort(function (a, b) { return (b.has_card ? 1 : 0) - (a.has_card ? 1 : 0); });
    } else if (sort === "nocard-first") {
      visible.sort(function (a, b) { return (a.has_card ? 1 : 0) - (b.has_card ? 1 : 0); });
    }

    if (!_allPapers.length) {
      list.innerHTML = '<div class="empty">Henüz makale yok. Yukarıdan PDF yükle.</div>';
      return;
    }
    if (!visible.length) {
      list.innerHTML = '<div class="empty">Filtreye uyan makale bulunamadı.</div>';
      return;
    }

    list.innerHTML = visible
      .map(function (p) {
        var btn = p.has_card
          ? '<button class="btn card-view" data-id="' + esc(p.paper_id) + '">✓ KARTI GÖR</button>'
          : '<button class="btn card-btn" data-id="' + esc(p.paper_id) + '">BİLGİ KARTI ÜRET</button>';
        var cached = _comprehensionCache[p.paper_id];
        var badgeLabel = (cached != null) ? cached + "%" : "?%";
        var badgeCls = "comp-badge" +
          (cached == null ? "" : (cached >= 80 ? " comp-high" : cached >= 50 ? " comp-mid" : " comp-low"));
        var badge = p.has_card
          ? '<button class="' + badgeCls + '" data-comp-id="' + esc(p.paper_id) + '" title="Anlama skoru — tıkla hesapla">' + badgeLabel + '</button>'
          : '';
        return (
          '<div class="paper-row' +
          (p.has_card ? " has-card" : "") +
          '"><div class="paper-meta">' +
          '<div class="paper-title">' +
          esc(p.title || "(başlıksız)") +
          "</div>" +
          '<div class="paper-id">' +
          esc(p.paper_id) +
          " · " +
          (p.n_chunks || 0) +
          " chunk" +
          (p.year ? " · " + esc(p.year) : "") +
          "</div></div>" +
          '<div class="paper-actions">' +
          badge +
          btn +
          "</div></div>"
        );
      })
      .join("");

    list.querySelectorAll(".card-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        makeCard(b.getAttribute("data-id"), b);
      });
    });
    list.querySelectorAll(".card-view").forEach(function (b) {
      b.addEventListener("click", function () {
        viewCard(b.getAttribute("data-id"));
      });
    });
    list.querySelectorAll(".comp-badge").forEach(function (b) {
      b.addEventListener("click", function () {
        var pid = b.getAttribute("data-comp-id");
        b.textContent = "…";
        b.disabled = true;
        api("/papers/" + encodeURIComponent(pid) + "/comprehension", { method: "POST" })
          .then(function (d) {
            _comprehensionCache[pid] = d.total != null ? Math.round(d.total) : null;
            renderPapers();
          })
          .catch(function () {
            b.textContent = "?%";
            b.disabled = false;
          });
      });
    });
  }

  function loadComprehensionScores() {
    api("/papers/comprehension/all", { method: "GET" })
      .then(function (d) {
        var scores = (d && d.scores) ? d.scores : {};
        var changed = false;
        Object.keys(scores).forEach(function (pid) {
          if (_comprehensionCache[pid] !== scores[pid]) {
            _comprehensionCache[pid] = scores[pid];
            changed = true;
          }
        });
        if (changed) renderPapers();
      })
      .catch(function () {});
  }

  // filtre/sıralama olayları
  document.getElementById("paperSearch").addEventListener("input", renderPapers);
  document.getElementById("paperSort").addEventListener("change", renderPapers);
  document.getElementById("cardFilter").addEventListener("click", function (e) {
    var btn = e.target.closest(".filter-btn");
    if (!btn) return;
    document.querySelectorAll(".filter-btn").forEach(function (b) { b.classList.remove("active"); });
    btn.classList.add("active");
    renderPapers();
  });

  function makeCard(paperId, btn) {
    btn.disabled = true;
    btn.classList.remove("btn-err");
    btn.innerHTML = '<span class="spinner"></span>ÜRETİLİYOR';
    api("/card/" + encodeURIComponent(paperId), { method: "POST" })
      .then(function (data) {
        toast("Bilgi kartı üretildi: " + paperId);
        if (data && data.card) renderCard(data.card, paperId); // hemen göster
        loadPapers(); // liste yenilensin → 'KARTI GÖR'
      })
      .catch(function (err) {
        toast(err.message, true);
        btn.classList.add("btn-err");
        btn.textContent = "✕ HATA — TEKRAR DENE";
        btn.disabled = false;
      });
  }

  // ---------- bilgi kartı görüntüleme ----------
  var cardModal = document.getElementById("cardModal");
  var cardBody = document.getElementById("cardBody");

  function showCardModal() {
    cardModal.classList.remove("hidden");
  }
  function hideCardModal() {
    cardModal.classList.add("hidden");
  }
  document.getElementById("cardClose").addEventListener("click", hideCardModal);
  cardModal.addEventListener("click", function (e) {
    if (e.target === cardModal) hideCardModal(); // dışına tıkla → kapat
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") hideCardModal();
  });

  function viewCard(paperId) {
    cardBody.innerHTML = '<div class="muted"><span class="spinner"></span> kart yükleniyor…</div>';
    showCardModal();
    api("/card/" + encodeURIComponent(paperId), { method: "GET" })
      .then(function (data) {
        renderCard(data.card, paperId);
      })
      .catch(function (err) {
        cardBody.innerHTML = '<div class="result-body">Hata: ' + esc(err.message) + "</div>";
      });
  }

  function _kcText(label, val) {
    if (!val) return "";
    return (
      '<div class="kc-field"><div class="kc-label">' +
      esc(label) +
      '</div><p>' +
      esc(val) +
      "</p></div>"
    );
  }
  function _kcList(label, arr) {
    if (!arr || !arr.length) return "";
    var items = arr
      .map(function (x) {
        return "<li>" + esc(x) + "</li>";
      })
      .join("");
    return (
      '<div class="kc-field"><div class="kc-label">' +
      esc(label) +
      '</div><ul>' +
      items +
      "</ul></div>"
    );
  }

  function _cardHasContent(c) {
    // Sunucudaki card_has_content ile aynı tanım: title VEYA main_claim alfanümerik.
    var re = /[0-9A-Za-zÇĞİÖŞÜçğıöşü]/;
    return re.test(String((c && c.title) || "")) || re.test(String((c && c.main_claim) || ""));
  }

  function renderCard(c, paperId) {
    c = c || {};
    if (!_cardHasContent(c)) {
      // Boş kabuk kartı "(başlıksız)" diye göstermek "kart var ama bozuk" yanılgısı
      // yaratıyordu; açıkça söyle ve yeniden üretme yolu ver.
      cardBody.innerHTML =
        '<h3 class="kc-title">Bilgi kartı içeriksiz</h3>' +
        '<div class="result-body">Bu makale için kaydedilmiş kart boş (LLM zaman aşımı / parse ' +
        "hatası). Ollama boşken yeniden üretmeyi deneyin.</div>" +
        (paperId
          ? '<div class="kc-bt-bar"><button class="btn" id="kcRegenBtn" data-id="' +
            esc(paperId) +
            '">↻ YENİDEN ÜRET</button></div>'
          : "");
      showCardModal();
      var regen = document.getElementById("kcRegenBtn");
      if (regen) regen.addEventListener("click", function () { makeCard(paperId, regen); });
      return;
    }
    var meta = [c.year, c.domain].filter(Boolean).map(esc).join(" · ");
    var hasHyps = c.possible_strategy_hypotheses && c.possible_strategy_hypotheses.length;
    var btBtn = paperId && hasHyps
      ? '<button class="btn btn-hyp-bt" id="hypBtBtn" data-id="' + esc(paperId) + '">⚡ HİPOTEZLERİ BACKTEST ET</button>'
      : "";
    cardBody.innerHTML =
      '<h3 class="kc-title">' +
      esc(c.title || "(başlıksız)") +
      "</h3>" +
      (meta ? '<div class="kc-meta">' + meta + "</div>" : "") +
      _kcText("Ana bulgu", c.main_claim) +
      _kcText("Trading ilgisi", c.trading_relevance) +
      _kcList("Yöntemler", c.methods) +
      _kcList("Veri setleri", c.datasets) +
      _kcList("Strateji hipotezleri", c.possible_strategy_hypotheses) +
      _kcList("Risk uyarıları", c.risk_warnings) +
      _kcList("Sınırlamalar", c.limitations) +
      _kcList("Uygulama notları", c.implementation_notes) +
      (btBtn ? '<div class="kc-bt-bar">' + btBtn + '<div id="hypBtResult"></div></div>' : "");
    showCardModal();

    if (paperId && hasHyps) {
      var hypBtBtn = document.getElementById("hypBtBtn");
      if (hypBtBtn) hypBtBtn.addEventListener("click", function () {
        backtestFromCard(paperId);
      });
    }
  }

  function backtestFromCard(paperId) {
    var btn = document.getElementById("hypBtBtn");
    var res = document.getElementById("hypBtResult");
    if (!btn || !res) return;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>ÇALIŞIYOR';
    res.innerHTML = "";
    api("/card/" + encodeURIComponent(paperId) + "/backtest", { method: "POST" })
      .then(function (data) {
        btn.textContent = "✓ TAMAMLANDI";
        res.innerHTML = renderHypBacktest(data);
      })
      .catch(function (err) {
        btn.disabled = false;
        btn.textContent = "⚡ HİPOTEZLERİ BACKTEST ET";
        res.innerHTML = '<div class="kc-bt-err">Hata: ' + esc(err.message) + "</div>";
      });
  }

  function renderHypBacktest(data) {
    if (!data.results || !data.results.length) return '<div class="muted">Sonuç yok.</div>';
    return data.results
      .map(function (r) {
        var m = r.metrics || {};
        var vClass = "verdict-" + (r.verdict || "inconclusive");
        var vLabel = { pass: "GEÇTİ", fail: "BAŞARISIZ", inconclusive: "SONUÇSUZ" }[r.verdict] || r.verdict;
        var reasons = (r.reasons || []).map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("");
        var keyMetrics = ["total_return_pct", "sharpe", "max_drawdown_pct", "win_rate_pct", "n_trades"];
        var metricHtml = keyMetrics
          .filter(function (k) { return k in m; })
          .map(function (k) {
            var labels = { total_return_pct: "getiri%", sharpe: "sharpe", max_drawdown_pct: "maxDD%", win_rate_pct: "win%", n_trades: "işlem" };
            var v = Number(m[k]).toFixed(k === "n_trades" ? 0 : 3);
            var cls = (k === "total_return_pct" || k === "sharpe") ? (m[k] > 0 ? " pos" : " neg") : (k === "max_drawdown_pct" ? " neg" : "");
            return '<div class="metric"><div class="k">' + (labels[k] || k) + '</div><div class="v' + cls + '">' + esc(v) + "</div></div>";
          })
          .join("");
        return (
          '<div class="hyp-row">' +
          '<div class="hyp-text">' + esc(r.hypothesis) + "</div>" +
          '<div class="metrics-grid">' + metricHtml + "</div>" +
          '<div class="verdict ' + vClass + '"><b>' + vLabel + "</b><ul>" + reasons + "</ul></div>" +
          "</div>"
        );
      })
      .join("");
  }

  // ---------- eğitim sekmesi ----------
  function loadTrainingStatus() {
    api("/training/status", { method: "GET" })
      .then(function (data) {
        var el = document.getElementById("trainingStatus");
        if (el) {
          el.innerHTML =
            '<div class="training-stat">' +
            '<span class="kc-label">Eğitim örnekleri</span> ' +
            '<strong>' + data.n_examples + '</strong>' +
            '</div>';
        }
        renderAdapters(data.adapters || []);
        populateAdapterSelects(data.adapters || []);
        loadExamples();
      })
      .catch(function (err) {
        var el = document.getElementById("trainingStatus");
        if (el) el.innerHTML = '<span class="conn-err">Hata: ' + esc(err.message) + "</span>";
      });
  }

  function renderAdapters(adapters) {
    var el = document.getElementById("adaptersList");
    if (!el) return;
    if (!adapters.length) {
      el.innerHTML = '<div class="empty">Henüz kayıtlı adapter yok.</div>';
      return;
    }
    el.innerHTML = adapters
      .map(function (a) {
        return (
          '<div class="adapter-row">' +
          '<div class="adapter-version">' + esc(a.version) + "</div>" +
          '<div class="adapter-meta">' +
          esc(a.base_model) +
          (a.created_at ? " · " + esc(a.created_at.slice(0, 10)) : "") +
          (a.notes ? " · " + esc(a.notes) : "") +
          "</div></div>"
        );
      })
      .join("");
  }

  // ---------- eğitim örnekleri ----------
  function loadExamples() {
    var el = document.getElementById("examplesList");
    if (!el) return;
    el.innerHTML = '<div class="empty"><span class="spinner"></span> yükleniyor…</div>';
    api("/training/examples", { method: "GET" })
      .then(function (data) {
        var examples = data.examples || [];
        if (!examples.length) {
          el.innerHTML = '<div class="empty">Henüz eğitim örneği yok.</div>';
          return;
        }
        el.innerHTML =
          '<div class="examples-count">' + examples.length + " örnek</div>" +
          examples
            .map(function (ex) {
              return (
                '<div class="example-row" data-id="' + esc(ex.example_id) + '">' +
                '<div class="example-header">' +
                '<span class="example-type">' + esc(ex.example_type) + "</span>" +
                (ex.source_paper_id ? '<span class="example-paper">' + esc(ex.source_paper_id) + "</span>" : "") +
                '<span class="example-date muted">' + esc((ex.created_at || "").slice(0, 10)) + "</span>" +
                '<button class="btn btn-del-example" data-id="' + esc(ex.example_id) + '" title="Sil">✕</button>' +
                "</div>" +
                '<div class="example-instruction">' + esc(ex.instruction.slice(0, 120)) + (ex.instruction.length > 120 ? "…" : "") + "</div>" +
                "</div>"
              );
            })
            .join("");
        el.querySelectorAll(".btn-del-example").forEach(function (b) {
          b.addEventListener("click", function (e) {
            e.stopPropagation();
            deleteExample(b.getAttribute("data-id"));
          });
        });
      })
      .catch(function (err) {
        el.innerHTML = '<div class="empty">Hata: ' + esc(err.message) + "</div>";
      });
  }

  function deleteExample(exampleId) {
    api("/training/examples/" + encodeURIComponent(exampleId), { method: "DELETE" })
      .then(function () {
        toast("Örnek silindi.");
        loadExamples();
        loadTrainingStatus();
      })
      .catch(function (err) {
        toast("Silinemedi: " + err.message, true);
      });
  }

  var refreshExamples = document.getElementById("refreshExamples");
  if (refreshExamples) refreshExamples.addEventListener("click", loadExamples);

  var buildDatasetBtn = document.getElementById("buildDatasetBtn");
  if (buildDatasetBtn) {
    buildDatasetBtn.addEventListener("click", function () {
      buildDatasetBtn.disabled = true;
      buildDatasetBtn.innerHTML = '<span class="spinner"></span>OLUŞTURULUYOR';
      var res = document.getElementById("datasetResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span></div>';
      api("/training/dataset", { method: "POST" })
        .then(function (data) {
          toast(data.message);
          res.innerHTML =
            '<div class="result-section"><div class="result-label">sonuç</div>' +
            '<div class="result-body">' +
            esc(data.message) +
            " · hash: " + esc(data.content_hash) +
            "</div></div>";
          loadTrainingStatus();
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          buildDatasetBtn.disabled = false;
          buildDatasetBtn.textContent = "DATASET OLUŞTUR";
        });
    });
  }

  // ---------- Canlı Eğitim UI ----------
  var _trainESS = null;

  function applyTrainProgress(d) {
    var stateMap = { idle: "boşta", running: "eğitiliyor…", completed: "tamamlandı ✓", failed: "hata ✗", stopped: "durduruldu" };
    var stateColors = { idle: "badge-info", running: "badge-warn", completed: "badge-success", failed: "badge-error", stopped: "badge-info" };
    var label = document.getElementById("trainStateLabel");
    if (label) {
      label.textContent = stateMap[d.state] || d.state;
      label.className = "badge " + (stateColors[d.state] || "badge-info");
    }
    var isRunning = d.state === "running";
    var startBtn = document.getElementById("startTrainBtn");
    var stopBtn = document.getElementById("stopTrainBtn");
    if (startBtn) startBtn.style.display = isRunning ? "none" : "";
    if (stopBtn) stopBtn.style.display = isRunning ? "" : "none";

    var section = document.getElementById("trainProgressSection");
    if (section && d.state !== "idle") section.style.display = "";

    var bar = document.getElementById("trainProgressBar");
    var pctLbl = document.getElementById("trainPctLabel");
    var iterLbl = document.getElementById("trainIterLabel");
    if (bar) bar.style.width = (d.pct || 0) + "%";
    if (pctLbl) pctLbl.textContent = (d.pct || 0).toFixed(1) + "%";
    if (iterLbl) iterLbl.textContent = (d.current_iter || 0) + " / " + (d.total_iters || 0) + " iter";

    var tl = document.getElementById("trainLossVal");
    var vl = document.getElementById("valLossVal");
    if (tl) tl.textContent = d.train_loss != null ? Number(d.train_loss).toFixed(4) : "—";
    if (vl) vl.textContent = d.val_loss != null ? Number(d.val_loss).toFixed(4) : "—";

    var meta = document.getElementById("trainMeta");
    if (meta && d.adapter_name) {
      meta.textContent = "Adapter: " + d.adapter_name +
        (d.started_at ? "  ·  Başladı: " + d.started_at : "") +
        (d.finished_at && d.state !== "running" ? "  ·  Bitti: " + d.finished_at : "");
    }
  }

  function appendLog(line) {
    var el = document.getElementById("trainLog");
    if (!el) return;
    el.textContent += line + "\n";
    el.scrollTop = el.scrollHeight;
  }

  function openTrainStream(url) {
    _trainESS = new EventSource(url);
    _trainESS.onmessage = function (e) {
      try {
        var d = JSON.parse(e.data);
        if (d.line) appendLog(d.line);
        applyTrainProgress(d);
        if (d.type === "done" || d.type === "stopped") {
          _trainESS.close();
          _trainESS = null;
          loadTrainingStatus();
        }
      } catch (_) {}
    };
    _trainESS.onerror = function () {
      // Bilet TEK kullanımlık → EventSource oto-yeniden bağlanamaz; kapat (yeni startSSE
      // taze bilet alır). İnsan api_token'ı artık query'ye KONMAZ (P7).
      if (_trainESS) { _trainESS.close(); _trainESS = null; }
    };
  }

  function startSSE() {
    if (_trainESS) { _trainESS.close(); _trainESS = null; }
    // İNSAN api_token'ı ARTIK query'ye konmaz (log/proxy/geçmiş sızıntısı — P7). Yerine
    // KISA ömürlü, TEK-kullanımlık bilet: normal auth ile al, EventSource'a query'de ver.
    api("/training/stream-ticket", { method: "POST" })
      .then(function (d) {
        var q = d && d.ticket ? "?ticket=" + encodeURIComponent(d.ticket) : "";
        openTrainStream("/api/training/stream" + q);
      })
      .catch(function () {
        // Bilet alınamadı (ör. token gerekli ama SİSTEM sekmesinde girilmemiş) → sessizce
        // vazgeç; loadTrainingStatus() zaten polling ile ilerlemeyi gösterir.
      });
  }

  var startTrainBtn = document.getElementById("startTrainBtn");
  if (startTrainBtn) {
    startTrainBtn.addEventListener("click", function () {
      // Phase 4D-1: gerçek eğitim tehlikeli → tek-tık yok, önce confirm.
      if (!window.confirm("Bu işlem gerçek LoRA training başlatabilir. Fresh manual "
          + "approval olmadan başlamamalıdır. Devam etmek istiyor musunuz?")) return;
      var payload = {
        base_model: (document.getElementById("drBaseModel") || {}).value || "",
        adapter_name: (document.getElementById("trAdapterName") || {}).value || "hektor_lora",
        iterations: parseInt((document.getElementById("drIterations") || {}).value, 10) || 500,
        // batch_size / num_layers BILEREK gonderilmiyor: gercek egitim yolu
        // (launch -> train --run) bunlari HIC kullanmiyor ve dry-run'da yalniz MLX
        // (macOS) dalinda gecerliler. Windows/Linux PEFT'te deger girmek kullaniciyi
        // yaniltiyordu -> alanlar arayuzden kaldirildi; sema varsayilanlari yeterli.
        learning_rate: 1e-4,
      };
      // GERÇEK eğitim onayı (tehlikeli aksiyon — agStopAll/agApprove ile aynı desen).
      // Not: bu yalnız UX uyarısıdır; gerçek kapı sunucudadır (TAZE tek-kullanımlık
      // onay — /api/training/run). İlk başlatma "onay gerekiyor" mesajı döndürür;
      // ONAYLAR sekmesinden onaylayıp tekrar başlat (CLAUDE.md Kural 8).
      if (!window.confirm(
        "GERÇEK LoRA eğitimi başlatılacak:\n" +
        "Adapter: " + payload.adapter_name + "\n" +
        "İterasyon: " + payload.iterations + "\n\n" +
        "Saatler sürebilir; bilgisayar açık kalmalı. Sunucu TAZE manuel onay ister " +
        "(ilk tık onay isteği oluşturur → ONAYLAR sekmesinden onayla → tekrar başlat).\n\n" +
        "Devam edilsin mi?"
      )) return;
      var section = document.getElementById("trainProgressSection");
      if (section) section.style.display = "";
      var logEl = document.getElementById("trainLog");
      if (logEl) logEl.textContent = "";
      api("/training/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
        .then(function (data) {
          // Phase 4D-1: backend taze manuel onay isteyebilir (needs_approval).
          if (data && data.status === "needs_approval") {
            var cmd = data.approve_command || ("uv run hektor approval-approve " + (data.approval_id || ""));
            if (logEl) {
              logEl.textContent = "Onay gerekli. Approval ID: " + (data.approval_id || "")
                + "\nCLI ile onayla: " + cmd
                + "\nOnayladıktan sonra eğitimi yeniden başlat.";
            }
            toast(data.message || "Onay gerekli — fresh manual approval şart.", true);
            return;
          }
          toast(data.message, !data.ok);
          // Detached eğitim: SSE yerine /api/training/live poll'u (mirrorTrainTab) besler.
          if (data.ok) refreshTrain();
        })
        .catch(function (err) { toast(err.message, true); });
    });
  }

  var stopTrainBtn = document.getElementById("stopTrainBtn");
  if (stopTrainBtn) {
    stopTrainBtn.addEventListener("click", function () {
      api("/training/stop", { method: "POST" })
        .then(function () { toast("Eğitim durduruldu."); })
        .catch(function () {});
    });
  }

  var clearLogBtn = document.getElementById("clearLogBtn");
  if (clearLogBtn) {
    clearLogBtn.addEventListener("click", function () {
      var el = document.getElementById("trainLog");
      if (el) el.textContent = "";
    });
  }

  var refreshAdapters = document.getElementById("refreshAdapters");
  if (refreshAdapters) refreshAdapters.addEventListener("click", loadTrainingStatus);

  // Sayfa açılışında mevcut durumu çek; training devam ediyorsa SSE başlat
  api("/training/progress", { method: "GET" })
    .then(function (d) {
      applyTrainProgress(d);
      if (d.state === "running") startSSE();
    })
    .catch(function () {});

  // ---------- toplu kart üretimi ----------
  var batchCardBtn = document.getElementById("batchCardBtn");
  if (batchCardBtn) {
    batchCardBtn.addEventListener("click", function () {
      batchCardBtn.disabled = true;
      batchCardBtn.innerHTML = '<span class="spinner"></span>ÜRETİLİYOR…';
      var res = document.getElementById("batchCardResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> Kartlar üretiliyor (LLM gerekli, her makale ~60s)…</div>';
      api("/cards/batch", { method: "POST" })
        .then(function (data) {
          toast(data.produced + " kart üretildi, " + data.skipped + " atlandı, " + data.errors + " hata.");
          var rows = (data.results || []).map(function (r) {
            var cls = r.status === "ok" ? "pos" : r.status === "error" ? "neg" : "muted";
            return (
              '<div class="batch-row">' +
              '<span class="' + cls + '">' + esc(r.status.toUpperCase()) + "</span> " +
              "<b>" + esc(r.title || r.paper_id) + "</b> — " +
              esc(r.message) +
              "</div>"
            );
          }).join("");
          res.innerHTML =
            '<div class="result-section"><div class="result-label">sonuç</div>' +
            "<div>" + rows + "</div></div>";
          loadPapers();
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          batchCardBtn.disabled = false;
          batchCardBtn.textContent = "⚡ TÜM KARTLARI ÜRET";
        });
    });
  }

  // ---------- toplu skor hesaplama ----------
  var batchScoreBtn = document.getElementById("batchScoreBtn");
  if (batchScoreBtn) {
    batchScoreBtn.addEventListener("click", function () {
      batchScoreBtn.disabled = true;
      batchScoreBtn.innerHTML = '<span class="spinner"></span>HESAPLANIYOR…';
      var res = document.getElementById("batchScoreResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> Skorlar hesaplanıyor (LLM gerekli, her makale ~30s)…</div>';
      api("/papers/comprehension/batch", { method: "POST" })
        .then(function (data) {
          toast(data.computed + " skor hesaplandı, " + data.skipped + " atlandı, " + data.errors + " hata.");
          var rows = (data.results || []).map(function (r) {
            var cls = r.status === "ok" ? "pos" : r.status === "error" ? "neg" : "muted";
            var scoreStr = r.score != null ? " — " + Math.round(r.score) + "%" : "";
            return (
              '<div class="batch-row">' +
              '<span class="' + cls + '">' + esc(r.status.toUpperCase()) + "</span> " +
              "<b>" + esc(r.title || r.paper_id) + "</b>" +
              esc(scoreStr) + " " + esc(r.message) +
              "</div>"
            );
          }).join("");
          res.innerHTML =
            '<div class="result-section"><div class="result-label">sonuç</div>' +
            "<div>" + rows + "</div></div>";
          _comprehensionCache = {};
          loadPapers();
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          batchScoreBtn.disabled = false;
          batchScoreBtn.textContent = "🧪 TÜM SKORLARI HESAPLA";
        });
    });
  }

  // ---------- objektif anlama skoru (L3/L4 sınav geçme oranı, kaba %'nin yerine) ----------
  var objUnderstandingEl = document.getElementById("objUnderstanding");
  if (objUnderstandingEl) {
    objUnderstandingEl.addEventListener("click", function () {
      var prev = objUnderstandingEl.textContent;
      objUnderstandingEl.textContent = "obj. anlama: …";
      // Tam merdiven (L5 + L3/L4) + KALICI kayıt: tıklayınca snapshot DB'ye yazılır
      // → anlama zaman içinde izlenir (understanding-history).
      api("/understanding-score?full=true&record=true", { method: "GET" })
        .then(function (s) {
          var brk = levelBreakdown(s.by_level);  // Taban→L5 sırasında (alfabetik değil)
          var snap = s.recorded && s.recorded.snapshot_id ? s.recorded.snapshot_id : null;
          if (s.status !== "scored" || s.pass_rate == null) {
            objUnderstandingEl.textContent = "obj. anlama: veri yok";
            objUnderstandingEl.title =
              "Objektif sınav notlanamadı (LLM çevrimdışı olabilir). Notlanan: " +
              s.graded + " · atlanan: " + s.skipped + (brk ? "\nSeviyeler: " + brk : "");
            toast("Objektif anlama: notlanan sınav yok (LLM çevrimdışı?).", true);
            return;
          }
          var pct = Math.round(s.pass_rate * 100);
          objUnderstandingEl.textContent = "obj. anlama %" + pct;
          objUnderstandingEl.title =
            "Objektif merdiven (Taban/L1/L2/L3/L4/L5) geçme oranı — geçti " +
            s.passed + "/" + s.graded + " (atlanan " + s.skipped + ", veri yok " +
            s.no_data + ")" + (brk ? "\nSeviyeler: " + brk : "") +
            (snap ? "\nKALICI kaydedildi: " + snap : "");
          toast("Objektif anlama: %" + pct + " (" + s.graded + " sınav) — kalıcı kaydedildi");
        })
        .catch(function (err) {
          objUnderstandingEl.textContent = prev;
          toast(err.message, true);
        });
    });
  }

  // ---------- çapraz makale sentezi ----------
  var crossSynthesisBtn = document.getElementById("crossSynthesisBtn");
  if (crossSynthesisBtn) {
    crossSynthesisBtn.addEventListener("click", function () {
      crossSynthesisBtn.disabled = true;
      crossSynthesisBtn.innerHTML = '<span class="spinner"></span>SENTEZLENİYOR…';
      var res = document.getElementById("crossSynthesisResult");
      res.className = "result";
      res.innerHTML =
        '<div class="result-section"><span class="spinner"></span> ' +
        'Formüller arası ilişkiler analiz ediliyor ve eğitim verisi üretiliyor…</div>';
      api("/synthesis/cross-paper", { method: "POST" })
        .then(function (data) {
          toast(data.message);
          var cls = data.produced > 0 ? "pos" : "muted";
          res.innerHTML =
            '<div class="result-section"><div class="result-label">sentez sonucu</div>' +
            '<div class="batch-row"><span class="' + cls + '">' +
            (data.produced > 0 ? "✓ " + data.produced + " yeni örnek" : "Güncel") +
            '</span> — ' + esc(data.message) + "</div></div>";
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML =
            '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          crossSynthesisBtn.disabled = false;
          crossSynthesisBtn.textContent = "🔗 ÇAPRAZ SENTEZ ÜRET";
        });
    });
  }

  document.getElementById("refreshPapers").addEventListener("click", loadPapers);
  document.getElementById("reindexBtn").addEventListener("click", function () {
    var b = this;
    b.disabled = true;
    b.innerHTML = '<span class="spinner"></span>İNDEKSLENİYOR';
    api("/ingest", { method: "POST" })
      .then(function (r) {
        toast(r.message);
        loadPapers();
        refreshStatus();
      })
      .catch(function (err) {
        toast(err.message, true);
      })
      .finally(function () {
        b.disabled = false;
        b.textContent = "⟳ TÜMÜNÜ İNDEKSLE";
      });
  });

  // ---------- upload ----------
  var zone = document.getElementById("uploadZone");
  var input = document.getElementById("pdfInput");
  document.getElementById("browseBtn").addEventListener("click", function (e) {
    e.stopPropagation();
    input.click();
  });
  zone.addEventListener("click", function () {
    input.click();
  });
  zone.addEventListener("dragover", function (e) {
    e.preventDefault();
    zone.classList.add("drag");
  });
  zone.addEventListener("dragleave", function () {
    zone.classList.remove("drag");
  });
  zone.addEventListener("drop", function (e) {
    e.preventDefault();
    zone.classList.remove("drag");
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      uploadFiles(e.dataTransfer.files);
    }
  });
  input.addEventListener("change", function () {
    if (input.files && input.files.length) uploadFiles(input.files);
    input.value = ""; // aynı dosyaları tekrar seçebilmek için sıfırla
  });

  // Birden çok PDF'i SIRAYLA yükler (8GB'ı zorlamamak için paralel DEĞİL).
  function uploadFiles(fileList) {
    var files = [];
    for (var k = 0; k < fileList.length; k++) {
      if (fileList[k].name.toLowerCase().endsWith(".pdf")) files.push(fileList[k]);
    }
    if (!files.length) {
      toast("Yalnız .pdf kabul edilir.", true);
      return;
    }
    var ok = 0,
      fail = 0,
      dup = 0;
    // Hız sınırına (429) takılan dosyayı SESSİZCE düşürme — bekle ve aynı dosyayı yeniden
    // dene. Sunucu üst sınırı kayan 60sn pencerede çalıştığı için birkaç saniye beklemek
    // pencereyi boşaltır. (Eski davranış: 429 → "hata" sayılıp dosya kayboluyordu; toplu
    // yüklemede "bir kısım gelmedi" sorununun kök nedeni buydu.)
    var MAX_RETRY = 40; // tek dosya için yeniden deneme üst sınırı (sonsuz döngü koruması)
    var RETRY_WAIT_MS = 3500;
    function sendOne(index, attempt) {
      if (index >= files.length) {
        done();
        return;
      }
      var f = files[index];
      if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
        fail++;
        toast(f.name + " > " + MAX_UPLOAD_MB + " MB, atlandı.", true);
        return sendOne(index + 1, 0);
      }
      toast("Yükleniyor (" + (index + 1) + "/" + files.length + "): " + f.name + " …");
      var fd = new FormData();
      fd.append("file", f);
      api("/papers/upload", { method: "POST", body: fd })
        .then(function (resp) {
          // skipped=1 → sunucu birebir aynı dosyayı zaten var diye atladı (dedup).
          if (resp && resp.skipped) dup++;
          else ok++;
          sendOne(index + 1, 0);
        })
        .catch(function (err) {
          if (err && err.message === "rate-limited" && attempt < MAX_RETRY) {
            toast(
              "Hız sınırı — bekleniyor, yeniden denenecek (" +
                (index + 1) + "/" + files.length + "): " + f.name,
              true
            );
            setTimeout(function () {
              sendOne(index, attempt + 1);
            }, RETRY_WAIT_MS);
          } else {
            fail++;
            sendOne(index + 1, 0);
          }
        });
    }

    function done() {
      var msg =
        ok +
        " PDF aktarıldı" +
        (dup ? ", " + dup + " zaten vardı (atlandı)" : "") +
        (fail ? ", " + fail + " hata" : "") +
        ". İndeksleniyor…";
      toast(msg);
      api("/papers")
        .then(function (data) {
          var prevCount = (data.papers || data || []).length;
          pollPapersUntilGrown(prevCount, 30);
        })
        .catch(function () { pollPapersUntilGrown(0, 30); });
    }

    function pollPapersUntilGrown(prevCount, attempts) {
      if (attempts <= 0) { loadPapers(); refreshStatus(); return; }
      setTimeout(function () {
        api("/papers")
          .then(function (data) {
            var list = data.papers || data || [];
            if (list.length > prevCount) {
              loadPapers();
              refreshStatus();
              toast("İndeksleme tamamlandı, makaleler güncellendi.");
            } else {
              pollPapersUntilGrown(prevCount, attempts - 1);
            }
          })
          .catch(function () { pollPapersUntilGrown(prevCount, attempts - 1); });
      }, 4000);
    }

    sendOne(0, 0);
  }

  // ---------- backtest ----------
  var btForm = document.getElementById("btForm");
  btForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var n = parseInt(document.getElementById("nBars").value, 10) || 2000;
    var seed = parseInt(document.getElementById("seed").value, 10) || 42;
    var btn = document.getElementById("btBtn");
    var res = document.getElementById("btResult");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>ÇALIŞIYOR';
    res.className = "result";
    res.innerHTML = '<div class="result-section"><span class="spinner"></span> backtest…</div>';

    api("/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ use_synthetic: true, n_bars: n, seed: seed }),
    })
      .then(function (data) {
        renderBacktest(res, data);
        loadBacktestHistory();
      })
      .catch(function (err) {
        res.innerHTML =
          '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
      })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "SENTETİK ÇALIŞTIR →";
      });
  });

  // ---------- backtest: gerçek veri (CSV) ----------
  var csvZone = document.getElementById("csvZone");
  var csvInput = document.getElementById("csvInput");
  if (csvZone && csvInput) {
    document.getElementById("csvBrowse").addEventListener("click", function (e) {
      e.stopPropagation();
      csvInput.click();
    });
    csvZone.addEventListener("click", function () {
      csvInput.click();
    });
    csvZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      csvZone.classList.add("drag");
    });
    csvZone.addEventListener("dragleave", function () {
      csvZone.classList.remove("drag");
    });
    csvZone.addEventListener("drop", function (e) {
      e.preventDefault();
      csvZone.classList.remove("drag");
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        runCsvBacktest(e.dataTransfer.files[0]);
      }
    });
    csvInput.addEventListener("change", function () {
      if (csvInput.files && csvInput.files.length) runCsvBacktest(csvInput.files[0]);
    });
  }

  function runCsvBacktest(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      toast("Yalnız .csv kabul edilir.", true);
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      toast("Dosya " + MAX_UPLOAD_MB + " MB sınırını aşıyor.", true);
      return;
    }
    var res = document.getElementById("btResult");
    res.className = "result";
    res.innerHTML =
      '<div class="result-section"><span class="spinner"></span> gerçek veri backtest: ' +
      esc(file.name) +
      " …</div>";
    var fd = new FormData();
    fd.append("file", file);
    api("/backtest/csv", { method: "POST", body: fd })
      .then(function (data) {
        renderBacktest(res, data);
        loadBacktestHistory();
      })
      .catch(function (err) {
        res.innerHTML =
          '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
      });
  }

  function renderBacktest(res, data) {
    var m = data.metrics || {};
    var order = [
      ["n_trades", "işlem"],
      ["total_return_pct", "getiri %"],
      ["sharpe", "sharpe"],
      ["sortino", "sortino"],
      ["max_drawdown_pct", "max dd %"],
      ["profit_factor", "profit factor"],
      ["win_rate_pct", "win %"],
    ];
    var cells = order
      .map(function (pair) {
        var key = pair[0];
        if (!(key in m)) return "";
        var val = m[key];
        var cls = "";
        if (typeof val === "number") {
          if (key.indexOf("return") >= 0 || key === "sharpe" || key === "sortino") {
            cls = val > 0 ? "pos" : val < 0 ? "neg" : "";
          }
          if (key === "max_drawdown_pct") cls = "neg";
          val = Number(val).toFixed(key === "n_trades" ? 0 : 4);
        }
        return (
          '<div class="metric"><div class="k">' +
          esc(pair[1]) +
          '</div><div class="v ' +
          cls +
          '">' +
          esc(val) +
          "</div></div>"
        );
      })
      .join("");

    var v = data.verdict || "inconclusive";
    var vClass = "verdict-" + v;
    var vLabel = { pass: "GEÇTİ", fail: "BAŞARISIZ", inconclusive: "SONUÇSUZ" }[v] || v;
    var reasons = (data.reasons || [])
      .map(function (r) {
        return "<li>" + esc(r) + "</li>";
      })
      .join("");

    var verdictNote =
      v === "pass"
        ? '<p class="verdict-note">Strateji bir <strong>adaydır</strong> — yatırım tavsiyesi/canlı sinyal değil; ileri doğrulama (örneklem-dışı, farklı seed) gerekir.</p>'
        : '<p class="verdict-note">Strateji <strong>aday değildir</strong> ("hazır" sayılmaz) — gözden geçirilmeli.</p>';

    res.innerHTML =
      '<div class="result-section"><div class="result-label">strateji</div>' +
      '<div class="result-body">' +
      esc(data.strategy_name) +
      (data.data_source
        ? '  <span class="muted small">· veri: ' +
          esc(data.data_source) +
          (data.n_bars ? " (" + data.n_bars + " bar)" : "") +
          "</span>"
        : "") +
      (data.backtest_id ? '  <span class="muted small">(' + esc(data.backtest_id) + ")</span>" : "") +
      "</div></div>" +
      '<div class="result-section"><div class="result-label">metrikler</div>' +
      '<div class="metrics-grid">' +
      cells +
      "</div>" +
      '<div class="cost-note">Tüm metrikler komisyon + slippage dahildir.</div>' +
      "</div>" +
      '<div class="result-section"><div class="verdict ' +
      vClass +
      '"><h4>YARGI: ' +
      vLabel +
      "</h4><ul>" +
      reasons +
      "</ul>" +
      verdictNote +
      "</div></div>";
  }

  // ---------- trader beyin ----------
  function loadTraderBrain() {
    loadFormulas();
    loadResearchSessions();
  }

  function loadFormulas() {
    api("/research/formulas", { method: "GET" })
      .then(function (formulas) {
        var el = document.getElementById("formulasList");
        if (!el) return;
        if (!formulas.length) {
          el.innerHTML = '<div class="empty muted small">Formül bulunamadı — "FORMÜL ÇIKAR" butonuna bas.</div>';
          return;
        }
        var byCat = {};
        formulas.forEach(function (f) {
          var cat = f.category || "other";
          if (!byCat[cat]) byCat[cat] = [];
          byCat[cat].push(f);
        });
        var html = "";
        Object.keys(byCat).sort().forEach(function (cat) {
          html += '<div class="formula-cat"><span class="formula-cat-label">' + esc(cat.toUpperCase()) + "</span>";
          byCat[cat].forEach(function (f) {
            html += '<span class="formula-chip" title="' + esc(f.description || "") + '">' + esc(f.name) + "</span>";
          });
          html += "</div>";
        });
        el.innerHTML = html;
      })
      .catch(function () {});
  }

  var extractFormulasBtn = document.getElementById("extractFormulasBtn");
  if (extractFormulasBtn) {
    extractFormulasBtn.addEventListener("click", function () {
      extractFormulasBtn.disabled = true;
      extractFormulasBtn.innerHTML = '<span class="spinner"></span>ÇIKARILIYOR…';
      var res = document.getElementById("extractResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> LLM formülleri çıkarıyor (~dakikalar)…</div>';
      api("/research/extract", { method: "POST" })
        .then(function (data) {
          toast(data.message);
          res.innerHTML = '<div class="result-section result-body">' + esc(data.message) + "</div>";
          loadFormulas();
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          extractFormulasBtn.disabled = false;
          extractFormulasBtn.textContent = "⚗ FORMÜL ÇIKAR";
        });
    });
  }

  var runResearchBtn = document.getElementById("runResearchBtn");
  if (runResearchBtn) {
    runResearchBtn.addEventListener("click", function () {
      var q = (document.getElementById("researchQuestion").value || "").trim();
      if (q.length < 10) { toast("Soru çok kısa (min 10 karakter).", true); return; }
      var maxIter = parseInt(document.getElementById("researchIter").value, 10) || 2;
      runResearchBtn.disabled = true;
      runResearchBtn.innerHTML = '<span class="spinner"></span>ARAŞTIRILYOR…';
      var res = document.getElementById("researchResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> Sentezleniyor, backtest ediliyor… (her iterasyon ~2 dk)</div>';
      api("/research/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, max_iterations: maxIter }),
      })
        .then(function (data) {
          res.innerHTML = renderResearchResult(data);
          loadResearchSessions();
        })
        .catch(function (err) {
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          runResearchBtn.disabled = false;
          runResearchBtn.textContent = "🧠 ARAŞTIR →";
        });
    });
  }

  function renderResearchResult(data) {
    var vClass = "verdict-" + (data.final_verdict || "inconclusive");
    var vLabel = { pass: "YAKINSADI ✓", fail: "BAŞARISIZ", inconclusive: "SONUÇSUZ" }[data.final_verdict] || data.final_verdict;
    var iters = (data.iterations || []).map(function (it) {
      var ivClass = "verdict-" + it.verdict;
      var m = it.metrics || {};
      var metHtml =
        '<div class="metric"><div class="k">getiri%</div><div class="v">' + (m.total_return_pct != null ? Number(m.total_return_pct).toFixed(3) : "—") + "</div></div>" +
        '<div class="metric"><div class="k">sharpe</div><div class="v">' + (m.sharpe != null ? Number(m.sharpe).toFixed(3) : "—") + "</div></div>" +
        '<div class="metric"><div class="k">işlem</div><div class="v">' + (m.n_trades || 0) + "</div></div>";
      return (
        '<div class="research-iter">' +
        '<div class="research-iter-head">' +
        '<span class="research-iter-num">İter ' + it.iteration + "</span>" +
        '<span class="research-ind-name">' + esc(it.indicator_name) + "</span>" +
        '<span class="verdict ' + ivClass + ' hist-verdict">' + it.verdict.toUpperCase() + "</span>" +
        "</div>" +
        '<div class="metrics-grid">' + metHtml + "</div>" +
        (it.reflection ? '<div class="research-reflection muted small">💭 ' + esc(it.reflection.slice(0, 200)) + "</div>" : "") +
        (it.improvement_notes ? '<div class="research-improvement muted small">→ ' + esc(it.improvement_notes) + "</div>" : "") +
        "</div>"
      );
    }).join("");
    return (
      '<div class="result-section"><div class="verdict ' + vClass + '"><b>' + vLabel + "</b></div></div>" +
      '<div class="result-section">' + iters + "</div>"
    );
  }

  function loadResearchSessions() {
    var el = document.getElementById("sessionsList");
    if (!el) return;
    api("/research/sessions", { method: "GET" })
      .then(function (rows) {
        if (!rows.length) {
          el.innerHTML = '<div class="empty">Henüz araştırma oturumu yok.</div>';
          return;
        }
        var vLabels = { pass: "✓ BAŞARILI", fail: "✗ BAŞARISIZ", inconclusive: "~ SONUÇSUZ" };
        var passed = rows.filter(function(r) { return r.verdict === "pass"; }).length;
        var failed = rows.filter(function(r) { return r.verdict === "fail"; }).length;
        var pending = rows.filter(function(r) { return !r.verdict; }).length;
        var summary = '<div class="sessions-summary">' +
          '<span class="verdict verdict-pass sess-count">✓ ' + passed + ' başarılı</span>' +
          '<span class="verdict verdict-fail sess-count">✗ ' + failed + ' başarısız</span>' +
          (pending ? '<span class="verdict verdict-inconclusive sess-count">~ ' + pending + ' bekliyor</span>' : '') +
          '</div>';
        el.innerHTML = summary + rows.map(function (r) {
          var vKey = r.verdict || "pending";
          var vClass = "verdict-" + (r.verdict || "inconclusive");
          var vLabel = vLabels[r.verdict] || "⏳ test bekleniyor";
          return (
            '<div class="hist-row">' +
            '<div class="hist-head">' +
            '<span class="hist-name">' + esc((r.indicator_name || "—")) + "</span>" +
            '<span class="muted small">iter ' + r.iteration + "</span>" +
            '<span class="verdict ' + vClass + ' hist-verdict">' + vLabel + "</span>" +
            '<span class="muted small">' + esc((r.created_at || "").slice(0, 10)) + "</span>" +
            "</div>" +
            '<div class="muted small">' + esc((r.question || "").slice(0, 80)) + "</div>" +
            "</div>"
          );
        }).join("");
      })
      .catch(function () {});
  }

  var refreshSessions = document.getElementById("refreshSessions");
  if (refreshSessions) refreshSessions.addEventListener("click", loadResearchSessions);

  var buildChainBtn = document.getElementById("buildChainBtn");
  if (buildChainBtn) {
    buildChainBtn.addEventListener("click", function () {
      var onlySuccessful = document.getElementById("onlySuccessful").checked;
      buildChainBtn.disabled = true;
      buildChainBtn.innerHTML = '<span class="spinner"></span>OLUŞTURULUYOR…';
      var res = document.getElementById("chainResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span></div>';
      api("/research/chain-dataset?only_successful=" + onlySuccessful, { method: "POST" })
        .then(function (data) {
          toast(data.n_records + " zincir kaydı üretildi.");
          res.innerHTML =
            '<div class="result-section result-body">' +
            data.n_records + " kayıt · hash: " + esc(data.content_hash) +
            "</div>";
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          buildChainBtn.disabled = false;
          buildChainBtn.textContent = "⚡ ZİNCİR DATASET OLUŞTUR";
        });
    });
  }

  // ---------- sentez makaleleri ----------
  function loadSynthPapers() {
    var list = document.getElementById("synthPapersList");
    if (!list) return;
    api("/synthesis/reports", { method: "GET" })
      .then(function (d) {
        var reports = (d && d.reports) || [];
        if (!reports.length) {
          list.innerHTML =
            '<p class="muted small">Henüz sentez makalesi yok — önce Agentic Araştırma çalıştır, sonra ÜRET.</p>';
          return;
        }
        list.innerHTML = reports
          .map(function (r) {
            return (
              '<div class="session-item">' +
              '<a href="/api/synthesis/reports/' + encodeURIComponent(r.name) +
              '" download>📄 ' + esc(r.name) + "</a>" +
              '<span class="muted small"> · ' + r.size_kb + " KB · " + esc(r.modified) + "</span>" +
              "</div>"
            );
          })
          .join("");
      })
      .catch(function () {});
  }
  var genSynthBtn = document.getElementById("genSynthPaperBtn");
  if (genSynthBtn) {
    genSynthBtn.addEventListener("click", function () {
      genSynthBtn.disabled = true;
      genSynthBtn.innerHTML = '<span class="spinner"></span>ÜRETİLİYOR…';
      var res = document.getElementById("synthPaperResult");
      res.className = "result";
      api("/synthesis/reports/generate", { method: "POST" })
        .then(function (d) {
          if (d.ok) {
            toast("Sentez makalesi üretildi: " + d.name);
            res.innerHTML =
              '<div class="result-section result-body">✓ ' + esc(d.name) + " üretildi.</div>";
            loadSynthPapers();
          } else {
            res.innerHTML =
              '<div class="result-section result-body">' + esc(d.message || "Üretilemedi") + "</div>";
          }
        })
        .catch(function (err) {
          toast(err.message, true);
          res.innerHTML =
            '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          genSynthBtn.disabled = false;
          genSynthBtn.textContent = "📄 SENTEZ MAKALESİ ÜRET";
        });
    });
  }
  var refreshSynthBtn = document.getElementById("refreshSynthPapers");
  if (refreshSynthBtn) refreshSynthBtn.addEventListener("click", loadSynthPapers);
  loadSynthPapers();

  // ---------- model değerlendirme ----------
  function loadEvalSets() {
    var sel = document.getElementById("evalSetSelect");
    if (!sel) return;
    api("/eval/sets", { method: "GET" })
      .then(function (sets) {
        while (sel.options.length > 0) sel.remove(0);
        if (!sets.length) {
          var opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "Eval seti bulunamadı";
          sel.appendChild(opt);
          return;
        }
        sets.forEach(function (s) {
          var opt = document.createElement("option");
          opt.value = s.name;
          opt.textContent = s.name + " (" + s.n_items + " soru)";
          sel.appendChild(opt);
        });
      })
      .catch(function () {});
  }

  var runEvalBtn = document.getElementById("runEvalBtn");
  if (runEvalBtn) {
    runEvalBtn.addEventListener("click", function () {
      var evalSet = document.getElementById("evalSetSelect").value;
      var adapterV = document.getElementById("evalAdapterSelect").value || null;
      if (!evalSet) { toast("Eval seti seç.", true); return; }
      runEvalBtn.disabled = true;
      runEvalBtn.innerHTML = '<span class="spinner"></span>ÇALIŞIYOR';
      var res = document.getElementById("evalResult");
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> LLM yanıtları alınıyor…</div>';
      api("/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eval_set: evalSet, adapter_version: adapterV }),
      })
        .then(function (data) {
          renderEvalResult(res, data);
        })
        .catch(function (err) {
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          runEvalBtn.disabled = false;
          runEvalBtn.textContent = "DEĞERLENDİR →";
        });
    });
  }

  function renderEvalResult(res, data) {
    var scorePct = Math.round(data.score * 100);
    var scoreClass = scorePct >= 80 ? "pos" : scorePct >= 60 ? "" : "neg";
    var rows = (data.rows || []).map(function (r) {
      var flagHtml = r.flags.length
        ? r.flags.map(function (f) { return '<span class="eval-flag">' + esc(f) + "</span>"; }).join(" ")
        : '<span class="muted small">temiz</span>';
      return (
        '<div class="eval-row">' +
        '<div class="eval-q"><b>S:</b> ' + esc(r.question) + "</div>" +
        '<div class="eval-a"><b>C:</b> ' + esc(r.answer.slice(0, 200)) + (r.answer.length > 200 ? "…" : "") + "</div>" +
        '<div class="eval-flags">' + flagHtml + "</div>" +
        "</div>"
      );
    }).join("");
    res.innerHTML =
      '<div class="result-section"><div class="result-label">özet</div>' +
      '<div class="metrics-grid">' +
      '<div class="metric"><div class="k">skor</div><div class="v ' + scoreClass + '">' + scorePct + '%</div></div>' +
      '<div class="metric"><div class="k">soru</div><div class="v">' + data.n_items + "</div></div>" +
      '<div class="metric"><div class="k">bayrak</div><div class="v' + (data.total_flags > 0 ? " neg" : " pos") + '">' + data.total_flags + "</div></div>" +
      "</div>" +
      (data.adapter_version ? '<div class="muted small">adapter: ' + esc(data.adapter_version) + "</div>" : "") +
      "</div>" +
      '<div class="result-section"><div class="result-label">sorular</div>' +
      '<div class="eval-rows">' + rows + "</div></div>";
  }

  // ---------- backtest geçmişi ----------
  function loadBacktestHistory() {
    var el = document.getElementById("btHistory");
    if (!el) return;
    el.innerHTML = '<div class="empty"><span class="spinner"></span> yükleniyor…</div>';
    api("/backtests?limit=30", { method: "GET" })
      .then(function (data) {
        var recs = data.records || [];
        if (!recs.length) {
          el.innerHTML = '<div class="empty">Henüz kayıtlı backtest yok.</div>';
          return;
        }
        el.innerHTML = recs
          .map(function (r) {
            var v = r.verdict || "inconclusive";
            var vClass = "verdict-" + v;
            var vLabel = { pass: "GEÇTİ", fail: "BAŞARISIZ", inconclusive: "SONUÇSUZ" }[v] || v;
            var ret = r.total_return_pct != null ? Number(r.total_return_pct).toFixed(3) : "—";
            var sh = r.sharpe != null ? Number(r.sharpe).toFixed(3) : "—";
            var dd = r.max_drawdown_pct != null ? Number(r.max_drawdown_pct).toFixed(3) : "—";
            var wr = r.win_rate_pct != null ? Number(r.win_rate_pct).toFixed(1) : "—";
            return (
              '<div class="hist-row">' +
              '<div class="hist-head">' +
              '<span class="hist-name">' + esc(r.strategy_name) + "</span>" +
              (r.market ? '<span class="muted small">' + esc(r.market) + (r.timeframe ? "/" + esc(r.timeframe) : "") + "</span>" : "") +
              '<span class="verdict ' + vClass + ' hist-verdict">' + vLabel + "</span>" +
              '<span class="muted small">' + esc((r.created_at || "").slice(0, 10)) + "</span>" +
              "</div>" +
              '<div class="metrics-grid">' +
              '<div class="metric"><div class="k">getiri%</div><div class="v' + (r.total_return_pct > 0 ? " pos" : r.total_return_pct < 0 ? " neg" : "") + '">' + esc(ret) + "</div></div>" +
              '<div class="metric"><div class="k">sharpe</div><div class="v' + (r.sharpe > 0 ? " pos" : r.sharpe < 0 ? " neg" : "") + '">' + esc(sh) + "</div></div>" +
              '<div class="metric"><div class="k">maxDD%</div><div class="v neg">' + esc(dd) + "</div></div>" +
              '<div class="metric"><div class="k">win%</div><div class="v">' + esc(wr) + "</div></div>" +
              '<div class="metric"><div class="k">işlem</div><div class="v">' + esc(r.n_trades) + "</div></div>" +
              "</div>" +
              (r.notes ? '<div class="hist-notes muted small">' + esc(r.notes) + "</div>" : "") +
              '<div class="hist-actions">' +
              '<button class="btn btn-pine" data-btid="' + esc(r.backtest_id) + '" title="Pine Script\'i göster ve kopyala">🌲 Pine Kopyala</button>' +
              '<button class="btn btn-risk" data-btid="' + esc(r.backtest_id) + '" title="Kelly + drawdown risk analizi">⚖ Risk Analizi</button>' +
              '<button class="btn btn-pkg" data-btid="' + esc(r.backtest_id) + '" data-stratname="' + esc(r.strategy_name) + '" title=".achpkg paketini indir">⬇ .achpkg</button>' +
              "</div>" +
              "</div>"
            );
          })
          .join("");

        el.querySelectorAll(".btn-pine").forEach(function (b) {
          b.addEventListener("click", function () { openPineModal(b.getAttribute("data-btid")); });
        });
        el.querySelectorAll(".btn-risk").forEach(function (b) {
          b.addEventListener("click", function () { openRiskModal(b.getAttribute("data-btid")); });
        });
        el.querySelectorAll(".btn-pkg").forEach(function (b) {
          b.addEventListener("click", function () {
            downloadAchpkg(b.getAttribute("data-btid"), b.getAttribute("data-stratname"));
          });
        });
      })
      .catch(function (err) {
        el.innerHTML = '<div class="empty">Hata: ' + esc(err.message) + "</div>";
      });
  }

  function downloadAchpkg(btId, stratName) {
    toast(".achpkg hazırlanıyor…");
    fetch("/api/backtest/" + btId + "/download-pkg", { headers: authHeaders() })
      .then(function (r) {
        if (r.status === 401) { toast("Yetkisiz — API token gerekli.", true); throw new Error("unauthorized"); }
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.blob();
      })
      .then(function (blob) {
        var safeName = stratName.replace(/[^\w\-]/g, "_") || btId;
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = safeName + ".achpkg";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast(".achpkg indirildi ✓");
      })
      .catch(function (err) { toast("İndirme hatası: " + esc(err.message || "Bilinmeyen hata"), true); });
  }

  // ---------- Pine Script modal ----------
  var pineModal = null;

  function openPineModal(btId) {
    if (!pineModal) {
      pineModal = document.createElement("div");
      pineModal.id = "pineModal";
      pineModal.className = "modal";
      pineModal.innerHTML =
        '<div class="modal-box" style="max-width:700px">' +
        '<button class="modal-close" id="pineClose">✕</button>' +
        '<div id="pineBody"></div>' +
        "</div>";
      document.body.appendChild(pineModal);
      document.getElementById("pineClose").addEventListener("click", function () {
        pineModal.className = "modal hidden";
      });
    }
    var body = document.getElementById("pineBody");
    body.innerHTML = '<div class="result-section"><span class="spinner"></span> yükleniyor…</div>';
    pineModal.className = "modal";

    api("/backtest/" + btId + "/pine", { method: "GET" })
      .then(function (d) {
        body.innerHTML =
          '<div class="result-section">' +
          '<div class="result-label">strateji</div>' +
          '<b>' + esc(d.strategy_name) + '</b> · ' + esc(d.market) + '/' + esc(d.timeframe) +
          '</div>' +
          '<div class="result-section">' +
          '<div class="result-label" style="display:flex;justify-content:space-between">' +
          '<span>Pine Script v5</span>' +
          '<button class="btn" id="copyPineBtn">📋 Panoya Kopyala</button>' +
          '</div>' +
          '<pre class="pine-code" id="pineCodePre" style="white-space:pre-wrap;font-size:12px;overflow:auto;max-height:400px">' +
          esc(d.pine_code) + '</pre>' +
          '</div>' +
          '<p class="muted small" style="margin-top:8px">' +
          'TradingView → Pine Script Editörü → Yeni → Kodu yapıştır → Derle → Strateji Tester' +
          '</p>';
        document.getElementById("copyPineBtn").addEventListener("click", function () {
          navigator.clipboard.writeText(d.pine_code).then(function () {
            toast("Pine Script panoya kopyalandı ✓");
          }).catch(function () {
            toast("Kopyalanamadı — kodu elle seç.", true);
          });
        });
      })
      .catch(function (err) {
        body.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
      });
  }

  // ---------- Risk Analizi modal ----------
  var riskModalEl = document.getElementById("riskModal");
  if (riskModalEl) {
    document.getElementById("riskClose").addEventListener("click", function () {
      riskModalEl.className = "modal hidden";
    });
  }

  function openRiskModal(btId) {
    var body = document.getElementById("riskBody");
    body.innerHTML = '<div class="result-section"><span class="spinner"></span> risk hesaplanıyor…</div>';
    riskModalEl.className = "modal";

    api("/backtest/" + btId + "/risk?equity_usd=10000&max_dd_pct=-20&risk_pct=1&stop_pct=2", { method: "GET" })
      .then(function (d) {
        var k = d.kelly;
        var dd = d.drawdown_scale;
        var fr = d.fixed_risk;
        var warnHtml = d.warnings.length
          ? '<div class="result-section"><div class="result-label">⚠ uyarılar</div>' +
            d.warnings.map(function (w) { return '<div class="muted small">' + esc(w) + '</div>'; }).join("") +
            '</div>'
          : "";
        body.innerHTML =
          '<div class="result-section"><div class="result-label">strateji — risk analizi</div>' +
          '<b>' + esc(d.strategy_name) + '</b>  ·  ' + d.n_trades + ' işlem' +
          '</div>' +

          '<div class="result-section"><div class="result-label">Kelly Kriteri</div>' +
          '<div class="risk-grid">' +
          '<div class="risk-item"><div class="k">Kazanma oranı</div><div class="v">' + (k.win_rate*100).toFixed(1) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Ort. kazanç</div><div class="v pos">' + (k.avg_win*100).toFixed(2) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Ort. kayıp</div><div class="v neg">-' + (k.avg_loss*100).toFixed(2) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Odds (b)</div><div class="v">' + k.odds.toFixed(2) + '</div></div>' +
          '<div class="risk-item"><div class="k">Tam Kelly</div><div class="v">' + (k.full_kelly*100).toFixed(1) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Yarı Kelly</div><div class="v pos">' + (k.half_kelly*100).toFixed(1) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Çeyrek Kelly</div><div class="v">' + (k.quarter_kelly*100).toFixed(1) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Sınırlı Kelly</div><div class="v pos"><b>' + (k.capped_kelly*100).toFixed(1) + '%</b></div></div>' +
          '</div></div>' +

          '<div class="result-section"><div class="result-label">Drawdown Ölçekleme (eşik: ' + dd.max_allowed_pct + '%)</div>' +
          '<div class="risk-grid">' +
          '<div class="risk-item"><div class="k">Anlık DD</div><div class="v ' + (dd.in_drawdown_zone ? 'neg' : '') + '">' + dd.current_drawdown_pct.toFixed(1) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Ölçek faktörü</div><div class="v ' + (dd.scale_factor < 1 ? 'neg' : 'pos') + '">' + (dd.scale_factor*100).toFixed(0) + '%</div></div>' +
          '<div class="risk-item"><div class="k">DD bölgesi</div><div class="v ' + (dd.in_drawdown_zone ? 'neg' : 'pos') + '">' + (dd.in_drawdown_zone ? '⚠ EVET' : '✓ HAYIR') + '</div></div>' +
          '</div></div>' +

          '<div class="result-section"><div class="result-label">Sabit Risk (sermaye: 10.000 $, risk: %1, stop: %2)</div>' +
          '<div class="risk-grid">' +
          '<div class="risk-item"><div class="k">Pozisyon %</div><div class="v">' + fr.position_size_pct.toFixed(1) + '%</div></div>' +
          '<div class="risk-item"><div class="k">Pozisyon $</div><div class="v"><b>' + fr.position_size_usd.toLocaleString() + ' $</b></div></div>' +
          '</div></div>' +

          warnHtml +
          '<div class="result-section"><div class="result-label">öneri</div>' +
          '<div class="result-body muted small">' + esc(d.recommendation) + '</div></div>' +
          (d.report_id
            ? '<div class="result-section"><span class="badge badge-ok">✓ Kaydedildi — ' + esc(d.report_id) + '</span></div>'
            : '');
        loadRiskReports();
      })
      .catch(function (err) {
        body.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + '</div>';
      });
  }

  function loadRiskReports() {
    var list = document.getElementById("riskReportsList");
    if (!list) return;
    list.innerHTML = '<span class="spinner"></span>';
    api("/risk-reports?limit=20", { method: "GET" })
      .then(function (d) {
        if (!d.reports || d.reports.length === 0) {
          list.innerHTML = '<p class="muted small">Henüz kaydedilmiş risk raporu yok.</p>';
          return;
        }
        list.innerHTML = d.reports.map(function (r) {
          return '<div class="bt-card">' +
            '<div class="bt-meta">' +
            '<b>' + esc(r.strategy_name) + '</b>' +
            '<span class="muted small"> · ' + r.n_trades + ' işlem · ' + new Date(r.created_at).toLocaleString("tr-TR") + '</span>' +
            '</div>' +
            '<div class="risk-grid" style="margin-top:6px">' +
            '<div class="risk-item"><div class="k">Kazanma</div><div class="v">' + (r.win_rate * 100).toFixed(1) + '%</div></div>' +
            '<div class="risk-item"><div class="k">Yarı Kelly</div><div class="v pos">' + (r.half_kelly * 100).toFixed(1) + '%</div></div>' +
            '<div class="risk-item"><div class="k">Sınırlı Kelly</div><div class="v pos"><b>' + (r.capped_kelly * 100).toFixed(1) + '%</b></div></div>' +
            '<div class="risk-item"><div class="k">Ölçek</div><div class="v ' + (r.scale_factor < 1 ? 'neg' : 'pos') + '">' + (r.scale_factor * 100).toFixed(0) + '%</div></div>' +
            '<div class="risk-item"><div class="k">Pozisyon %</div><div class="v">' + r.position_size_pct.toFixed(1) + '%</div></div>' +
            '<div class="risk-item"><div class="k">Pozisyon $</div><div class="v">' + r.position_size_usd.toLocaleString() + ' $</div></div>' +
            '</div>' +
            '<div class="muted small" style="margin-top:4px">backtest: ' + esc(r.backtest_id) + '</div>' +
            '</div>';
        }).join("");
      })
      .catch(function () {
        list.innerHTML = '<p class="muted small">Risk raporları yüklenemedi.</p>';
      });
  }

  var refreshHistory = document.getElementById("refreshHistory");
  if (refreshHistory) {
    refreshHistory.addEventListener("click", loadBacktestHistory);
  }

  var refreshRiskReports = document.getElementById("refreshRiskReports");
  if (refreshRiskReports) {
    refreshRiskReports.addEventListener("click", loadRiskReports);
  }

  // ---------- özel strateji IR backtest ----------
  var customBtBtn = document.getElementById("customBtBtn");
  if (customBtBtn) {
    customBtBtn.addEventListener("click", function () {
      var raw = document.getElementById("customIR").value.trim();
      var n = parseInt(document.getElementById("customNBars").value, 10) || 2000;
      var seed = parseInt(document.getElementById("customSeed").value, 10) || 42;
      var strategyIr = null;
      if (raw) {
        try {
          strategyIr = JSON.parse(raw);
        } catch (e) {
          toast("Geçersiz JSON: " + e.message, true);
          return;
        }
      }
      var res = document.getElementById("btResult");
      customBtBtn.disabled = true;
      customBtBtn.innerHTML = '<span class="spinner"></span>ÇALIŞIYOR';
      res.className = "result";
      res.innerHTML = '<div class="result-section"><span class="spinner"></span> özel IR backtest…</div>';
      api("/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_synthetic: true, n_bars: n, seed: seed, strategy_ir: strategyIr }),
      })
        .then(function (data) {
          renderBacktest(res, data);
          loadBacktestHistory();
        })
        .catch(function (err) {
          res.innerHTML = '<div class="result-section result-body">Hata: ' + esc(err.message) + "</div>";
        })
        .finally(function () {
          customBtBtn.disabled = false;
          customBtBtn.textContent = "ÖZEL IR ÇALIŞTIR →";
        });
    });
  }

  // ---------- kart onay ----------
  function loadPendingCards() {
    var container = document.getElementById("pending-cards-list");
    if (!container) return;
    container.innerHTML = '<div class="empty"><span class="spinner"></span> yükleniyor…</div>';
    api("/cards/pending", { method: "GET" })
      .then(function (data) {
        if (!data.cards || data.cards.length === 0) {
          container.innerHTML = '<p style="color:#94a3b8">Onay bekleyen kart yok.</p>';
        } else {
          var html = '<p style="color:#94a3b8;margin-bottom:8px">' + data.total + ' kart onay bekliyor</p>';
          data.cards.forEach(function (card) {
            var diffPct = Math.round(card.difficulty * 100);
            var stageLabel = card.stage || "—";
            // Boş/placeholder kart: backend içerik guard'ı bunu onaylamaz (title ve
            // main_claim alfanümerik taşımıyor → eğitime boş kart sokulmaz). Kullanıcıya
            // Onayla'nın neden çalışmadığını ÖNCEDEN göster ve butonu devre dışı bırak;
            // tek geçerli eylem Reddet'tir (kuyruğu temizler).
            var hasContent = /[0-9A-Za-zÇĞİÖŞÜçğıöşü]/.test(
              (card.title || "") + " " + (card.main_claim || "")
            );
            var emptyBadge = hasContent
              ? ""
              : '<span class="badge badge-warning" style="margin-left:4px">boş içerik — onaylanamaz</span>';
            var approveBtn = hasContent
              ? '<button class="btn btn-success btn-approve" data-card-id="' + esc(card.card_id) + '" style="padding:4px 12px">✓ Onayla</button>'
              : '<button class="btn btn-approve" data-card-id="' + esc(card.card_id) + '" style="padding:4px 12px;opacity:.45;cursor:not-allowed" disabled title="Boş/içeriksiz kart onaylanamaz — Reddet ile kuyruktan çıkar">✓ Onayla</button>';
            html +=
              '<div style="border:1px solid var(--border);border-radius:var(--radius);padding:10px;margin-bottom:8px">' +
              '<div style="display:flex;justify-content:space-between;align-items:center">' +
              '<div>' +
              '<strong>' + esc(card.title || card.paper_id) + "</strong>" +
              '<span class="badge badge-warning" style="margin-left:8px">' + esc(stageLabel) + "</span>" +
              '<span class="badge badge-info" style="margin-left:4px">zorluk %' + diffPct + "</span>" +
              emptyBadge +
              "</div>" +
              '<div style="display:flex;gap:8px">' +
              approveBtn +
              '<button class="btn btn-danger btn-reject" data-card-id="' + esc(card.card_id) + '" style="padding:4px 12px">✗ Reddet</button>' +
              "</div></div>" +
              '<p style="color:#94a3b8;margin-top:6px;font-size:12px">' + esc(card.main_claim || "") + "</p>" +
              '<small style="color:#94a3b8">kart: ' + esc(card.card_id) + " · model: " + esc(card.model) + "</small>" +
              "</div>";
          });
          container.innerHTML = html;
          container.querySelectorAll(".btn-approve").forEach(function (btn) {
            btn.addEventListener("click", function () {
              var cid = btn.getAttribute("data-card-id");
              api("/card/" + encodeURIComponent(cid) + "/approve", { method: "POST" })
                .then(function (d) {
                  toast(d.message);
                  loadPendingCards();
                })
                .catch(function (err) { toast(err.message, true); });
            });
          });
          container.querySelectorAll(".btn-reject").forEach(function (btn) {
            btn.addEventListener("click", function () {
              var cid = btn.getAttribute("data-card-id");
              api("/card/" + encodeURIComponent(cid) + "/reject", { method: "POST" })
                .then(function (d) {
                  toast(d.message);
                  loadPendingCards();
                })
                .catch(function (err) { toast(err.message, true); });
            });
          });
        }
        // Onaylananlar özeti
        api("/cards/approved", { method: "GET" })
          .then(function (d) {
            var el = document.getElementById("approved-cards-summary");
            if (el) el.textContent = d.total + " onaylı kart · LoRA'ya girebilir";
          })
          .catch(function () {});
      })
      .catch(function (err) {
        container.innerHTML = '<div class="empty">Hata: ' + esc(err.message) + "</div>";
      });
  }

  var btnLoadPending = document.getElementById("btn-load-pending");
  if (btnLoadPending) {
    btnLoadPending.addEventListener("click", loadPendingCards);
  }

  // ---------- sistem durumu ----------
  function loadSystemStatus() {
    var el = document.getElementById("systemStatusDisplay");
    if (!el) return;
    el.innerHTML = '<span class="spinner"></span> güncelleniyor…';
    api("/status", { method: "GET" })
      .then(function (s) {
        el.innerHTML =
          '<div class="metrics-grid" style="margin-top:8px">' +
          // "LLM model" tek basina yaniltiyordu: bu OLLAMA CIKARIM modelidir (RAG / bilgi
          // karti / sohbet), EGITILEN model DEGILDIR. Egitim temel modeli ayri gosterilir
          // (05 · EGITIM sekmesi, canli ilerleme).
          '<div class="metric"><div class="k">LLM modeli (RAG/sohbet)</div><div class="v">' + esc(s.llm_model) + '</div></div>' +
          '<div class="metric"><div class="k">Ollama</div><div class="v ' + (s.ollama_available ? "pos" : "neg") + '">' + (s.ollama_available ? "bağlı ✓" : "kapalı ✗") + '</div></div>' +
          '<div class="metric"><div class="k">Embed modu</div><div class="v">' + esc(s.embedding_mode) + '</div></div>' +
          '<div class="metric"><div class="k">Makale sayısı</div><div class="v">' + s.n_papers + '</div></div>' +
          '<div class="metric"><div class="k">Chunk sayısı</div><div class="v">' + s.n_chunks + '</div></div>' +
          '<div class="metric"><div class="k">Max yükleme</div><div class="v">' + s.max_upload_mb + ' MB</div></div>' +
          '</div>';
      })
      .catch(function () {
        el.innerHTML = '<span class="conn-err">Durum alınamadı.</span>';
      });
  }

  // ---------- token ----------
  var tokenInput = document.getElementById("tokenInput");
  if (tokenInput) tokenInput.value = getToken();
  var saveTokenBtn = document.getElementById("saveToken");
  if (saveTokenBtn) saveTokenBtn.addEventListener("click", function () {
    setToken(tokenInput ? tokenInput.value.trim() : "");
    toast("Token kaydedildi.");
    refreshStatus();
  });

  // ---------- auto-lora pipeline ----------
  var _STAGE_LABELS = {
    idle: 'Boşta', checking: 'Gate kontrol ediliyor…', gate_failed: 'Gate başarısız',
    ready_to_train: 'Gate geçti — Eğitim onayı bekleniyor', training: 'Eğitim sürüyor…',
    train_failed: 'Eğitim başarısız', evaluating: 'Eval çalışıyor…',
    eval_failed: 'Eval başarısız', eval_passed: 'Eval geçti — Production onayı bekleniyor',
    promoted: 'Production\'a terfi edildi ✓'
  };
  var _STAGE_COLORS = {
    idle: '#94a3b8', checking: '#facc15', gate_failed: '#f87171',
    ready_to_train: '#60a5fa', training: '#a78bfa', train_failed: '#f87171',
    evaluating: '#a78bfa', eval_failed: '#f87171', eval_passed: '#34d399', promoted: '#4ade80'
  };

  function renderAutoLoraStatus(d) {
    var label = _STAGE_LABELS[d.stage] || d.stage;
    var color = _STAGE_COLORS[d.stage] || '#94a3b8';
    var rows = [
      ['Durum', '<strong style="color:' + color + '">' + esc(label) + '</strong>'],
      ['Onaylı kart', esc(d.approved_cards_at_last_check) + ' / min ' + esc(d.min_eligible_cards)],
      ['Son kontrol', d.last_check ? esc(d.last_check.slice(0,16).replace('T',' ')) : '—'],
      ['Gate özeti', esc(d.gate_summary || '—')],
      ['Adapter', esc(d.adapter_id || '—')],
      ['Hata', esc(d.last_error || '—')]
    ];
    var html = '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    rows.forEach(function(r) {
      html += '<tr><td style="padding:2px 10px 2px 0;color:#94a3b8;white-space:nowrap">' + r[0] + '</td><td>' + r[1] + '</td></tr>';
    });
    html += '</table>';
    var trainBtn = document.getElementById('autoLoraTrainBtn');
    var promoteBtn = document.getElementById('autoLoraPromoteBtn');
    var enableLabel = document.getElementById('autoLoraEnabledLabel');
    if (trainBtn) trainBtn.disabled = d.stage !== 'ready_to_train';
    if (promoteBtn) promoteBtn.disabled = d.stage !== 'eval_passed';
    if (enableLabel) {
      enableLabel.textContent = d.auto_enabled ? 'açık' : 'kapalı';
      enableLabel.className = 'badge ' + (d.auto_enabled ? 'badge-success' : 'badge-info');
    }
    return html;
  }

  function loadAutoLoraStatus() {
    var body = document.getElementById('autoLoraStatusBody');
    if (!body) return;
    api('/auto-lora/status', { method: 'GET' })  // api() = auth header + hata yönetimi
      .then(function(d) { body.innerHTML = renderAutoLoraStatus(d); })
      .catch(function() { body.innerHTML = '<span class="muted small">Durum alınamadı</span>'; });
  }

  function autoLoraMsg(text, ok) {
    var el = document.getElementById('autoLoraMsg');
    if (!el) return;
    el.textContent = text;
    el.className = 'result ' + (ok ? '' : 'error');
  }

  var autoLoraRefreshBtn = document.getElementById('autoLoraRefreshBtn');
  var autoLoraEnableBtn  = document.getElementById('autoLoraEnableBtn');
  var autoLoraDisableBtn = document.getElementById('autoLoraDisableBtn');
  var autoLoraCheckBtn   = document.getElementById('autoLoraCheckBtn');
  var autoLoraTrainBtn   = document.getElementById('autoLoraTrainBtn');
  var autoLoraPromoteBtn = document.getElementById('autoLoraPromoteBtn');
  var autoLoraResetBtn   = document.getElementById('autoLoraResetBtn');

  if (autoLoraRefreshBtn) autoLoraRefreshBtn.addEventListener('click', loadAutoLoraStatus);

  if (autoLoraEnableBtn) autoLoraEnableBtn.addEventListener('click', function() {
    api('/auto-lora/enable?enabled=true', { method: 'POST' })
      .then(function() { autoLoraMsg('Otomatik kontrol açıldı.', true); loadAutoLoraStatus(); })
      .catch(function(e) { autoLoraMsg('Bağlantı hatası: ' + e.message, false); });
  });
  if (autoLoraDisableBtn) autoLoraDisableBtn.addEventListener('click', function() {
    api('/auto-lora/enable?enabled=false', { method: 'POST' })
      .then(function() { autoLoraMsg('Otomatik kontrol kapatıldı.', true); loadAutoLoraStatus(); })
      .catch(function(e) { autoLoraMsg('Bağlantı hatası: ' + e.message, false); });
  });
  if (autoLoraCheckBtn) autoLoraCheckBtn.addEventListener('click', function() {
    autoLoraMsg('Gate 0-8 kontrol ediliyor…', true);
    api('/auto-lora/check', { method: 'POST' })
      .then(function(d) {
        autoLoraMsg(d.ok ? 'Gate geçti: ' + (d.summary||'') : 'Gate başarısız: ' + (d.reason||''), d.ok);
        loadAutoLoraStatus();
      }).catch(function(e) { autoLoraMsg('Bağlantı hatası: ' + e.message, false); });
  });
  if (autoLoraTrainBtn) autoLoraTrainBtn.addEventListener('click', function() {
    var nameEl = document.getElementById('autoLoraAdapterName') || document.getElementById('trAdapterName');
    var itersEl = document.getElementById('autoLoraIters') || document.getElementById('drIterations');
    var name = (nameEl && nameEl.value.trim()) || ('hektor_auto_' + Date.now());
    var iters = parseInt((itersEl && itersEl.value) || '300', 10);
    // Backend ile aynı kısıt (path-traversal/arg güvenliği) — erken, net geri bildirim.
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) {
      autoLoraMsg('Adapter adı yalnız harf, rakam, _ ve - içerebilir (en çok 64).', false); return;
    }
    if (iters < 50 || iters > 5000) { autoLoraMsg('İterasyon 50–5000 arasında olmalı.', false); return; }
    if (!confirm('Eğitim başlatılacak:\nAdapter: ' + name + '\nİterasyon: ' + iters + '\n\nOnaylıyor musun?')) return;
    autoLoraMsg('Eğitim başlatılıyor…', true);
    api('/auto-lora/train?adapter_name=' + encodeURIComponent(name) + '&iters=' + iters, { method: 'POST' })
      .then(function(d) {
        autoLoraMsg(d.ok ? '✓ Eğitim başladı: ' + esc(d.adapter_name) + ' (' + iters + ' iter)' : 'Hata: ' + (d.reason || 'bilinmiyor'), d.ok);
        loadAutoLoraStatus();
      }).catch(function(e) { autoLoraMsg('Bağlantı hatası: ' + e.message, false); });
  });
  if (autoLoraPromoteBtn) autoLoraPromoteBtn.addEventListener('click', function() {
    if (!confirm('Bu adapter production\'a terfi ettirilecek. Onaylıyor musun?')) return;
    api('/auto-lora/promote', { method: 'POST' })
      .then(function(d) {
        autoLoraMsg(d.ok ? 'Terfi edildi: ' + esc(d.adapter_id) : 'Hata: ' + (d.reason || 'bilinmiyor'), d.ok);
        loadAutoLoraStatus();
      }).catch(function(e) { autoLoraMsg('Bağlantı hatası: ' + e.message, false); });
  });
  if (autoLoraResetBtn) autoLoraResetBtn.addEventListener('click', function() {
    if (!confirm('Pipeline IDLE\'a sıfırlanacak. Devam?')) return;
    api('/auto-lora/reset', { method: 'POST' })
      .then(function() { autoLoraMsg('Sıfırlandı.', true); loadAutoLoraStatus(); })
      .catch(function(e) { autoLoraMsg('Bağlantı hatası: ' + e.message, false); });
  });

  loadAutoLoraStatus();

  // ---------- donanım profili + model önerisi ----------
  function renderHwProfile(profile, recs) {
    var loraBackend = profile.lora_backend || (profile.lora_supported ? 'mlx' : 'peft_cpu');
    var loraNote = loraBackend === 'mlx'
      ? '<span style="color:#4ade80">&#10003; LoRA egitimi destekleniyor (MLX - Apple Silicon, hizli)</span>'
      : loraBackend === 'peft_cuda'
        ? '<span style="color:#4ade80">&#10003; LoRA egitimi destekleniyor (CUDA GPU, hizli)</span>'
        : '<span style="color:#facc15">&#9888; LoRA egitimi CPU ile calisir (yavas ~2-4 saat) &mdash; Hizli egitim icin Egitim sekmesindeki "Colab Notebook Indir" dugmesini kullanin.</span>';
    var recsHtml = recs.recommended.length
      ? recs.recommended.map(function (r) {
          return (
            '<tr><td><strong>' + r.name + '</strong></td>' +
            '<td><code>' + r.ollama + '</code></td>' +
            '<td>' + r.confidence + '%</td></tr>'
          );
        }).join('')
      : '<tr><td colspan="3" class="muted">Öneri bulunamadı.</td></tr>';

    return (
      '<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:10px">' +
      '<tr><td style="padding:3px 8px 3px 0"><strong>İşletim Sistemi</strong></td><td>' + profile.os + ' ' + profile.arch + '</td></tr>' +
      '<tr><td style="padding:3px 8px 3px 0"><strong>CPU</strong></td><td>' + profile.cpu + ' (' + profile.cores + ' çekirdek)</td></tr>' +
      '<tr><td style="padding:3px 8px 3px 0"><strong>RAM</strong></td><td>' + profile.ram_gb + ' GB</td></tr>' +
      '<tr><td style="padding:3px 8px 3px 0"><strong>GPU</strong></td><td>' + profile.gpu + '</td></tr>' +
      '</table>' +
      loraNote +
      '<h4 style="margin:12px 0 6px">Önerilen Modeller</h4>' +
      '<table style="width:100%;border-collapse:collapse;font-size:13px">' +
      '<thead><tr><th style="text-align:left;padding-bottom:4px">Model</th><th style="text-align:left">Ollama Komutu</th><th style="text-align:left">Uyum</th></tr></thead>' +
      '<tbody>' + recsHtml + '</tbody></table>'
    );
  }

  function loadHwProfile(targetEl) {
    Promise.all([
      fetch('/api/profile').then(function (r) { return r.json(); }),
      fetch('/api/recommend').then(function (r) { return r.json(); })
    ]).then(function (results) {
      targetEl.innerHTML = renderHwProfile(results[0], results[1]);
    }).catch(function () {
      targetEl.innerHTML = '<span class="muted small">Profil alınamadı.</span>';
    });
  }

  var hwProfileBody = document.getElementById('hwProfileBody');
  if (hwProfileBody) loadHwProfile(hwProfileBody);

  var setupModal = document.getElementById('setupModal');
  var setupModalBody = document.getElementById('setupModalBody');
  var setupModalClose = document.getElementById('setupModalClose');
  var setupModalDismiss = document.getElementById('setupModalDismiss');

  if (setupModal && !localStorage.getItem('hektor_setup_seen')) {
    loadHwProfile(setupModalBody);
    setupModal.classList.remove('hidden');
  }

  function closeSetupModal() {
    if (setupModal) setupModal.classList.add('hidden');
    localStorage.setItem('hektor_setup_seen', '1');
  }
  if (setupModalClose) setupModalClose.addEventListener('click', closeSetupModal);
  if (setupModalDismiss) setupModalDismiss.addEventListener('click', closeSetupModal);

  // ---------- ilk-açılış "nasıl çalışır" şeridi (3 adım) ----------
  var introBanner = document.getElementById("introBanner");
  var introClose = document.getElementById("introClose");
  if (introBanner && !localStorage.getItem("hektor_intro_seen")) {
    introBanner.style.display = "block";
  }
  if (introClose) {
    introClose.addEventListener("click", function () {
      if (introBanner) introBanner.style.display = "none";
      try {
        localStorage.setItem("hektor_intro_seen", "1");
      } catch (e) {}
    });
  }

  // ---------- 09 · ÖĞRENME dashboard ----------

  function _svgLinePath(points, xScale, yScale, viewW, viewH, padX, padY) {
    if (!points.length) return '';
    return points.map((p, i) => {
      const x = padX + p.x * xScale;
      const y = viewH - padY - p.y * yScale;
      return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1);
    }).join(' ');
  }

  function renderLossCurve(curve) {
    const svg = document.getElementById('lrnLossSvg');
    if (!svg) return;
    if (!curve || !curve.length) {
      svg.innerHTML = '<text x="350" y="115" text-anchor="middle" fill="#666" font-size="13">Henüz eğitim verisi yok</text>';
      return;
    }
    const W = 700, H = 220, padX = 48, padY = 20;
    const iters = curve.map(d => d.iter);
    const trains = curve.map(d => d.train_loss);
    const vals = curve.filter(d => d.val_loss != null).map(d => d.val_loss);
    const maxIter = Math.max(...iters) || 1;
    const maxLoss = Math.max(...trains, ...vals, 0.01);
    const xS = (W - padX * 2) / maxIter;
    const yS = (H - padY * 2) / maxLoss;
    const trainPts = curve.map(d => ({ x: d.iter, y: d.train_loss }));
    const valPts = curve.filter(d => d.val_loss != null).map(d => ({ x: d.iter, y: d.val_loss }));
    const trainPath = _svgLinePath(trainPts, xS, yS, W, H, padX, padY);
    const valPath = _svgLinePath(valPts, xS, yS, W, H, padX, padY);
    // axis labels
    const yLabels = [0, maxLoss * 0.5, maxLoss].map(v => {
      const y = H - padY - v * yS;
      return `<text x="${padX - 4}" y="${y.toFixed(1)}" text-anchor="end" fill="#888" font-size="10">${v.toFixed(2)}</text>`;
    }).join('');
    const xLabels = [0, Math.round(maxIter / 2), maxIter].map(it => {
      const x = padX + it * xS;
      return `<text x="${x.toFixed(1)}" y="${H - 4}" text-anchor="middle" fill="#888" font-size="10">${it}</text>`;
    }).join('');
    svg.innerHTML = yLabels + xLabels +
      (trainPath ? `<path d="${trainPath}" stroke="#4af" stroke-width="1.8" fill="none"/>` : '') +
      (valPath   ? `<path d="${valPath}"   stroke="#fa4" stroke-width="1.8" fill="none" stroke-dasharray="4 2"/>` : '');
  }

  function renderCardGrowth(rows) {
    const svg = document.getElementById('lrnGrowthSvg');
    if (!svg) return;
    if (!rows || !rows.length) {
      svg.innerHTML = '<text x="350" y="95" text-anchor="middle" fill="#666" font-size="13">Henüz onaylı kart yok</text>';
      return;
    }
    const W = 700, H = 180, padX = 48, padY = 20;
    const maxCum = Math.max(...rows.map(r => r.cumulative), 1);
    const barW = Math.max(4, Math.floor((W - padX * 2) / rows.length) - 2);
    const xStep = (W - padX * 2) / rows.length;
    const yS = (H - padY * 2) / maxCum;
    const bars = rows.map((r, i) => {
      const bh = Math.max(2, r.cumulative * yS);
      const x = padX + i * xStep;
      const y = H - padY - bh;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${bh.toFixed(1)}" fill="#4af" opacity="0.7" rx="2"/>`;
    }).join('');
    const lastRow = rows[rows.length - 1];
    const topLabel = `<text x="${W - padX}" y="${padY}" text-anchor="end" fill="#4af" font-size="11">${lastRow.cumulative} toplam</text>`;
    svg.innerHTML = bars + topLabel;
  }

  function renderEvalTable(rows) {
    const el = document.getElementById('lrnEvalTable');
    if (!el) return;
    if (!rows || !rows.length) {
      el.innerHTML = '<span class="muted small">Henüz eval kaydı yok</span>';
      return;
    }
    const grouped = {};
    rows.forEach(r => {
      if (!grouped[r.adapter_name]) grouped[r.adapter_name] = [];
      grouped[r.adapter_name].push(r);
    });
    let html = '<table style="width:100%;border-collapse:collapse;font-size:.85rem"><thead><tr>' +
      '<th style="text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Adapter</th>' +
      '<th style="text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Eval Set</th>' +
      '<th style="text-align:right;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Geçiş %</th>' +
      '<th style="text-align:right;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Geçen/Toplam</th>' +
      '<th style="padding:.3rem .5rem;border-bottom:1px solid var(--border)">Tarih</th>' +
      '</tr></thead><tbody>';
    rows.slice().reverse().forEach(r => {
      const pct = (r.pass_rate * 100).toFixed(1);
      const color = r.pass_rate >= 0.5 ? '#4a4' : '#a44';
      html += `<tr>
        <td style="padding:.3rem .5rem;font-family:monospace">${esc(r.adapter_name)}</td>
        <td style="padding:.3rem .5rem">${esc(r.eval_set)}</td>
        <td style="text-align:right;padding:.3rem .5rem;color:${color}"><strong>${pct}%</strong></td>
        <td style="text-align:right;padding:.3rem .5rem">${r.passed_items || 0}/${r.total_items || 0}</td>
        <td style="padding:.3rem .5rem;color:#888;font-size:.75rem">${esc((r.scored_at||'').slice(0,10))}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    el.innerHTML = html;
  }

  // ---- Anlama merdiveni seviye sıralaması (Taban→L1→…→L5; alfabetik DEĞİL) ----
  var LADDER_ORDER = ['Taban', 'L1', 'L2', 'L3', 'L4', 'L5'];
  function orderedLevels(byLevel) {
    return Object.keys(byLevel || {}).sort(function (a, b) {
      var ia = LADDER_ORDER.indexOf(a); if (ia === -1) ia = LADDER_ORDER.length;
      var ib = LADDER_ORDER.indexOf(b); if (ib === -1) ib = LADDER_ORDER.length;
      return ia - ib || a.localeCompare(b);
    });
  }
  function levelBreakdown(byLevel) {
    return orderedLevels(byLevel).map(function (k) {
      var v = byLevel[k] || {};
      return k + ': ' + (v.passed || 0) + '/' + ((v.passed || 0) + (v.failed || 0));
    }).join(' · ');
  }
  function ctxSummary(ctx) {
    ctx = ctx || {};
    var parts = [ctx.model_kind || '', String(ctx.llm_model || '')];
    if (ctx.with_rag) parts.push('rag');
    return parts.filter(Boolean).join('·') || '—';
  }

  function renderUnderstandingHistory(rows) {
    var svg = document.getElementById('lrnUndSvg');
    var tbl = document.getElementById('lrnUndTable');
    if (!rows || !rows.length) {
      if (svg) svg.innerHTML = '<text x="350" y="85" text-anchor="middle" fill="#666" font-size="13">Henüz kayıt yok — "obj. anlama" rozetine tıkla</text>';
      if (tbl) tbl.innerHTML = '<span class="muted small">Henüz kayıtlı anlama skoru yok</span>';
      return;
    }
    var chrono = rows.slice().reverse();  // rows en yeni→eski; grafik eski→yeni
    if (svg) {
      var W = 700, H = 160, padX = 40, padY = 20, pts = [];
      chrono.forEach(function (r, i) {
        if (r.pass_rate == null) return;  // notlanmamış (graded=0) → sahte 0% çizme (Kural 2)
        var x = padX + (chrono.length > 1 ? i * (W - padX * 2) / (chrono.length - 1) : (W - padX * 2) / 2);
        var y = H - padY - r.pass_rate * (H - padY * 2);
        pts.push({ x: x, y: y, pr: r.pass_rate });
      });
      if (!pts.length) {
        svg.innerHTML = '<text x="350" y="85" text-anchor="middle" fill="#666" font-size="13">Notlanmış skor yok (LLM çevrimdışı olabilir)</text>';
      } else {
        var midY = (H - padY - 0.5 * (H - padY * 2)).toFixed(1);
        var ref = '<line x1="' + padX + '" y1="' + midY + '" x2="' + (W - padX) + '" y2="' + midY +
          '" stroke="#555" stroke-dasharray="4 4" stroke-width="1"/>' +
          '<text x="' + (W - padX) + '" y="' + (parseFloat(midY) - 4) + '" text-anchor="end" fill="#777" font-size="10">%50</text>';
        var line = '<polyline fill="none" stroke="#4af" stroke-width="2" points="' +
          pts.map(function (p) { return p.x.toFixed(1) + ',' + p.y.toFixed(1); }).join(' ') + '"/>';
        var dots = pts.map(function (p) {
          return '<circle cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="3" fill="' +
            (p.pr >= 0.5 ? '#4a4' : '#a44') + '"/>';
        }).join('');
        svg.innerHTML = ref + line + dots;
      }
    }
    if (tbl) {
      var html = '<table style="width:100%;border-collapse:collapse;font-size:.85rem"><thead><tr>' +
        '<th style="text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Zaman</th>' +
        '<th style="text-align:right;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Oran</th>' +
        '<th style="text-align:right;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Notlanan</th>' +
        '<th style="text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Seviyeler</th>' +
        '<th style="text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--border)">Bağlam</th>' +
        '</tr></thead><tbody>';
      rows.forEach(function (r) {
        var rate = r.pass_rate == null ? '—' : Math.round(r.pass_rate * 100) + '%';
        var color = r.pass_rate == null ? '#888' : (r.pass_rate >= 0.5 ? '#4a4' : '#a44');
        html += '<tr>' +
          '<td style="padding:.3rem .5rem;color:#888;font-size:.75rem">' + esc(String(r.created_at || '').slice(0, 19)) + '</td>' +
          '<td style="text-align:right;padding:.3rem .5rem;color:' + color + '"><strong>' + rate + '</strong></td>' +
          '<td style="text-align:right;padding:.3rem .5rem">' + (r.graded || 0) + '</td>' +
          '<td style="padding:.3rem .5rem;font-size:.78rem">' + esc(levelBreakdown(r.by_level)) + '</td>' +
          '<td style="padding:.3rem .5rem;font-size:.75rem;color:#888">' + esc(ctxSummary(r.context)) + '</td>' +
          '</tr>';
      });
      tbl.innerHTML = html + '</tbody></table>';
    }
  }

  let _lrnRuns = [];

  async function loadLearningDashboard() {
    try {
      const [sum, evalData, runsData, growthData, undData] = await Promise.all([
        api('/learning/summary', { method: 'GET' }),       // api() = auth header + hata
        api('/learning/eval-history', { method: 'GET' }),
        api('/learning/training-runs', { method: 'GET' }),
        api('/learning/card-growth', { method: 'GET' }),
        api('/understanding-score/history?limit=30', { method: 'GET' }).catch(function () { return { history: [] }; }),
      ]);
      document.getElementById('lrn-papers').textContent  = sum.n_papers   ?? '—';
      document.getElementById('lrn-chunks').textContent  = sum.n_chunks   ?? '—';
      document.getElementById('lrn-approved').textContent= sum.n_approved_cards ?? '—';
      document.getElementById('lrn-pending').textContent = sum.n_pending_cards  ?? '—';

      renderEvalTable(evalData.rows || []);

      _lrnRuns = runsData.runs || [];
      const sel = document.getElementById('lrnRunSelect');
      if (sel) {
        sel.innerHTML = _lrnRuns.length
          ? _lrnRuns.map((r, i) => `<option value="${i}">${esc(r.adapter_name)}</option>`).join('')
          : '<option value="">— yok —</option>';
        if (_lrnRuns.length) renderLossCurve(_lrnRuns[0].curve);
      }

      renderCardGrowth(growthData.rows || []);
      renderUnderstandingHistory((undData && undData.history) || []);
    } catch (e) {
      console.error('learning dashboard yüklenemedi', e);
    }
  }

  const lrnRunSel = document.getElementById('lrnRunSelect');
  if (lrnRunSel) {
    lrnRunSel.addEventListener('change', () => {
      const i = parseInt(lrnRunSel.value, 10);
      if (_lrnRuns[i]) renderLossCurve(_lrnRuns[i].curve);
    });
  }

  const lrnRefreshBtn = document.getElementById('lrnRefreshBtn');
  if (lrnRefreshBtn) lrnRefreshBtn.addEventListener('click', loadLearningDashboard);

  // Auto-load when tab is clicked
  document.querySelectorAll('.tab[data-tab="learning"]').forEach(btn => {
    btn.addEventListener('click', loadLearningDashboard);
    btn.addEventListener('click', loadRagLoopStatus);
  });

  // ---------- RAG öğrenme döngüsü (otonom, sunucu-taraflı) ----------
  var _RAG_STAGE_LABELS = {
    idle: 'Boşta', fetching: 'Makale çekiliyor…', carding: 'Kart üretiliyor…',
    scoring: 'Skorlanıyor…', paused_training: 'LoRA eğitimi sürüyor — duraklatıldı',
    error: 'Hata'
  };
  var _RAG_STAGE_COLORS = {
    idle: '#94a3b8', fetching: '#facc15', carding: '#a78bfa',
    scoring: '#60a5fa', paused_training: '#fb923c', error: '#f87171'
  };

  function renderRagLoopStatus(d) {
    var label = _RAG_STAGE_LABELS[d.stage] || d.stage;
    var color = _RAG_STAGE_COLORS[d.stage] || '#94a3b8';
    var running = d.running ? ' <span class="badge badge-warn">çalışıyor</span>' : '';
    var mastery = (d.mastery_percent === null || d.mastery_percent === undefined)
      ? '—' : (d.mastery_percent + '%');
    var rows = [
      ['Durum', '<strong style="color:' + color + '">' + esc(label) + '</strong>' + running],
      ['RAG ustalığı', esc(mastery)],
      ['Tamamlanan tur', esc(d.cycles_completed)],
      ['Son tur', d.last_cycle_at ? esc(d.last_cycle_at.slice(0, 16).replace('T', ' ')) : '—'],
      ['Son tur özeti', '+' + esc(d.last_ingested) + ' makale · +' + esc(d.last_cards) +
        ' kart · +' + esc(d.last_rebuilt) + ' yeniden · +' + esc(d.last_scored) + ' skor'],
      ['Toplam', esc(d.total_fetched) + ' makale · ' + esc(d.total_cards) + ' kart · ' +
        esc(d.total_rebuilt) + ' yeniden · ' + esc(d.total_scored) + ' skor'],
      ['Son çekim', d.last_fetch_at ? esc(d.last_fetch_at.slice(0, 16).replace('T', ' ')) : '—'],
      ['Hata', esc(d.last_error || '—')]
    ];
    var html = '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    rows.forEach(function (r) {
      html += '<tr><td style="padding:2px 10px 2px 0;color:#94a3b8;white-space:nowrap">' +
        r[0] + '</td><td>' + r[1] + '</td></tr>';
    });
    html += '</table>';
    var enLabel = document.getElementById('ragLoopEnabledLabel');
    if (enLabel) {
      enLabel.textContent = d.enabled ? 'açık' : 'kapalı';
      enLabel.className = 'badge ' + (d.enabled ? 'badge-success' : 'badge-info');
    }
    return html;
  }

  function fillRagLoopConfig(d) {
    var map = {
      ragLoopInterval: d.interval_min, ragLoopFetchHours: d.fetch_interval_hours,
      ragLoopMaxFetch: d.max_fetch_per_cycle, ragLoopCards: d.cards_per_cycle,
      ragLoopScores: d.scores_per_cycle
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      // Kullanıcı o an düzenliyorsa değerini ezme.
      if (el && map[id] != null && document.activeElement !== el) el.value = map[id];
    });
    var fe = document.getElementById('ragLoopFetchEnabled');
    if (fe && document.activeElement !== fe) fe.checked = !!d.fetch_enabled;
    var ul = document.getElementById('ragLoopUseLlm');
    if (ul && document.activeElement !== ul) ul.checked = !!d.score_use_llm;
    var re = document.getElementById('ragLoopRebuildEmpty');
    if (re && document.activeElement !== re) re.checked = !!d.rebuild_empty;
  }

  function loadRagLoopStatus() {
    var body = document.getElementById('ragLoopStatusBody');
    if (!body) return;
    api('/rag-loop/status', { method: 'GET' })  // api() = auth header + hata yönetimi
      .then(function (d) { body.innerHTML = renderRagLoopStatus(d); fillRagLoopConfig(d); })
      .catch(function () { body.innerHTML = '<span class="muted small">Durum alınamadı</span>'; });
  }

  function ragLoopMsg(text, ok) {
    var el = document.getElementById('ragLoopMsg');
    if (!el) return;
    el.textContent = text;
    el.className = 'result ' + (ok ? '' : 'error');
  }

  var ragLoopRefreshBtn = document.getElementById('ragLoopRefreshBtn');
  if (ragLoopRefreshBtn) ragLoopRefreshBtn.addEventListener('click', loadRagLoopStatus);

  var ragLoopEnableBtn = document.getElementById('ragLoopEnableBtn');
  if (ragLoopEnableBtn) ragLoopEnableBtn.addEventListener('click', function () {
    api('/rag-loop/enable?enabled=true', { method: 'POST' })
      .then(function () { ragLoopMsg('Döngü açıldı — ilk tur birazdan başlar.', true); loadRagLoopStatus(); })
      .catch(function (e) { ragLoopMsg('Hata: ' + e.message, false); });
  });

  var ragLoopDisableBtn = document.getElementById('ragLoopDisableBtn');
  if (ragLoopDisableBtn) ragLoopDisableBtn.addEventListener('click', function () {
    api('/rag-loop/enable?enabled=false', { method: 'POST' })
      .then(function () { ragLoopMsg('Döngü kapatıldı (yürüyen tur biter, yenisi başlamaz).', true); loadRagLoopStatus(); })
      .catch(function (e) { ragLoopMsg('Hata: ' + e.message, false); });
  });

  var ragLoopRunOnceBtn = document.getElementById('ragLoopRunOnceBtn');
  if (ragLoopRunOnceBtn) ragLoopRunOnceBtn.addEventListener('click', function () {
    ragLoopMsg('Bir tur başlatılıyor…', true);
    api('/rag-loop/run-once', { method: 'POST' })
      .then(function (d) {
        ragLoopMsg(d.ok ? 'Tur başlatıldı — ilerlemeyi yukarıdaki durumdan izle.'
          : ('Çalıştırılamadı: ' + (d.reason || 'bilinmiyor')), d.ok);
        loadRagLoopStatus();
      })
      .catch(function (e) { ragLoopMsg('Hata: ' + e.message, false); });
  });

  // makale-okuyucu: "tüm PDF'leri okut" — bütçeli tek koşu (arka plan; koşu 10·AGENTS'ta izlenir)
  var readerRunBtn = document.getElementById('readerRunBtn');
  if (readerRunBtn) readerRunBtn.addEventListener('click', function () {
    ragLoopMsg('Makale okuyucu başlatılıyor (20 kart + 20 skor, hızlı skor modu)…', true);
    // use_llm_score=false: skorda LLM doğrulaması atlanır (doluluk + RAG precision) — ölçüldü,
    // LLM'li skor kart kadar pahalı; toplu okumada hızlı mod, tek makale isteyince CLI --llm-score.
    api('/reader/run?cards=20&scores=20&use_llm_score=false', { method: 'POST' })
      .then(function (d) {
        ragLoopMsg(d.started ? 'Okuyucu koşuyor — ilerleme 10 · AGENTS koşularında ve haritada.'
          : 'Başlatılamadı.', !!d.started);
        if (typeof loadAgRuns === 'function') loadAgRuns();
      })
      .catch(function (e) { ragLoopMsg('Hata: ' + e.message, false); });
  });
  // kaynak-tamamlayici: okunamayan makaleye alaka kapılı arXiv ön-koşul kaynağı
  var kaynakTamamlaBtn = document.getElementById('kaynakTamamlaBtn');
  if (kaynakTamamlaBtn) kaynakTamamlaBtn.addEventListener('click', function () {
    ragLoopMsg('Kaynak tamamlayıcı başlatılıyor…', true);
    api('/kaynak-tamamla/run?esik=50&max_paper=5&max_per_paper=2', { method: 'POST' })
      .then(function (d) {
        ragLoopMsg(d.started ? 'Kaynak tamamlayıcı koşuyor — sonuç 10 · AGENTS koşu detayında.'
          : 'Başlatılamadı.', !!d.started);
        if (typeof loadAgRuns === 'function') loadAgRuns();
      })
      .catch(function (e) { ragLoopMsg('Hata: ' + e.message, false); });
  });
  var ragLoopSaveCfgBtn = document.getElementById('ragLoopSaveCfgBtn');
  if (ragLoopSaveCfgBtn) ragLoopSaveCfgBtn.addEventListener('click', function () {
    function val(id) { var el = document.getElementById(id); return el ? el.value : ''; }
    function chk(id) { var el = document.getElementById(id); return el && el.checked ? 'true' : 'false'; }
    var qs = [
      'interval_min=' + encodeURIComponent(val('ragLoopInterval')),
      'fetch_interval_hours=' + encodeURIComponent(val('ragLoopFetchHours')),
      'max_fetch_per_cycle=' + encodeURIComponent(val('ragLoopMaxFetch')),
      'cards_per_cycle=' + encodeURIComponent(val('ragLoopCards')),
      'scores_per_cycle=' + encodeURIComponent(val('ragLoopScores')),
      'fetch_enabled=' + chk('ragLoopFetchEnabled'),
      'score_use_llm=' + chk('ragLoopUseLlm'),
      'rebuild_empty=' + chk('ragLoopRebuildEmpty')
    ].join('&');
    api('/rag-loop/config?' + qs, { method: 'POST' })
      .then(function () { ragLoopMsg('Ayarlar kaydedildi.', true); loadRagLoopStatus(); })
      .catch(function (e) { ragLoopMsg('Hata: ' + e.message, false); });
  });

  loadRagLoopStatus();
  // ÖĞRENME sekmesi açıkken döngü durumunu canlı tut.
  setInterval(function () {
    var panel = document.getElementById('panel-learning');
    if (panel && panel.classList.contains('active')) loadRagLoopStatus();
  }, 15000);

  // ---------- egitim rozeti (ust bar) ----------
  // Uc durum: CANLI (nabizli) · HAZIR (tek tikla baslat) · YOK.
  // Gercek egitimi web'den VEYA detached/CLI'dan algilar (/api/training/live).
  var _trainStarting = false;

  function startTrainingFromBadge(examples) {
    if (_trainStarting) return;
    var msg =
      "LoRA eğitimi başlatılsın mı?\n\n" +
      examples + " örnek · ~1 epoch.\n" +
      "Uzun sürer (saatlerce); bilgisayar açık kalmalı.\n" +
      "Eğitim, web/terminal kapansa da arka planda sürer.";
    if (!window.confirm(msg)) return;
    _trainStarting = true;
    var stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
    api("/training/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ adapter_name: "hektor_auto_" + stamp, iterations: 0 }),
    })
      .then(function (d) {
        toast(d.message, !d.ok);
        refreshTrain();
      })
      .catch(function (e) {
        toast(e.message, true);
      })
      .finally(function () {
        _trainStarting = false;
      });
  }

  // Detached eğitimde SSE beslenmediği için EĞİTİM sekmesi ilerlemesini de
  // /api/training/live'dan aynala (sekme boş kalmasın).
  function mirrorTrainTab(t) {
    var section = document.getElementById("trainProgressSection");
    var bar = document.getElementById("trainProgressBar");
    var pctLbl = document.getElementById("trainPctLabel");
    var iterLbl = document.getElementById("trainIterLabel");
    var stateLbl = document.getElementById("trainStateLabel");
    if (section) section.style.display = "";
    if (bar) bar.style.width = (t.pct || 0) + "%";
    if (pctLbl) pctLbl.textContent = (t.pct || 0) + "%";
    if (iterLbl) iterLbl.textContent = (t.step || 0) + " / " + (t.total || 0) + " iter";
    if (stateLbl) {
      stateLbl.textContent = "eğitiliyor… (detached)";
      stateLbl.className = "badge badge-warn";
    }
    var meta = document.getElementById("trainMeta");
    if (meta && t.adapter) meta.textContent = "Adapter: " + t.adapter + "  ·  kaynak: detached";
    // Detached eğitimde SSE beslenmez → son log satırlarını çekip göster (boş kalmasın).
    var logEl = document.getElementById("trainLog");
    if (logEl) {
      api("/training/logs?lines=40", { method: "GET" })
        .then(function (d) {
          if (d && d.lines && d.lines.length) {
            logEl.textContent = d.lines.join("\n");
            logEl.scrollTop = logEl.scrollHeight;
          }
        })
        .catch(function () {});
    }
  }

  function refreshTrain() {
    var el = document.getElementById("trainBadge");
    if (!el) return;
    // Klavye/erişilebilirlik ipuçlarını her döngüde sıfırla (yalnız HAZIR iken aktif).
    function clearInteractive() {
      el.onclick = null;
      el.onkeydown = null;
      el.style.cursor = "";
      el.removeAttribute("role");
      el.removeAttribute("tabindex");
    }
    api("/training/live", { method: "GET" })
      .then(function (t) {
        clearInteractive();
        if (t && t.running) {
          el.className = "train-live"; // nabız atan nokta (CSS) + turuncu — CVD-safe
          var who = t.adapter ? (" " + t.adapter) : "";
          var eta = t.eta ? ("  ETA " + t.eta) : "";
          el.textContent = "EĞİTİM:" + who + " " + t.step + "/" + t.total + " (%" + t.pct + ")" + eta;
          el.title = "Eğitim çalışıyor (" + (t.source === "web" ? "web" : "detached/CLI") + ") — adapter: " + (t.adapter || "LoRA");
          mirrorTrainTab(t);
        } else if (t && t.ready) {
          el.className = "train-ready"; // teal + ▶ — tıklanabilir davet (CVD-safe)
          el.textContent = "▶ EĞİTİME HAZIR (" + t.ready_examples + " örnek) — BAŞLAT";
          el.title = "Tıkla / Enter → LoRA eğitimini başlat (onay sorulur). " + t.ready_label;
          el.style.cursor = "pointer";
          // Klavye erişimi: span'i buton gibi davranıştır (Tab + Enter/Space).
          el.setAttribute("role", "button");
          el.setAttribute("tabindex", "0");
          var start = function () { startTrainingFromBadge(t.ready_examples); };
          el.onclick = start;
          el.onkeydown = function (e) {
            if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
              e.preventDefault();
              start();
            }
          };
        } else {
          el.className = "train-idle";
          el.textContent = "eğitim yok" + (t && t.ready_label ? " (" + t.ready_label + ")" : "");
          el.title = "Aktif LoRA eğitimi yok";
        }
      })
      .catch(function () {
        clearInteractive();
        el.className = "train-idle";
        el.textContent = "eğitim: —";
      });
  }

  // ---------- AGENTS / OTOMASYON dashboard (Phase 3) ----------
  // Yalnız Phase 1/2 endpointlerini kullanır (yeni tehlikeli backend YOK). Tüm dinamik
  // metin esc() ile kaçırılır; innerHTML yalnız escape edilmiş veriyle kullanılır.
  // Tehlikeli aksiyonlar (STOP_ALL, approve, reject, cancel) confirm() ister — tek tık yok.
  var AG_RISK_CLS = { low: "ag-r-low", medium: "ag-r-med", high: "ag-r-high", critical: "ag-r-crit" };
  var _agState = { agentFilter: null };

  function agStatusBadge(status) {
    var s = String(status || "");
    var cls = "ag-gray";
    if (s === "completed" || s === "approved" || s === "ok" || s === "healthy") cls = "ag-green";
    else if (s === "running" || s === "pending" || s === "claimed" || s === "blocked_approval")
      cls = "ag-yellow";
    else if (s === "failed" || s === "rejected" || s === "blocked_stop_all") cls = "ag-red";
    return '<span class="ag-badge ' + cls + '">' + esc(s || "—") + "</span>";
  }
  function agShort(id) {
    return id ? esc(String(id).slice(0, 18)) : "—";
  }
  function agTime(t) {
    return t ? esc(String(t).slice(0, 19).replace("T", " ")) : "—";
  }

  function loadAgentsDashboard() {
    loadAgSupervisorStatus();
    loadAgApprovals();
    loadAgAgents();
    loadAgRuns();
    loadAgTasks();
    loadAgEvents();
    updateAgFilterHint();
  }

  // A) Supervisor / health
  function loadAgSupervisorStatus() {
    var grid = document.getElementById("agStatusGrid");
    if (!grid) return;
    Promise.all([
      api("/supervisor/status").catch(function () { return {}; }),
      api("/healthz").catch(function () { return {}; }),
      api("/approvals?status=pending&limit=200").catch(function () { return { approvals: [] }; }),
      api("/agents/runs?status=running&limit=200").catch(function () { return { runs: [] }; }),
      api("/events?limit=1").catch(function () { return { events: [] }; }),
    ]).then(function (res) {
      var sup = res[0] || {}, hz = res[1] || {}, apr = res[2] || {}, runs = res[3] || {}, ev = res[4] || {};
      var stopAll = !!sup.stop_all_active;
      var pending = (apr.approvals || []).length;
      var running = (runs.runs || []).length;
      var lastEv = ev.events && ev.events[0] ? ev.events[0].ts : null;
      var rows = [
        ["STOP_ALL", stopAll ? '<span class="ag-badge ag-red">AKTİF</span>'
          : '<span class="ag-badge ag-green">kapalı</span>'],
        ["Danger gate", '<span class="ag-badge ag-yellow">her zaman aktif (taze onay)</span>'],
        ["Bekleyen onay", '<span class="ag-badge ' + (pending ? "ag-yellow" : "ag-green") + '">'
          + pending + "</span>"],
        ["Çalışan koşu", '<span class="ag-badge ' + (running ? "ag-yellow" : "ag-gray") + '">'
          + running + "</span>"],
        ["Son olay", agTime(lastEv)],
        ["Sağlık", agStatusBadge(hz.status || (hz.ok ? "healthy" : "?"))
          + ' <span class="muted small">' + agTime(hz.time) + "</span>"],
      ];
      var html = "";
      rows.forEach(function (it) {
        html += '<div class="ag-stat"><span class="ag-stat-k">' + esc(it[0])
          + '</span><span class="ag-stat-v">' + it[1] + "</span></div>";
      });
      grid.innerHTML = html;
      var card = grid.closest(".card");
      if (card) card.classList.toggle("ag-stopall-active", stopAll);
    }).catch(function (e) {
      grid.innerHTML = '<span class="ag-red">Durum yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  // B) Agents
  function loadAgAgents() {
    var el = document.getElementById("agAgentsTable");
    if (!el) return;
    api("/agents").then(function (d) {
      var agents = d.agents || [];
      if (!agents.length) { el.innerHTML = '<span class="muted small">Ajan yok.</span>'; return; }
      var rows = "";
      agents.forEach(function (a) {
        var danger = a.dangerous ? '<span class="ag-badge ag-red">⚠ tehlikeli</span>'
          : '<span class="muted small">—</span>';
        rows += "<tr" + (a.dangerous ? ' class="ag-danger-row"' : "") + ">"
          + '<td><button class="ag-link" data-ag-agent="' + esc(a.agent_id) + '">'
          + esc(a.agent_id) + "</button></td>"
          + "<td>" + esc(a.autonomy) + "</td>"
          + "<td>" + danger + "</td>"
          + "<td>" + (a.approval_required ? "✓" : "—") + "</td>"
          + "<td>" + (a.default_enabled ? "✓" : "—") + "</td>"
          + '<td class="muted small">' + esc(a.status_location || "—") + "</td></tr>";
      });
      el.innerHTML = '<button class="ag-link" data-ag-agent="" style="margin-bottom:.4rem">'
        + "↺ filtreyi temizle</button>"
        + '<table class="ag-table"><thead><tr><th>Agent</th><th>Autonomy</th><th>Tehlikeli</th>'
        + "<th>Onay</th><th>Vars. açık</th><th>Status Location</th></tr></thead><tbody>"
        + rows + "</tbody></table>";
    }).catch(function (e) {
      el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  // C) Runs + detail
  function loadAgRuns() {
    var el = document.getElementById("agRunsTable");
    if (!el) return;
    var path = "/agents/runs?limit=50";
    if (_agState.agentFilter) path += "&agent_id=" + encodeURIComponent(_agState.agentFilter);
    api(path).then(function (d) {
      var runs = d.runs || [];
      if (!runs.length) { el.innerHTML = '<span class="muted small">Koşu yok.</span>'; return; }
      var rows = "";
      runs.forEach(function (r) {
        rows += "<tr>"
          + '<td><button class="ag-link" data-ag-run="' + esc(r.run_id) + '">'
          + agShort(r.run_id) + "</button></td>"
          + "<td>" + esc(r.agent_id) + "</td>"
          + "<td>" + agStatusBadge(r.status) + "</td>"
          + "<td>" + agTime(r.started_at) + "</td>"
          + "<td>" + agTime(r.finished_at) + "</td>"
          + '<td class="ag-red small">' + esc(r.error || "") + "</td></tr>";
      });
      el.innerHTML = '<table class="ag-table"><thead><tr><th>Run ID</th><th>Agent</th><th>Durum</th>'
        + "<th>Başladı</th><th>Bitti</th><th>Hata</th></tr></thead><tbody>" + rows + "</tbody></table>";
    }).catch(function (e) {
      el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  function loadAgRunDetail(runId) {
    var el = document.getElementById("agRunDetail");
    if (!el) return;
    el.classList.remove("hidden");
    el.innerHTML = '<span class="muted small">Yükleniyor…</span>';
    api("/agents/runs/" + encodeURIComponent(runId)).then(function (d) {
      var r = d.run || {}, evs = d.events || [];
      var meta = '<div class="ag-detail-head"><strong>' + esc(r.run_id) + "</strong> "
        + agStatusBadge(r.status)
        + ' <button class="ag-link" id="agRunDetailClose">kapat ✕</button></div>'
        + '<div class="muted small">agent: ' + esc(r.agent_id) + " · tetik: "
        + esc(r.trigger_type || "—") + " · başladı: " + agTime(r.started_at)
        + " · bitti: " + agTime(r.finished_at) + "</div>";
      if (r.error) meta += '<div class="ag-red small">hata: ' + esc(r.error) + "</div>";
      if (r.summary) meta += '<div class="muted small">özet: ' + esc(JSON.stringify(r.summary)) + "</div>";
      if (r.outputs) meta += '<div class="muted small">çıktı: ' + esc(JSON.stringify(r.outputs)) + "</div>";
      var tl = "";
      evs.forEach(function (ev) {
        var lvl = ev.level || "info";
        tl += '<li class="ag-tl ag-lvl-' + esc(lvl) + '"><span class="ag-tl-ts">'
          + agTime(ev.ts) + '</span> <span class="ag-badge ag-gray">' + esc(ev.kind) + "</span> "
          + esc(ev.message || "") + "</li>";
      });
      el.innerHTML = meta + '<ul class="ag-timeline">'
        + (tl || '<li class="muted small">olay yok</li>') + "</ul>";
      var cl = document.getElementById("agRunDetailClose");
      if (cl) cl.addEventListener("click", function () {
        el.classList.add("hidden");
        el.innerHTML = "";
      });
    }).catch(function (e) {
      el.innerHTML = '<span class="ag-red">Detay yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  // D) Approvals (pending önce)
  function loadAgApprovals() {
    var el = document.getElementById("agApprovalsTable");
    if (!el) return;
    api("/approvals?limit=100").then(function (d) {
      var items = (d.approvals || []).slice();
      items.sort(function (a, b) {
        return (a.status === "pending" ? 0 : 1) - (b.status === "pending" ? 0 : 1);
      });
      if (!items.length) { el.innerHTML = '<span class="muted small">Onay isteği yok.</span>'; return; }
      var rows = "";
      items.forEach(function (a) {
        var riskCls = AG_RISK_CLS[a.risk] || "ag-r-med";
        var actions = a.status === "pending"
          ? ('<button class="btn-sm ag-approve" data-ag-action="approve" data-ag-id="'
              + esc(a.approval_id) + '">Onayla</button> '
            + '<button class="btn-sm ag-reject" data-ag-action="reject" data-ag-id="'
              + esc(a.approval_id) + '">Reddet</button>')
          : '<span class="muted small">—</span>';
        rows += "<tr>"
          + "<td>" + agShort(a.approval_id) + "</td>"
          + "<td>" + esc(a.agent_id) + "</td>"
          + "<td>" + esc(a.action) + "</td>"
          + '<td><span class="ag-badge ' + riskCls + '">' + esc(a.risk) + "</span></td>"
          + "<td>" + agStatusBadge(a.status) + "</td>"
          + "<td>" + agTime(a.requested_at) + "</td>"
          + "<td>" + actions + "</td></tr>";
      });
      el.innerHTML = '<table class="ag-table"><thead><tr><th>Approval</th><th>Agent</th><th>Action</th>'
        + "<th>Risk</th><th>Durum</th><th>İstendi</th><th>Aksiyon</th></tr></thead><tbody>"
        + rows + "</tbody></table>";
    }).catch(function (e) {
      el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  // E) Tasks
  function loadAgTasks() {
    var el = document.getElementById("agTasksTable");
    if (!el) return;
    api("/automation/tasks?limit=50").then(function (d) {
      var items = d.tasks || [];
      if (!items.length) { el.innerHTML = '<span class="muted small">Görev yok.</span>'; return; }
      var rows = "";
      items.forEach(function (t) {
        var terminal = t.status === "completed" || t.status === "failed" || t.status === "cancelled";
        var actions = terminal ? '<span class="muted small">—</span>'
          : '<button class="btn-sm ag-reject" data-ag-action="cancel" data-ag-id="'
            + esc(t.task_id) + '">İptal</button>';
        rows += "<tr>"
          + "<td>" + agShort(t.task_id) + "</td>"
          + "<td>" + esc(t.agent_id) + "</td>"
          + "<td>" + esc(t.title || "") + "</td>"
          + "<td>" + agStatusBadge(t.status) + "</td>"
          + "<td>" + (t.requires_approval ? "✓" : "—") + "</td>"
          + "<td>" + agTime(t.created_at) + "</td>"
          + "<td>" + actions + "</td></tr>";
      });
      el.innerHTML = '<table class="ag-table"><thead><tr><th>Task</th><th>Agent</th><th>Başlık</th>'
        + "<th>Durum</th><th>Onay</th><th>Oluşturuldu</th><th>Aksiyon</th></tr></thead><tbody>"
        + rows + "</tbody></table>";
    }).catch(function (e) {
      el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  // F) Global events
  function loadAgEvents() {
    var el = document.getElementById("agEventsTable");
    if (!el) return;
    api("/events?limit=100").then(function (d) {
      var items = d.events || [];
      if (!items.length) { el.innerHTML = '<span class="muted small">Olay yok.</span>'; return; }
      var rows = "";
      items.forEach(function (ev) {
        var lvl = ev.level || "info";
        var who = ev.run_id || "";
        if (ev.payload && ev.payload.agent_id) who = ev.payload.agent_id;
        var lvlCls = lvl === "error" ? "ag-red" : lvl === "warning" ? "ag-yellow" : "ag-gray";
        rows += '<tr class="ag-lvl-' + esc(lvl) + '">'
          + "<td>" + agTime(ev.ts) + "</td>"
          + '<td><span class="ag-badge ' + lvlCls + '">' + esc(lvl) + "</span></td>"
          + "<td>" + esc(ev.kind) + "</td>"
          + '<td class="small">' + esc(String(who).slice(0, 24)) + "</td>"
          + "<td>" + esc(ev.message || "") + "</td></tr>";
      });
      el.innerHTML = '<table class="ag-table"><thead><tr><th>Zaman</th><th>Level</th><th>Kind</th>'
        + "<th>Agent/Run</th><th>Mesaj</th></tr></thead><tbody>" + rows + "</tbody></table>";
    }).catch(function (e) {
      el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
    });
  }

  function updateAgFilterHint() {
    var h = document.getElementById("agAgentFilterHint");
    if (!h) return;
    h.textContent = _agState.agentFilter
      ? "Filtre: " + _agState.agentFilter + " — temizlemek için ajan tablosunda 'filtreyi temizle'."
      : "Bir ajana tıkla → koşularını filtrele.";
  }

  // ---- dangerous actions (her biri confirm() ister — tek tık yok) ----
  function agStopAll() {
    if (!window.confirm("Bu işlem tüm dangerous agent action'larını durdurur. Emin misiniz?")) return;
    api("/supervisor/stop-all", { method: "POST" })
      .then(function () { toast("STOP_ALL etkin."); loadAgSupervisorStatus(); })
      .catch(function (e) { toast("Hata: " + e.message, true); });
  }
  function agClearStopAll() {
    if (!window.confirm("STOP_ALL kaldırılacak; tehlikeli aksiyonlar yeniden (onaya tabi) "
        + "çalışabilir. Emin misiniz?")) return;
    api("/supervisor/clear-stop-all", { method: "POST" })
      .then(function () { toast("STOP_ALL kaldırıldı."); loadAgSupervisorStatus(); })
      .catch(function (e) { toast("Hata: " + e.message, true); });
  }
  function agApprove(id) {
    if (!id) return;
    if (!window.confirm("Bu onay tek kullanımlıktır ve dangerous action çalıştırabilir. "
        + "Emin misiniz?")) return;
    api("/approvals/" + encodeURIComponent(id) + "/approve", { method: "POST" })
      .then(function () { toast("Onaylandı."); loadAgApprovals(); loadAgSupervisorStatus(); })
      .catch(function (e) { toast("Hata: " + e.message, true); });
  }
  function agReject(id) {
    if (!id) return;
    if (!window.confirm("Bu approval request reddedilecek. Emin misiniz?")) return;
    api("/approvals/" + encodeURIComponent(id) + "/reject", { method: "POST" })
      .then(function () { toast("Reddedildi."); loadAgApprovals(); loadAgSupervisorStatus(); })
      .catch(function (e) { toast("Hata: " + e.message, true); });
  }
  function agCancelTask(id) {
    if (!id) return;
    if (!window.confirm("Bu görev iptal edilecek. Emin misiniz?")) return;
    api("/automation/tasks/" + encodeURIComponent(id) + "/cancel", { method: "POST" })
      .then(function () { toast("Görev iptal edildi."); loadAgTasks(); })
      .catch(function (e) { toast("Hata: " + e.message, true); });
  }
  function agCreateTask() {
    var agent = (document.getElementById("agTaskAgent").value || "").trim();
    var title = (document.getElementById("agTaskTitle").value || "").trim();
    var desc = (document.getElementById("agTaskDesc").value || "").trim();
    var appr = document.getElementById("agTaskApproval").checked;
    if (!agent || !title) { toast("agent_id ve başlık gerekli.", true); return; }
    var qs = "?agent_id=" + encodeURIComponent(agent) + "&title=" + encodeURIComponent(title)
      + "&requires_approval=" + (appr ? "true" : "false");
    if (desc) qs += "&description=" + encodeURIComponent(desc);
    api("/automation/tasks" + qs, { method: "POST" }).then(function () {
      toast("Görev oluşturuldu.");
      document.getElementById("agTaskAgent").value = "";
      document.getElementById("agTaskTitle").value = "";
      document.getElementById("agTaskDesc").value = "";
      loadAgTasks();
    }).catch(function (e) { toast("Hata: " + e.message, true); });
  }

  // ---- wiring (script body sonunda; panel DOM'u zaten parse edilmiş) ----
  (function wireAgents() {
    var panel = document.getElementById("panel-agents");
    if (!panel) return;
    var byId = function (id) { return document.getElementById(id); };
    if (byId("agRefreshBtn")) byId("agRefreshBtn").addEventListener("click", loadAgentsDashboard);
    if (byId("agStopAllBtn")) byId("agStopAllBtn").addEventListener("click", agStopAll);
    if (byId("agClearStopAllBtn")) byId("agClearStopAllBtn").addEventListener("click", agClearStopAll);
    if (byId("agTaskCreateBtn")) byId("agTaskCreateBtn").addEventListener("click", agCreateTask);
    // delegated: dinamik satır butonları (CSP-safe; inline onclick yok)
    panel.addEventListener("click", function (e) {
      var t = e.target;
      if (!t || t.nodeType !== 1) return;
      var act = t.getAttribute("data-ag-action");
      if (act === "approve") return agApprove(t.getAttribute("data-ag-id"));
      if (act === "reject") return agReject(t.getAttribute("data-ag-id"));
      if (act === "cancel") return agCancelTask(t.getAttribute("data-ag-id"));
      var runId = t.getAttribute("data-ag-run");
      if (runId) return loadAgRunDetail(runId);
      if (t.hasAttribute("data-ag-agent")) {
        _agState.agentFilter = t.getAttribute("data-ag-agent") || null;
        loadAgRuns();
        updateAgFilterHint();
      }
    });
  })();

  // ---------- RLM dashboard (salt-okuma; /api/rlm/*) ----------
  function rlmStatusBadge(s) {
    var cls = "ag-gray";
    if (s === "answered") cls = "ag-green";
    else if (s === "answered_with_limitation" || s === "no_llm") cls = "ag-yellow";
    else if (s === "abstained" || s === "failed") cls = "ag-red";
    return '<span class="ag-badge ' + cls + '">' + esc(s) + "</span>";
  }

  function rlmCfgRow(k, v) {
    return '<tr><td class="muted">' + esc(k) + "</td><td>" + esc(String(v)) + "</td></tr>";
  }
  // Çalışma-zamanı paneli: /api/rlm/config (salt-okuma, sır YOK). Motor tektir ve
  // yereldir (RlmController + Ollama) — sağlayıcı/adapter seçimi yoktur.
  function loadRlmEnginePanel() {
    var el = document.getElementById("rlmEnginePanel");
    if (!el) return;
    el.innerHTML = '<span class="muted small">Yükleniyor…</span>';
    api("/rlm/config")
      .then(function (d) {
        var c = (d && d.config) || {};
        var rows =
          rlmCfgRow("Motor", c.engine) +
          rlmCfgRow("LLM backend", c.llm_backend) +
          rlmCfgRow("Model", c.llm_model) +
          rlmCfgRow("Üretim modu", c.production_mode ? "açık" : "kapalı") +
          rlmCfgRow("Local exec", c.allow_local_exec ? "AÇIK" : "kapalı") +
          rlmCfgRow("Seed (determinizm)", c.seed) +
          rlmCfgRow("İzinli tool'lar", (c.allowed_tools || []).join(", "));
        el.innerHTML = '<table class="ag-table"><tbody>' + rows + "</tbody></table>";
      })
      .catch(function (e) {
        el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
      });
  }

  function loadRlmDashboard() {
    loadRlmEnginePanel();
    var el = document.getElementById("rlmRunsTable");
    if (!el) return;
    el.innerHTML = '<span class="muted small">Yükleniyor…</span>';
    api("/rlm/runs?limit=50")
      .then(function (d) {
        var runs = d.runs || [];
        if (!runs.length) {
          el.innerHTML =
            '<span class="muted small">Henüz RLM koşusu yok. CLI: hektor rlm-answer "soru"</span>';
          return;
        }
        var rows = "";
        runs.forEach(function (r) {
          rows +=
            "<tr><td><button class=\"ag-link\" data-rlm-run=\"" +
            esc(r.run_id) +
            '">' +
            agShort(r.run_id) +
            "</button></td><td>" +
            esc((r.user_query || "").slice(0, 60)) +
            "</td><td>" +
            esc(r.task_type) +
            "</td><td>" +
            rlmStatusBadge(r.status) +
            "</td><td>" +
            esc(r.evidence_score) +
            "</td><td>" +
            esc(r.final_confidence) +
            "</td><td>" +
            agTime(r.created_at) +
            "</td></tr>";
        });
        el.innerHTML =
          '<table class="ag-table"><thead><tr><th>Run</th><th>Soru</th><th>Görev</th>' +
          "<th>Durum</th><th>Kanıt</th><th>Güven</th><th>Tarih</th></tr></thead><tbody>" +
          rows +
          "</tbody></table>";
      })
      .catch(function (e) {
        el.innerHTML = '<span class="ag-red">Yüklenemedi: ' + esc(e.message) + "</span>";
      });
  }

  function rlmClaimList(title, claims, red) {
    if (!claims || !claims.length) return "";
    var items = claims
      .map(function (c) {
        return "<li>" + esc(c) + "</li>";
      })
      .join("");
    return (
      '<div class="small' +
      (red ? " ag-red" : "") +
      '"><strong>' +
      esc(title) +
      "</strong><ul>" +
      items +
      "</ul></div>"
    );
  }

  function loadRlmRunDetail(runId) {
    var el = document.getElementById("rlmDetail");
    if (!el) return;
    el.classList.remove("hidden");
    el.innerHTML = '<span class="muted small">Yükleniyor…</span>';
    api("/rlm/runs/" + encodeURIComponent(runId))
      .then(function (d) {
        var r = d.run || {},
          steps = d.steps || [],
          ev = d.evidence || [],
          v = d.verification || null;
        var head =
          '<div class="ag-detail-head"><strong>' +
          esc(r.run_id) +
          "</strong> " +
          rlmStatusBadge(r.status) +
          ' <button class="ag-link" id="rlmDetailClose">kapat ✕</button></div>' +
          '<div class="muted small">görev: ' +
          esc(r.task_type) +
          " · kanıt: " +
          esc(r.evidence_score) +
          " · güven: " +
          esc(r.final_confidence) +
          " · model: " +
          esc(r.model_name || "—") +
          " · " +
          agTime(r.created_at) +
          "</div>" +
          '<div style="white-space:pre-wrap;background:rgba(127,127,127,.12);padding:.5rem;' +
          'border-radius:4px;margin:.4rem 0;font-size:.85rem">' +
          esc(r.final_answer || "") +
          "</div>";
        var sl = "";
        steps.forEach(function (s) {
          var txt =
            (s.tool_used ? s.tool_used + " — " : "") + (s.output_text || s.input_text || "");
          sl +=
            '<li class="ag-tl"><span class="ag-badge ag-gray">' +
            esc(s.step_type) +
            "</span> " +
            esc(txt.slice(0, 220)) +
            "</li>";
        });
        var er = "";
        ev.forEach(function (e2) {
          er +=
            "<tr><td>" +
            esc(e2.paper_id) +
            "</td><td>" +
            esc(e2.chunk_id) +
            "</td><td>" +
            esc(e2.relevance_score) +
            "</td><td>" +
            (e2.used_in_final_answer ? "✓" : "—") +
            "</td></tr>";
        });
        var vb;
        if (v) {
          vb =
            '<div class="muted small">citation: ' +
            esc(v.citation_score) +
            " · grounding: " +
            esc(v.grounding_score) +
            " · sufficiency: " +
            esc(v.context_sufficiency_score) +
            " · karar: " +
            esc(v.final_decision) +
            "</div>" +
            rlmClaimList("Desteklenen iddialar:", v.supported_claims, false) +
            rlmClaimList("Desteklenmeyen (atılan):", v.unsupported_claims, true) +
            (v.contradictions && v.contradictions.length
              ? "<div class=\"small\"><strong>Çelişki:</strong> " +
                esc(v.contradictions.join(", ")) +
                "</div>"
              : "");
        } else {
          vb = '<span class="muted small">doğrulama kaydı yok</span>';
        }
        el.innerHTML =
          head +
          '<h4 style="margin:.6rem 0 .2rem">Adımlar</h4><ul class="ag-timeline">' +
          (sl || '<li class="muted small">adım yok</li>') +
          "</ul>" +
          '<h4 style="margin:.6rem 0 .2rem">Kanıt (chunk)</h4>' +
          (er
            ? '<table class="ag-table"><thead><tr><th>paper</th><th>chunk</th>' +
              "<th>relevance</th><th>kullanıldı</th></tr></thead><tbody>" +
              er +
              "</tbody></table>"
            : '<span class="muted small">kanıt satırı yok</span>') +
          '<h4 style="margin:.6rem 0 .2rem">Doğrulama</h4>' +
          vb;
        var cl = document.getElementById("rlmDetailClose");
        if (cl)
          cl.addEventListener("click", function () {
            el.classList.add("hidden");
            el.innerHTML = "";
          });
      })
      .catch(function (e) {
        el.innerHTML = '<span class="ag-red">Detay yüklenemedi: ' + esc(e.message) + "</span>";
      });
  }

  (function () {
    var panel = document.getElementById("panel-rlm");
    if (!panel) return;
    panel.addEventListener("click", function (e) {
      var t = e.target;
      if (!t || t.nodeType !== 1) return;
      if (t.id === "rlmRefreshBtn") return loadRlmDashboard();
      var runId = t.getAttribute("data-rlm-run");
      if (runId) return loadRlmRunDetail(runId);
    });
  })();

  // ---------- Orkestrasyon (dayanıklı eğitim hattı) ----------
  var ORC_GLYPH = {
    completed: "✓",
    running: "▶",
    blocked: "⏸",
    failed: "✕",
    skipped: "–",
    pending: "·",
  };
  var orcCurrentRun = null;

  function orcShortTs(ts) {
    return (ts || "").replace("T", " ").slice(0, 19);
  }

  function orcRenderGraph(snap) {
    var el = document.getElementById("orcGraph");
    if (!el) return;
    var stages = (snap && snap.stages) || [];
    if (!stages.length) {
      el.innerHTML = '<span class="muted small">Koşu başlatın…</span>';
      return;
    }
    el.innerHTML = stages
      .map(function (s) {
        return (
          '<div class="orc-stage ' + esc(s.status) + '">' +
          '<div class="orc-st-name">' + (ORC_GLYPH[s.status] || "") + " " + esc(s.name) + "</div>" +
          '<div class="orc-st-msg">' + esc(s.message || "") + "</div></div>"
        );
      })
      .join("");
    var run = (snap && snap.run) || {};
    var lbl = document.getElementById("orcRunLabel");
    if (lbl) lbl.textContent = run.run_id ? run.run_id + " — " + run.status : "";
    var resumeBtn = document.getElementById("orcResumeBtn");
    if (resumeBtn) resumeBtn.disabled = !(run.status === "blocked" || run.status === "failed");
    var autoBtn = document.getElementById("orcAutodriveBtn");
    if (autoBtn)
      autoBtn.disabled = !(run.status === "blocked" && run.current_stage === "deep-hunt");
  }

  function orcRenderTimeline(events) {
    var el = document.getElementById("orcTimeline");
    if (!el) return;
    if (!events || !events.length) {
      el.innerHTML = '<span class="muted small">—</span>';
      return;
    }
    el.innerHTML = events
      .map(function (e) {
        return (
          '<div class="ev ' + esc(e.level) + '">' +
          esc(orcShortTs(e.created_at)) + " · [" + esc(e.stage || "-") + "] " + esc(e.message) +
          "</div>"
        );
      })
      .join("");
    el.scrollTop = el.scrollHeight;
  }

  function orcLoadTimeline(runId) {
    if (!runId) return;
    api("/orchestration/timeline/" + encodeURIComponent(runId))
      .then(function (d) {
        orcRenderTimeline(d.events);
      })
      .catch(function () {});
  }

  function orcRefreshStatus(runId) {
    if (!runId) return;
    api("/orchestration/status/" + encodeURIComponent(runId))
      .then(function (snap) {
        orcRenderGraph(snap);
        orcLoadTimeline(runId);
      })
      .catch(function () {});
  }

  function startOrchestration() {
    var adapter = (document.getElementById("orcAdapter") || {}).value || "hektor_lora";
    var iters = parseInt((document.getElementById("orcIters") || {}).value, 10) || 300;
    var huntAck = !!(document.getElementById("orcHuntAck") || {}).checked;
    var btn = document.getElementById("orcStartBtn");
    if (btn) btn.disabled = true;
    api("/orchestration/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        adapter_name: adapter,
        iters: iters,
        hunt_ack: huntAck,
        auto_run: true,
      }),
    })
      .then(function (snap) {
        orcCurrentRun = snap.run_id;
        orcRenderGraph(snap);
        orcLoadTimeline(orcCurrentRun);
        loadOrchestrationRuns();
        toast("Koşu başladı: " + snap.run_id);
      })
      .catch(function (e) {
        toast("Başlatılamadı: " + e.message, true);
      })
      .finally(function () {
        if (btn) btn.disabled = false;
      });
  }

  function resumeOrchestration() {
    if (!orcCurrentRun) {
      toast("Önce bir koşu başlat ya da listeden seç", true);
      return;
    }
    var huntAck = !!(document.getElementById("orcHuntAck") || {}).checked;
    api("/orchestration/resume/" + encodeURIComponent(orcCurrentRun), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hunt_ack: huntAck }),
    })
      .then(function (snap) {
        orcRenderGraph(snap);
        orcLoadTimeline(orcCurrentRun);
        loadOrchestrationRuns();
      })
      .catch(function (e) {
        toast("Sürdürülemedi: " + e.message, true);
      });
  }

  function autodriveOrchestration() {
    if (!orcCurrentRun) {
      toast("Önce bir koşu başlat ya da listeden seç", true);
      return;
    }
    if (
      !window.confirm(
        "Otonom AV: deep-hunt aşamasını (zorunlu Kademe-2 derin av, salt rapor) abonelikli " +
          "`claude -p` ile çalıştıracak (dakikalar sürebilir). Gerçek eğitim yine onay bekler " +
          "(Kural 8). Başlatılsın mı?"
      )
    )
      return;
    // ⚠️ Bu buton AV (hunt) tetikleyicisidir — zorunlu Kademe-2 derin avı sürer (MCP KAPALI,
    // salt rapor). ⚡ RUN (15·AJAN HARİTASI) ise SÜR (drive) modunu doğurur (MCP'li veri hattı).
    api("/orchestration/autodrive/" + encodeURIComponent(orcCurrentRun), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ execute: true, mode: "hunt" }),
    })
      .then(function (d) {
        if (d.status === "autodrive_started") {
          toast("Otonom sürüş başladı — timeline'ı izle.");
        } else if (d.dry_run) {
          toast("DRY-RUN: claude -p komutu hazır (execute kapalı).");
        } else {
          toast("Sürüş yanıtı: " + (d.reason || JSON.stringify(d).slice(0, 80)), !d.ok);
        }
        if (orcCurrentRun) orcRefreshStatus(orcCurrentRun);
      })
      .catch(function (e) {
        toast("Otonom sürüş hatası: " + e.message, true);
      });
  }

  function recoverOrchestration() {
    api("/orchestration/recover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timeout_min: 30 }),
    })
      .then(function (d) {
        toast((d.recovered || []).length + " stale aşama kurtarıldı");
        loadOrchestrationRuns();
        if (orcCurrentRun) orcRefreshStatus(orcCurrentRun);
      })
      .catch(function (e) {
        toast("Recovery hatası: " + e.message, true);
      });
  }

  function loadOrchestrationRuns() {
    var el = document.getElementById("orcRuns");
    api("/orchestration/runs?limit=10")
      .then(function (d) {
        var runs = d.runs || [];
        if (!runs.length) {
          if (el) el.innerHTML = '<span class="muted small">Henüz koşu yok.</span>';
          return;
        }
        if (el)
          el.innerHTML = runs
            .map(function (r) {
              return (
                '<div class="orc-run-row" data-run="' + esc(r.run_id) + '">' +
                "<code>" + esc(r.run_id) + "</code> · " + esc(r.status) + " · " +
                esc(r.current_stage || "-") +
                ' <span class="muted small">' + esc(orcShortTs(r.created_at)) + "</span></div>"
              );
            })
            .join("");
        if (el)
          el.querySelectorAll(".orc-run-row").forEach(function (row) {
            row.addEventListener("click", function () {
              orcCurrentRun = row.getAttribute("data-run");
              orcRefreshStatus(orcCurrentRun);
            });
          });
      })
      .catch(function () {
        if (el) el.innerHTML = '<span class="muted small">Yüklenemedi.</span>';
      });
  }

  function loadOrchestration() {
    loadOrchestrationRuns();
    if (orcCurrentRun) orcRefreshStatus(orcCurrentRun);
    var sb = document.getElementById("orcStartBtn");
    if (sb && !sb._wired) {
      sb._wired = true;
      sb.addEventListener("click", startOrchestration);
      document.getElementById("orcResumeBtn").addEventListener("click", resumeOrchestration);
      document.getElementById("orcAutodriveBtn").addEventListener("click", autodriveOrchestration);
      document.getElementById("orcRecoverBtn").addEventListener("click", recoverOrchestration);
      document.getElementById("orcRefreshBtn").addEventListener("click", function () {
        loadOrchestrationRuns();
        if (orcCurrentRun) orcRefreshStatus(orcCurrentRun);
      });
    }
  }

  // ---------- Geri bildirim (Echo) ----------
  var FB_STATUS_CLS = {
    pending: "warning",
    approved: "",
    rejected: "error",
    exported: "",
  };

  function fbRenderSummary(s) {
    var el = document.getElementById("fbSummary");
    if (!el || !s) return;
    el.textContent =
      "bekleyen " + (s.pending || 0) +
      " · onaylı " + (s.approved || 0) +
      " · reddedilen " + (s.rejected || 0) +
      " · export " + (s.exported || 0);
  }

  function fbRenderList(items) {
    var el = document.getElementById("fbList");
    if (!el) return;
    if (!items || !items.length) {
      el.innerHTML = '<span class="muted small">Henüz düzeltme yok.</span>';
      return;
    }
    el.innerHTML = items
      .map(function (c) {
        var cls = FB_STATUS_CLS[c.status] || "";
        var actions =
          c.status === "pending"
            ? '<button class="btn btn-sm" data-fb-approve="' + esc(c.correction_id) + '">✓ Onayla</button>' +
              '<button class="btn btn-sm" data-fb-reject="' + esc(c.correction_id) + '">✕ Reddet</button>'
            : "";
        var reason = c.reject_reason
          ? ' <span class="ev ' + cls + '">(' + esc(c.reject_reason) + ")</span>"
          : "";
        return (
          '<div class="orc-run-row">' +
          "<strong>" + esc(c.status) + "</strong> · " + esc(c.correction_type) + " · " +
          esc((c.correction || "").slice(0, 90)) + reason +
          ' <span style="float:right">' + actions + "</span></div>"
        );
      })
      .join("");
    el.querySelectorAll("[data-fb-approve]").forEach(function (b) {
      b.addEventListener("click", function () {
        fbAct("approve", b.getAttribute("data-fb-approve"));
      });
    });
    el.querySelectorAll("[data-fb-reject]").forEach(function (b) {
      b.addEventListener("click", function () {
        fbAct("reject", b.getAttribute("data-fb-reject"));
      });
    });
  }

  function loadFeedback() {
    api("/feedback/summary")
      .then(fbRenderSummary)
      .catch(function () {});
    api("/feedback/list?limit=50")
      .then(function (d) {
        fbRenderList(d.items);
      })
      .catch(function () {});
    var sb = document.getElementById("fbSubmitBtn");
    if (sb && !sb._wired) {
      sb._wired = true;
      sb.addEventListener("click", submitFeedback);
      document.getElementById("fbExportBtn").addEventListener("click", exportFeedback);
      document.getElementById("fbRefreshBtn").addEventListener("click", loadFeedback);
    }
  }

  function submitFeedback() {
    var correction = (document.getElementById("fbCorrection") || {}).value || "";
    if (!correction.trim()) {
      toast("Düzeltme metni boş olamaz", true);
      return;
    }
    var body = {
      correction: correction,
      question: (document.getElementById("fbQuestion") || {}).value || "",
      bad_answer: (document.getElementById("fbBad") || {}).value || "",
      correction_type: (document.getElementById("fbType") || {}).value || "other",
      source: "manual",
    };
    api("/feedback/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (rec) {
        if (rec.status === "rejected") {
          toast("Reddedildi: " + (rec.reject_reason || "Kural-1 zehiri"), true);
        } else {
          toast("Kaydedildi (" + rec.status + ")");
          document.getElementById("fbCorrection").value = "";
        }
        loadFeedback();
      })
      .catch(function (e) {
        toast("Kaydedilemedi: " + e.message, true);
      });
  }

  function fbAct(action, id) {
    api("/feedback/" + action + "/" + encodeURIComponent(id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    })
      .then(function (d) {
        if (action === "approve" && d.ok === false) {
          toast("Onaylanamadı: " + (d.reason || ""), true);
        }
        loadFeedback();
      })
      .catch(function (e) {
        toast("İşlem hatası: " + e.message, true);
      });
  }

  function exportFeedback() {
    api("/feedback/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    })
      .then(function (d) {
        toast(d.n_exported + " aday export edildi (eğitim başlamaz — Kural 8)");
        loadFeedback();
      })
      .catch(function (e) {
        toast("Export hatası: " + e.message, true);
      });
  }

  // ---------- Nöbetçi (Sentinel — sağlık monitörü) ----------
  // Probe kartları orc-stage stilini yeniden kullanır (yeni CSS gerekmez):
  var SEN_CLS = { ok: "completed", warn: "blocked", fail: "failed", skip: "pending" };
  var SEN_GLYPH = { ok: "✓", warn: "≈", fail: "✕", skip: "–" };

  function senRenderReport(report) {
    var overallEl = document.getElementById("senOverall");
    if (overallEl && report) {
      overallEl.textContent = (report.overall || "?").toUpperCase() + " — " + (report.summary || "");
    }
    var el = document.getElementById("senProbes");
    if (!el) return;
    var probes = (report && report.probes) || [];
    if (!probes.length) {
      el.innerHTML = '<span class="muted small">Yoklama yok.</span>';
      return;
    }
    el.innerHTML = probes
      .map(function (p) {
        var advice = p.advice
          ? '<div class="orc-st-msg"><strong>Öneri:</strong> ' + esc(p.advice) + "</div>"
          : "";
        return (
          '<div class="orc-stage ' + (SEN_CLS[p.status] || "pending") + '">' +
          '<div class="orc-st-name">' + (SEN_GLYPH[p.status] || "?") + " " + esc(p.name) + "</div>" +
          '<div class="orc-st-msg">' + esc(p.detail || "") + "</div>" + advice + "</div>"
        );
      })
      .join("");
  }

  function senRenderHistory(items) {
    var el = document.getElementById("senHistory");
    if (!el) return;
    if (!items || !items.length) {
      el.innerHTML = '<span class="muted small">—</span>';
      return;
    }
    el.innerHTML = items
      .map(function (h) {
        var cls = h.overall === "fail" ? "error" : h.overall === "warn" ? "warning" : "";
        return (
          '<div class="ev ' + cls + '">' +
          esc((h.created_at || "").replace("T", " ").slice(0, 19)) +
          " · " + esc(h.overall) + " — " + esc(h.summary || "") + "</div>"
        );
      })
      .join("");
  }

  function runSentinel() {
    var btn = document.getElementById("senRunBtn");
    if (btn) btn.disabled = true;
    api("/sentinel/overview")
      .then(function (d) {
        senRenderReport(d.report);
        senRenderHistory(d.history);
      })
      .catch(function (e) {
        toast("Nöbetçi koşulamadı: " + e.message, true);
      })
      .finally(function () {
        if (btn) btn.disabled = false;
      });
  }

  function loadSentinel() {
    var rb = document.getElementById("senRunBtn");
    if (rb && !rb._wired) {
      rb._wired = true;
      rb.addEventListener("click", runSentinel);
      document.getElementById("senRefreshBtn").addEventListener("click", runSentinel);
    }
    // sekme açılınca geçmişi göster (canlı koşu butonla — sekme gezinmesi probe koşturmasın)
    api("/sentinel/history?limit=10")
      .then(function (d) {
        senRenderHistory(d.history);
      })
      .catch(function () {});
  }

  // ---------- 15 · AJAN HARİTASI (canlı "ışıklı yol" grafiği) ----------
  // Salt-okuma: /agents/graph (düğüm+kenar+canlı durum) poll edilir; harita hiçbir şeyi
  // kendisi başlatmaz. ⚡ tetik ise ANA ajanı (orchestration-autodrive) devreye sokar —
  // gerçek `claude -p` yalnız confirm() sonrası (Kural 8: eğitim onay kapısında durur).
  var amPollTimer = null;
  var amCurrentRun = null;
  var amSelectedId = null;
  var amSig = null; // düğüm-kümesi imzası → yeniden-kurma yalnız küme değişince
  var amLastData = null;
  var amMainId = null; // ana ajan id (alt tam-genişlik yeşil buton olarak render edilir)
  var amGateRun = null; // hattın son koşusu — onay kapısında mı diye bakılır (iki-tık akışı)
  var amResizeTimer = null; // pencere yeniden boyutlanınca yerleşimi tazeler (debounce)
  var amEngines = []; // /api/engines özeti (salt bilgi; KİMLİK BİLGİSİ İÇERMEZ)
  var amConfirmResolve = null; // açık onay diyaloğunun promise resolver'ı
  var amDriving = false; // bu oturumda sürüş başlatıldı mı → canlı şerit + DURDUR görünür
  var AM_STLBL = {
    idle: "boşta",
    running: "çalışıyor",
    blocked: "onay bekliyor",
    error: "hata",
    done: "bitti",
  };

  function amPanelActive() {
    var p = document.getElementById("panel-agentmap");
    return !!(p && p.classList.contains("active"));
  }

  function amShort(s, n) {
    s = String(s || "");
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  function amStatusClass(n) {
    var s = (n && n.status) || "idle";
    if (["running", "blocked", "error"].indexOf(s) === -1) s = "idle"; // done/bilinmeyen → idle
    return "status-" + s;
  }

  // Deterministik yerleşim: gruba göre yatay lane; lane içinde id'ye göre stabil x
  // (aynı girdi → aynı konum; poll'de düğümler zıplamaz).
  // "DERLİ TOPLU" yerleşim: gruplar YATAY ŞERİTLER (sol→sağ katman akışı); her şeritte
  // EŞİT-BOY KARTLAR sütun hizalı (id'ye göre stabil, satır taşarsa alta sarar). Ana ajan
  // şeritlerden ÇIKARILIR → ALT ŞERİTTE tam-genişlik yeşil buton; ondan yükselen yeşil
  // "devreye soktukları" (aktivasyon) yolları zincir-köklerine gider. Tüm konum deterministik.
  var AM_CARD_W = 152,
    AM_CARD_H = 44,
    AM_CARD_GX = 14,
    AM_CARD_GY = 10,
    AM_MAX_PER_ROW = 12, // sert üst sınır (aşırı gerilmesin)
    AM_MIN_PER_ROW = 4, // dar ekranda bile okunur kalsın (altında yatay kaydırma devreye girer)
    AM_PAD = 16,
    AM_LANE_PAD = 12,
    AM_LANE_LABEL_H = 22,
    AM_LANE_GAP = 16,
    AM_MAIN_H = 58,
    AM_MAIN_GAP = 40;

  // Satır başına kaç kart sığar? Tuvalin GERÇEK genişliğinden hesaplanır → geniş ekranda
  // 8 ajanlı şerit tek satıra sığar (sarma yok, yatay kaydırma yok = "tüm ajanlar görünsün").
  // Ölçüm alınamazsa 6'ya düşer (eski davranış). Belirli bir genişlik için sonuç deterministik.
  function amPerRowCap() {
    var el = document.getElementById("amCanvas");
    var avail = (el && el.clientWidth) || 0;
    if (!avail) return 6;
    var inner = avail - AM_PAD * 2 - AM_LANE_PAD * 2 - 10; // kenarlık/kaydırma payı
    var n = Math.floor((inner + AM_CARD_GX) / (AM_CARD_W + AM_CARD_GX));
    return Math.max(AM_MIN_PER_ROW, Math.min(AM_MAX_PER_ROW, n));
  }

  function amLayout(data) {
    var order = (data.groups || []).map(function (g) {
      return g.key;
    });
    var mainNode = null;
    var byGroup = {};
    (data.nodes || []).forEach(function (n) {
      if (n.is_main) {
        mainNode = n; // ana ajan şeritlerde DEĞİL, altta buton
        return;
      }
      (byGroup[n.group] = byGroup[n.group] || []).push(n);
    });
    var lanes = order.filter(function (k) {
      return (byGroup[k] || []).length;
    });
    Object.keys(byGroup).forEach(function (k) {
      if (lanes.indexOf(k) === -1) lanes.push(k); // haritada olmayan grup → sona
    });
    // Satır kapasitesi tuval genişliğinden gelir (geniş panelde 8'lik şerit tek satır olur).
    var perRowCap = amPerRowCap();
    // Global en-geniş satır (tüm şeritler aynı genişlikte kutu; kartlar sola hizalı).
    var maxRow = 1;
    lanes.forEach(function (k) {
      maxRow = Math.max(maxRow, Math.min(byGroup[k].length, perRowCap));
    });
    var innerW = maxRow * AM_CARD_W + (maxRow - 1) * AM_CARD_GX;
    var sectionW = innerW + AM_LANE_PAD * 2;
    var W = sectionW + AM_PAD * 2;

    var pos = {};
    var laneMeta = [];
    var y = AM_PAD;
    lanes.forEach(function (gk) {
      var arr = byGroup[gk].slice().sort(function (a, b) {
        return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
      });
      var perRow = Math.min(arr.length, perRowCap);
      var rows = Math.ceil(arr.length / perRow);
      var cardsTop = y + AM_LANE_PAD + AM_LANE_LABEL_H;
      var laneInnerH = rows * AM_CARD_H + (rows - 1) * AM_CARD_GY;
      var sectionH = AM_LANE_PAD + AM_LANE_LABEL_H + laneInnerH + AM_LANE_PAD;
      arr.forEach(function (n, i) {
        var row = Math.floor(i / perRow),
          col = i % perRow;
        var cx0 = AM_PAD + AM_LANE_PAD + col * (AM_CARD_W + AM_CARD_GX);
        var cy0 = cardsTop + row * (AM_CARD_H + AM_CARD_GY);
        pos[n.id] = {
          x: cx0,
          y: cy0,
          w: AM_CARD_W,
          h: AM_CARD_H,
          cx: cx0 + AM_CARD_W / 2,
          cy: cy0 + AM_CARD_H / 2,
          left: cx0,
          right: cx0 + AM_CARD_W,
          top: cy0,
          bottom: cy0 + AM_CARD_H,
          node: n,
        };
      });
      laneMeta.push({
        label: (arr[0] && arr[0].group_label) || gk,
        x: AM_PAD,
        y: y,
        w: sectionW,
        h: sectionH,
      });
      y += sectionH + AM_LANE_GAP;
    });

    // Alt ana-ajan butonu + yüksekliği.
    var mainBar = null;
    var H = y - AM_LANE_GAP + AM_PAD;
    if (mainNode) {
      var mainTop = y - AM_LANE_GAP + AM_MAIN_GAP;
      mainBar = {
        x: AM_PAD,
        y: mainTop,
        w: W - AM_PAD * 2,
        h: AM_MAIN_H,
        cx: W / 2,
        cy: mainTop + AM_MAIN_H / 2,
        node: mainNode,
      };
      H = mainTop + AM_MAIN_H + AM_PAD;
    }

    // Zincir kökleri (gelen chain-kenarı olmayan; ana ajanın "devreye soktukları").
    var hasIncoming = {};
    (data.edges || []).forEach(function (e) {
      if (e.kind !== "data") hasIncoming[e.to] = true;
    });
    var roots = [];
    lanes.forEach(function (gk) {
      byGroup[gk].forEach(function (n) {
        if (!hasIncoming[n.id]) roots.push(n.id);
      });
    });

    return { pos: pos, lanes: laneMeta, mainBar: mainBar, roots: roots, W: W, H: H };
  }

  // Yuvarlatılmış dikdörtgen YOLU (path) — "gezen LED" halkası bunun üstünde döner.
  // <path> kullanılır (<rect> değil): pathLength her tarayıcıda path'te güvenilir çalışır →
  // dash birimleri %0-100'e normalize olur, kart boyutundan BAĞIMSIZ animasyon.
  function amRoundRectPath(x, y, w, h, r) {
    var x2 = x + w,
      y2 = y + h;
    return (
      "M" + (x + r) + " " + y +
      " H" + (x2 - r) + " A" + r + " " + r + " 0 0 1 " + x2 + " " + (y + r) +
      " V" + (y2 - r) + " A" + r + " " + r + " 0 0 1 " + (x2 - r) + " " + y2 +
      " H" + (x + r) + " A" + r + " " + r + " 0 0 1 " + x + " " + (y2 - r) +
      " V" + (y + r) + " A" + r + " " + r + " 0 0 1 " + (x + r) + " " + y + " Z"
    );
  }

  // Dik-açılı (ortogonal) yol: aşağı → yatay → aşağı (baskın akış yönü). Aynı satırda yan-yana.
  function amOrthPath(a, b) {
    if (Math.abs(a.cy - b.cy) < 6) {
      // aynı satır → yatay
      return b.cx >= a.cx
        ? "M" + a.right + " " + a.cy + " H" + b.left
        : "M" + a.left + " " + a.cy + " H" + b.right;
    }
    if (b.cy > a.cy) {
      var my = ((a.bottom + b.top) / 2).toFixed(1);
      return "M" + a.cx + " " + a.bottom + " V" + my + " H" + b.cx + " V" + b.top;
    }
    var my2 = ((b.bottom + a.top) / 2).toFixed(1);
    return "M" + a.cx + " " + a.top + " V" + my2 + " H" + b.cx + " V" + b.bottom;
  }

  function amBuildSvg(data) {
    var L = amLayout(data);
    amMainId = L.mainBar ? L.mainBar.node.id : null;
    var out = [];
    out.push(
      '<svg class="am-svg" width="' + L.W + '" height="' + L.H + '" viewBox="0 0 ' +
        L.W + " " + L.H + '" role="img" aria-label="Ajan etkileşim haritası (ışıklı yol)">'
    );
    out.push(
      "<defs>" +
        '<marker id="am-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" ' +
        'markerHeight="7" orient="auto-start-reverse">' +
        '<path class="am-arrow-head" d="M0,0 L10,5 L0,10 z" /></marker>' +
        '<marker id="am-arrow-act" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" ' +
        'markerHeight="7" orient="auto-start-reverse">' +
        '<path class="am-arrow-act-head" d="M0,0 L10,5 L0,10 z" /></marker>' +
        "</defs>"
    );
    // 1) Şerit kutuları + etiketler (en altta)
    L.lanes.forEach(function (lm) {
      out.push(
        '<rect class="am-lane-box" x="' + lm.x + '" y="' + lm.y + '" width="' + lm.w +
          '" height="' + lm.h + '" rx="10" />'
      );
      out.push(
        '<text class="am-lane-label" x="' + (lm.x + AM_LANE_PAD) + '" y="' +
          (lm.y + AM_LANE_PAD + 12) + '">' + esc(lm.label) + "</text>"
      );
    });
    // 2) Aktivasyon yolları (ana buton → zincir kökleri; yükselen yeşil "ışıklı yol")
    if (L.mainBar) {
      L.roots.forEach(function (rid) {
        var r = L.pos[rid];
        if (!r) return;
        var sx = Math.max(L.mainBar.x + 24, Math.min(L.mainBar.x + L.mainBar.w - 24, r.cx));
        var my = ((L.mainBar.y + r.bottom) / 2).toFixed(1);
        var d = "M" + sx + " " + L.mainBar.y + " V" + my + " H" + r.cx + " V" + r.bottom;
        out.push(
          '<path class="am-edge am-edge-activate" data-from="' + esc(amMainId) +
            '" data-to="' + esc(rid) + '" d="' + d + '" marker-end="url(#am-arrow-act)" />'
        );
      });
    }
    // 3) Normal kenarlar (kartların ALTINDA) — dik açılı
    (data.edges || []).forEach(function (e) {
      var a = L.pos[e.from],
        b = L.pos[e.to];
      if (!a || !b) return;
      var kind = e.kind === "data" ? "data" : e.kind === "control" ? "control" : "chain";
      var arrow = kind === "data" ? "" : ' marker-end="url(#am-arrow)"';
      out.push(
        '<path class="am-edge am-edge-' + kind + '" data-from="' + esc(e.from) +
          '" data-to="' + esc(e.to) + '" d="' + amOrthPath(a, b) + '"' + arrow + " />"
      );
    });
    // 4) Kartlar (kenarların ÜSTÜNDE)
    (data.nodes || []).forEach(function (n) {
      var p = L.pos[n.id];
      if (!p) return; // ana ajan burada yok (altta buton)
      var cls = "am-node " + amStatusClass(n) + (n.dangerous ? " am-danger" : "");
      out.push(
        '<g class="' + cls + '" data-id="' + esc(n.id) + '" tabindex="0" role="button" ' +
          'aria-label="' + esc(n.name || n.id) + " — " +
          esc(AM_STLBL[n.status] || n.status || "boşta") + '">'
      );
      out.push(
        '<rect class="am-card" x="' + p.x + '" y="' + p.y + '" width="' + p.w + '" height="' +
          p.h + '" rx="7" />'
      );
      out.push(
        '<rect class="am-card-edge" x="' + p.x + '" y="' + (p.y + 3) + '" width="5" height="' +
          (p.h - 6) + '" rx="2" />'
      );
      // "gezen LED" halkası — yalnız çalışırken görünür (CSS), kartın çevresinde döner.
      // İki katman: geniş yarı-saydam hale + ince parlak LED (filtresiz → ucuz render).
      var orbitD = amRoundRectPath(p.x, p.y, p.w, p.h, 7);
      out.push('<path class="am-card-orbit-glow" pathLength="100" d="' + orbitD + '" />');
      out.push('<path class="am-card-orbit" pathLength="100" d="' + orbitD + '" />');
      out.push(
        '<text class="am-card-label" x="' + (p.x + 16) + '" y="' + p.cy.toFixed(1) + '">' +
          esc(amShort(n.name || n.id, 17)) + "</text>"
      );
      out.push("</g>");
    });
    // 5) Ana ajan — ALT tam-genişlik yeşil buton (tıkla → ⚡ tetik)
    if (L.mainBar) {
      var mb = L.mainBar,
        mn = mb.node;
      out.push(
        '<g class="am-mainbar ' + amStatusClass(mn) + '" data-main="1" data-id="' + esc(mn.id) +
          '" tabindex="0" role="button" aria-label="⚡ Eğitimi devreye sok — ' +
          esc(mn.name || mn.id) + '">'
      );
      out.push(
        '<rect class="am-mainbar-box" x="' + mb.x + '" y="' + mb.y + '" width="' + mb.w +
          '" height="' + mb.h + '" rx="12" />'
      );
      // ana ajan çalışırken butonun çevresinde dönen beyaz LED'ler (hale + LED, filtresiz)
      var mOrbitD = amRoundRectPath(mb.x, mb.y, mb.w, mb.h, 12);
      out.push('<path class="am-mainbar-orbit-glow" pathLength="100" d="' + mOrbitD + '" />');
      out.push('<path class="am-mainbar-orbit" pathLength="100" d="' + mOrbitD + '" />');
      out.push(
        '<text class="am-mainbar-label" x="' + mb.cx + '" y="' + mb.cy.toFixed(1) + '">' +
          "⚡ EĞİTİMİ DEVREYE SOK · " + esc(amShort(mn.name || mn.id, 26)) + "</text>"
      );
      out.push("</g>");
    }
    out.push("</svg>");
    return out.join("");
  }

  function amApplyStatus(data) {
    var canvas = document.getElementById("amCanvas");
    if (!canvas) return;
    var byId = {};
    (data.nodes || []).forEach(function (n) {
      byId[n.id] = n;
    });
    // "İLİŞKİLİ" küme: çalışan bir ajana kenarla bağlı komşular → o an hangi ajanların
    // birbiriyle iş gördüğü tek bakışta anlaşılsın (çalışan = gezen LED, komşu = ışıklı çerçeve).
    // Kenarlar DOM'dan okunur: böylece sentetik aktivasyon yolları (ana ajan → kökler) de sayılır.
    var linked = {};
    canvas.querySelectorAll(".am-edge").forEach(function (ln) {
      var f = ln.getAttribute("data-from"),
        t = ln.getAttribute("data-to");
      var a = byId[f],
        b = byId[t];
      if (a && a.status === "running") linked[t] = true;
      if (b && b.status === "running") linked[f] = true;
    });
    canvas.querySelectorAll(".am-node").forEach(function (g) {
      var id = g.getAttribute("data-id");
      var n = byId[id];
      g.classList.remove("status-idle", "status-running", "status-blocked", "status-error");
      g.classList.add(amStatusClass(n || { status: "idle" }));
      g.classList.toggle("am-selected", id === amSelectedId);
      // çalışanın kendisi zaten LED'li; komşuysa "ilişkili" parıltısı al
      g.classList.toggle("am-linked", !!linked[id] && !(n && n.status === "running"));
    });
    // Ana ajan alt butonu: durum sınıfı + etiketi canlı tut.
    var mainG = canvas.querySelector(".am-mainbar");
    if (mainG) {
      var mainId = mainG.getAttribute("data-id");
      var mn = byId[mainId];
      mainG.classList.remove("status-idle", "status-running", "status-blocked", "status-error");
      mainG.classList.add(amStatusClass(mn || { status: "idle" }));
      mainG.classList.toggle("am-linked", !!linked[mainId] && !(mn && mn.status === "running"));
      var lbl = mainG.querySelector(".am-mainbar-label");
      if (lbl && mn) {
        lbl.textContent =
          "⚡ EĞİTİMİ DEVREYE SOK · " + amShort(mn.name || mn.id, 26) +
          " (" + (AM_STLBL[mn.status] || mn.status || "boşta") + ")";
      }
    }
    canvas.querySelectorAll(".am-edge").forEach(function (ln) {
      var from = ln.getAttribute("data-from");
      var to = ln.getAttribute("data-to");
      var src = byId[from];
      // "ışıklı yol": kaynak düğüm çalışıyorsa kenar akar
      ln.classList.toggle("flowing", !!(src && src.status === "running"));
      if (amSelectedId) {
        var inc = from === amSelectedId || to === amSelectedId;
        ln.classList.toggle("am-edge-hi", inc);
        ln.classList.toggle("am-edge-dim", !inc);
      } else {
        ln.classList.remove("am-edge-hi", "am-edge-dim");
      }
    });
  }

  function amUpdateStatusLine(data) {
    var el = document.getElementById("amStatusLine");
    if (!el) return;
    var nodes = data.nodes || [];
    var run = 0,
      blk = 0,
      err = 0,
      main = "idle";
    nodes.forEach(function (n) {
      if (n.status === "running") run++;
      else if (n.status === "blocked") blk++;
      else if (n.status === "error") err++;
      if (n.is_main) main = n.status || "idle";
    });
    el.textContent =
      nodes.length + " ajan · " + run + " çalışıyor · " + blk + " onay bekliyor" +
      (err ? " · " + err + " hata" : "") + " · ana ajan: " + (AM_STLBL[main] || main);
  }

  function amRenderDetail(id) {
    var el = document.getElementById("amDetail");
    if (!el) return;
    var n = null;
    ((amLastData && amLastData.nodes) || []).some(function (x) {
      if (x.id === id) {
        n = x;
        return true;
      }
      return false;
    });
    if (!n) {
      el.innerHTML = '<span class="muted small">Ajan bulunamadı.</span>';
      return;
    }
    function list(arr, cap) {
      arr = arr || [];
      if (!arr.length) return '<span class="muted small">—</span>';
      var shown = arr
        .slice(0, cap)
        .map(function (x) {
          return "<code>" + esc(x) + "</code>";
        })
        .join(" ");
      return (
        shown +
        (arr.length > cap
          ? ' <span class="muted small">+' + (arr.length - cap) + " diğer</span>"
          : "")
      );
    }
    var tags = [];
    if (n.is_main) tags.push('<span class="am-tag am-tag-main">ANA AJAN · claude -p</span>');
    if (n.dangerous) tags.push('<span class="am-tag am-tag-danger">TEHLİKELİ</span>');
    if (n.approval_required) tags.push('<span class="am-tag am-tag-warn">ONAY ŞART</span>');
    tags.push('<span class="am-tag">' + esc(n.autonomy || "manual") + "</span>");
    el.innerHTML =
      '<div class="am-d-head"><span class="am-d-name">' + esc(n.name || n.id) + "</span>" +
      '<span class="am-d-status am-st-' + esc(n.status || "idle") + '">● ' +
      esc(AM_STLBL[n.status] || n.status || "boşta") + "</span></div>" +
      '<div class="am-d-tags">' + tags.join(" ") + "</div>" +
      '<div class="am-d-grp muted small">' + esc(n.group_label || n.group || "") + "</div>" +
      '<dl class="am-d-dl">' +
      "<dt>Tetikleyici</dt><dd>" +
      (n.trigger ? esc(n.trigger) : '<span class="muted small">—</span>') + "</dd>" +
      "<dt>Okur</dt><dd>" + list(n.reads, 4) + "</dd>" +
      "<dt>Yazar</dt><dd>" + list(n.writes, 4) + "</dd>" +
      "<dt>Güvenlik kapıları</dt><dd>" + list(n.safety_gates, 5) + "</dd>" +
      "</dl>";
  }

  function amSelectNode(id) {
    amSelectedId = id;
    amRenderDetail(id);
    if (amLastData) amApplyStatus(amLastData);
  }

  function amRender(data) {
    amLastData = data;
    var canvas = document.getElementById("amCanvas");
    if (!canvas) return;
    var sig = (data.nodes || [])
      .map(function (n) {
        return n.id;
      })
      .sort()
      .join("|");
    if (sig !== amSig || !canvas.querySelector("svg")) {
      canvas.innerHTML = amBuildSvg(data); // yalnız düğüm-kümesi değişince yeniden kur
      amSig = sig;
    }
    amApplyStatus(data);
    amUpdateStatusLine(data);
    if (amSelectedId) amRenderDetail(amSelectedId); // canlı durum ayrıntıda da tazelensin
  }

  function amRefreshOnce() {
    return api("/agents/graph")
      .then(function (d) {
        amRender(d);
      })
      .catch(function (e) {
        var c = document.getElementById("amCanvas");
        if (c) c.innerHTML = '<span class="muted small">Harita yüklenemedi: ' + esc(e.message) + "</span>";
      });
  }

  // ── İKİ-TIK EĞİTİM KAPISI ──────────────────────────────────────────────────
  // ⚡ RUN salt-okuma aşamalarını + zorunlu derin avı sürer, sonra `approval`/`train`
  // aşamasında DURUR (Kural 8). Burada hattın son koşusuna bakıp, kapıda bekliyorsa
  // "ONAYLA VE EĞİTİMİ BAŞLAT" kutusunu açarız — onayı İNSAN verir, motor değil.
  function amRefreshGate() {
    return api("/orchestration/runs?limit=1")
      .then(function (d) {
        var run = ((d && d.runs) || [])[0] || null;
        amGateRun = run;
        var box = document.getElementById("amGate");
        var info = document.getElementById("amGateInfo");
        var waiting = !!(
          run &&
          run.status === "blocked" &&
          (run.current_stage === "approval" || run.current_stage === "train")
        );
        if (box) box.hidden = !waiting;
        if (info && run) {
          info.textContent =
            "· koşu " + (run.run_id || "?") + " · aşama: " + (run.current_stage || "-");
        }
        // Orkestrasyon insan kapısında `blocked` olabilirken motor hâlâ çalışabilir.
        // Şeridi DB durumundan değil, sunucunun gerçek alt-süreç kaydından sür.
        if (amDriving) {
          if (run && run.driver_running) {
            var live = document.getElementById("amLiveText");
            if (live) {
              live.textContent =
                "Sürüş devrede — aşama: " + (run.current_stage || "-") + " · koşu " +
                (run.run_id || "?");
            }
          } else {
            amSetLive(false);
          }
        }
      })
      .catch(function () {
        var box = document.getElementById("amGate");
        if (box) box.hidden = true; // durum bilinmiyorsa kapıyı AÇMA (güvenli taraf)
      });
  }

  // ✅ İkinci tık: insan onayı + GERÇEK detached eğitim (mevcut /training/run kapısı).
  // Akış: run → needs_approval ise approvals/{id}/approve → run (onay TÜKETİLİR, eğitim başlar).
  function amApproveAndTrain() {
    var btn = document.getElementById("amApproveTrainBtn");
    var adapter = (amGateRun && amGateRun.adapter_name) || "hektor_lora";
    var iters = parseInt((amGateRun && amGateRun.iters) || 300, 10) || 300;
    if (
      !window.confirm(
        "✅ GERÇEK LoRA EĞİTİMİ BAŞLATILACAK.\n\n" +
          "Adapter: " + adapter + " · ~" + iters + " adım.\n" +
          "Uzun sürer (saatlerce) ve CPU'yu doldurur; bilgisayar açık kalmalı.\n" +
          "Eğitim DETACHED başlar — web/terminal kapansa da sürer.\n\n" +
          "Bu senin TAZE onayındır (Kural 8): tek kullanımlıktır ve yalnız bu eğitimi yetkilendirir.\n" +
          "Başlatılsın mı?"
      )
    )
      return;
    if (btn) btn.disabled = true;
    var payload = { adapter_name: adapter, iterations: iters };
    function runTraining() {
      return api("/training/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }
    runTraining()
      .then(function (d) {
        if (d && d.status === "needs_approval" && d.approval_id) {
          // Onayı İNSAN yüzeyinden ver, sonra eğitimi tekrar çağır (onay tüketilir).
          return api("/approvals/" + encodeURIComponent(d.approval_id) + "/approve", {
            method: "POST",
          }).then(runTraining);
        }
        return d;
      })
      .then(function (d) {
        if (d && d.status === "started") {
          toast("🚀 Eğitim başladı (detached) — harita canlı yanacak.");
        } else if (d && d.status === "blocked") {
          toast("Engellendi: " + (d.message || "STOP_ALL aktif olabilir"), true);
        } else {
          toast((d && d.message) || "Eğitim başlatılamadı", !(d && d.ok));
        }
        amRefreshGate();
        amRefreshOnce();
        if (typeof refreshTrain === "function") refreshTrain(); // üst şerit rozetini tazele
      })
      .catch(function (e) {
        toast("Eğitim başlatılamadı: " + e.message, true);
      })
      .finally(function () {
        if (btn) btn.disabled = false;
      });
  }

  function amStartPoll() {
    if (amPollTimer) return;
    amPollTimer = setInterval(function () {
      if (!amPanelActive()) {
        clearInterval(amPollTimer); // sekme kapanınca poll dursun
        amPollTimer = null;
        return;
      }
      amRefreshOnce();
      amRefreshGate();
    }, 7000);
  }

  function amShowDryRun(cmd) {
    var box = document.getElementById("amDryRun");
    var el = document.getElementById("amDryRunCmd");
    if (el) el.textContent = cmd || ""; // textContent → XSS yok
    if (box) box.hidden = !cmd; // hidden özelliği (inline style değil) — CSP-güvenli
  }

  function amCanvasClick(ev) {
    var t = ev.target;
    if (!t || !t.closest) return;
    if (t.closest(".am-mainbar")) {
      amTriggerTraining(); // alt yeşil buton = ⚡ tetik (grafik-içi)
      return;
    }
    var g = t.closest(".am-node");
    if (g) amSelectNode(g.getAttribute("data-id"));
  }

  function amCanvasKey(ev) {
    if (ev.key !== "Enter" && ev.key !== " ") return;
    var t = ev.target;
    if (!t || !t.closest) return;
    if (t.closest(".am-mainbar")) {
      ev.preventDefault();
      amTriggerTraining();
      return;
    }
    var g = t.closest(".am-node");
    if (g) {
      ev.preventDefault();
      amSelectNode(g.getAttribute("data-id"));
    }
  }

  // ---------- motor seçici (salt bilgi — /api/engines) ----------
  // ⛔ Bu blok KİMLİK BİLGİSİ İŞLEMEZ: uç token/mail/anahtar döndürmez, UI de istemez.
  // Kurulu olmayan motor <option disabled> → tarayıcı seçtirmez; sunucu da ayrıca reddeder
  // (`engines.run_blocked_reason`, çift kapı — gri buton güvenlik sınırı DEĞİLDİR).
  function amSelectedEngine() {
    var sel = document.getElementById("amEngineSel");
    var name = (sel && sel.value) || "";
    for (var i = 0; i < amEngines.length; i++) {
      if (amEngines[i].name === name) return amEngines[i];
    }
    return null;
  }

  function amRenderEngines(d) {
    amEngines = (d && d.engines) || [];
    var sel = document.getElementById("amEngineSel");
    if (!sel) return;
    var prev = sel.value;
    sel.innerHTML = "";
    amEngines.forEach(function (e) {
      var o = document.createElement("option"); // DOM + textContent → XSS yok
      o.value = e.name;
      o.textContent = e.label + (e.selectable ? "" : " — kullanılamaz");
      o.disabled = !e.selectable; // kurulu değil / süreç başlatmaz / sertleştirilemez
      sel.appendChild(o);
    });
    // Seçim: önceki geçerliyse korunur, değilse ilk SEÇİLEBİLİR motora düşer.
    var keep = amEngines.some(function (e) {
      return e.name === prev && e.selectable;
    });
    if (!keep) {
      var first = amEngines.filter(function (e) {
        return e.selectable;
      })[0];
      sel.value = first ? first.name : "";
    } else {
      sel.value = prev;
    }
    amRenderEngineNote();
  }

  function amRenderEngineNote() {
    var box = document.getElementById("amEngineNote");
    if (!box) return;
    var cur = amSelectedEngine();
    var parts = [];
    if (cur) {
      // Giriş durumu UYDURULMAZ: uç `logged_in: null` döner, biz de "bilinmiyor" deriz.
      parts.push(
        "<strong>" +
          esc(cur.label) +
          "</strong> · kurulu: " +
          (cur.installed ? "evet" : "hayır") +
          " · giriş: bilinmiyor (" +
          esc(cur.login_note || "") +
          ")"
      );
      if (cur.quota_warning) parts.push("⚠ " + esc(cur.quota_warning));
    } else {
      parts.push("Seçilebilir motor yok — aşağıdaki kurulum ipuçlarına bak.");
    }
    // Kullanılamayan motorlar için "nasıl kurulur" (kimlik formu DEĞİL — kendi terminalinde
    // çalıştıracağın kurulum/giriş komutu).
    amEngines
      .filter(function (e) {
        return !e.selectable;
      })
      .forEach(function (e) {
        parts.push(
          "⛔ <strong>" +
            esc(e.label) +
            "</strong>: " +
            esc(e.blocked_reason || "kullanılamaz") +
            (e.install_hint ? " — " + esc(e.install_hint) : "")
        );
      });
    box.innerHTML = parts.join("<br>");
  }

  function amLoadEngines(rescan) {
    return api("/engines" + (rescan ? "/rescan" : ""), rescan ? { method: "POST" } : undefined)
      .then(amRenderEngines)
      .catch(function (e) {
        var box = document.getElementById("amEngineNote");
        if (box) box.textContent = "Motor listesi alınamadı: " + e.message;
      });
  }

  // ---------- ATLANAMAZ onay diyaloğu ----------
  // Geri alınamaz iş (gerçek motor spawn'ı) YALNIZ buradan geçer: `execute:true` isteği
  // bu promise `true` çözülmeden GÖNDERİLMEZ. Tek tıkla spawn YOKTUR.
  function amConfirmOpen(eng) {
    var modal = document.getElementById("amConfirmModal");
    var elEngine = document.getElementById("amConfirmEngine");
    var elQuota = document.getElementById("amConfirmQuota");
    if (!modal) return Promise.resolve(false); // diyalog yoksa ASLA çalıştırma (fail-closed)
    // textContent → sunucudan gelen metin işaretleme olarak yorumlanmaz.
    if (elEngine) {
      elEngine.textContent =
        eng.label + " (" + eng.name + ") · abonelik CLI'si — API anahtarı KULLANILMAZ";
    }
    if (elQuota) elQuota.textContent = eng.quota_warning || "—";
    // Her açılışta onay kutusu SIFIRLANIR ve başlat butonu KİLİTLENİR — önceki
    // diyalogdan kalan işaret bir sonrakini asla açmasın.
    var ack = document.getElementById("amConfirmAck");
    var go = document.getElementById("amConfirmGoBtn");
    if (ack) ack.checked = false;
    if (go) go.disabled = true;
    modal.classList.remove("hidden");
    // ⚠️ ODAK GÜVENLİ TARAFA: yıkıcı butona ASLA odaklanma. Odaklanmış bir "başlat"
    // butonu tek bir dalgın Enter'la gerçek motor doğurur — bu geliştirme sırasında
    // FİİLEN yaşandı (5 istenmeyen `claude -p` spawn'ı). Odak "Vazgeç"e verilir.
    var cancel = document.getElementById("amConfirmCancelBtn");
    if (cancel) cancel.focus();
    return new Promise(function (resolve) {
      amConfirmResolve = resolve;
    });
  }

  function amConfirmClose(ok) {
    var modal = document.getElementById("amConfirmModal");
    if (modal) modal.classList.add("hidden");
    var resolve = amConfirmResolve;
    amConfirmResolve = null;
    if (resolve) resolve(!!ok);
  }

  function amWireConfirm() {
    var modal = document.getElementById("amConfirmModal");
    if (!modal || modal._wired) return;
    modal._wired = true;
    ["amConfirmCancelBtn", "amConfirmCloseBtn"].forEach(function (id) {
      var b = document.getElementById(id);
      if (b) b.addEventListener("click", function () { amConfirmClose(false); });
    });
    // Onay kutusu → başlat butonunun kilidi (ikinci bilinçli hareket).
    var ack = document.getElementById("amConfirmAck");
    var go = document.getElementById("amConfirmGoBtn");
    if (ack && go) {
      ack.addEventListener("change", function () {
        go.disabled = !ack.checked;
      });
    }
    if (go) {
      go.addEventListener("click", function (ev) {
        // Üç kapı birden: (1) buton kilidi açık mı, (2) kutu gerçekten işaretli mi,
        // (3) olay GERÇEK kullanıcı hareketinden mi geliyor (script `.click()` değil).
        // `isTrusted` yalnız tarayıcının girdi hattından gelen olaylarda true'dur.
        if (go.disabled) return;
        if (!ack || !ack.checked) return;
        if (ev && ev.isTrusted === false) return;
        amConfirmClose(true);
      });
    }
    // Arka plana tıklama ve Esc → İPTAL (varsayılan güvenli taraf).
    modal.addEventListener("click", function (ev) {
      if (ev.target === modal) amConfirmClose(false);
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && amConfirmResolve) amConfirmClose(false);
    });
  }

  // ---------- canlı şerit + görünür DURDUR ----------
  function amSetLive(on, text) {
    amDriving = !!on;
    var box = document.getElementById("amLive");
    var el = document.getElementById("amLiveText");
    if (el && text) el.textContent = text;
    if (box) box.hidden = !on;
  }

  function amStopAll() {
    if (
      !window.confirm(
        "⛔ STOP_ALL etkinleştirilecek: tüm tehlikeli ajan aksiyonları durur ve " +
          "eğitim başlatılamaz.\n\nDevam edilsin mi?"
      )
    )
      return;
    api("/supervisor/stop-all", { method: "POST" })
      .then(function () {
        amSetLive(false);
        toast("⛔ STOP_ALL etkin — tehlikeli aksiyonlar durduruldu.");
        amRefreshOnce();
        amRefreshGate();
      })
      .catch(function (e) {
        toast("Durdurulamadı: " + e.message, true);
      });
  }

  // ⚡ ANA AJANI DEVREYE SOK: (gerekirse) koşu başlat → ATLANAMAZ onay diyaloğu → autodrive.
  // execute değeri = diyalog sonucu; onaylanmazsa DRY-RUN komutu gösterilir (spawn YOK).
  function amTriggerTraining() {
    var btn = document.getElementById("amTriggerBtn");
    var eng = amSelectedEngine();
    // ⛔ TEKRAR-GİRİŞ KİLİDİ: sürüş devredeyken yeni tetik ÜST ÜSTE motor doğurur
    // (geliştirme sırasında 5 eşzamanlı `claude -p` bu yüzden oluştu — her biri aynı
    // abonelik penceresinden yer). Önce mevcut sürüşü DURDUR.
    if (amDriving) {
      toast("Sürüş zaten devrede — yenisini başlatmadan önce DURDUR.", true);
      return;
    }
    // Diyalog zaten açıksa ikinci bir akış başlatma (resolver tek; ikincisi öncekini
    // sahipsiz bırakırdı).
    if (amConfirmResolve) return;
    if (!eng || !eng.selectable) {
      toast(
        "Kullanılabilir motor seçili değil" +
          (eng && eng.blocked_reason ? ": " + eng.blocked_reason : "."),
        true
      );
      return;
    }
    if (btn) btn.disabled = true;
    var ensure = amCurrentRun
      ? Promise.resolve({ run_id: amCurrentRun })
      : api("/orchestration/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            adapter_name: "hektor_lora",
            iters: 300,
            hunt_ack: false,
            auto_run: true,
          }),
        });
    ensure
      .then(function (snap) {
        amCurrentRun = snap.run_id || amCurrentRun;
        if (!amCurrentRun) throw new Error("koşu kimliği alınamadı");
        return amConfirmOpen(eng); // ← geri alınamaz iş için TEK kapı
      })
      .then(function (go) {
        return api("/orchestration/autodrive/" + encodeURIComponent(amCurrentRun), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ execute: go, engine: eng.name }),
        }).then(function (d) {
          return { d: d, go: go };
        });
      })
      .then(function (r) {
        if (r.go && r.d && r.d.status === "autodrive_started") {
          amShowDryRun("");
          amSetLive(true, eng.label + " sürüyor — veri hattı MCP ile ilerletiliyor "
            + "(eğitim onay kapısında duracak).");
          toast("⚡ Ana ajan devrede — ajanlar aydınlanıyor (timeline: 12·ORKESTRASYON).");
        } else if (r.d && r.d.command) {
          amShowDryRun(Array.isArray(r.d.command) ? r.d.command.join(" ") : r.d.command);
          toast(
            r.go ? "Sürüş gerekmedi — komut gösterildi." : "KURU-ÇALIŞTIRMA: komut hazır (spawn YOK)."
          );
        } else {
          amShowDryRun("");
          toast(
            "Yanıt: " + ((r.d && (r.d.reason || r.d.status)) || "bilinmiyor"),
            !(r.d && r.d.ok)
          );
        }
        amRefreshOnce();
        amRefreshGate(); // sürüş bitince hat onay kapısına gelmiş olabilir → kutuyu aç
      })
      .catch(function (e) {
        toast("Devreye sokulamadı: " + e.message, true);
      })
      .finally(function () {
        if (btn) btn.disabled = false;
      });
  }

  function loadAgentMap() {
    var canvas = document.getElementById("amCanvas");
    if (canvas && !canvas._wired) {
      canvas._wired = true; // delegasyon: #amCanvas kalıcı (innerHTML yenilense de listener durur)
      canvas.addEventListener("click", amCanvasClick);
      canvas.addEventListener("keydown", amCanvasKey);
    }
    var tb = document.getElementById("amTriggerBtn");
    if (tb && !tb._wired) {
      tb._wired = true;
      tb.addEventListener("click", amTriggerTraining);
      document.getElementById("amRefreshBtn").addEventListener("click", function () {
        amRefreshOnce();
        amRefreshGate();
      });
    }
    var ab = document.getElementById("amApproveTrainBtn");
    if (ab && !ab._wired) {
      ab._wired = true;
      ab.addEventListener("click", amApproveAndTrain);
    }
    amWireConfirm(); // onay diyaloğu — ⚡ tetikten ÖNCE bağlanmalı (fail-closed)
    var es = document.getElementById("amEngineSel");
    if (es && !es._wired) {
      es._wired = true;
      es.addEventListener("change", amRenderEngineNote);
    }
    var er = document.getElementById("amEngineRescanBtn");
    if (er && !er._wired) {
      er._wired = true;
      er.addEventListener("click", function () {
        amLoadEngines(true).then(function () {
          toast("Motorlar yeniden tarandı.");
        });
      });
    }
    var sb = document.getElementById("amStopBtn");
    if (sb && !sb._wired) {
      sb._wired = true;
      sb.addEventListener("click", amStopAll);
    }
    if (!window._amResizeWired) {
      window._amResizeWired = true;
      // Genişlik değişince satır kapasitesi değişir → yerleşimi yeniden kur (debounce'lu).
      window.addEventListener("resize", function () {
        if (amResizeTimer) clearTimeout(amResizeTimer);
        amResizeTimer = setTimeout(function () {
          amResizeTimer = null;
          if (!amPanelActive() || !amLastData) return;
          amSig = null; // imzayı sıfırla → amRender tam yeniden kurar
          amRender(amLastData);
        }, 250);
      });
    }
    amLoadEngines(false);
    amRefreshOnce();
    amRefreshGate();
    amStartPoll();
  }

  // ---------- init ----------
  refreshStatus();
  setInterval(refreshStatus, 30000);
  refreshTrain();
  setInterval(refreshTrain, 15000);
})();
