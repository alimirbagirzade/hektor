# Hektor LoRA egitimi -- DETACHED (Claude Code/terminal kapansa da surer; PC acik kaldikca)
#
# Kullanim:
#   .\scripts\start-train.ps1                         -- bf16, hektor_lora_v5, 1 epoch (1203)
#   .\scripts\start-train.ps1 -Iterations 2406        -- 2 epoch
#   .\scripts\start-train.ps1 -Dtype fp32             -- fp32 (hizli ama web/Ollama kapatilmali)
#   .\scripts\start-train.ps1 -SkipGate               -- kalite kapisini ATLA (acik insan karari)
#   .\scripts\start-train.ps1 -Stop                   -- egitimi durdur
#   .\scripts\start-train.ps1 -Status                 -- durum
#
# KALITE KAPISI: egitim, veri `pretrain-gate` (GO/NO-GO) ve `lora-audit` (Gate 0-7)
# kapilarindan gecmeden BASLAMAZ. NO-GO/FAIL -> cikis 1, egitim yok. Kapi calistirilamazsa
# da baslamaz (Kural 2). Bilincli atlamak icin -SkipGate.
#
# Start-Process ile baslatildigi icin baslatan kabuk (Claude Code dahil) kapansa da
# egitim bagimsiz surer. KOSULLAR: bilgisayar acik kalmali (uyku/hazirda bekleme YOK),
# kullanici oturumu acik kalmali (logoff egitimi durdurur).

param(
    [string]$Adapter = "hektor_lora_v5",
    # Adim sayisi. 0 = PROFIL PLANINDAN hesapla (onerilen; plan_iterations ile
    # ornek_sayisi x epochs). Elle buyuk bir sayi vermek, ornek tavani sabit oldugu
    # icin AYNI kucuk alt-kume uzerinde sessizce coklu epoch demektir (ezber riski;
    # 2026-09-08 v8 kosusu: 600 adim / 300 ornek = 2 epoch).
    [int]$Iterations = 0,
    # Egitilecek ornek TAVANI (0 = profildeki max_examples gecerli). Daha COK VERI
    # istiyorsan adim sayisini degil BUNU buyut; adim sayisi plandan turer.
    [int]$MaxExamples = 0,
    [string]$Dtype = "bf16",
    # LoRA recete profili. VARSAYILAN discipline_safe_local (assistant_only_loss maskeleme +
    # NEFTune). --profile GECMEZSEK `train --run` vanilya varsayilanla (maskesiz, lr=2e-4)
    # kosar -> prompt kalibi ezberlenir = v5 disiplin-regresyon recetesi (Kademe-2 av bulgusu).
    # Profili tamamen atlamak icin -Profile "" ver.
    [string]$Profile = "discipline_safe_local",
    # Temel model. BOS ise ayardaki varsayilan (Qwen3-4B) kullanilir. Yerel CPU'da
    # 4B bf16 ~8 GB tutar; dusuk RAM'li makinede kucuk model (or. Qwen2.5-1.5B-Instruct)
    # sec. Secim train_status.json'a yazilir ki nobetci yeniden baslatirken UNUTMASIN.
    [string]$BaseModel = "",
    # Cokme-kurtarma: son checkpoint'ten DEVAM et. Trainer'da resume artik ACIK TERCIH
    # (varsayilan KAPALI) -- ayni adapter adiyla ikinci kosu eskiden sessizce eski
    # agirliklari yeniden kullaniyor, hatta eski adim >= hedef ise SIFIR adim egitip
    # "basarili" doneyordu (Kademe-2 av bulgusu; Kural 2). Watchdog olen egitimi
    # surdururken bu switch'i gecer; sifir-adim durumu artik trainer'da hata verir.
    [switch]$Resume,
    # Kaliteyi kapiyi ATLA -- ACIK INSAN KARARI. Varsayilan KAPALI: egitim, veri
    # `pretrain-gate` (GO/NO-GO) ve `lora-audit` (Gate 0-7) kapilarindan gecmeden
    # baslamaz. Bu kapi, 31 ajanlik denetim mimarisi ile FIILEN egitilen veri
    # arasindaki tek zorunlu bagdir (bkz. reports/agent-inventory/, bulgu B2:
    # v7/v8 kosulari bu betikle baslatildi ve hicbir kapidan gecmedi).
    # Kapi ARIZALANIRSA da egitim baslamaz (Kural 2: dogrulanmadan devam etme);
    # uzun bir kosuyu kurtarmak icin bilinctli override olarak bunu gec.
    [switch]$SkipGate,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = "Continue"
# uv her `uv run`'da paketi yeniden senkronlar; calisan web sunucusunun kilitledigi
# hektor-web.exe'yi silmeye ugrasip "os error 32" ile patlar -> egitim baslamaz.
# Senkronu kapat; bagimliliklar zaten kurulu. (bkz. continuous-learning.sh)
$env:UV_NO_SYNC = "1"
$ScriptDir  = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectDir = Split-Path -Parent $ScriptDir
$LogOut     = Join-Path $ProjectDir "logs\train-full.log"
$LogErr     = Join-Path $ProjectDir "logs\train-full-err.log"
$StatusFile = Join-Path $ProjectDir "storage\train_status.json"

function Find-Uv {
    $fromPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if ($fromPath -and (Test-Path $fromPath)) { return $fromPath }
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\uv.exe")
    )
    foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}

function Get-TrainProcs {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*peft_lora_train*' -or $_.CommandLine -like '*train*--run*' }
}

if ($Status) {
    $p = Get-TrainProcs
    if ($p) { Write-Host "  Egitim: CALISIYOR ($(($p | Measure-Object).Count) surec)" -ForegroundColor Green }
    else    { Write-Host "  Egitim: calismyor" -ForegroundColor Yellow }
    if (Test-Path $LogErr) { Write-Host "  Son satir:"; Get-Content $LogErr -Tail 1 }
    exit 0
}

if ($Stop) {
    $p = Get-TrainProcs
    if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Write-Host "  [OK] Egitim durduruldu." -ForegroundColor Yellow }
    else    { Write-Host "  Zaten calismiyor." -ForegroundColor Gray }
    Remove-Item $StatusFile -Force -ErrorAction SilentlyContinue
    exit 0
}

if (Get-TrainProcs) {
    Write-Host "  [OK] Egitim zaten calisiyor (Status icin -Status)." -ForegroundColor Green
    exit 0
}

$uv = Find-Uv
if (-not $uv) { Write-Host "  [HATA] uv bulunamadi." -ForegroundColor Red; exit 1 }

$null = New-Item -ItemType Directory -Path (Split-Path $LogOut) -Force
$env:HEKTOR_TRAIN_DTYPE = $Dtype
# Resume yalniz ACIKCA istendiginde; aksi halde ortamda kalmis eski degeri TEMIZLE
# (kalici HEKTOR_TRAIN_RESUME=1 devami yeniden ortuk hale getirirdi).
if ($Resume) { $env:HEKTOR_TRAIN_RESUME = "1" } else { $env:HEKTOR_TRAIN_RESUME = "0" }
# Bu script yalnız merkezi unattended eğitim servisi tarafından kullanılır. STOP_ALL
# yine CLI içinde zorunludur; tekrar başlatmalarda tek kullanımlık insan onayı aranmaz.
$env:HEKTOR_TRAIN_SUPERVISED = "1"
# Temel model secimi alt surece ORTAM uzerinden gecer (Start-Process ortami miras alir).
if ($BaseModel -and $BaseModel.Trim() -ne "") { $env:HEKTOR_PEFT_BASE_MODEL = $BaseModel }
# ---------------------------------------------------------------------------
# KALITE KAPISI -- lora-split'ten ONCE. pretrain-gate kaynak dosyayi
# (data/lora_sft/lora_sft.jsonl) denetler; split o dosyadan turer, dolayisiyla
# kapi bolmeden once calismali. Ikisi de LLM'siz ve deterministik.
# ---------------------------------------------------------------------------
$gateSummary = "atlandi (-SkipGate)"
if ($SkipGate) {
    Write-Host "  [UYARI] Kalite kapisi ATLANDI (-SkipGate). Egitilen veri denetimden GECMEDI." -ForegroundColor Yellow
} else {
    Write-Host "  Kalite kapisi calisiyor (pretrain-gate + lora-audit)..." -ForegroundColor DarkGray

    # 1) On egitim kalite kapisi: garanti-vaadi regex, acilis-ezberi, minimum boyut.
    $pg = $null
    try { $pg = & $uv run --project "$ProjectDir" hektor pretrain-gate --json | ConvertFrom-Json } catch { $pg = $null }
    if (-not $pg -or -not $pg.verdict) {
        Write-Host "  [HATA] pretrain-gate calistirilamadi -- kapi sonucu OKUNAMADI." -ForegroundColor Red
        Write-Host "         Dogrulanmamis veriyle egitim baslatilmaz (Kural 2)." -ForegroundColor Red
        Write-Host "         Elle bak: uv run hektor pretrain-gate" -ForegroundColor Gray
        Write-Host "         Bilincli atlamak icin: -SkipGate" -ForegroundColor Gray
        exit 1
    }
    if ($pg.verdict -ne "GO") {
        Write-Host "  [ENGEL] pretrain-gate: $($pg.verdict) -- egitim BASLATILMADI." -ForegroundColor Red
        if ($pg.blockers) { foreach ($b in $pg.blockers) { Write-Host "          - $b" -ForegroundColor Red } }
        Write-Host "         Ayrinti: uv run hektor pretrain-gate" -ForegroundColor Gray
        exit 1
    }

    # 2) LoRA denetim hatti (Gate 0-7): kaynak butunlugu, sema, kopya, alan, guvenlik.
    $la = $null
    try { $la = & $uv run --project "$ProjectDir" hektor lora-audit --json | ConvertFrom-Json } catch { $la = $null }
    if ($null -eq $la -or $null -eq $la.passed) {
        Write-Host "  [HATA] lora-audit calistirilamadi -- kapi sonucu OKUNAMADI." -ForegroundColor Red
        Write-Host "         Dogrulanmamis veriyle egitim baslatilmaz (Kural 2)." -ForegroundColor Red
        Write-Host "         Elle bak: uv run hektor lora-audit" -ForegroundColor Gray
        Write-Host "         Bilincli atlamak icin: -SkipGate" -ForegroundColor Gray
        exit 1
    }
    # Bos girdi kapiyi BOSUNA gecer: `passed` = tum kapilarin AND'i, hicbir kart
    # yoksa reddedilecek kart da yoktur -> True. Sifir kartla egitim anlamsizdir ve
    # kartlarin sifir olmasi, jsonl doluysa katmanlar arasi tutarsizlik demektir.
    if ([int]$la.total_input -le 0 -or [int]$la.total_approved -le 0) {
        Write-Host "  [ENGEL] lora-audit girdi/onay sayisi SIFIR (girdi=$($la.total_input), onay=$($la.total_approved))." -ForegroundColor Red
        Write-Host "          Bos denetim 'gecti' sayilmaz; korpus/kart katmanini kontrol et." -ForegroundColor Red
        Write-Host "          Elle bak: uv run hektor lora-audit" -ForegroundColor Gray
        exit 1
    }
    if (-not $la.passed) {
        Write-Host "  [ENGEL] lora-audit BASARISIZ -- egitim BASLATILMADI." -ForegroundColor Red
        foreach ($s in $la.stages) {
            if (-not $s.passed) { Write-Host "          - Gate $($s.gate_id) $($s.name): red=$($s.rejected_count) inceleme=$($s.review_count)" -ForegroundColor Red }
        }
        Write-Host "         Rapor: $($la.report_path)" -ForegroundColor Gray
        exit 1
    }

    $gateSummary = "GO (pretrain-gate: $($pg.total) ornek; lora-audit: $($la.total_approved)/$($la.total_input) kart)"
    Write-Host "  [OK] Kapi GECILDI -- $gateSummary" -ForegroundColor Green
}

# Egitim verisi: lora_sft.jsonl -> train/valid (clobber-proof; bos train.jsonl onarilir)
& $uv run --project "$ProjectDir" hektor lora-split | Out-Null

# Adim plani: hesabi burada TEKRARLAMA -- kanonik kaynak plan_iterations
# (app/training/detached_launch.py). Plan = min(train_satiri, tavan) x profil epochs.
$nTrain = 0
$trainJsonl = Join-Path $ProjectDir "data\training\jsonl\train.jsonl"
if (Test-Path $trainJsonl) { $nTrain = (Get-Content $trainJsonl | Measure-Object -Line).Lines }
$profArg = if ($Profile -and $Profile.Trim() -ne "") { "'$Profile'" } else { "None" }
$pyPlan = "import json;from app.training.detached_launch import plan_iterations as p;i,n,e=p($nTrain,$MaxExamples,$profArg);print(json.dumps({'iters':i,'n':n,'epochs':e}))"
$plan = $null
try { $plan = & $uv run --project "$ProjectDir" python -c $pyPlan | ConvertFrom-Json } catch { $plan = $null }

if ($plan) {
    Write-Host "  Plan: $($plan.n) ornek x $($plan.epochs) epoch = $($plan.iters) adim (egitim havuzu: $nTrain satir)." -ForegroundColor DarkGray
    if ($Iterations -le 0) { $Iterations = [int]$plan.iters }
    elseif ($Iterations -gt [int]$plan.iters) {
        # Adim sayisi plandan buyukse fark SESSIZ coklu epoch'tur; profilin epochs
        # vaadi asilir (v5 disiplin-regresyonunun sinifi) -- gorunur uyar.
        $effEpochs = [math]::Round($Iterations / [math]::Max(1, [int]$plan.n), 2)
        Write-Host "  [UYARI] -Iterations $Iterations plandan ($($plan.iters)) BUYUK -> ayni $($plan.n) ornek uzerinde ~$effEpochs epoch." -ForegroundColor Yellow
        Write-Host "          Daha cok VERI icin adim sayisini degil -MaxExamples degerini buyut." -ForegroundColor Yellow
    }
} elseif ($Iterations -le 0) {
    Write-Host "  [HATA] Adim plani hesaplanamadi (plan_iterations cagrilamadi) ve -Iterations verilmedi." -ForegroundColor Red
    Write-Host "         Kor adim sayisiyla egitim baslatilmaz; -Iterations <n> ile acikca belirt." -ForegroundColor Red
    exit 1
}

# Temel argumanlar + (profil verildiyse) --profile. Profil bos ise EKLENMEZ (vanilya).
$trainArgs = @("run", "--project", "`"$ProjectDir`"", "hektor", "train", "--run",
               "--backend", "peft", "--adapter-name", $Adapter, "--iterations", "$Iterations")
if ($Profile -and $Profile.Trim() -ne "") { $trainArgs += @("--profile", $Profile) }
# Ornek tavani: verilmezse profildeki max_examples gecerli olur. Bayrak GECMEZSEK
# -MaxExamples sessizce YOK SAYILIR (2026-09-08 v8: 600 ornek istendi, 300 egitildi).
if ($MaxExamples -gt 0) { $trainArgs += @("--max-examples", "$MaxExamples") }
Start-Process -FilePath $uv `
    -ArgumentList $trainArgs `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $LogOut `
    -RedirectStandardError $LogErr `
    -WindowStyle Hidden
# Rozet/durum icin: adapter adini storage'a yaz (web /api/training/live okur)
$null = New-Item -ItemType Directory -Path (Split-Path $StatusFile) -Force
# Recetenin TAMAMI yazilir: nobetci (training-watchdog.ps1) coken egitimi YALNIZ bu
# dosyadan diriltir; profil/ornek tavani eksik kalirsa yeniden baslatma sessizce
# BASKA bir receteye kayar (bkz. _status_payload, detached_launch.py).
([ordered]@{
    adapter      = $Adapter
    dtype        = $Dtype
    iterations   = $Iterations
    base_model   = $BaseModel
    profile      = $Profile
    max_examples = $MaxExamples
} | ConvertTo-Json -Compress) |
    Out-File -FilePath $StatusFile -Encoding ascii -Force
$profLabel = if ($Profile -and $Profile.Trim() -ne "") { $Profile } else { "(vanilya)" }
$resumeLabel = if ($Resume) { "devam(checkpoint)" } else { "sifirdan" }
$modelLabel = if ($BaseModel) { $BaseModel } else { "ayardaki varsayilan" }
Write-Host "  Temel model: $modelLabel" -ForegroundColor DarkGray
Write-Host "  [OK] Egitim DETACHED baslatildi (dtype=$Dtype, adapter=$Adapter, iters=$Iterations, profil=$profLabel, mod=$resumeLabel)." -ForegroundColor Green
Write-Host "       Kalite kapisi: $gateSummary" -ForegroundColor DarkGray
Write-Host "       Claude Code/terminal kapansa da surer. PC acik + oturum acik kalmali." -ForegroundColor Cyan
Write-Host "       Ilerleme: logs\train-full-err.log" -ForegroundColor Gray
