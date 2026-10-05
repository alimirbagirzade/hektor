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
    # LoRA recete profili. VARSAYILAN moe30b_attn_local (Qwen3-30B-A3B MoE: yalniz attention +
    # assistant_only_loss maskeleme + NEFTune; app.config.DEFAULT_TRAIN_PROFILE ile ayni).
    # --profile GECMEZSEK `train --run` vanilya varsayilanla (maskesiz) kosar -> prompt kalibi
    # ezberlenir = v5 disiplin-regresyon recetesi. Dense/4B temel model icin
    # -Profile discipline_safe_local. Profili tamamen atlamak icin -Profile "" ver.
    [string]$Profile = "moe30b_attn_local",
    # Temel model. BOS ise ayardaki varsayilan (Qwen3-30B-A3B-Instruct-2507, bf16 ~61 GB RAM)
    # kullanilir. Dusuk RAM'li makinede -BaseModel Qwen/Qwen3-4B-Instruct-2507 +
    # -Profile discipline_safe_local. Secim train_status.json'a yazilir ki nobetci UNUTMASIN.
    [string]$BaseModel = "",
    # Cokme-kurtarma: son checkpoint'ten DEVAM et. Trainer'da resume artik ACIK TERCIH
    # (varsayilan KAPALI) -- ayni adapter adiyla ikinci kosu eskiden sessizce eski
    # agirliklari yeniden kullaniyor, hatta eski adim >= hedef ise SIFIR adim egitip
    # "basarili" doneyordu (Kademe-2 av bulgusu; Kural 2). Watchdog olen egitimi
    # surdururken bu switch'i gecer; sifir-adim durumu artik trainer'da hata verir.
    [switch]$Resume,
    # KURAL 8 MUAFIYETI -- yalniz COKME-KURTARMA icin. Bu anahtar HEKTOR_TRAIN_SUPERVISED=1
    # gecirir, yani alt surecteki taze-onay kapisi ATLANIR. Varsayilan KAPALI: betik eskiden
    # bu degiskeni KOSULSUZ veriyordu ama kendisi hicbir onay istegi acmiyor/tuketmiyordu ->
    # "ust katman onayi kullanildi" yaziyor, ustte onay YOK (Kademe-2 av bulgusu K8-b; v7/v8
    # kosulari boyle basladi). Nobetci (training-watchdog.ps1) bunu gecer: dirilttigi kosu
    # zaten onaylanmisti ve onay TUKETILMISTIR, ikinci kez istenemez.
    [switch]$Supervised,
    # K8-b (zaman penceresi YERINE): bu kosuyu yetkilendiren TUKETILMIS onayin kimligi.
    # -Supervised (nobetci kurtarmasi) ile birlikte gecilir -- train-recovery-check'in
    # zaten dogruladigi onayi durum dosyasina TASIR ki bir sonraki teshis/kurtarma bunu
    # bir ZAMAN PENCERESI ("son N dakikada tuketilmis onay" tahmini) yerine DOGRUDAN kimlik
    # eslesmesiyle bulsun (bkz. app/training/train_guard.py: find_run_approval). Taze
    # (-Supervised OLMAYAN) baslatmada BOS birakilir -- alt surec kendi onayini tuketir ve
    # asagida basari sonrasi log'dan okunup durum dosyasina ISLENIR.
    [string]$ApprovalId = "",
    # Kademe-2 A2: KURTARMADA karisim agirligi. Nobetci bunlari durum dosyasindan (ilk
    # kosunun TUKETTIGI karar) okuyup gecer -> alt surec `--mix-weights`/`--mix-profile`
    # ile AYNI agirlikla egitir; bir SONRAKI egitim icin kaydedilmis bekleyen karari
    # tuketmez. Taze baslatmada BOS birakilir (alt surec bekleyen karari kendisi tuketir).
    [string]$MixWeights = "",
    [string]$MixProfile = "",
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
# HEKTOR_TRAIN_SUPERVISED + HEKTOR_PEFT_BASE_MODEL burada DEGIL, yalniz Start-Process
# aninda ayarlanip hemen geri alinir (asagida) -- kalici birakilirsa kabuk kirlenir.
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

    # 1) On egitim kalite kapisi (NO-GO): garanti-vaadi, acilis/kapanis ezberi, bos veya
    #    okunamayan satir, sir/kisisel veri. Az ornek yalniz UYARI verir.
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

# Egitim verisi: lora_sft.jsonl -> train/valid. `train --run` ile AYNI bolucu
# (ensure_train_split, kaynak-gruplu). Eskiden burada `lora-split` (satir-sirali) cagriliyordu;
# iki bolucu farkli sayida satir uretiyordu (1702 vs 1680) ve plan fazla adim veriyordu ->
# ~22 ornek ikinci kez goruluyordu (v10'da ~24). 2026-09-30: train --run artik plani asan
# adimi REDDEDER, bu yuzden plan ayni bolmeden alinmali.
$pySplit = "from app.training.detached_launch import ensure_train_split as s;print(s()[0])"
$nTrain = 0
try { $nTrain = [int](& $uv run --project "$ProjectDir" python -c $pySplit | Select-Object -Last 1) } catch { $nTrain = 0 }

# Adim plani: hesabi burada TEKRARLAMA -- kanonik kaynak plan_iterations
# (app/training/detached_launch.py). Plan = min(train_satiri, tavan) x profil epochs.
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
# Kurtarmada agirlik BAYRAKLA geri verilir (bosluk icermez: "math=0.3,statistics=0.2,...").
if ($MixWeights -and $MixWeights.Trim() -ne "") { $trainArgs += @("--mix-weights", $MixWeights.Trim()) }
elseif ($MixProfile -and $MixProfile.Trim() -ne "") { $trainArgs += @("--mix-profile", $MixProfile.Trim()) }
# Kademe-2 A5: kanonik kaynagin (lora_sft.jsonl) icerik hash'i durum dosyasina yazilir; veri
# kaymasi train.jsonl mtime'i yerine bununla olculur (kurtarma train.jsonl'i yeniden yazar).
$srcData = Join-Path $ProjectDir "data\lora_sft\lora_sft.jsonl"
$dataSha = ""
if (Test-Path $srcData) {
    try { $dataSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $srcData).Hash.ToLowerInvariant() } catch { $dataSha = "" }
}
# Kademe-2 A3: started_at (UTC ISO 8601, detached_launch ile ayni bicim). Yoksa train_guard
# onay/veri-kaymasi kontrollerini atlar ve kurtarma HER ZAMAN reddedilir.
$startedAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffff", [Globalization.CultureInfo]::InvariantCulture) + "+00:00"
# Kademe-2 N1 (2026-09-28): nobetci KURTARMASINDA ilk kosunun started_at'i KORUNUR ve deneme
# sayaci artar. Eskiden her kurtarma started_at'i yeniliyordu -> 72 saat bayatlik siniri hic
# devreye girmiyor, yukleme-sonrasi dusen kosu SONSUZA dek (her turda 61 GB) diriltiliyordu.
# train_guard.recovery_allowed recovery_attempts >= 3 ise reddeder.
$recoveryAttempts = 0
if ($Supervised -and (Test-Path $StatusFile)) {
    try {
        $prev = Get-Content $StatusFile -Raw | ConvertFrom-Json
        if ($prev.adapter -eq $Adapter) {
            $pp = $prev.PSObject.Properties.Name
            if ($pp -contains "started_at" -and "$($prev.started_at)".Trim() -ne "") { $startedAt = [string]$prev.started_at }
            if ($pp -contains "recovery_attempts") { $recoveryAttempts = [int]$prev.recovery_attempts }
        }
    } catch { }
    $recoveryAttempts += 1
}
# Alt surece ORTAMLA gecenler yalniz Start-Process ANINDA ayarlanir, hemen geri alinir.
# $env: surec-geneldir: kalici kalirsa ayni kabukta sonradan elle calistirilan
# `hektor train --run` taze onay kapisini ATLAR ve eski -BaseModel ile egitir (Kademe-2 av).
$prevSupervised = $env:HEKTOR_TRAIN_SUPERVISED
$prevRecovery   = $env:HEKTOR_TRAIN_RECOVERY
$prevBaseModel  = $env:HEKTOR_PEFT_BASE_MODEL
$proc = $null
try {
    # Varsayilan: SUPERVISED GECILMEZ -> alt surecteki `train --run` taze onayi KENDISI
    # tuketir (yoksa pending istek acip 3 ile cikar, egitim baslamaz). Yalniz -Supervised
    # ile atlanir (nobetci kurtarmasi).
    if ($Supervised) { $env:HEKTOR_TRAIN_SUPERVISED = "1" }
    else { Remove-Item Env:HEKTOR_TRAIN_SUPERVISED -ErrorAction SilentlyContinue }
    # RECOVERY: alt surec agirligi YALNIZ bayraktan alir, bekleyen karari tuketmez.
    if ($Supervised) { $env:HEKTOR_TRAIN_RECOVERY = "1" }
    else { Remove-Item Env:HEKTOR_TRAIN_RECOVERY -ErrorAction SilentlyContinue }
    if ($BaseModel -and $BaseModel.Trim() -ne "") { $env:HEKTOR_PEFT_BASE_MODEL = $BaseModel }
    # Start-Process -Redirect* dosyalari KESER: kurtarma/yeniden baslatmada onceki kosunun
    # cokme izi (traceback) silinirdi (Kademe 2 F2-3; v13'te kesinti nedeni bu yuzden
    # bilinmiyor). Dolu eski loglar zaman damgasiyla arsivlenir.
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($lf in @($LogOut, $LogErr)) {
        if ((Test-Path $lf) -and ((Get-Item $lf).Length -gt 0)) {
            $arch = [System.IO.Path]::ChangeExtension($lf, "$stamp.log")
            Move-Item -LiteralPath $lf -Destination $arch -Force
            Write-Host "Onceki log arsivlendi: $arch"
        }
    }
    $proc = Start-Process -FilePath $uv `
        -ArgumentList $trainArgs `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $LogOut `
        -RedirectStandardError $LogErr `
        -WindowStyle Hidden `
        -PassThru
    # Handle'i hemen onbellege al: aksi halde surec bittikten sonra ExitCode bos donebilir.
    if ($proc) { $null = $proc.Handle }
} finally {
    $env:HEKTOR_TRAIN_SUPERVISED = $prevSupervised
    $env:HEKTOR_TRAIN_RECOVERY   = $prevRecovery
    $env:HEKTOR_PEFT_BASE_MODEL  = $prevBaseModel
}
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
    # pid: web "durdur" (request_stop_detached_training) sureci agaciyla oldurebilsin;
    # pid yoksa yalniz STOP_TRAINING dosyasi birakiyor ve egitim suruyordu.
    pid          = $(if ($proc) { [int]$proc.Id } else { 0 })
    # K8-b: bu kosuyu yetkilendiren TUKETILMIS onayin kimligi -- kurtarmada (-Supervised)
    # cagiran (nobetci) bunu ONCEDEN train-recovery-check ile dogrulayip gecirir; taze
    # baslatmada burada henuz BILINMEZ (alt surec kendi onayini asagida tuketir) --
    # basari sonrasi log'dan okunup asagida bu alana ISLENIR (zaman penceresi DEGIL,
    # gercek kimlik; bkz. app/training/train_guard.py: find_run_approval).
    approval_id  = $ApprovalId
    started_at   = $startedAt
    recovery_attempts = $recoveryAttempts
    # Kurtarmada param'dan; taze baslatmada asagida log'daki MIX_DECISION satirindan islenir.
    mix_weights  = $MixWeights
    mix_profile  = $MixProfile
    mix_decision_id = ""
    data_sha256  = $dataSha
} | ConvertTo-Json -Compress) |
    Out-File -FilePath $StatusFile -Encoding ascii -Force
# KURAL 8 KAPISI (fail-closed): -Supervised YOKSA onayi alt surec tuketir. Onay yoksa
# `train --run` pending istek acip 3 ile cikar -- ama Start-Process ile baslatildigi icin
# betik bunu GORMEZSE "egitim basladi" der ve durum dosyasi kalir (nobetci de olu kosuyu
# diriltmeye calisir). Bu yuzden kisa sure bekleyip erken cikisi yakala.
# Kademe-2 A2: -Supervised (nobetci kurtarmasi) icin de ayni kontrol -- kurtarma kosusu
# agirlik/sizinti/yuk kapisinda dusse bile eskiden "baslatildi" deniyordu.
if ($proc) {
    if ($proc.WaitForExit(60000)) {
        $code = $proc.ExitCode
        Remove-Item $StatusFile -Force -ErrorAction SilentlyContinue
        $aprId = (Select-String -Path $LogOut, $LogErr -Pattern 'apr_[0-9a-f]{8,}' -ErrorAction SilentlyContinue |
                  Select-Object -Last 1).Matches.Value
        Write-Host "  [ENGEL] Egitim BASLAMADI (cikis kodu $code)." -ForegroundColor Red
        if ($Supervised) {
            Write-Host "          (nobetci kurtarmasi -Supervised; durum kaydi silindi, yeniden dirilme yok)" -ForegroundColor Yellow
        }
        if ($code -eq 3 -and $aprId) {
            Write-Host "          Taze insan onayi gerekiyor (Kural 8). Onay istegi: $aprId" -ForegroundColor Yellow
            Write-Host "          Onayla:  uv run --no-sync hektor approval-approve $aprId" -ForegroundColor Cyan
            Write-Host "          Sonra bu betigi TEKRAR calistir (onay tek kullanimliktir)." -ForegroundColor Cyan
        } elseif ($code -eq 4) {
            # train-load-doctor NO-GO paneli STDOUT'a yazilir (LogErr'e degil).
            Write-Host "          train-load-doctor NO-GO (rakip GPU/LLM yuku). Onay TUKETILMEDI." -ForegroundColor Yellow
            Write-Host "          Ayrinti (NO-GO paneli): $LogOut" -ForegroundColor Cyan
            Write-Host "          Elle bak:  uv run --no-sync hektor train-load-doctor" -ForegroundColor Cyan
        } elseif ($code -eq 5) {
            Write-Host "          Karisim agirligi sorulmadi (her egitimden once zorunlu)." -ForegroundColor Yellow
            Write-Host "          Once:  uv run --no-sync hektor mix weights" -ForegroundColor Cyan
            Write-Host "          Sonra bu betigi TEKRAR calistir (karar tek kullanimliktir)." -ForegroundColor Cyan
        } elseif ($code -eq 6) {
            Write-Host "          Egitim verisinde eval (golden/validation) sizintisi var." -ForegroundColor Yellow
            Write-Host "          Ayrinti:  uv run --no-sync hektor mix leakage" -ForegroundColor Cyan
        } elseif ($code -eq 8) {
            Write-Host "          Ortak agir is kilidi tutuluyor ya da sohbet cevabi uretiliyor." -ForegroundColor Yellow
            Write-Host "          Durum:  uv run --no-sync python -m app.training.resource_lock status" -ForegroundColor Cyan
        } elseif ($code -eq 10) {
            Write-Host "          Kademe 2 kaydi yok/gecersiz (her egitimden once zorunlu). Onay TUKETILMEDI." -ForegroundColor Yellow
            Write-Host "          Derin avdan sonra:  uv run --no-sync hektor kademe2-kayit --findings f.json --data-sha <sha> --evidence ..." -ForegroundColor Cyan
            Write-Host "          Ayrinti: $LogOut" -ForegroundColor Gray
        } else {
            Write-Host "          Ayrinti: $LogErr" -ForegroundColor Gray
        }
        exit 3
    }
    # 60 sn icinde cikmadi -> egitim GERCEKTEN basladi; alt surecteki `train --run` kendi
    # taze onayini TUKETTI. K8-b: o kimligi log'dan al ve durum dosyasina ISLE -- bir
    # sonraki teshis/kurtarma (train-recovery-check) bunu ZAMAN PENCERESI ile TAHMIN
    # ETMEK yerine dogrudan kullanabilsin.
    $consumedId = ""
    if (-not $Supervised) {
        $consumedId = (Select-String -Path $LogOut, $LogErr -Pattern 'apr_[0-9a-f]{8,}' -ErrorAction SilentlyContinue |
                       Select-Object -Last 1).Matches.Value
    }
    # Kademe-2 A2: alt surecin TUKETTIGI agirlik karari (main.py `MIX_DECISION` satiri) ->
    # durum dosyasina; nobetci kurtarmada bunu -MixWeights ile geri verir.
    $mix = Select-String -Path $LogOut, $LogErr -Pattern 'MIX_DECISION id=(\S+) profile=(\S+) weights=(\S+)' -ErrorAction SilentlyContinue |
           Select-Object -Last 1
    if (($consumedId -or $mix) -and (Test-Path $StatusFile)) {
        try {
            $cur = Get-Content $StatusFile -Raw | ConvertFrom-Json
            if ($consumedId) { $cur | Add-Member -NotePropertyName approval_id -NotePropertyValue $consumedId -Force }
            if ($mix) {
                $g = $mix.Matches[0].Groups
                $cur | Add-Member -NotePropertyName mix_decision_id -NotePropertyValue $g[1].Value -Force
                $cur | Add-Member -NotePropertyName mix_profile -NotePropertyValue $g[2].Value -Force
                $cur | Add-Member -NotePropertyName mix_weights -NotePropertyValue $g[3].Value -Force
            }
            ($cur | ConvertTo-Json -Compress) | Out-File -FilePath $StatusFile -Encoding ascii -Force
        } catch {
            Write-Host "  [UYARI] approval_id/agirlik durum dosyasina islenemedi (kurtarma eksik kalir)." -ForegroundColor Yellow
        }
    }
    if (-not $mix -and -not $MixWeights -and -not $MixProfile) {
        Write-Host "  [UYARI] MIX_DECISION satiri log'da bulunamadi -- agirlik durum dosyasina yazilmadi; nobetci bu kosuyu diriltemez." -ForegroundColor Yellow
    }
}
$profLabel = if ($Profile -and $Profile.Trim() -ne "") { $Profile } else { "(vanilya)" }
$resumeLabel = if ($Resume) { "devam(checkpoint)" } else { "sifirdan" }
$modelLabel = if ($BaseModel) { $BaseModel } else { "ayardaki varsayilan" }
Write-Host "  Temel model: $modelLabel" -ForegroundColor DarkGray
Write-Host "  [OK] Egitim DETACHED baslatildi (dtype=$Dtype, adapter=$Adapter, iters=$Iterations, profil=$profLabel, mod=$resumeLabel)." -ForegroundColor Green
Write-Host "       Kalite kapisi: $gateSummary" -ForegroundColor DarkGray
Write-Host "       Claude Code/terminal kapansa da surer. PC acik + oturum acik kalmali." -ForegroundColor Cyan
Write-Host "       Ilerleme: logs\train-full-err.log" -ForegroundColor Gray
