# Hektor Web Server -- Windows kalici arka plan servisi
# Kullanim:
#   .\scripts\start-server.ps1            -- simdi baslat (zaten calisiyorsa DOKUNMAZ)
#   .\scripts\start-server.ps1 -Restart   -- durdur + tekrar baslat (git pull SONRASI bunu kullan)
#   .\scripts\start-server.ps1 -Install   -- Windows acilisina ekle + simdi baslat
#   .\scripts\start-server.ps1 -Repair    -- autostart/03:00 gorev yollarini BU repoya yeniden bagla
#   .\scripts\start-server.ps1 -Uninstall -- Windows acilisindan kaldir + durdur
#   .\scripts\start-server.ps1 -Stop      -- durdur
#   .\scripts\start-server.ps1 -Status    -- durum goster (gorev yolu bu repoyu mu isaret ediyor?)
#
# NOT: Yeni backend rotalari (orn. /api/agents, AGENTS sekmesi) yalniz proses
# ACILISINDA yuklenir. `git pull` ettikten sonra duz `start-server.ps1` ESKI
# prosesi calisiyor birakir -> tarayicida yeni bolumler GORUNMEZ. Cekilen kodu
# uygulamak icin: `.\scripts\start-server.ps1 -Restart` ya da `.\update.ps1`,
# ardindan tarayicida Ctrl+Shift+R (sert yenileme).

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status,
    [switch]$Stop,
    [switch]$Restart,     # durdur + tekrar baslat (git pull sonrasi yeni kodu uygula)
    [switch]$Repair,      # autostart/guncelleme gorevlerinin yolunu BU repoya yeniden bagla
    [switch]$SkipVerify   # -Install'da cevrimdisi dogrulama kapisini atla (acil kacis)
)

$ErrorActionPreference = "Continue"

$ScriptDir  = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectDir = Split-Path -Parent $ScriptDir
$LogDir     = Join-Path $ProjectDir "logs"
$LogOut     = Join-Path $LogDir "hektor-web.log"
$LogErr     = Join-Path $LogDir "hektor-web-err.log"
$PidFile    = Join-Path $ProjectDir ".web.pid"
$VbsFile    = Join-Path $ScriptDir "hektor-autostart.vbs"
$RegPath    = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$RegKey     = "HektorWeb"
$WebExe     = Join-Path $ProjectDir ".venv\Scripts\hektor-web.exe"
$WebTaskScript = Join-Path $ScriptDir "run-web-service.ps1"

# ---------------------------------------------------------------- uv bul
function Find-Uv {
    $fromPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if ($fromPath -and (Test-Path $fromPath)) { return $fromPath }

    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"),
        (Join-Path $env:APPDATA "uv\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\uv.exe"),
        "C:\uv\uv.exe"
    )
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

$UvPath = Find-Uv
if (-not $UvPath) {
    Write-Host "  [HATA] uv bulunamadi." -ForegroundColor Red
    Write-Host "  Kur: winget install astral-sh.uv" -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------- yardimci
function Get-WebPid {
    if (Test-Path $PidFile) {
        $stored = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($stored -match '^\d+$') {
            $proc = Get-Process -Id ([int]$stored) -ErrorAction SilentlyContinue
            if ($proc) { return [int]$stored }
        }
    }
    return $null
}

function Start-OllamaIfNeeded {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" `
            -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        return
    } catch {}

    $candidates = @(
        (Get-Command ollama -ErrorAction SilentlyContinue).Source,
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:LOCALAPPDATA "Ollama\ollama.exe")
    )
    foreach ($exe in $candidates) {
        if ($exe -and (Test-Path $exe)) {
            Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 4
            Write-Host "  [OK] Ollama baslatildi" -ForegroundColor Green
            return
        }
    }
    Write-Host "  [!] Ollama bulunamadi -- RAG/LLM calismaaz" -ForegroundColor Yellow
}

function Start-HektorServer {
    $running = Get-WebPid
    if ($running) {
        Write-Host "  [OK] Zaten calisiyor (PID $running) -- http://127.0.0.1:8765" -ForegroundColor Green
        Write-Host "       (git pull ettiyseniz yeni kodu uygulamak icin: start-server.ps1 -Restart)" -ForegroundColor Gray
        return
    }
    $null = New-Item -ItemType Directory -Path $LogDir -Force

    # KRITIK: uv her `uv run` cagrisinda paketi yeniden senkronlamaya calisir; bu da
    # calisan (veya yari-olu) bir sunucunun kilitledigi hektor-web.exe'yi silmeye
    # ugrasip "os error 32" ile patlar -> SUNUCU HIC BASLAMAZ. Senkronu kapat;
    # bagimliliklar zaten kurulu, sunucu yalniz mevcut ortami kullanir.
    # (Ayni cozum scripts/continuous-learning.sh icinde de uygulanmis durumda.)
    $env:UV_NO_SYNC = "1"
    # Yalniz taze kurulum (.venv yok) ise bir defaliga senkronla — aksi halde
    # --no-sync ile hektor-web giris noktasi bulunamaz.
    $venvDir = Join-Path $ProjectDir ".venv"
    if (-not (Test-Path $venvDir)) {
        Write-Host "  [..] Ilk kurulum: uv sync" -ForegroundColor Gray
        & $UvPath sync --project "$ProjectDir" | Out-Null
    }

    $proc = $null
    try {
        $proc = Start-Process `
            -FilePath $UvPath `
            -ArgumentList "run", "--no-sync", "--project", "`"$ProjectDir`"", "hektor-web" `
            -WorkingDirectory $ProjectDir `
            -RedirectStandardOutput $LogOut `
            -RedirectStandardError  $LogErr `
            -WindowStyle Hidden `
            -PassThru
    } catch {
        Write-Host "  [HATA] Sunucu baslatilamadi: $_" -ForegroundColor Red
        return
    }
    if (-not $proc) {
        Write-Host "  [HATA] Sunucu baslatilamadi (proses olusmadi)." -ForegroundColor Red
        return
    }
    $proc.Id | Out-File $PidFile -Force -Encoding ascii
    Start-Sleep -Seconds 5
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:8765/api/status" `
            -TimeoutSec 8 -UseBasicParsing -ErrorAction Stop
        Write-Host "  [OK] Hektor Web calisiyor -- http://127.0.0.1:8765" -ForegroundColor Green
    } catch {
        Write-Host "  [!] Basliyor... log: $LogErr" -ForegroundColor Yellow
    }
}

function Stop-HektorServer {
    $webPid = Get-WebPid
    if ($webPid) {
        # PID, `uv run` sarmalayicisina aittir; gercek sunucu onun ALT prosesi
        # olan python+uvicorn'dur. Yalniz uv'yi oldurmek (Stop-Process) gercek
        # sunucuyu birakir -> port 8765 tutulmaya devam eder, yeniden baslatma
        # bozulur. Bu yuzden tum prosess agacini oldur (/T).
        & taskkill /PID $webPid /T /F 2>$null | Out-Null
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Host "  [OK] Durduruldu (PID $webPid + alt prosesler)" -ForegroundColor Yellow
        return
    }
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -like "*app.web.server*" -or
            $_.CommandLine -like "*hektor-web*"
        }
    if ($procs) {
        $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Host "  [OK] Durduruldu." -ForegroundColor Yellow
    } else {
        Write-Host "  Zaten calismiyor." -ForegroundColor Gray
    }
}

# ---------------------------------------------------------------- self-heal yardimcilari
# Bir scheduled task'in action'indan gomulu .vbs/.ps1 hedef yolunu cikar (salt-okuma)
function Get-EmbeddedTaskPath {
    param([string]$TaskName)
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) { return $null }
    $arg = ($t.Actions | Select-Object -First 1).Arguments
    if ($arg -and ($arg -match '"([^"]+\.(?:vbs|ps1))"')) { return $matches[1] }
    return ($t.Actions | Select-Object -First 1).WorkingDirectory
}

# Gomulu yol bu repoyu mu isaret ediyor? (normalize edilmis, case-insensitive tam yol)
function Test-PathMatchesRepo {
    param([string]$Embedded, [string]$Expected)
    if (-not $Embedded -or -not $Expected) { return $false }
    try {
        $e = [System.IO.Path]::GetFullPath($Embedded).TrimEnd('\')
        $x = [System.IO.Path]::GetFullPath($Expected).TrimEnd('\')
        return ($e -ieq $x)
    } catch { return $false }
}

# Status ekraninda kayitli hedefi ve checkout ile uyumunu tek satirda goster.
function Format-PathMatch {
    param([string]$Embedded, [string]$Expected)
    if (-not $Embedded) { return "kayitli yol yok [UYUMSUZ]" }
    $match = if (Test-PathMatchesRepo $Embedded $Expected) { "[OK]" } else { "[UYUMSUZ]" }
    return "$Embedded $match"
}

# VBS + Registry Run + HektorWeb/HektorUpdate gorevlerini MEVCUT checkout'a yaz
# (idempotent kendini-onarma). Cagrildigi $ProjectDir/$ScriptDir'e gomer. git'e DOKUNMAZ.
# Yukseltilmemis (non-admin) oturumda Register-ScheduledTask basarisiz olabilir; sessiz
# "[OK]" yerine GERCEK sonucu raporlamak icin $script:AutostartOk izlenir.
function Sync-Autostart {
    $script:AutostartOk = $true

    # VBS olustur: uv tam yolunu degil, start-server.ps1'i (PATH-bagimsiz) cagirir
    $thisScript = Join-Path $ScriptDir "start-server.ps1"
    $vbsContent = @"
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -NonInteractive -File """ & "$($thisScript -replace '"','\"')" & """", 0, False
Set sh = Nothing
"@
    $vbsContent | Out-File -FilePath $VbsFile -Encoding ascii -Force

    # Registry Run anahtari (her login'de otomatik calisir)
    Set-ItemProperty -Path $RegPath -Name $RegKey -Value "wscript.exe `"$VbsFile`"" -Force
    Write-Host "  [OK] Windows acilisina eklendi (Registry Run)" -ForegroundColor Green
    Write-Host "       $VbsFile" -ForegroundColor Gray

    # Task Scheduler web servisinin GERCEK sahibi olmali. WScript yalnizca
    # Start-Process yapip hemen cikarsa gorev Ready durumuna doner ve Windows
    # onun alt prosesini temizleyebilir. uv'yi dogrudan action yapmak gorevi
    # sunucu yasadigi surece Running tutar.
    $taskExe = "powershell.exe"
    $taskArgs = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$WebTaskScript`""
    $action = New-ScheduledTaskAction -Execute $taskExe -Argument $taskArgs -WorkingDirectory $ProjectDir
    $trigger  = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -StartWhenAvailable `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 2)
    try {
        Register-ScheduledTask -TaskName "HektorWeb" -Action $action `
            -Trigger $trigger -Settings $settings -RunLevel Highest -Force -ErrorAction Stop | Out-Null
        Write-Host "  [OK] Gorev Zamanlayici yedegi eklendi" -ForegroundColor Green
    } catch {
        $script:AutostartOk = $false
        Write-Host "  [!] HektorWeb gorevi KAYDEDILEMEDI (Yonetici PowerShell gerekebilir)." -ForegroundColor Yellow
    }

    # Eğitim web'den bağımsızdır. Watchdog yalnız PID yoksa train_status.json'daki
    # reçeteyi yeniden başlatır; PEFT son checkpoint'i otomatik bulup sürdürür.
    $watchdogScript = Join-Path $ScriptDir "training-watchdog.ps1"
    $watchdogAction = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$watchdogScript`"" `
        -WorkingDirectory $ProjectDir
    $watchdogTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 5)
    try {
        Register-ScheduledTask -TaskName "HektorTrainingWatchdog" -Action $watchdogAction `
            -Trigger $watchdogTrigger -Settings $settings -Force -ErrorAction Stop | Out-Null
        Write-Host "  [OK] Egitim watchdog eklendi (5 dakikada bir)" -ForegroundColor Green
    } catch {
        $script:AutostartOk = $false
        Write-Host "  [!] Egitim watchdog KAYDEDILEMEDI." -ForegroundColor Yellow
    }

    # Gunluk otomatik guncelleme gorevi (her gun 03:00) -- BU repodaki update.ps1
    $updateScript = Join-Path $ProjectDir "update.ps1"
    $updateAction = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -NonInteractive -File `"$updateScript`"" `
        -WorkingDirectory $ProjectDir
    $updateTrigger  = New-ScheduledTaskTrigger -Daily -At "03:00"
    $updateSettings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
        -StartWhenAvailable
    try {
        Register-ScheduledTask -TaskName "HektorUpdate" -Action $updateAction `
            -Trigger $updateTrigger -Settings $updateSettings -RunLevel Highest -Force -ErrorAction Stop | Out-Null
        Write-Host "  [OK] Gunluk otomatik guncelleme eklendi (her gun 03:00)" -ForegroundColor Green
    } catch {
        $script:AutostartOk = $false
        Write-Host "  [!] HektorUpdate gorevi KAYDEDILEMEDI (Yonetici PowerShell gerekebilir)." -ForegroundColor Yellow
    }
}

# Kayitli gorev/Registry yolu bu repodan farkliysa (veya yoksa) yeniden gom. git'e DOKUNMAZ.
function Repair-Autostart {
    $webEmb = Get-EmbeddedTaskPath -TaskName "HektorWeb"
    $webTask = Get-ScheduledTask -TaskName "HektorWeb" -ErrorAction SilentlyContinue
    $watchdogTask = Get-ScheduledTask -TaskName "HektorTrainingWatchdog" -ErrorAction SilentlyContinue
    $expectedWebExe = "powershell.exe"
    $updEmb = Get-EmbeddedTaskPath -TaskName "HektorUpdate"
    $regVal = (Get-ItemProperty -Path $RegPath -Name $RegKey -ErrorAction SilentlyContinue).$RegKey
    $needs = $false
    if (-not (Test-PathMatchesRepo $webEmb $ProjectDir)) { $needs = $true }
    if (-not $webTask -or $webTask.Actions[0].Execute -ine $expectedWebExe) { $needs = $true }
    if ($webTask -and $webTask.Actions[0].Arguments -notlike "*$WebTaskScript*") { $needs = $true }
    if (-not $watchdogTask) { $needs = $true }
    if (-not (Test-PathMatchesRepo $updEmb (Join-Path $ProjectDir 'update.ps1'))) { $needs = $true }
    if (-not $regVal -or ($regVal -notlike "*$VbsFile*")) { $needs = $true }
    if (-not $needs) {
        Write-Host "  [OK] Gorevler ve Registry zaten bu repoya isaret ediyor -- onarim gereksiz" -ForegroundColor Green
        return
    }
    Write-Host "  [..] Gorev/Registry yollari bu repoyla uyumsuz -- yeniden gomuluyor" -ForegroundColor Yellow
    Sync-Autostart
    if ($script:AutostartOk) {
        Write-Host "  [OK] Onarildi -- gorevler $ProjectDir konumuna isaret ediyor" -ForegroundColor Green
        Write-Host "       (dogrulamak icin: .\scripts\start-server.ps1 -Status)" -ForegroundColor Gray
    } else {
        Write-Host "  [!] KISMEN onarildi -- bazi gorevler kaydedilemedi (Yonetici olarak tekrar deneyin)." -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- kurulum
# Neden Registry + VBS?
# Task Scheduler PATH'i olmadan calisir -- uv bulunamaz.
# Registry Run anahtari WScript ile calisir, uv tam yolu VBS icinde gomerek
# PATH'e bagimliligi tamamen ortadan kaldirir.
function Install-Autostart {
    # DOGRULAMA KAPISI: autostart kurmadan ONCE cevrimdisi duman testi calistir.
    # "Test edilmeden calisiyor deme" (CLAUDE.md Kural 2): kanitlanmamis kurulumu
    # Windows acilisina bagdamak, sessizce bozuk bir sistemi her loginde ayaga
    # kaldirmaya calisir. Dogrulama kalirsa kurulumu DURDUR (-SkipVerify ile atlanir).
    if (-not $SkipVerify) {
        $verifyScript = Join-Path $ScriptDir "verify-install.ps1"
        if (Test-Path $verifyScript) {
            Write-Host "  [..] Kurulum dogrulamasi calisiyor (verify-install.ps1)..." -ForegroundColor Gray
            & powershell.exe -ExecutionPolicy Bypass -NonInteractive -File "$verifyScript"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [HATA] Dogrulama KALDI (cikis $LASTEXITCODE) -- autostart KURULMADI." -ForegroundColor Red
                Write-Host "         Duzeltip tekrar deneyin, ya da: start-server.ps1 -Install -SkipVerify" -ForegroundColor Yellow
                exit 1
            }
            Write-Host "  [OK] Dogrulama GECTI -- autostart kuruluyor" -ForegroundColor Green
        } else {
            Write-Host "  [!] verify-install.ps1 yok -- dogrulama atlandi" -ForegroundColor Yellow
        }
    }

    # VBS + Registry Run + HektorWeb/HektorUpdate gorevlerini BU checkout'a yaz
    # (tek dogru kaynak; -Install her cagrildiginda mevcut $ProjectDir'e yeniden gomer).
    Sync-Autostart

    # Hemen baslat
    Start-OllamaIfNeeded
    Start-HektorServer

    Write-Host ""
    Write-Host "  Artik PowerShell'i kapatabilirsiniz." -ForegroundColor Cyan
    Write-Host "  Web arayuz her Windows acilisinda otomatik baslar." -ForegroundColor Cyan
    Write-Host "  Guncelleme her gece 03:00'de otomatik yapilir." -ForegroundColor Cyan
}

function Uninstall-Autostart {
    Stop-HektorServer
    Remove-ItemProperty -Path $RegPath -Name $RegKey -ErrorAction SilentlyContinue
    Remove-Item $VbsFile -Force -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "HektorWeb"    -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "HektorUpdate" -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "HektorTrainingWatchdog" -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "  [OK] Otomatik baslatma ve guncelleme kaldirildi." -ForegroundColor Yellow
}

function Show-Status {
    $webPid = Get-WebPid
    if ($webPid) {
        Write-Host "  Proses     : CALISIYOR (PID $webPid)" -ForegroundColor Green
        Write-Host "  Adres      : http://127.0.0.1:8765" -ForegroundColor Cyan
    } else {
        Write-Host "  Proses     : CALISMYOR" -ForegroundColor Red
    }
    $reg = Get-ItemProperty -Path $RegPath -Name $RegKey -ErrorAction SilentlyContinue
    Write-Host "  Registry   : $(if ($reg) { 'kayitli' } else { 'kayitli degil' })" -ForegroundColor Gray
    $task = Get-ScheduledTask -TaskName "HektorWeb" -ErrorAction SilentlyContinue
    Write-Host "  Zamanlayici: $(if ($task) { $task.State } else { 'kayitli degil' })" -ForegroundColor Gray
    $upd = Get-ScheduledTask -TaskName "HektorUpdate" -ErrorAction SilentlyContinue
    Write-Host "  Guncelleme : $(if ($upd) { 'kayitli (her gece 03:00)' } else { 'kayitli degil' })" -ForegroundColor Gray
    # Gomulu gorev yollari BU repoyu mu isaret ediyor? (olu/yabanci yol tespiti)
    Write-Host "  Web yolu   : $(Format-PathMatch (Get-EmbeddedTaskPath 'HektorWeb') $ProjectDir)" -ForegroundColor Gray
    Write-Host "  Upd yolu   : $(Format-PathMatch (Get-EmbeddedTaskPath 'HektorUpdate') (Join-Path $ProjectDir 'update.ps1'))" -ForegroundColor Gray
    Write-Host "  Bu repo    : $ProjectDir" -ForegroundColor Gray
    Write-Host "  Log        : $LogOut" -ForegroundColor Gray
    Write-Host "  uv yolu    : $UvPath" -ForegroundColor Gray
}

# ---------------------------------------------------------------- ana
if ($Install)   { Install-Autostart;   exit 0 }
if ($Repair)    { Repair-Autostart;    exit 0 }
if ($Uninstall) { Uninstall-Autostart; exit 0 }
if ($Stop)      { Stop-HektorServer; exit 0 }
if ($Status)    { Show-Status;         exit 0 }
if ($Restart)   {
    # git pull SONRASI: eski prosesi kesin durdur (yeni rotalar yalniz acilista
    # yuklenir), portun serbest kalmasini bekle, sonra taze baslat.
    Stop-HektorServer
    Start-Sleep -Seconds 1
    Start-OllamaIfNeeded
    Start-HektorServer
    Write-Host "  >> Tarayicida son halini gormek icin: Ctrl+Shift+R (sert yenileme)" -ForegroundColor Cyan
    exit 0
}

Start-OllamaIfNeeded
Start-HektorServer
