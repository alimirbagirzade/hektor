# Achilles Trader AI -- Windows cevrimdisi kurulum dogrulama kapisi.
#
# Amac: autostart KURULMADAN ONCE "sistem gercekten ayaga kalkiyor mu" kanitla.
# bootstrap.sh'in Windows karsiligi: Ollama GEREKTIRMEZ -- fake embedding +
# sentetik veri ile uctan uca duman testi (init -> status -> gen-data ->
# backtest -> offline pytest).
#
# Tasinabilir: $PSScriptRoot tabanli, hardcoded yol YOK. Idempotent.
# Cikis kodu sozlesmesi (cagiran buna gore kapi uygular):
#   0  = GECTI  (autostart kurulabilir)
#   1  = KALDI  (bir adim basarisiz -- autostart KURULMAMALI)
#   2  = ORTAM  (uv bulunamadi -- on kosul eksik)
#
# Kullanim:
#   .\scripts\verify-install.ps1            -- tam dogrulama (gerekirse uv sync)
#   .\scripts\verify-install.ps1 -SkipSync  -- bagimlilik senkronunu atla (cevrimdisi/CI)

param(
    [switch]$SkipSync
)

$ScriptDir  = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectDir = Split-Path -Parent $ScriptDir

# ---------------------------------------------------------------- uv bul
# start-server.ps1 ile AYNI desen: PATH'te yoksa bilinen konumlari dene.
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
    Write-Host "[HATA] uv bulunamadi. Kur: winget install astral-sh.uv" -ForegroundColor Red
    exit 2
}

# ---------------------------------------------------------------- cevrimdisi mod
# Fake embedding: Ollama/ag olmadan RAG/embedding kod yollari calisabilsin.
$env:ACHILLES_ALLOW_FAKE_EMBEDDINGS = "true"

# CWD'yi proje koküne sabitle (backtest goreli yol 'data/market/raw/synthetic.csv' icin sart).
Set-Location -Path $ProjectDir

Write-Host "Achilles cevrimdisi kurulum dogrulamasi" -ForegroundColor White
Write-Host "  Proje : $ProjectDir" -ForegroundColor Gray
Write-Host "  uv    : $UvPath" -ForegroundColor Gray

# ---------------------------------------------------------------- adim koscusu
$script:stepNo  = 0
$script:failed  = $null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$UvArgs
    )
    if ($script:failed) { return }   # onceki adim kaldiysa gerisini atla (fail-fast)
    $script:stepNo++
    Write-Host ""
    Write-Host "==> [$($script:stepNo)] $Name" -ForegroundColor Cyan
    & $UvPath @UvArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [KALDI] $Name (cikis kodu $LASTEXITCODE)" -ForegroundColor Red
        $script:failed = $Name
    } else {
        Write-Host "  [OK] $Name" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- 0) bagimliliklar
# Taze makinede bir defaliga senkronla. Cevrimdisiyse -SkipSync ile atlanir;
# senkron basarisiz olsa bile mevcut ortamla devam edip testte yakalariz.
if (-not $SkipSync) {
    Write-Host ""
    Write-Host "==> [0] Bagimliliklar (uv sync --extra dev)" -ForegroundColor Cyan
    & $UvPath sync --extra dev --project "$ProjectDir"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] uv sync basarisiz (cevrimdisi olabilir) -- mevcut ortamla devam." -ForegroundColor Yellow
    }
}
# Sonraki `uv run` cagrilari yeniden senkron DENEMESIN: calisan bir sunucu
# achilles-web.exe'yi kilitliyse 'os error 32' ile patlar (start-server.ps1 dersi).
$env:UV_NO_SYNC = "1"

# ---------------------------------------------------------------- duman testi zinciri
Invoke-Step "Sistem baslat (init)"       @("run", "--no-sync", "achilles", "init")
Invoke-Step "Durum (status)"             @("run", "--no-sync", "achilles", "status")
Invoke-Step "Sentetik veri (gen-data)"   @("run", "--no-sync", "achilles", "gen-data")
Invoke-Step "Backtest (ornek strateji)"  @("run", "--no-sync", "achilles", "backtest", "data/market/raw/synthetic.csv")
# --basetemp: pytest'in varsayilan global Temp'i (AppData\Local\Temp\pytest-of-*)
# Windows'ta 'WinError 5 erisim engellendi' verebilir; proje-yerel .pytest_tmp
# bu workaround icin .gitignore'da zaten tanimli (commit'e sizmaz).
Invoke-Step "Testler (offline)"          @("run", "--no-sync", "pytest", "-q", "-m", "not ollama and not slow", "--basetemp", ".pytest_tmp")

# ---------------------------------------------------------------- sonuc
Write-Host ""
if ($script:failed) {
    Write-Host "SONUC: KALDI -- '$($script:failed)' adiminda basarisiz." -ForegroundColor Red
    Write-Host "       Autostart KURULMAMALI. Hatayi duzeltip tekrar calistirin." -ForegroundColor Yellow
    exit 1
}
Write-Host "SONUC: GECTI -- $($script:stepNo) adim cevrimdisi dogrulandi. Autostart icin hazir." -ForegroundColor Green
exit 0
