param(
    [int]$Target = 1300,
    [int]$Batch = 3,
    [string]$Output = "data/lora_sft",
    [int]$StartSeed = 100
)

# Tasinabilir kok cozumu (hardcoded yol YOK): script scripts/ altinda → kok bir ust.
$ScriptDir   = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectRoot = Split-Path -Parent $ScriptDir

# uv'yi bul (start-server.ps1 / verify-install.ps1 ile ayni desen): PATH'te yoksa bilinen konumlar.
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
    foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}
$UvExe = Find-Uv
if (-not $UvExe) {
    Write-Host "[HATA] uv bulunamadi. Kur: winget install astral-sh.uv" -ForegroundColor Red
    exit 2
}
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Get-CurrentCount {
    # PS 5.1 Join-Path yalnız 2 yol argümanı alır (3. pozisyonel → hata); iç içe çağır.
    $jsonlPath = Join-Path (Join-Path $ProjectRoot $Output) "synthetic_qa.jsonl"
    if (-not (Test-Path $jsonlPath)) { return 0 }
    return (Get-Content $jsonlPath -Encoding utf8 | Where-Object { $_ -ne "" }).Count
}

$seeds = @(100, 200, 300, 400) | Where-Object { $_ -ge $StartSeed }
foreach ($seed in $seeds) {
    $count = Get-CurrentCount
    Write-Host "$(Get-Date -Format 'HH:mm:ss') — Mevcut: $count örnek"
    if ($count -ge $Target) {
        Write-Host "Hedef $Target'a ulaşıldı — çıkılıyor."
        break
    }
    $logFile = Join-Path $LogDir "synth_qa_seed${seed}.log"
    Write-Host "$(Get-Date -Format 'HH:mm:ss') — seed=$seed başlatılıyor (hedef=$Target)…"
    $proc = Start-Process -FilePath $UvExe `
        -ArgumentList "run","achilles","synth-qa-bulk","--seed","$seed","--target","$Target","--output",$Output,"--batch","$Batch" `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -PassThru -NoNewWindow
    Write-Host "PID: $($proc.Id)"
    $proc.WaitForExit()
    $count = Get-CurrentCount
    Write-Host "$(Get-Date -Format 'HH:mm:ss') — seed=$seed bitti → $count örnek"
}

$finalCount = Get-CurrentCount
Write-Host "$(Get-Date -Format 'HH:mm:ss') — TAMAMLANDI: $finalCount örnek"
if ($finalCount -ge $Target) {
    Write-Host "SONRAKI ADIM: uv run achilles lora-split — ardından Kaggle 'Run All' tıkla"
}
