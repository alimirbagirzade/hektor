$ErrorActionPreference = "Continue"
$projectDir = Split-Path -Parent $PSScriptRoot
$statusFile = Join-Path $projectDir "storage\train_status.json"
$startScript = Join-Path $PSScriptRoot "start-train.ps1"
$trainLog = Join-Path $projectDir "logs\train-full-err.log"
$mutex = [Threading.Mutex]::new($false, "Local\AchillesTrainingWatchdog")
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
    if (-not (Test-Path $statusFile)) { exit 0 }
    $status = Get-Content $statusFile -Raw | ConvertFrom-Json
    if ((Test-Path $trainLog) -and ((Get-Date) - (Get-Item $trainLog).LastWriteTime).TotalMinutes -lt 10) { exit 0 }
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*peft_lora_train*' -or $_.CommandLine -like '*train*--run*' }
    if ($running) { exit 0 }
    & $startScript -Adapter $status.adapter -Iterations ([int]$status.iterations) -Dtype $status.dtype -Profile "discipline_safe_local"
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}

