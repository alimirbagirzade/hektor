# Hektor nightly loop -- end-of-day run (ASCII-only, Windows PS 5.1 safe).
#
#   Manual run:  powershell -ExecutionPolicy Bypass -File scripts\nightly.ps1
#   Scheduled:   scripts\install-nightly-task.ps1 registers this daily.
#
# WHAT IT DOES (uv run hektor gece):
#   1. local Ollama judge reads learning candidates; suspicious/unclear -> quarantine
#   2. LLM-30 measurement of the active model (flags + numeric key trace; no score),
#      only when the model digest changed or weekly
#   3. CSV lab: new CSV files in data\market\raw -> indicator/strategy CANDIDATES
#      (selected on the development period, validated out-of-sample once; final untouched)
#   4. training readiness REPORT only
#
# WHAT IT DOES NOT DO: it never starts training, never promotes a model and never
# sends anything to a cloud service (CLAUDE.md Rule 8; docs/TASARIM_GECE_DONGUSU.md).
# storage\STOP_ALL or storage\STOP_LEARNING makes it a no-op.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "reports\nightly"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$log = Join-Path $logDir ("nightly-" + $stamp + ".log")

Write-Output ("[" + (Get-Date -Format "HH:mm:ss") + "] nightly loop starting...")
try {
    & uv run --no-sync hektor gece 2>&1 | Tee-Object -FilePath $log
    Write-Output ("Done. Log: " + $log)
}
catch {
    # A failed night must never be fatal; the next night retries.
    $_ | Out-String | Tee-Object -FilePath $log -Append | Out-Null
    Write-Output ("Nightly run failed (non-fatal). See log: " + $log)
}
