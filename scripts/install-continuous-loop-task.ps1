# Registers the Hektor continuous training loop as a Windows Scheduled Task.
# ASCII-only (Windows PS 5.1 safe). User-level task (no admin needed).
#
#   Install:   powershell -ExecutionPolicy Bypass -File scripts\install-continuous-loop-task.ps1
#   Interval:  ... -EveryMinutes 15
#   Uninstall: powershell -ExecutionPolicy Bypass -File scripts\install-continuous-loop-task.ps1 -Uninstall
#   Test now:  schtasks /Run /TN Hektor-ContinuousLoop
#
# Runs scripts\continuous-loop.ps1 every $EveryMinutes minutes. Each run takes ONE step
# (see app/orchestration/continuous_loop.py); heavy work runs as hidden background processes.

param(
    [switch]$Uninstall,
    [int]$EveryMinutes = 30
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "Hektor-ContinuousLoop"
$loopScript = Join-Path $repo "scripts\continuous-loop.ps1"

if ($Uninstall) {
    schtasks /Delete /TN $taskName /F
    Write-Output "Task removed: $taskName"
    return
}
if (-not (Test-Path $loopScript)) { throw "Loop script not found: $loopScript" }
if ($EveryMinutes -lt 5) { throw "EveryMinutes must be >= 5" }

$action = 'powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $loopScript + '"'
schtasks /Create /TN $taskName /TR $action /SC MINUTE /MO $EveryMinutes /F

Write-Output ("Task installed: " + $taskName + " -- every " + $EveryMinutes + " minutes.")
Write-Output "Status:    uv run hektor surekli-dongu durum"
Write-Output "Pause:     uv run hektor surekli-dongu duraklat --gerekce ""..."""
Write-Output ("Uninstall: powershell -File scripts\install-continuous-loop-task.ps1 -Uninstall")
