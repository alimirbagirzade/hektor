# Registers the Achilles weekly bug-hunt scan (Tier 1, report-only) as a Windows
# Scheduled Task.  ASCII-only (Windows PS 5.1 safe).
#
#   Install:   powershell -ExecutionPolicy Bypass -File scripts\install-bug-scan-task.ps1
#   Uninstall: powershell -ExecutionPolicy Bypass -File scripts\install-bug-scan-task.ps1 -Uninstall
#   Test now:  schtasks /Run /TN Achilles-WeeklyBugScan
#
# Runs scripts\weekly-bug-scan.ps1 every Monday 09:00. The scan does NOT edit code or
# push -- it only writes reports\bug-scan\. User-level task (no admin needed).

param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "Achilles-WeeklyBugScan"
$scanScript = Join-Path $repo "scripts\weekly-bug-scan.ps1"

if ($Uninstall) {
    schtasks /Delete /TN $taskName /F
    Write-Output "Task removed: $taskName"
    return
}

if (-not (Test-Path $scanScript)) {
    throw "Scan script not found: $scanScript"
}

# /F updates an existing task (idempotent). Weekly, Monday 09:00.
$action = 'powershell -NoProfile -ExecutionPolicy Bypass -File "' + $scanScript + '"'
schtasks /Create /TN $taskName /TR $action /SC WEEKLY /D MON /ST 09:00 /F

Write-Output "Task installed: $taskName -- every Monday 09:00 (Tier 1 report-only scan)."
Write-Output "Test now:   schtasks /Run /TN $taskName"
Write-Output "Uninstall:  powershell -File scripts\install-bug-scan-task.ps1 -Uninstall"
