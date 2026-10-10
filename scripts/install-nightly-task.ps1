# Registers the Hektor nightly loop as a Windows Scheduled Task.
# ASCII-only (Windows PS 5.1 safe).
#
#   Install:   powershell -ExecutionPolicy Bypass -File scripts\install-nightly-task.ps1
#   Other time: powershell -ExecutionPolicy Bypass -File scripts\install-nightly-task.ps1 -At 22:45
#   Uninstall: powershell -ExecutionPolicy Bypass -File scripts\install-nightly-task.ps1 -Uninstall
#   Test now:  schtasks /Run /TN Hektor-Nightly
#
# Runs scripts\nightly.ps1 every day at $At (default 23:30). The run judges learning
# candidates with the LOCAL model, quarantines suspicious ones, builds CSV indicator
# candidates and writes a readiness report. It never trains (Rule 8).
# User-level task (no admin needed).

param(
    [switch]$Uninstall,
    [string]$At = "23:30"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "Hektor-Nightly"
$nightlyScript = Join-Path $repo "scripts\nightly.ps1"

if ($Uninstall) {
    schtasks /Delete /TN $taskName /F
    Write-Output "Task removed: $taskName"
    return
}

if (-not (Test-Path $nightlyScript)) {
    throw "Nightly script not found: $nightlyScript"
}

# -WindowStyle Hidden: no console window pops up on the desktop at night.
$action = 'powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $nightlyScript + '"'
schtasks /Create /TN $taskName /TR $action /SC DAILY /ST $At /F

Write-Output ("Task installed: " + $taskName + " -- every day at " + $At + ".")
Write-Output "It judges candidates locally, runs the CSV lab and reports readiness; it never trains."
Write-Output ("Drop CSV files into: " + (Join-Path $repo "data\market\raw"))
Write-Output ("Test now:   schtasks /Run /TN " + $taskName)
Write-Output "Uninstall:  powershell -File scripts\install-nightly-task.ps1 -Uninstall"
