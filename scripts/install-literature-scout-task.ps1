# Registers the Achilles literature scout (discovery-only) as a Windows Scheduled Task.
# ASCII-only (Windows PS 5.1 safe).
#
#   Install:   powershell -ExecutionPolicy Bypass -File scripts\install-literature-scout-task.ps1
#   Uninstall: powershell -ExecutionPolicy Bypass -File scripts\install-literature-scout-task.ps1 -Uninstall
#   Test now:  schtasks /Run /TN Achilles-LiteratureScout
#
# Runs scripts\literature-scout.ps1 every day at 08:30 (before the Monday 09:00 bug
# scan, so the two never overlap). The scout only downloads candidate PDFs into the
# inbox and updates watchlists -- it never ingests into RAG and never trains (Rule 8).
# User-level task (no admin needed).

param(
    [switch]$Uninstall,
    [string]$At = "08:30"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "Achilles-LiteratureScout"
$scoutScript = Join-Path $repo "scripts\literature-scout.ps1"

if ($Uninstall) {
    schtasks /Delete /TN $taskName /F
    Write-Output "Task removed: $taskName"
    return
}

if (-not (Test-Path $scoutScript)) {
    throw "Scout script not found: $scoutScript"
}

# /F updates an existing task (idempotent). Daily at $At.
$action = 'powershell -NoProfile -ExecutionPolicy Bypass -File "' + $scoutScript + '"'
schtasks /Create /TN $taskName /TR $action /SC DAILY /ST $At /F

Write-Output ("Task installed: " + $taskName + " -- every day at " + $At + " (discovery only).")
Write-Output "It downloads candidate PDFs to the inbox; RAG ingest + training stay manual."
Write-Output ("Inbox: " + $(if ($env:ACHILLES_SCOUT_INBOX_DIR) { $env:ACHILLES_SCOUT_INBOX_DIR } else { Join-Path $repo "data\literature_inbox" }))
Write-Output ("Test now:   schtasks /Run /TN " + $taskName)
Write-Output "Uninstall:  powershell -File scripts\install-literature-scout-task.ps1 -Uninstall"
