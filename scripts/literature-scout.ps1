# Achilles literature scout -- daily discovery run (ASCII-only, Windows PS 5.1 safe).
#
#   Manual run:  powershell -ExecutionPolicy Bypass -File scripts\literature-scout.ps1
#   Scheduled:   scripts\install-literature-scout-task.ps1 registers this daily.
#
# WHAT IT DOES: searches arXiv for LoRA / RAG / RLM / math-physics METHOD papers,
# scores them offline (no LLM, no API quota), appends new ones to the per-topic
# watchlists and downloads the top few PDFs into the inbox folder.
#
# WHAT IT DOES NOT DO: it never ingests into RAG and never starts training.
# Downloaded != ingested != trained -- those stay manual (CLAUDE.md Rule 8).
#
# Inbox location: $env:ACHILLES_SCOUT_INBOX_DIR, else <repo>\data\literature_inbox

param(
    [int]$TopN = 2,
    [switch]$NoDownload
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "reports\literature-scout"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$log = Join-Path $logDir ("scout-" + $stamp + ".log")

$scoutArgs = @("run", "--no-sync", "achilles", "lit-scan", "--top-n", $TopN)
if ($NoDownload) { $scoutArgs += "--no-download" }

Write-Output ("[" + (Get-Date -Format "HH:mm:ss") + "] literature scout starting...")
try {
    & uv @scoutArgs 2>&1 | Tee-Object -FilePath $log
    Write-Output ("Done. Log: " + $log)
}
catch {
    # A failed discovery round must never be fatal -- it is a report-only agent.
    $_ | Out-String | Tee-Object -FilePath $log -Append | Out-Null
    Write-Output ("Scout run failed (non-fatal). See log: " + $log)
}
