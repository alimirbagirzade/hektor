# Hektor continuous training loop -- ONE step per run (ASCII-only, Windows PS 5.1 safe).
#
#   Manual run:  powershell -ExecutionPolicy Bypass -File scripts\continuous-loop.ps1
#   Scheduled:   scripts\install-continuous-loop-task.ps1 registers this every 30 minutes.
#
# Each run calls `uv run hektor surekli-dongu tik`: verify the last candidate, convert it to
# Ollama, run LLM-30 (candidate + reference), score with the LOCAL judge, record the gate; when
# ALL questions are correct the loop pauses for a human decision. When data is new and every
# gate passes, it starts the next training WITHOUT a human click (user decision 2026-10-15,
# CLAUDE.md Rule 8) -- only if HEKTOR_CONTINUOUS_LOOP_ENABLED and
# HEKTOR_UNATTENDED_TRAINING_ENABLED are both true. It never promotes a model.
# storage\STOP_ALL or storage\STOP_LEARNING makes it a no-op.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "storage\continuous_loop"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$log = Join-Path $logDir "ticks.log"

$ErrorActionPreference = "Continue"
$line = & uv run --no-sync hektor surekli-dongu tik 2>&1 | Select-Object -Last 1
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Add-Content -Path $log -Value ("[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] (" + $code + ") " + $line) -Encoding UTF8
