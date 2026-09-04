# Achilles weekly bug-hunt scan -- TIER 1 (REPORT-ONLY).  ASCII-only (Windows PS 5.1 safe).
#
# What it does:
#   1) Validation gate: ruff + mypy + pytest (offline, --no-sync avoids the web .exe lock).
#   2) Lightweight LLM bug scan: one `claude -p` call over recent diff + core (best-effort).
#   3) Writes reports/bug-scan/scan-<date>.md + one-line summary to HANDOFF.md.
#
# Does NOT edit code, does NOT commit/push. Findings are fixed in the next SUPERVISED
# session (see CLAUDE.md "Bug-avi kadansi" -- Tier 2 deep hunt is human-supervised).
#
# Run: Windows Task Scheduler (weekly) or manually:  powershell scripts\weekly-bug-scan.ps1

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$outDir = Join-Path $repo "reports\bug-scan"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$report = Join-Path $outDir "scan-$stamp.md"

function Append([string]$text) { $text | Out-File -FilePath $report -Append -Encoding utf8 }
function LastLine([string]$s) {
    ($s -split "`r?`n" | Where-Object { $_.Trim() -ne "" } | Select-Object -Last 1)
}

"# Achilles bug-hunt scan -- $stamp (Tier 1, report-only)" | Out-File -FilePath $report -Encoding utf8
Append ""

# --- 1) Validation gate ------------------------------------------------------
Append "## Validation gate"
$ruff   = (uv run --no-sync ruff check . 2>&1 | Out-String).Trim()
$mypy   = (uv run --no-sync mypy app 2>&1 | Out-String).Trim()
$pytest = (uv run --no-sync pytest -q --basetemp=.pytest_tmp -p no:cacheprovider 2>&1 | Out-String).Trim()
Append '```'
Append ("RUFF   : " + (LastLine $ruff))
Append ("MYPY   : " + (LastLine $mypy))
Append ("PYTEST : " + (LastLine $pytest))
Append '```'

# --- 2) Lightweight LLM bug scan (best-effort) -------------------------------
Append ""
Append "## LLM bug scan (Tier 1)"
$claude = Get-Command claude -ErrorAction SilentlyContinue
if ($claude) {
    # SECURITY (see docs/SCOPE_ISOLATION.md): this prompt used to say "DO NOT edit code,
    # DO NOT run git" -- A PROMPT INSTRUCTION IS NOT A BOUNDARY. The scan reads recent
    # diffs, so hostile content in a commit can redirect the agent (prompt injection).
    # With Bash it could run the UNAUTHENTICATED CLI (`achilles approval-approve`) or hit
    # 127.0.0.1:8765 and approve its own training (CLAUDE.md Rule 8).
    #
    # The restriction is now TECHNICAL, matching AutoDriver (app/orchestration/engines.py):
    #   --safe-mode          : hooks/plugins/MCP/custom agents off. These run OUTSIDE the
    #                          tool layer, so a deny-list alone never sees them.
    #   --strict-mcp-config  : no --mcp-config given => zero MCP servers.
    #   --disallowedTools    : variadic => MUST stay last, single comma-joined arg.
    # Read/Grep/Glob stay enabled -- enough for a read-only scan.
    $denied = "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task"

    # Bash is denied, so the agent cannot run git itself -- compute the diff HERE and embed
    # it. Truncated: a huge diff would blow the context window and add nothing.
    $diff = (git diff --stat HEAD~15..HEAD 2>&1 | Out-String).Trim()
    if ($diff.Length -gt 6000) { $diff = $diff.Substring(0, 6000) + "`n...(truncated)" }

    $prompt = @"
Run a LIGHTWEIGHT, REPORT-ONLY bug scan on the Achilles trading-RESEARCH project.
Only find and summarize -- you have read-only tools (Read/Grep/Glob) by design.
FIRST read CLAUDE.md at the repo root (safe-mode disables auto-discovery; the rules are there).
Focus: the changed files below + core (app/trading, app/brain, app/memory,
app/verification, app/lora, app/training). Highest priority = CLAUDE.md absolute rules:
look-ahead bias, costs (commission+slippage), determinism (seed), no eval/exec, no source fabrication.
List each REAL, code-verified bug: [severity] file:line -- why -- suggested fix.
If none, write "temiz". Output: short plain markdown.

Recent changes (git diff --stat HEAD~15..HEAD):
$diff
"@
    try {
        $scan = (claude -p $prompt --safe-mode --strict-mcp-config --disallowedTools $denied 2>&1 | Out-String).Trim()
        Append $scan
    } catch {
        Append ("_LLM scan failed: " + $_.Exception.Message + "_")
    }
} else {
    Append "_claude CLI not on PATH -- LLM scan skipped (gate results above)._"
}

# --- 3) HANDOFF summary ------------------------------------------------------
$handoff = Join-Path $repo "HANDOFF.md"
if (Test-Path $handoff) {
    "`n> [bug-scan $stamp] Weekly Tier-1 scan done -> reports/bug-scan/scan-$stamp.md" |
        Out-File -FilePath $handoff -Append -Encoding utf8
}

Write-Output "Bug-hunt scan done -> $report"
