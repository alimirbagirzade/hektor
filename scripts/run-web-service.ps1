$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectDir ".venv\Scripts\python.exe"
Set-Location $projectDir
$env:UV_NO_SYNC = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:HEKTOR_WEB_ISOLATE_CHROMA = "1"
& $pythonExe -c "from app.web.server import run; run()"
exit $LASTEXITCODE
