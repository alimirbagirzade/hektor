# LoRA adapter -> birlesik model -> GGUF (bf16) -> Q4_K_M -> Ollama modeli.
# Ollama 0.34.x calisma-ani ADAPTER destegini kaldirdi; yol base ile birlestirmektir
# (HANDOFF 2026-09-28 (3), Kademe-2 C9). HF onbellegindeki base DEGISMEZ.
#
# Kullanim:
#   powershell -File scripts\adapter_to_ollama.ps1 -Adapter hektor_lora_v12_30b `
#       -OllamaName hektor-v12-30b -TemplateFrom qwen3:30b-a3b-instruct-2507-q4_K_M
#
# UYARI (30B): birlestirme adimi base'i bf16 yukler (~116 GB surec bellegi olculdu) —
# bu sirada eval/egitim/web LoRA sohbeti/Ollama modeli CALISTIRMA.
# Web'in modelini DEGISTIRMEZ: .env HEKTOR_LLM_MODEL ayri, bilincli adimdir.
param(
    [Parameter(Mandatory = $true)][string]$Adapter,
    [Parameter(Mandatory = $true)][string]$OllamaName,
    # Sablon/parametreler (TEMPLATE, stop, temperature...) base'in Ollama etiketinden alinir.
    [Parameter(Mandatory = $true)][string]$TemplateFrom,
    [string]$Quant = "Q4_K_M",
    [string]$Tools = "C:\HP\tools",
    # Bos = base etiketindeki deger (Qwen3-2507: 1 = KAPALI). v12'de dejenere tekrar
    # vetosu goruldu (2026-09-30) -> 1.1 ile yeniden olusturuldu. Uygulama (LocalLLM)
    # repeat_penalty GONDERMEZ, yani Modelfile degeri gecerlidir.
    [string]$RepeatPenalty = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:UV_NO_SYNC = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$merged = Join-Path $root "models\merged\$Adapter"
$ggufDir = Join-Path $root "models\gguf"
$bf16 = Join-Path $ggufDir "$Adapter-bf16.gguf"
$quantOut = Join-Path $ggufDir "$Adapter-$Quant.gguf"
$modelfile = Join-Path $ggufDir "Modelfile.$OllamaName"
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
$null = New-Item -ItemType Directory -Force -Path $ggufDir
function Step($m) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[HATA] $m" -ForegroundColor Red; exit 1 }

# 1) Birlestir (PEFT merge_and_unload + oncesi/sonrasi logit dogrulamasi)
if (Test-Path (Join-Path $merged "merge_info.json")) {
    Step "1/5 birlesik model zaten var: $merged (atlaniyor)"
} else {
    Step "1/5 birlestirme: $Adapter -> $merged"
    & $py scripts\merge_adapter.py $Adapter --out $merged
    if ($LASTEXITCODE -ne 0) { Fail "birlestirme basarisiz (cikis $LASTEXITCODE)" }
}

# 2) GGUF (bf16; f16 tasma riskine karsi bf16 korunur)
if (Test-Path $bf16) {
    Step "2/5 bf16 GGUF zaten var: $bf16 (atlaniyor)"
} else {
    Step "2/5 GGUF donusumu -> $bf16"
    $env:PYTHONPATH = Join-Path $Tools "spstub"
    & $py (Join-Path $Tools "llama.cpp\convert_hf_to_gguf.py") $merged --outfile $bf16 --outtype bf16
    $rc = $LASTEXITCODE
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if ($rc -ne 0 -or -not (Test-Path $bf16)) { Fail "GGUF donusumu basarisiz (cikis $rc)" }
}

# 3) Nicemleme
if (Test-Path $quantOut) {
    Step "3/5 $Quant zaten var: $quantOut (atlaniyor)"
} else {
    Step "3/5 nicemleme $Quant -> $quantOut"
    & (Join-Path $Tools "llama-bin\llama-quantize.exe") $bf16 $quantOut $Quant
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $quantOut)) { Fail "nicemleme basarisiz" }
}

# 4) Modelfile: sablon/parametreler base etiketinden, FROM yeni GGUF
Step "4/5 Modelfile ($TemplateFrom sablonuyla) -> $modelfile"
$base = & $ollama show $TemplateFrom --modelfile
if ($LASTEXITCODE -ne 0) { Fail "ollama show $TemplateFrom basarisiz" }
$lines = @("# $OllamaName - $Adapter birlesik GGUF $Quant; sablon/parametreler $TemplateFrom'dan.")
$lines += "FROM $quantOut"
# Yalniz BASTAKI yorum satirlari ve FROM satiri atilir; TEMPLATE """ blogu icindeki '#'
# ile baslayan satirlar (or. "# Tools") KORUNUR.
$inHeader = $true
foreach ($l in $base) {
    if ($inHeader -and ($l -match '^\s*#' -or $l -match '^\s*$')) { continue }
    $inHeader = $false
    if ($l -match '^FROM\s') { continue }
    if ($RepeatPenalty -and $l -match '^PARAMETER\s+repeat_penalty\s') { continue }
    $lines += $l
}
if ($RepeatPenalty) { $lines += "PARAMETER repeat_penalty $RepeatPenalty" }
# PowerShell 5.1 Set-Content UTF8 BOM ekler; Modelfile BOM'suz yazilir.
[IO.File]::WriteAllLines($modelfile, [string[]]$lines, (New-Object Text.UTF8Encoding($false)))

# 5) Ollama modeli
Step "5/5 ollama create $OllamaName"
& $ollama create $OllamaName -f $modelfile
if ($LASTEXITCODE -ne 0) { Fail "ollama create basarisiz" }
$sha = (Get-FileHash -Algorithm SHA256 $quantOut).Hash.ToLowerInvariant()
Step "TAMAM: $OllamaName hazir · GGUF sha256=$($sha.Substring(0,16))… · $quantOut"
