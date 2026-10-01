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
    [string]$RepeatPenalty = "",
    # Attention tensorlerinin nicemleme tipi. LoRA yalniz attention'i degistiriyor; Q4_K
    # gurultusu bu farkin 3-7 kati olculdu (Kademe-2 D2, 2026-09-30) -> q8_0 (~+0.5 GB).
    # Bos = hepsi $Quant (eski davranis).
    [string]$AttnQuant = "q8_0",
    # Ayni adda Ollama modeli varsa ya da ad .env HEKTOR_LLM_MODEL ise uzerine yazmak icin.
    [switch]$Force,
    # Adapter'SIZ base'i AYNI tarifle (GGUF + $Quant + attention $AttnQuant + sablon) Ollama'ya
    # koy: 2x2 karsilastirmada base ile LoRA'nin nicemlemesi esit olsun diye (protokol Asama 2).
    # Verilirse birlestirme atlanir, HF onbellegindeki snapshot dogrudan donusturulur;
    # -Adapter yalniz dosya adi etiketidir (or. base_qwen3_30b_a3b).
    [string]$BaseRepo = "",
    # Bos = merge_adapter.py varsayilan KL kapisi (0.01). Yalniz bilincli insan karariyla;
    # kullanilan deger merge_info.json'a yazilir.
    [string]$MaxKL = ""
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

# Koken anahtari (Kademe 2 F4-2): ara ciktilar (birlesik model, bf16 GGUF, nicemli GGUF) eskiden
# YALNIZ var olduklari icin yeniden kullaniliyordu -> ayni adla yeniden egitimde Ollama ESKI
# agirliklari servis eder, llm30 manifesti ise YENI adapter sha'sini kaydederdi. Her ciktinin
# yanina kaynagini yazan bir .src dosyasi konur; anahtar uyusmazsa (ya da .src yoksa) zincir
# durur, eski dosyayi silmen istenir (sessiz yeniden kullanim yok).
if ($BaseRepo) {
    $srcKey = "base:$BaseRepo"
} else {
    $w = Join-Path $root "models\adapters\$Adapter\adapter_model.safetensors"
    if (-not (Test-Path $w)) { Fail "adapter agirligi yok: $w" }
    $srcKey = "adapter:" + (Get-FileHash -Algorithm SHA256 -LiteralPath $w).Hash.ToLower()
}
$attnKey = if ($AttnQuant) { $AttnQuant } else { "-" }
function Assert-Source($artifact, $key) {
    $side = "$artifact.src"
    if (-not (Test-Path $side)) {
        Fail "$artifact var ama koken kaydi ($side) yok - hangi adapter'dan uretildigi bilinmiyor. Sil ve yeniden calistir."
    }
    $have = (Get-Content -LiteralPath $side -Raw).Trim()
    if ($have -ne $key) {
        Fail "$artifact BASKA bir kaynaktan uretilmis ($have != $key). Eski dosyayi sil ve yeniden calistir."
    }
}
function Write-Source($artifact, $key) {
    [System.IO.File]::WriteAllText("$artifact.src", $key)
}

# 0) Ad cakismasi (Kademe-2 D5): `ollama create` ayni adi SESSIZCE ezer; ad web'in canli
# modeliyse (.env HEKTOR_LLM_MODEL) zincir onaysiz olarak canli modeli degistirirdi.
$liveModel = ""
if (Test-Path (Join-Path $root ".env")) {
    $m = Select-String -Path (Join-Path $root ".env") -Pattern '^\s*HEKTOR_LLM_MODEL\s*=\s*(\S+)' |
        Select-Object -First 1
    if ($m) { $liveModel = $m.Matches[0].Groups[1].Value }
}
$existing = (& $ollama list 2>$null) -match ("^" + [regex]::Escape($OllamaName) + "(:latest)?\s")
if (-not $Force -and ($existing -or $OllamaName -eq $liveModel -or "$OllamaName`:latest" -eq $liveModel)) {
    Fail "Ollama adi '$OllamaName' zaten var ya da web'in canli modeli (.env) - uzerine yazmak icin -Force."
}

# 1) Birlestir (PEFT merge_and_unload + oncesi/sonrasi logit dogrulamasi)
if ($BaseRepo) {
    $merged = (& $py -c "from huggingface_hub import snapshot_download as s; print(s('$BaseRepo', local_files_only=True))").Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $merged "config.json"))) {
        Fail "base snapshot HF onbelleginde yok: $BaseRepo (indirme YAPILMAZ)"
    }
    Step "1/5 base-only: birlestirme yok, snapshot -> $merged"
} elseif (Test-Path (Join-Path $merged "merge_info.json")) {
    $mi = Get-Content -LiteralPath (Join-Path $merged "merge_info.json") -Raw | ConvertFrom-Json
    if ("adapter:$($mi.adapter_sha256)" -ne $srcKey) {
        Fail "birlesik model $merged BASKA bir adapter agirligindan ($($mi.adapter_sha256)) - sil ve yeniden calistir."
    }
    Step "1/5 birlesik model zaten var ve adapter sha eslesiyor: $merged (atlaniyor)"
} else {
    Step "1/5 birlestirme: $Adapter -> $merged"
    $margs = @()
    if ($MaxKL) { $margs += "--max-kl"; $margs += $MaxKL }
    & $py scripts\merge_adapter.py $Adapter --out $merged @margs
    if ($LASTEXITCODE -ne 0) { Fail "birlestirme basarisiz (cikis $LASTEXITCODE)" }
}

# 2) GGUF (bf16; f16 tasma riskine karsi bf16 korunur)
if (Test-Path $bf16) {
    Assert-Source $bf16 $srcKey
    Step "2/5 bf16 GGUF zaten var ve kaynagi eslesiyor: $bf16 (atlaniyor)"
} else {
    Step "2/5 GGUF donusumu -> $bf16"
    $env:PYTHONPATH = Join-Path $Tools "spstub"
    # Yarim kalan cikti "zaten var" sanilmasin (Kademe-2 D4): once .partial, basarida ad degistir.
    $part = "$bf16.partial"
    Remove-Item $part -ErrorAction SilentlyContinue
    & $py (Join-Path $Tools "llama.cpp\convert_hf_to_gguf.py") $merged --outfile $part --outtype bf16
    $rc = $LASTEXITCODE
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    if ($rc -ne 0 -or -not (Test-Path $part)) { Fail "GGUF donusumu basarisiz (cikis $rc)" }
    Move-Item $part $bf16
    Write-Source $bf16 $srcKey
}

# 3) Nicemleme (anahtar: kaynak + nicemleme tarifi; dosya adi AttnQuant'i tasimaz)
$quantKey = "$srcKey|$Quant|attn=$attnKey"
if (Test-Path $quantOut) {
    Assert-Source $quantOut $quantKey
    Step "3/5 $Quant zaten var ve tarif/kaynak eslesiyor: $quantOut (atlaniyor)"
} else {
    Step "3/5 nicemleme $Quant (attention: $(if ($AttnQuant) { $AttnQuant } else { $Quant })) -> $quantOut"
    $part = "$quantOut.partial"
    Remove-Item $part -ErrorAction SilentlyContinue
    $qargs = @()
    if ($AttnQuant) {
        foreach ($t in "attn_q", "attn_k", "attn_v", "attn_output") { $qargs += "--tensor-type"; $qargs += "$t=$AttnQuant" }
    }
    & (Join-Path $Tools "llama-bin\llama-quantize.exe") @qargs $bf16 $part $Quant
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $part)) { Fail "nicemleme basarisiz" }
    Move-Item $part $quantOut
    Write-Source $quantOut $quantKey
}

# 4) Modelfile: sablon/parametreler base etiketinden, FROM yeni GGUF
Step "4/5 Modelfile ($TemplateFrom sablonuyla) -> $modelfile"
$base = & $ollama show $TemplateFrom --modelfile
if ($LASTEXITCODE -ne 0) { Fail "ollama show $TemplateFrom basarisiz" }
$kind = if ($BaseRepo) { "base-only ($BaseRepo @ $(Split-Path -Leaf $merged))" } else { "birlesik" }
$lines = @("# $OllamaName - $Adapter $kind GGUF $Quant; sablon/parametreler $TemplateFrom'dan.")
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
# BOM'suz ve LF: WriteAllLines Windows'ta CRLF yazar; Ollama CR karakterini TEMPLATE'e
# aynen alir ve Qwen'de CRLF (token 319) != LF (198) -> egitimde hic gorulmeyen istem
# (Kademe-2 D1: hektor-v12-30b sablonunda 47 CR vardi).
$text = (($lines | ForEach-Object { $_ -replace "`r", "" }) -join "`n") + "`n"
[IO.File]::WriteAllText($modelfile, $text, (New-Object Text.UTF8Encoding($false)))

# 5) Ollama modeli
Step "5/5 ollama create $OllamaName"
& $ollama create $OllamaName -f $modelfile
if ($LASTEXITCODE -ne 0) { Fail "ollama create basarisiz" }
$tpl = (& $ollama show $OllamaName --template) -join "`n"
if ($tpl -match "`r") { Fail "olusturulan sablonda CR karakteri var - egitim sablonuyla uyusmaz" }
$sha = (Get-FileHash -Algorithm SHA256 $quantOut).Hash.ToLowerInvariant()
Step "TAMAM: $OllamaName hazir · GGUF sha256=$($sha.Substring(0,16))… · $quantOut"
