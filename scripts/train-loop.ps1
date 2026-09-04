# Achilles - Sürekli (periyodik) LoRA eğitim döngüsü (Windows)
#
# Kullanım:
#   .\scripts\train-loop.ps1                       # varsayılan: 30 iter, 180sn cooldown
#   .\scripts\train-loop.ps1 -Iterations 60 -CooldownSec 180
#
# Durdurma (graceful): storage\STOP_TRAINING dosyası oluştur:
#   New-Item storage\STOP_TRAINING
# Döngü o anki eğitimi bitirir, sonra durur. (Veya pencereyi kapat.)
#
# Her döngü: dataset'i tazele (yeni onaylı kart varsa) → eğit → cooldown.
# Tek beyin: Qwen3-4B (.env → ACHILLES_PEFT_BASE_MODEL).

param(
    [int]$Iterations = 20,
    [int]$CooldownSec = 120,
    [int]$MaxHours = 24,
    [string]$Adapter = "achilles_auto"
)

$ErrorActionPreference = "Continue"
# uv her `uv run`'da paketi yeniden senkronlar; calisan web sunucusunun kilitledigi
# achilles-web.exe'yi silmeye ugrasip "os error 32" ile patlar -> egitim baslamaz.
# Senkronu kapat; bagimliliklar zaten kurulu. (bkz. continuous-learning.sh)
$env:UV_NO_SYNC = "1"
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Log = Join-Path $LogDir "train-loop.log"
$StopFile = Join-Path $ProjectDir "storage\STOP_TRAINING"

# uv bul (PATH veya bilinen konumlar)
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $Uv) { $Uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
if (-not (Test-Path $Uv)) {
    Write-Host "[HATA] uv bulunamadi." -ForegroundColor Red
    exit 1
}

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

$EndTime = (Get-Date).AddHours($MaxHours)
Write-Log "Surekli egitim baslatildi (iters=$Iterations, cooldown=${CooldownSec}sn, adapter=$Adapter, max=${MaxHours}sa)"
Write-Log "Durdurmak icin: New-Item '$StopFile'"

$cycle = 0
while (-not (Test-Path $StopFile) -and (Get-Date) -lt $EndTime) {
    $cycle++
    Write-Log "=== Dongu ${cycle}: dataset tazeleniyor (birlesik: synth-qa + kart + %25 disiplin) ==="
    # ÖNEMLI: `lora-dataset` SADECE karttan uretir (synth-qa + disiplin DAHIL DEGIL) ve
    # lora_sft.jsonl'i clobber eder -> v5 regresyonunun veri tarafi. Bunun yerine kanonik
    # birlesik assembly (assemble_sft.py = lora-cloud-prep/pretrain-gate ile ayni yol) + split.
    & $Uv run python scripts/assemble_sft.py *>> $Log
    & $Uv run achilles lora-split *>> $Log
    Write-Log "=== Dongu ${cycle}: egitim ($Iterations iter) ==="
    # --profile ZORUNLU: geçilmezse vanilya reçete (maskesiz, NEFTune'suz) koşar = v5 tuzağı.
    & $Uv run achilles train --run --backend peft --adapter-name $Adapter --iterations $Iterations --profile discipline_safe_local *>> $Log
    Write-Log "=== Dongu $cycle bitti -> ${CooldownSec}sn cooldown ==="
    # Cooldown sirasinda da durdurma kontrolu
    $waited = 0
    while ($waited -lt $CooldownSec -and -not (Test-Path $StopFile)) {
        Start-Sleep -Seconds 5
        $waited += 5
    }
}

Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
Write-Log "STOP_TRAINING algilandi - surekli egitim durdu ($cycle dongu calisti)."
