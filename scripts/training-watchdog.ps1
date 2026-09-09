$ErrorActionPreference = "Continue"
$projectDir = Split-Path -Parent $PSScriptRoot
$statusFile = Join-Path $projectDir "storage\train_status.json"
$startScript = Join-Path $PSScriptRoot "start-train.ps1"
$trainLog = Join-Path $projectDir "logs\train-full-err.log"
$mutex = [Threading.Mutex]::new($false, "Local\HektorTrainingWatchdog")
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
    if (-not (Test-Path $statusFile)) { exit 0 }
    $status = Get-Content $statusFile -Raw | ConvertFrom-Json
    if ((Test-Path $trainLog) -and ((Get-Date) - (Get-Item $trainLog).LastWriteTime).TotalMinutes -lt 10) { exit 0 }
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*peft_lora_train*' -or $_.CommandLine -like '*train*--run*' }
    if ($running) { exit 0 }
    # -Resume: bu bir COKME-KURTARMA yeniden baslatmasidir; yarim kalan kosu sifirdan
    # baslamasin (trainer'da resume artik varsayilan KAPALI -- Kademe-2 av bulgusu).
    # Sifir-adim ("ogrenilecek bir sey kalmamis") durumu trainer'da hata verir, sessizce
    # "basarili" donmez; dolayisiyla kurtarma guvenli.
    # Temel model status dosyasindan GERI OKUNUR: yoksa varsayilana (4B) doner ve
    # dusuk RAM'li makinede OOM olur -- kurtarma sessizce basarisiz olurdu.
    # Receteyi durum dosyasindan OLDUGU GIBI geri oku. Eksik okunan her alan, kurtarma
    # kosusunu SESSIZCE baska bir receteye kaydirir:
    #   * base_model yoksa -> varsayilan 4B, dusuk RAM'li makinede OOM,
    #   * profile sabit yazilirsa -> baska profille baslamis kosu profil degistirir,
    #   * max_examples unutulursa -> ornek tavani profil varsayilanina duser, adim sayisi
    #     ayni kalir, yani ayni kucuk alt-kume uzerinde COKLU EPOCH (ezber; 2026-09-08 v8).
    $prop = $status.PSObject.Properties.Name
    $bm = if ($prop -contains "base_model") { [string]$status.base_model } else { "" }
    $prof = if ($prop -contains "profile" -and "$($status.profile)".Trim() -ne "") { [string]$status.profile } else { "discipline_safe_local" }
    $mx = if ($prop -contains "max_examples") { [int]$status.max_examples } else { 0 }
    & $startScript -Adapter $status.adapter -Iterations ([int]$status.iterations) -Dtype $status.dtype -Profile $prof -BaseModel $bm -MaxExamples $mx -Resume
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
