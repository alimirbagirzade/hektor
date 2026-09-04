# Achilles Surekli Ogrenme Dongusu -- Windows baslatici + otomatik acilis
#
# Kullanim:
#   .\scripts\start-loop.ps1             -- simdi baslat (egitim calismiyorsa)
#   .\scripts\start-loop.ps1 -Install    -- Windows acilisina ekle (+ simdi baslat)
#   .\scripts\start-loop.ps1 -Uninstall  -- acilistan kaldir + durdur
#   .\scripts\start-loop.ps1 -Stop       -- durdur
#   .\scripts\start-loop.ps1 -Status     -- durum goster
#
# Dongu kendi kendine internetten makale CEKMEZ (o ozellik kapali); yalnizca
# kullanicinin yukledigi makaleler uzerinde kart/kavrama/sentetik-veri uretir.

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Stop,
    [switch]$Status,
    [int]$Hours = 72
)

$ErrorActionPreference = "Continue"
$ScriptDir  = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectDir = Split-Path -Parent $ScriptDir
$RegPath    = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$RegKey     = "AchillesLoop"
$VbsFile    = Join-Path $ScriptDir "achilles-loop-autostart.vbs"
$StopFile   = Join-Path $ProjectDir "storage\STOP_LEARNING"

function Find-Bash {
    $c = @(
        (Get-Command bash -ErrorAction SilentlyContinue).Source,
        "C:\Program Files\Git\bin\bash.exe",
        "C:\Program Files\Git\usr\bin\bash.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Git\bin\bash.exe")
    )
    foreach ($p in $c) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}

function Test-LoopRunning {
    $n = (Get-CimInstance Win32_Process -Filter "Name='bash.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*continuous-learning*' } | Measure-Object).Count
    return ($n -gt 0)
}

function Test-TrainRunning {
    $n = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*peft_lora_train*' -or $_.CommandLine -like '*train*--run*' } |
        Measure-Object).Count
    return ($n -gt 0)
}

function Start-Loop {
    if (Test-LoopRunning) { Write-Host "  [OK] Dongu zaten calisiyor." -ForegroundColor Green; return }
    if (Test-TrainRunning) {
        Write-Host "  [!] Egitim suruyor -- dongu ertelendi (cakismamasi icin)." -ForegroundColor Yellow
        return
    }
    $bash = Find-Bash
    if (-not $bash) {
        Write-Host "  [HATA] bash (Git Bash) bulunamadi." -ForegroundColor Red
        return
    }
    Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
    $hrs = "$Hours"
    Start-Process -FilePath $bash `
        -ArgumentList "scripts/continuous-learning.sh", $hrs `
        -WorkingDirectory $ProjectDir -WindowStyle Hidden
    Write-Host "  [OK] Surekli ogrenme dongusu baslatildi ($hrs saat)." -ForegroundColor Green
}

function Stop-Loop {
    New-Item -ItemType File $StopFile -Force | Out-Null
    Get-CimInstance Win32_Process -Filter "Name='bash.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*continuous-learning*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "  [OK] Dongu durduruldu (STOP_LEARNING)." -ForegroundColor Yellow
}

function Install-Autostart {
    $thisScript = Join-Path $ScriptDir "start-loop.ps1"
    $vbsContent = @"
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -NonInteractive -File ""$thisScript""", 0, False
Set sh = Nothing
"@
    $vbsContent | Out-File -FilePath $VbsFile -Encoding ascii -Force
    Set-ItemProperty -Path $RegPath -Name $RegKey -Value ('wscript.exe "' + $VbsFile + '"') -Force
    Write-Host "  [OK] Windows acilisina eklendi (HKCU Run). Her login'de dongu baslar." -ForegroundColor Green
    Write-Host "       (Egitim calisiyorsa otomatik ertelenir -- cakisma yok.)" -ForegroundColor Gray
    Start-Loop
}

function Uninstall-Autostart {
    Stop-Loop
    Remove-ItemProperty -Path $RegPath -Name $RegKey -ErrorAction SilentlyContinue
    Remove-Item $VbsFile -Force -ErrorAction SilentlyContinue
    Write-Host "  [OK] Otomatik baslatma kaldirildi." -ForegroundColor Yellow
}

function Show-Status {
    $loop = if (Test-LoopRunning) { 'CALISIYOR' } else { 'durdu' }
    $train = if (Test-TrainRunning) { 'CALISIYOR (dongu ertelenir)' } else { 'yok' }
    $reg = Get-ItemProperty -Path $RegPath -Name $RegKey -ErrorAction SilentlyContinue
    $auto = if ($reg) { 'kayitli' } else { 'kayitli degil' }
    Write-Host "  Dongu     : $loop" -ForegroundColor Cyan
    Write-Host "  Egitim    : $train" -ForegroundColor Gray
    Write-Host "  Autostart : $auto" -ForegroundColor Gray
}

if ($Install)   { Install-Autostart;   exit 0 }
if ($Uninstall) { Uninstall-Autostart; exit 0 }
if ($Stop)      { Stop-Loop;           exit 0 }
if ($Status)    { Show-Status;         exit 0 }

Start-Loop
