# Achilles Trader AI -- TEK KOMUT guncelleme (KURULU makinede calistir)
#
#   .\update.ps1           -- normal: origin/main'e GUVENLI yakinsama (ff-only)
#   .\update.ps1 -Force    -- yereli AT, origin/main ile birebir esitle (salt-kopya kurulum)
#
# Yapar: web sunucusunu durdur -> 'main' dalina yakinsa (origin/main) -> uv sync --extra dev ->
#        web'i yeniden baslat -> saglik kontrolu.  EGITIME DOKUNMAZ.
#
# NOT (kok-neden duzeltmesi): Bu betik artik MEVCUT dal ne olursa olsun makineyi
# 'main' dalina + origin/main'e yakinsatir. Eskiden bir feature dalina parklanmis
# makinede 'git pull origin main' origin/main'i o dala MERGE ediyor, makine asla
# main'e gecmiyordu -> "guncelleme oturmuyor". Tani icin:  uv run achilles doctor
#
# Tarayicida son halini gormek icin sonunda: Ctrl+Shift+R (sert yenileme).

param([switch]$Force)

$ErrorActionPreference = "Continue"
$ProjectDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $ProjectDir

function Find-Uv {
    $fromPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if ($fromPath -and (Test-Path $fromPath)) { return $fromPath }
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"),
        (Join-Path $env:APPDATA "uv\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\uv.exe")
    )
    foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}
function Find-Git {
    $fromPath = (Get-Command git -ErrorAction SilentlyContinue).Source
    if ($fromPath -and (Test-Path $fromPath)) { return $fromPath }
    $candidates = @(
        "C:\Program Files\Git\bin\git.exe",
        "C:\Program Files (x86)\Git\bin\git.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Git\bin\git.exe")
    )
    foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}

$UvPath  = Find-Uv
$GitPath = Find-Git
if (-not $UvPath)  { Write-Host "[HATA] uv bulunamadi (once install.ps1)." -ForegroundColor Red; exit 1 }
if (-not $GitPath) { Write-Host "[HATA] git bulunamadi." -ForegroundColor Red; exit 1 }

& $GitPath rev-parse --is-inside-work-tree 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[HATA] Bu klasor bir git deposu degil: $ProjectDir" -ForegroundColor Red
    Write-Host "       Dogru klasorde calistir (Achilles deposu)." -ForegroundColor Yellow
    exit 1
}

$LogDir = Join-Path $ProjectDir "logs"
$null   = New-Item -ItemType Directory -Path $LogDir -Force
$LogFile = Join-Path $LogDir "update.log"
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Guncelleme basliyor (Force=$Force)..." | Add-Content $LogFile

# --------------------------------------------------------------------------
# Yardimci fonksiyonlar (cagrilmadan ONCE tanimli olmali)
# --------------------------------------------------------------------------
# 'main' baska bir worktree'de checkout mu? (oyle ise bu kopya main'e gecemez)
function Test-MainElsewhere {
    $wt = & $GitPath worktree list --porcelain 2>$null
    return [bool]($wt | Select-String -Pattern 'branch refs/heads/main$' -Quiet)
}

# Dal + HEAD + origin/main'e gore ahead/behind raporla (drift gorunur olsun)
function Show-Drift {
    $b = (& $GitPath rev-parse --abbrev-ref HEAD 2>$null)
    if ($b) { $b = $b.Trim() }
    $h = (& $GitPath rev-parse --short HEAD 2>$null)
    if ($h) { $h = $h.Trim() }
    $behind = 0; $ahead = 0
    $c = (& $GitPath rev-list --left-right --count origin/main...HEAD 2>$null)
    if ($c -and ($c -match '^\s*(\d+)\s+(\d+)')) { $behind = [int]$matches[1]; $ahead = [int]$matches[2] }
    $col = if (($b -eq 'main') -and ($ahead -eq 0) -and ($behind -eq 0)) { 'Green' } else { 'Yellow' }
    Write-Host "[DURUM] dal=$b HEAD=$h | origin/main'e gore: +$ahead / -$behind" -ForegroundColor $col
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] DURUM dal=$b HEAD=$h ahead=$ahead behind=$behind" | Add-Content $LogFile
}

# Mevcut dal ne olursa olsun 'main' + origin/main'e DETERMINISTIK yakinsa.
# Kullanici verisini ASLA atmaz (Force haric); iraksak dali AUTO-MERGE ETMEZ.
function Sync-ToMain {
    & $GitPath fetch origin main 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[UYARI] fetch basarisiz (cevrimdisi?) -- yerel ref'lerle devam." -ForegroundColor Yellow
    }

    $curBranch = (& $GitPath rev-parse --abbrev-ref HEAD 2>$null)
    if ($curBranch) { $curBranch = $curBranch.Trim() }
    $dirty = [bool]((& $GitPath status --porcelain 2>$null) | Where-Object { $_ })

    if ($curBranch -ne "main") {
        Write-Host "[UYARI] Bu makine 'main' DEGIL, '$curBranch' dalinda PARKLANMIS." -ForegroundColor Yellow
        Write-Host "        origin/main'e yakinsamak icin 'main' dalina geciliyor..." -ForegroundColor Yellow
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Parkli dal '$curBranch' -> main gecisi deneniyor." | Add-Content $LogFile

        if ($dirty -and -not $Force) {
            Write-Host "[HATA] Yerel (commit'lenmemis) degisiklik var; 'main'e guvenle gecemiyorum." -ForegroundColor Red
            Write-Host "       Cozum: degisiklikleri commit/stash et, ya da yereli ATMAK icin: .\update.ps1 -Force" -ForegroundColor Cyan
            return
        }
        if (Test-MainElsewhere) {
            Write-Host "[HATA] 'main' baska bir worktree'de checkout -- bu kopya main'e gecemez." -ForegroundColor Red
            Write-Host "       O worktree'yi kapat ya da 'git worktree' ile duzelt. origin/main'e DOKUNULMADI." -ForegroundColor Cyan
            return
        }

        & $GitPath show-ref --verify --quiet refs/heads/main
        $mainExists = ($LASTEXITCODE -eq 0)
        if ($Force) {
            if ($mainExists) { & $GitPath switch -f main 2>&1 | Out-Null }
            else             { & $GitPath switch -C main --track origin/main 2>&1 | Out-Null }
        } else {
            if ($mainExists) { & $GitPath switch main 2>&1 | Out-Null }
            else             { & $GitPath switch -c main --track origin/main 2>&1 | Out-Null }
        }

        $now = (& $GitPath rev-parse --abbrev-ref HEAD 2>$null)
        if ($now) { $now = $now.Trim() }
        if ($now -ne "main") {
            Write-Host "[HATA] 'main' dalina gecilemedi -- origin/main'e MERGE EDILMEDI (veri korundu)." -ForegroundColor Red
            return
        }
        Write-Host "[OK] 'main' dalina gecildi." -ForegroundColor Green
    }

    $remoteHash = (& $GitPath rev-parse origin/main 2>$null)
    if ($remoteHash) { $remoteHash = $remoteHash.Trim() }
    $headHash = (& $GitPath rev-parse HEAD 2>$null)
    if ($headHash) { $headHash = $headHash.Trim() }

    if ($Force) {
        Write-Host "[!] -Force: yerel kod degisiklikleri ATILIYOR, origin/main'e (HEAD=main) esitleniyor." -ForegroundColor Yellow
        & $GitPath reset --hard origin/main 2>&1 | Out-Null
    } elseif (-not $remoteHash) {
        Write-Host "[UYARI] origin/main yerel ref'i yok (ilk fetch basarisiz olabilir) -- atlandi." -ForegroundColor Yellow
    } elseif ($headHash -ne $remoteHash) {
        & $GitPath pull --ff-only origin main 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[HATA] ff-only ilerleyemedi: yerel 'main' ile origin/main IRAKSAK." -ForegroundColor Red
            Write-Host "       Cozum (yereli ATAR): .\update.ps1 -Force" -ForegroundColor Cyan
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] ff-only IRAKSAK HATASI." | Add-Content $LogFile
        }
    } else {
        Write-Host "[OK] Kod zaten guncel ($($headHash.Substring(0,7)))." -ForegroundColor Cyan
    }
}

# --- 1. Web sunucusunu KESIN durdur (port 8765 + achilles-web). EGITIME dokunma. ---
function Stop-Web {
    $pidFile = Join-Path $ProjectDir ".web.pid"
    if (Test-Path $pidFile) {
        $stored = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($stored -match '^\d+$') { Stop-Process -Id ([int]$stored) -Force -ErrorAction SilentlyContinue }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    try {
        Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique |
            ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    } catch {}
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'achilles-web|achilles_web' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Stop-Web
Start-Sleep -Seconds 1

# --- 2. origin/main'e DETERMINISTIK yakinsama (parklanmis dali zorla main'e al) ---
$localHash = (& $GitPath rev-parse HEAD 2>$null).Trim()
Sync-ToMain
Show-Drift

$newHash = (& $GitPath rev-parse HEAD 2>$null).Trim()
$updated = ($newHash -ne $localHash)
if ($updated) {
    Write-Host "[OK] Kod guncellendi: $($localHash.Substring(0,7)) -> $($newHash.Substring(0,7))" -ForegroundColor Green
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Kod $($localHash.Substring(0,7)) -> $($newHash.Substring(0,7))." | Add-Content $LogFile
}

# --- 3. Bagimliliklar (WEB extra DAHIL -- duz 'uv sync' web paketlerini budar) ---
if ($updated -or $Force) {
    Write-Host "[..] Bagimliliklar esitleniyor (uv sync --extra dev)..." -ForegroundColor Gray
    & $UvPath sync --extra dev 2>&1 | Out-Null
}

# --- 4. Web'i yeniden baslat ---
$LogOut = Join-Path $LogDir "achilles-web.log"
$LogErr = Join-Path $LogDir "achilles-web-err.log"
$proc = Start-Process `
    -FilePath $UvPath `
    -ArgumentList "run", "--project", "`"$ProjectDir`"", "achilles-web" `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $LogOut `
    -RedirectStandardError  $LogErr `
    -WindowStyle Hidden `
    -PassThru
$proc.Id | Out-File (Join-Path $ProjectDir ".web.pid") -Force -Encoding ascii
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Sunucu baslatildi (PID $($proc.Id))." | Add-Content $LogFile

# --- 5. Saglik kontrolu (port dinliyor mu) ---
$ok = $false
for ($i = 0; $i -lt 15; $i++) {
    if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) { $ok = $true; break }
    Start-Sleep -Seconds 2
}
Write-Host ""
if ($ok) {
    Write-Host "[OK] Web calisiyor: http://127.0.0.1:8765" -ForegroundColor Green
    Write-Host "     >> Son halini gormek icin tarayicida: Ctrl+Shift+R (sert yenileme!)" -ForegroundColor Cyan
} else {
    Write-Host "[UYARI] Web 30 sn'de acilmadi -- log: logs\achilles-web-err.log" -ForegroundColor Yellow
}
