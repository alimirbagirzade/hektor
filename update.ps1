# Hektor Trader AI -- TEK KOMUT guncelleme (KURULU makinede calistir)
#
#   .\update.ps1           -- normal: origin/main'e GUVENLI yakinsama (ff-only)
#   .\update.ps1 -Force    -- yereli AT, origin/main ile birebir esitle (salt-kopya kurulum)
#
# Yapar: web sunucusunu durdur -> 'main' dalina yakinsa (origin/main) ->
#        uv sync --extra dev --extra train-cpu -> web'i yeniden baslat -> saglik kontrolu.
#
# EGITIM KOSARKEN ATLANIR. (Eskiden burada 'EGITIME DOKUNMAZ' yaziyordu; YANLISTI:
# bu betik egitimin bagli oldugu venv'i degistirir.)
#
# OLAY + KOK NEDEN (2026-09-13'te DOGRULANDI; onceki iki teshis YANLIS adima bagliydi):
# 2026-09-09, 09-11 ve 09-12 kosularinin UCUNDE de tokenizers 0.23.2 -> 0.22.2 dustu ve
# 'import transformers' kirildi. Ucunde de 3. adimdaki acik 'uv sync' ATLANMISTI
# (update.log: kod guncellenmedi / ff-only iraksama; "Kod x -> y" satiri yok). Dusuren,
# 4. adimdaki web baslatmanin '--no-sync'SIZ 'uv run'i idi: uv run ortami kilide gore
# ORTUK senkronlar (inexact) -> taban kumedeki tokenizers kilide cekilir, extra'daki
# transformers dokunulmaz -> cift bozulur. dist-info damgalari "Sunucu baslatildi"
# satiriyla ayni dakikadadir.
#   Duzeltme: (a) web baslatma '--no-sync' (tek senkron noktasi 3. adim),
#             (b) 3. adim '--extra dev --extra train-cpu',
#             (c) pyproject 'train-cpu' surumleri v8'i egiten yigina sabit; kilit = kurulu.
#   Egitim korumasi ise yalniz egitim CANLIYKEN devreye girer; bitmis kosunun
#   DEGERLENDIRILMESINI (lora-eval) korumaz -- kok neden o degil, ama o da yetmezdi.
#
# NOT (kok-neden duzeltmesi): Bu betik artik MEVCUT dal ne olursa olsun makineyi
# 'main' dalina + origin/main'e yakinsatir. Eskiden bir feature dalina parklanmis
# makinede 'git pull origin main' origin/main'i o dala MERGE ediyor, makine asla
# main'e gecmiyordu -> "guncelleme oturmuyor". Tani icin:  uv run hektor doctor
#
# Tarayicida son halini gormek icin sonunda: Ctrl+Shift+R (sert yenileme).

param([switch]$Force)

$ErrorActionPreference = "Continue"
$ProjectDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $ProjectDir

# --- EGITIM KORUMASI ---------------------------------------------------------
# Bu betik venv'i degistirir: 3. adimda acik 'uv sync --extra dev --extra train-cpu'
# (yalniz kod guncellendiyse ya da -Force). Kosan egitim bellekteki moduller sayesinde
# ayakta kalir ama COKME SONRASI nobetci (training-watchdog.ps1) degismis ortamla
# karsilasabilir -> otomatik kurtarma sessizce basarisiz olur. Bu yuzden egitim varken
# guncelleme YAPILMAZ; atlamak hata degildir (exit 0), bir sonraki turda tekrar denenir.
# (tokenizers olayinin gercek tetikleyicisi bu adim DEGIL, 4. adimdaki ortuk senkrondu;
# bkz. basliktaki OLAY + KOK NEDEN.)
$egitimProc = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*peft_lora_train*' -or $_.CommandLine -like '*train*--run*' }
if ($egitimProc) {
    Write-Host "  [ATLANDI] LoRA egitimi kosuyor -> guncelleme yapilmadi (venv'e dokunulmaz)." -ForegroundColor Yellow
    Write-Host "            Egitim bitince elle calistir: update.ps1" -ForegroundColor DarkGray
    exit 0
}

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
    Write-Host "       Dogru klasorde calistir (Hektor deposu)." -ForegroundColor Yellow
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

# --- 1. Web sunucusunu KESIN durdur (port 8765 + hektor-web). EGITIME dokunma. ---
# Durdurma DOGRULANIR. Eskiden uc yontem de '-ErrorAction SilentlyContinue' ile sessizce
# basarisiz olabiliyordu: 2026-09-13'te HektorUpdate gorevinin (RunLevel=Highest) 08:40'ta
# baslattigi web, 20:31 kosusunca OLDURULEMEDI; yeni sunucu port 8765'e baglanamadi
# ([Errno 10048]) ama saglik kontrolu eski sureci gorup "[OK]" dedi. Artik port
# bosalmazsa betik senkrona ve yeniden baslatmaya GECMEZ, sifir-disi cikar.
$script:Hatalar = New-Object System.Collections.Generic.List[string]
function Get-PortSahibi {
    Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
}
function Stop-Web {
    $pidFile = Join-Path $ProjectDir ".web.pid"
    $hedefler = @()
    if (Test-Path $pidFile) {
        $stored = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($stored -match '^\d+$') { $hedefler += [int]$stored }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    $hedefler += @(Get-PortSahibi)
    $hedefler += @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'hektor-web|hektor_web|app\.web\.server' } |
        Select-Object -ExpandProperty ProcessId)
    foreach ($h in ($hedefler | Where-Object { $_ } | Select-Object -Unique)) {
        if (-not (Get-Process -Id $h -ErrorAction SilentlyContinue)) { continue }
        & taskkill /PID $h /T /F 2>$null | Out-Null
        if (Get-Process -Id $h -ErrorAction SilentlyContinue) {
            try { Stop-Process -Id $h -Force -ErrorAction Stop } catch {
                Write-Host "  [!] PID $h durdurulamadi: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
    }
    for ($i = 0; $i -lt 10; $i++) {
        if (-not (Get-PortSahibi)) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}
if (-not (Stop-Web)) {
    $sahip = (Get-PortSahibi) -join ','
    $msg = "Port 8765 BOSALMADI (PID $sahip) -- eski web durdurulamadi. Yukseltilmis (Yonetici) surec olabilir: HektorUpdate/HektorWeb gorevleri RunLevel=Highest ile baslatir. Senkron ve yeniden baslatma YAPILMADI."
    Write-Host "[HATA] $msg" -ForegroundColor Red
    Write-Host "       Yonetici PowerShell'den: Stop-Process -Id $sahip -Force ; sonra .\update.ps1" -ForegroundColor Gray
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] HATA: $msg" | Add-Content $LogFile
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] SONUC: HATA (web durdurulamadi)" | Add-Content $LogFile
    exit 1
}

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

# --- 3. Bagimliliklar (WEB + EGITIM extra'lari DAHIL) -------------------------
# 'train-cpu' de senkronlanir: egitim yigini (torch/transformers/peft/accelerate)
# kilidin kapsamindadir ve kilit v8'i egiten surumleri tutar -> bu senkron o yigini
# KORUR. (tokenizers olaylarinin tetikleyicisi bu adim DEGILDI; bkz. 4. adim.)
#
# Cikis kodu DENETLENIR, cikti logs\uv-sync*.log'a yazilir. 2026-09-13'te bu senkron
# exit 2 ile basarisiz oldu; cikti '| Out-Null' ile yutuldugu icin betik "[OK]" dedi.
# O sirada ayni venv'den BASKA bir python sureci (baska bir oturumun betigi) kosuyordu:
# Windows'ta yuklu .pyd silinemez, tam senkron yarida kalip venv'i YARIM birakabilir ->
# boyle bir surec varsa senkron ERTELENIR (sonraki turda tekrar denenir).
$SyncOut = Join-Path $LogDir "uv-sync.log"
$SyncErr = Join-Path $LogDir "uv-sync-err.log"
if ($updated -or $Force) {
    $venvDir = Join-Path $ProjectDir ".venv"
    $kullananlar = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath -like "$venvDir*" })
    if ($kullananlar.Count -gt 0) {
        $ids = ($kullananlar | ForEach-Object { $_.ProcessId }) -join ','
        $msg = "Senkron ERTELENDI: venv'i kullanan baska python sureci var (PID $ids); yuklu .pyd dosyalari kilitli."
        Write-Host "[UYARI] $msg" -ForegroundColor Yellow
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] UYARI: $msg" | Add-Content $LogFile
        $script:Hatalar.Add("senkron ertelendi (PID $ids)")
    } else {
        Write-Host "[..] Bagimliliklar esitleniyor (uv sync --extra dev --extra train-cpu)..." -ForegroundColor Gray
        $sp = Start-Process -FilePath $UvPath -ArgumentList "sync --extra dev --extra train-cpu" -WorkingDirectory $ProjectDir -RedirectStandardOutput $SyncOut -RedirectStandardError $SyncErr -NoNewWindow -Wait -PassThru
        if ($sp.ExitCode -ne 0) {
            $son = (Get-Content $SyncErr -Tail 5 -ErrorAction SilentlyContinue) -join ' | '
            $msg = "uv sync BASARISIZ (cikis=$($sp.ExitCode)): $son"
            Write-Host "[HATA] $msg" -ForegroundColor Red
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] HATA: $msg" | Add-Content $LogFile
            $script:Hatalar.Add("uv sync cikis=$($sp.ExitCode)")
        } else {
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Senkron OK." | Add-Content $LogFile
        }
    }
}

# --- 4. Web'i yeniden baslat ---
# '--no-sync' ZORUNLU. 'uv run' varsayilan olarak ortami kilide gore ORTUK senkronlar
# (inexact: fazlaligi silmez ama kilitli surumleri ayarlar). 2026-09-09, 09-11 ve
# 09-12 kosularinin UCUNDE de 3. adimdaki acik 'uv sync' ATLANMISTI (kod guncellenmedi
# / iraksama) -- tokenizers'i 0.23.2 -> 0.22.2 dusuren BU ortuk senkrondu. Extra'daki
# transformers dokunulmadan kaldigi icin cift bozuldu. Tek mesru senkron noktasi
# 3. adimdir; start-server.ps1 de ayni sebeple '--no-sync' kullanir.
$LogOut = Join-Path $LogDir "hektor-web.log"
$LogErr = Join-Path $LogDir "hektor-web-err.log"
$proc = Start-Process `
    -FilePath $UvPath `
    -ArgumentList "run", "--no-sync", "--project", "`"$ProjectDir`"", "hektor-web" `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $LogOut `
    -RedirectStandardError  $LogErr `
    -WindowStyle Hidden `
    -PassThru
$proc.Id | Out-File (Join-Path $ProjectDir ".web.pid") -Force -Encoding ascii
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Sunucu baslatildi (PID $($proc.Id))." | Add-Content $LogFile

# --- 5. Saglik kontrolu: port BIZIM baslattigimiz surecte mi ---
# Eskiden yalniz "port dinliyor mu"ya bakiliyordu: 2026-09-13'te yeni sunucu
# baglanamayip kapandi, port hala ESKI surecteydi ve betik "[OK]" dedi.
function Test-BizimSurec([int]$Sahip, [int]$Kok) {
    $cur = $Sahip
    for ($d = 0; ($d -lt 6) -and ($cur -gt 0); $d++) {
        if ($cur -eq $Kok) { return $true }
        $w = Get-CimInstance Win32_Process -Filter "ProcessId=$cur" -ErrorAction SilentlyContinue
        if (-not $w) { return $false }
        $cur = [int]$w.ParentProcessId
    }
    return $false
}
$ok = $false
$sahip = $null
for ($i = 0; $i -lt 15; $i++) {
    $sahip = Get-PortSahibi | Select-Object -First 1
    if ($sahip -and (Test-BizimSurec ([int]$sahip) $proc.Id)) { $ok = $true; break }
    if ($proc.HasExited) { break }
    Start-Sleep -Seconds 2
}
Write-Host ""
if ($ok) {
    Write-Host "[OK] Web calisiyor: http://127.0.0.1:8765 (PID $sahip)" -ForegroundColor Green
    Write-Host "     >> Son halini gormek icin tarayicida: Ctrl+Shift+R (sert yenileme!)" -ForegroundColor Cyan
} else {
    if ($sahip) { $msg = "Port 8765 BASKA bir surecte (PID $sahip); baslatilan sunucu (PID $($proc.Id)) baglanamadi." } else { $msg = "Web 30 sn'de acilmadi." }
    Write-Host "[HATA] $msg -- log: logs\hektor-web-err.log" -ForegroundColor Red
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] HATA: $msg" | Add-Content $LogFile
    $script:Hatalar.Add("web")
}

# --- 6. Sonuc: yalniz her adim basariliysa 0 (zamanlanmis gorevin sonucu gercegi yansitsin) ---
if ($script:Hatalar.Count -gt 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] SONUC: HATA ($($script:Hatalar -join '; '))" | Add-Content $LogFile
    exit 1
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] SONUC: OK" | Add-Content $LogFile
exit 0
