# Achilles Trader AI -- Windows Yukleyici
# Bu dosyayi HERHANGI BIR YERDEN calistirabilirsiniz.
# Her zaman dogru konuma (%USERPROFILE%\achilles) kurar.
#
# Kullanim (PowerShell):
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
#   .\install.ps1
#
# Veya tek satirda (internetten direkt):
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force; irm https://raw.githubusercontent.com/alimirbagirzade/achilles2.0/main/install.ps1 | iex

$ErrorActionPreference = "Continue"

$TARGET = Join-Path $env:USERPROFILE "achilles2.0"

# --- Mevcut kurulumu ana dala (main) yakinsatarak ve GORUNUR sekilde guncelle ---
# Eski 'git pull --ff-only 2>&1 | Out-Null' SESSIZ basarisiz oluyordu (main disi/iraksak
# dalda kullanici hicbir uyari gormeden "guncelledim" saniyordu). Artik: dali main'e
# zorla, sonucu goster, ff basarisizsa ACIK talimat ver. AUTO-MERGE/AUTO-RESET YOK.
function Update-ExistingCheckout {
    Write-Host "  >> Guncellemeler indiriliyor..." -ForegroundColor White
    Push-Location $TARGET
    try {
        $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
        if ($branch) { $branch = $branch.Trim() }

        git fetch origin main 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [UYARI] GitHub'a ulasilamadi (fetch basarisiz)." -ForegroundColor Yellow
            Write-Host "          Internet baglantinizi kontrol edip tekrar deneyin." -ForegroundColor Yellow
            return
        }

        # main disi bir dala parklanmissa once main'e gec (kullanici commit'i kaybolmaz:
        # yalniz HEAD tasinir, hicbir dal silinmez; kaydedilmemis degisiklik varsa git
        # checkout'u reddeder ve asagida stash talimati verilir).
        if ($branch -ne "main") {
            Write-Host "  [!] Aktif dal 'main' degil: '$branch' -- 'main'e geciliyor..." -ForegroundColor Yellow
            git checkout main 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [HATA] 'main' dalina gecilemedi (kaydedilmemis degisiklik olabilir)." -ForegroundColor Red
                Write-Host "         Su klasorde elle cozun:  $TARGET" -ForegroundColor Yellow
                Write-Host "         Oneri:  git stash; git checkout main; git pull --ff-only origin main" -ForegroundColor Cyan
                Write-Host "         Onceki dala donmek icin:  git checkout $branch" -ForegroundColor Cyan
                return
            }
        }

        # Sadece ileri-sarim (ff-only): asla otomatik merge/rebase yapma.
        $pullOut = (git pull --ff-only origin main 2>&1 | Out-String)
        if ($LASTEXITCODE -eq 0) {
            $head = (git rev-parse --short HEAD 2>$null)
            if ($head) { $head = $head.Trim() }
            Write-Host "  [OK] Guncel main alindi ($head)." -ForegroundColor Green
        } else {
            Write-Host "  [UYARI] Otomatik guncelleme yapilamadi (ileri-sarim degil)." -ForegroundColor Yellow
            Write-Host "          Yerel kopya origin/main ile iraksamis olabilir. git ciktisi:" -ForegroundColor Yellow
            Write-Host ("          " + $pullOut.Trim()) -ForegroundColor DarkGray
            Write-Host "          Kurulum MEVCUT kodla devam edecek. Zorla esitlemek icin (YERELI ATAR):" -ForegroundColor Cyan
            Write-Host "            cd `"$TARGET`"; .\update.ps1 -Force" -ForegroundColor Cyan
        }
    } finally {
        Pop-Location
    }
}

# --- Ikinci (yabanci) klon tespiti: scheduled-task / Registry Run hedefini oku ---
# install.ps1 her zaman $TARGET'a (~\achilles) kurar; ama makinede gecmiste BASKA bir
# konuma (orn ~\Development\achilles) kurulmus, autostart/scheduled-task'lari ORAYI isaret
# eden bir klon olabilir -> iki-kopya bolunmesi (biri guncellenir, autostart digerini sunar).
# Tespit edip UYAR (degistirme; salt-okuma).
function Find-ForeignCheckout {
    # DIKKAT: ~\achilles henuz yoksa Resolve-Path THROW ETMEZ, $null DONER. catch hic
    # calismaz; bu yuzden ACIK null-kontrolu sart (yoksa -notlike "*" her zaman $false ->
    # tespit sessizce olur, tam da en cok gerektigi anda).
    $targetFull = (Resolve-Path $TARGET -ErrorAction SilentlyContinue).Path
    if (-not $targetFull) { $targetFull = $TARGET }
    $foreign = New-Object System.Collections.Generic.HashSet[string]

    foreach ($name in @("AchillesWeb", "AchillesUpdate")) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if (-not $task) { continue }
        foreach ($act in $task.Actions) {
            foreach ($c in @($act.WorkingDirectory, $act.Arguments) | Where-Object { $_ }) {
                $m = [regex]::Match($c, '(?i)([A-Za-z]:\\[^"]*?achilles)(?:\\|"|\s|$)')
                if ($m.Success) {
                    $dir = $m.Groups[1].Value.TrimEnd('\')
                    if ($dir -and ($dir -notlike "$targetFull*")) { [void]$foreign.Add($dir) }
                }
            }
        }
    }
    $reg = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" `
        -Name "AchillesWeb" -ErrorAction SilentlyContinue
    if ($reg -and $reg.AchillesWeb) {
        $m = [regex]::Match($reg.AchillesWeb, '(?i)([A-Za-z]:\\[^"]*?achilles)(?:\\|"|\s|$)')
        if ($m.Success) {
            $dir = $m.Groups[1].Value.TrimEnd('\')
            if ($dir -and ($dir -notlike "$targetFull*")) { [void]$foreign.Add($dir) }
        }
    }

    if ($foreign.Count -gt 0) {
        Write-Host ""
        Write-Host "  [UYARI] Bu makinede BASKA bir Achilles kopyasi tespit edildi:" -ForegroundColor Yellow
        Write-Host "          Kurulum konumu : $targetFull" -ForegroundColor Yellow
        foreach ($d in $foreign) {
            $tag = if (Test-Path (Join-Path $d ".git")) { "(canli kopya)" } else { "(klasor yok / olu yol)" }
            Write-Host "          Diger kopya    : $d  $tag" -ForegroundColor Yellow
        }
        Write-Host "          Iki kopya birbirini ezebilir; autostart/03:00 guncelleme yanlis" -ForegroundColor Yellow
        Write-Host "          konumu isaret ediyor olabilir. Tek kopyada kalin." -ForegroundColor Cyan
        Write-Host "          Kurulum sonunda -Install gorevleri $targetFull'a yeniden yazar." -ForegroundColor Cyan
        Write-Host ""
    }
}

Write-Host ""
Write-Host "  ====================================================" -ForegroundColor Magenta
Write-Host "    Achilles Trader AI  --  Yukleyici" -ForegroundColor Magenta
Write-Host "  ====================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Kurulum konumu: $TARGET" -ForegroundColor Cyan
Write-Host ""

# --- Git kontrolu ---
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "  >> Git kuruluyor (winget)..." -ForegroundColor White
    winget install --id Git.Git -e --source winget --silent
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "  [HATA] Git kurulamadi." -ForegroundColor Red
        Write-Host "  Lutfen https://git-scm.com adresinden Git'i indirip kurun," -ForegroundColor Yellow
        Write-Host "  sonra bu scripti tekrar calistirin." -ForegroundColor Yellow
        Read-Host "  Cikmak icin Enter'a basin"
        exit 1
    }
    Write-Host "  [OK] Git hazir" -ForegroundColor Green
}

# --- Mevcut kurulum kontrolu ---
if (Test-Path (Join-Path $TARGET ".git")) {
    Write-Host "  [OK] Mevcut kurulum bulundu: $TARGET" -ForegroundColor Green
    Find-ForeignCheckout       # iki-kopya cakismasi uyarisi (salt-okuma)
    Update-ExistingCheckout    # main'e yakinsa + GORUNUR ff-only guncelleme
} else {
    if (Test-Path $TARGET) {
        Write-Host "  >> Eski klasor temizleniyor..." -ForegroundColor White
        Remove-Item $TARGET -Recurse -Force
    }
    Write-Host "  >> Proje indiriliyor..." -ForegroundColor White
    git clone https://github.com/alimirbagirzade/achilles2.0.git "$TARGET"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  [HATA] Indirme basarisiz. Internet baglantinizi kontrol edin." -ForegroundColor Red
        Read-Host "  Cikmak icin Enter'a basin"
        exit 1
    }
    Write-Host "  [OK] Proje indirildi: $TARGET" -ForegroundColor Green
}

# --- Kurulumu dogru dizinden baslat ---
Write-Host ""
Write-Host "  >> Kurulum baslatiliyor..." -ForegroundColor Cyan
Write-Host ""

Set-Location $TARGET
& powershell.exe -ExecutionPolicy RemoteSigned -File (Join-Path $TARGET "setup.ps1")

# Kurulum bitti -- servisi arka plana al ve Windows acilisina ekle
Write-Host ""
Write-Host "  >> Web servisi arka planda baslatiliyor..." -ForegroundColor Cyan
& powershell.exe -ExecutionPolicy RemoteSigned -File (Join-Path $TARGET "scripts\start-server.ps1") -Install
