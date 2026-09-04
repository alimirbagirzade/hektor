# Hektor Trader AI -- Windows Kurulum Scripti
# Gereksinim: Windows 10/11, PowerShell 5.1+, internet baglantisi
# Kullanim: PowerShell'i YONETICI olarak ac -> cd proje_klasoru -> .\setup.ps1

param([switch]$SkipOllama, [switch]$SkipModels)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step($n, $msg) { Write-Host "`n[$n/3] $msg" -ForegroundColor Cyan }
function Write-OK($msg)       { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)     { Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Write-Info($msg)     { Write-Host "  >>   $msg" -ForegroundColor White }
function Write-Link($msg)     { Write-Host "       $msg" -ForegroundColor DarkCyan }
function Write-Sep           { Write-Host "  +------------------------------------------------------------------+" -ForegroundColor DarkGray }

Write-Host ""
Write-Host "  ====================================================" -ForegroundColor Magenta
Write-Host "    Hektor Trader AI  -  Windows Kurulum" -ForegroundColor Magenta
Write-Host "  ====================================================" -ForegroundColor Magenta
Write-Host ""

# ==========================================================================
# DIZIN KONTROLU — sistem klasorlerinde calismayi otomatik duzelt
# ==========================================================================
# Proje klasoru kontrolu -- pyproject.toml ve app/ olmadan calisma
$_scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} elseif ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $PWD.Path
}

$_hasProject = (Test-Path (Join-Path $_scriptDir "pyproject.toml")) -and
               (Test-Path (Join-Path $_scriptDir "app"))

if (-not $_hasProject) {
    Write-Host ""
    Write-Host "  ================================================================" -ForegroundColor Red
    Write-Host "   HATA: setup.ps1 yanlis klasorden calistirildi!" -ForegroundColor Red
    Write-Host "  ================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Mevcut konum  : $_scriptDir" -ForegroundColor Yellow
    Write-Host "  Beklenen dosya: pyproject.toml (bu klasorde yok)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Dogru kurulum icin asagidaki komutu PowerShell'e kopyalayip calistirin:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force; irm https://raw.githubusercontent.com/alimirbagirzade/hektor/main/install.ps1 | iex" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Bu komut projeyi otomatik olarak dogru konuma indirir ve kurar." -ForegroundColor White
    Write-Host "  ================================================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "  Enter'a basin ve bu pencereyi kapatin"
    exit 1
}

# CWD'yi proje kokune sabitle (her kosulda tutarli calissin)
Set-Location $_scriptDir

# ==========================================================================
# MODEL SECIM MENUSU
# ==========================================================================
Write-Host "  +------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "  |  Hangi yerel yapay zeka modelini kullanmak istiyorsunuz?         |" -ForegroundColor Cyan
Write-Host "  |  (Model seciminden sonra size adim adim yol gosterilecektir)     |" -ForegroundColor Cyan
Write-Host "  +------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "  |                                                                  |" -ForegroundColor Cyan
Write-Host "  |  YEREL MODELLER  (internet gerekmez, ucretsiz, bilgisayarda)     |" -ForegroundColor Yellow
Write-Host "  |  (Ollama uygulamasi + model dosyasi otomatik indirilir)          |" -ForegroundColor DarkGray
Write-Host "  |  Bu proje API anahtari istemez; bulut modeli yoktur.             |" -ForegroundColor DarkGray
Write-Host "  |                                                                  |" -ForegroundColor Cyan
Write-Host "  |  -- Qwen3 (Alibaba) --                                           |" -ForegroundColor White
Write-Host "  |  [1] qwen3:4b     ~2.5 GB disk    8 GB+ RAM  Hizli  [ONERILEN]   |" -ForegroundColor Green
Write-Host "  |  [2] qwen3:8b     ~5 GB disk     16 GB+ RAM  Dengeli             |" -ForegroundColor White
Write-Host "  |  [3] qwen3:14b    ~9 GB disk     32 GB+ RAM  Guclu               |" -ForegroundColor White
Write-Host "  |  [4] qwen3:30b    ~20 GB disk    32 GB+ RAM  Cok guclu           |" -ForegroundColor White
Write-Host "  |                                                                  |" -ForegroundColor Cyan
Write-Host "  |  -- Llama 3.1 (Meta) --                                          |" -ForegroundColor White
Write-Host "  |  [5] llama3.1:8b     ~5 GB disk  16 GB+ RAM                      |" -ForegroundColor White
Write-Host "  |  [6] llama3.1:70b   ~40 GB disk  80 GB+ RAM  Cok guclu           |" -ForegroundColor White
Write-Host "  |                                                                  |" -ForegroundColor Cyan
Write-Host "  |  -- Mistral --                                                   |" -ForegroundColor White
Write-Host "  |  [7] mistral:7b      ~4 GB disk   8 GB+ RAM  Hizli               |" -ForegroundColor White
Write-Host "  |                                                                  |" -ForegroundColor Cyan
Write-Host "  |  -- DeepSeek --                                                  |" -ForegroundColor White
Write-Host "  |  [8] deepseek-r1:8b     ~5 GB disk  16 GB+ RAM  Akil yurutme     |" -ForegroundColor White
Write-Host "  |  [9] deepseek-r1:14b    ~9 GB disk  32 GB+ RAM  Guclu            |" -ForegroundColor White
Write-Host "  |                                                                  |" -ForegroundColor Cyan
Write-Host "  +------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "  Seciminiz [1-9] (Enter = 1 / qwen3:4b)"
if ($choice -eq "") { $choice = "1" }

# Bu proje YALNIZ yerel Ollama kullanir; API anahtari istemez.
$llmModel    = "qwen3:4b"
$modelEnv    = "HEKTOR_LLM_MODEL"
$needOllama  = $true
$ollamaRamGB = 8
$ollamaDskGB = 3

switch ($choice) {
    "1" { $llmModel="qwen3:4b";        $ollamaRamGB=8;  $ollamaDskGB=3  }
    "2" { $llmModel="qwen3:8b";        $ollamaRamGB=16; $ollamaDskGB=5  }
    "3" { $llmModel="qwen3:14b";       $ollamaRamGB=32; $ollamaDskGB=9  }
    "4" { $llmModel="qwen3:30b";       $ollamaRamGB=32; $ollamaDskGB=20 }
    "5" { $llmModel="llama3.1:8b";     $ollamaRamGB=16; $ollamaDskGB=5  }
    "6" { $llmModel="llama3.1:70b";    $ollamaRamGB=80; $ollamaDskGB=40 }
    "7" { $llmModel="mistral:7b";      $ollamaRamGB=8;  $ollamaDskGB=4  }
    "8" { $llmModel="deepseek-r1:8b";  $ollamaRamGB=16; $ollamaDskGB=5  }
    "9" { $llmModel="deepseek-r1:14b"; $ollamaRamGB=32; $ollamaDskGB=9  }
    default { $llmModel="qwen3:4b"; $ollamaRamGB=8; $ollamaDskGB=3 }
}

# ==========================================================================
# YEREL MODELLER: SISTEM KONTROLU
# ==========================================================================
if ($needOllama) {
    Write-Host ""
    Write-Host "  ====================================================" -ForegroundColor Cyan
    Write-Host "    SISTEM KONTROLU  --  $llmModel" -ForegroundColor Cyan
    Write-Host "  ====================================================" -ForegroundColor Cyan
    Write-Host ""

    $ramGB  = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $disk   = Get-PSDrive -Name C | Select-Object -ExpandProperty Free
    $diskGB = [math]::Round($disk / 1GB)

    Write-Info "Bilgisayariniz:  RAM = $ramGB GB  |  C: bos disk = $diskGB GB"
    Write-Info "Bu model icin:   RAM = $ollamaRamGB GB+  |  Disk = ~$ollamaDskGB GB"

    if ($ramGB -lt $ollamaRamGB) {
        Write-Warn "RAM yetersiz olabilir ($ramGB GB < $ollamaRamGB GB). Model yavas calisabilir."
        $cont = Read-Host "  Devam etmek istiyor musunuz? [E/H]"
        if ($cont -ne "E" -and $cont -ne "e") { exit 0 }
    } else {
        Write-OK "RAM yeterli ($ramGB GB)"
    }

    if ($diskGB -lt ($ollamaDskGB + 2)) {
        Write-Warn "Disk alani yetersiz olabilir ($diskGB GB bos, gerekli ~$ollamaDskGB GB)."
        $cont = Read-Host "  Devam etmek istiyor musunuz? [E/H]"
        if ($cont -ne "E" -and $cont -ne "e") { exit 0 }
    } else {
        Write-OK "Disk alani yeterli ($diskGB GB bos)"
    }
}

# ==========================================================================
# [1/3] PYTHON
# ==========================================================================
Write-Step 1 "Python 3.12 kurulumu..."
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Info "Python bulunamadi. Otomatik kurulum deneniyor..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Info "winget ile Python 3.12 kuruluyor..."
        winget install --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        $py = Get-Command python -ErrorAction SilentlyContinue
    }
    if (-not $py) {
        Write-Warn "Python otomatik kurulamadi. Elle kurulum gerekiyor:"
        Write-Host ""
        Write-Info "1. Tarayici aciliyor: python.org/downloads"
        Write-Info "2. 'Download Python 3.12' butonuna tiklayin"
        Write-Info "3. Indirilen dosyayi calistirin"
        Write-Info "4. KURULUMDA: 'Add Python to PATH' kutusunu ISARETLE (cok onemli!)"
        Write-Info "5. 'Install Now' tiklayin, bitmesini bekleyin"
        Write-Info "6. Bu pencereyi KAPAT, yeni PowerShell ac (Yonetici), tekrar calistir"
        Start-Process "https://www.python.org/downloads/"
        exit 1
    }
}
Write-OK "Python: $(python --version 2>&1)"

# ==========================================================================
# [2/3] UV + BAGIMLILIKLAR + OLLAMA (gerekirse)
# ==========================================================================
Write-Step 2 "Kurulum devam ediyor..."

# uv
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Info "uv paket yoneticisi kuruluyor (python.org/pypa alternatifi)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $uvPath = "$env:USERPROFILE\.local\bin"
    if (Test-Path $uvPath) { $env:PATH = "$uvPath;$env:PATH" }
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Warn "uv PATH'e eklenemedi. Bu pencereyi kapat, yeni PowerShell ac, tekrar calistir."
        exit 1
    }
}
Write-OK "uv: $(uv --version)"

Write-Info "Python kutuphane bagimliliklar yukleniyor..."
uv sync
Write-OK "Kutuphaneler tamam"

# PEFT / LoRA egitim paketleri (Windows icin)
Write-Host ""
$installPeft = Read-Host "  LoRA model egitimi icin ek paketler kurulsun mu? (~2 GB) [E/H] (Enter = H)"
if ($installPeft -eq "E" -or $installPeft -eq "e") {
    Write-Info "PEFT paketleri kuruluyor (torch, transformers, peft, datasets)..."
    Write-Info "Bu islem 5-15 dakika surebilir, internet hizinize gore degisir."
    # Once sadece torch (PyTorch CPU index) -- transformers/peft PyPI'da yok
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    # Diger paketler normal PyPI'dan
    uv pip install transformers peft datasets accelerate
    Write-OK "LoRA egitim paketleri kuruldu (CPU modu)"
    Write-Warn "NVIDIA GPU icin: uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
} else {
    Write-Info "LoRA paketleri atlandı. Sonradan kurmak icin:"
    Write-Host "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Yellow
    Write-Host "  uv pip install transformers peft datasets accelerate" -ForegroundColor Yellow
}

# Ollama bolumu
if ($needOllama -and -not $SkipOllama) {

    Write-Host ""
    Write-Host "  ---- Ollama Kurulumu ----" -ForegroundColor Cyan
    Write-Host ""
    Write-Info "Ollama: bilgisayarda yapay zeka modeli calistiran ucretsiz program"
    Write-Info "Resmi site: https://ollama.com"
    Write-Host ""

    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {

        $installed = $false

        # Yontem 1: winget
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-Info "Yontem 1: winget ile sessiz kurulum deneniyor..."
            try {
                winget install --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
                $installed = $true
                Write-OK "winget ile Ollama kuruldu"
            } catch {
                Write-Warn "winget kurulumu basarisiz, alternatif deneniyor..."
            }
        }

        # Yontem 2: Resmi exe
        if (-not $installed) {
            $installer = "$env:TEMP\OllamaSetup.exe"
            Write-Info "Yontem 2: Resmi kurulum dosyasi indiriliyor (~500 MB)..."
            Write-Info "Kaynak: https://ollama.com/download/OllamaSetup.exe"
            try {
                Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
                    -OutFile $installer -UseBasicParsing -TimeoutSec 600
                Write-Info "Kuruluyor... Acilan pencerede 'Install' tiklayin."
                Start-Process -FilePath $installer -ArgumentList "/S" -Wait
                $installed = $true
                Write-OK "Ollama kuruldu"
            } catch {
                Write-Warn "Otomatik indirme basarisiz oldu."
                Write-Host ""
                Write-Host "  ELLE KURULUM GEREKIYOR:" -ForegroundColor Yellow
                Write-Info "1. Tarayici aciliyor: ollama.com/download"
                Write-Info "2. 'Download for Windows' butonuna tiklayin"
                Write-Info "3. Indirilen 'OllamaSetup.exe' dosyasini calistirin"
                Write-Info "4. Acilan sihirbazda 'Install' > 'Finish' tiklayin"
                Write-Info "5. Kurulum bittikten sonra ENTER'a basin"
                Start-Process "https://ollama.com/download"
                Read-Host "`n  Ollama kurulumunu tamamladiktan sonra Enter'a basin"
            }
        }

        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        Start-Sleep -Seconds 3
        $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollamaCmd) {
            Write-Warn "Ollama kuruldu ancak bu oturumda taninamadi."
            Write-Info "Bu pencereyi kapat, yeni PowerShell (Yonetici) ac, tekrar calistir."
            exit 1
        }
    } else {
        Write-OK "Ollama zaten yuklu: $(ollama --version 2>&1)"
    }

    # Servis baslat
    Write-Info "Ollama arka plan servisi baslatiliyor..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3

    Write-Info "Servis hazir olana kadar bekleniyor (max 30 saniye)..."
    $ready = $false
    for ($i = 1; $i -le 15; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 2
    }

    if (-not $ready) {
        Write-Warn "Ollama servisi yanit vermiyor."
        Write-Info "Baska bir terminal acin, 'ollama serve' yazin, Enter'a basin."
        Read-Host "  Ollama calisir hale gelince burada Enter'a basin"
    } else {
        Write-OK "Ollama servisi aktif (http://localhost:11434)"
    }

    # Model indir
    if (-not $SkipModels) {
        Write-Host ""
        Write-Host "  ---- Model Indirme ----" -ForegroundColor Cyan
        Write-Host ""
        Write-Info "Model: $llmModel  (boyut: ~$ollamaDskGB GB)"
        Write-Info "Bu islem internet hizinize gore 5-30 dakika surebilir."
        Write-Info "Bilgisayari kapatmayin, interneti kesmeyin."
        Write-Host ""
        ollama pull $llmModel
        Write-Host ""
        Write-Info "Yaziya donusturme modeli indiriliyor: nomic-embed-text (~270 MB)..."
        ollama pull nomic-embed-text
        Write-OK "Tum modeller indirildi ve hazir"
    }
}

# ==========================================================================
# [3/3] .ENV + VERITABANI
# ==========================================================================
Write-Step 3 ".env yapilandirmasi ve veritabani olusturuluyor..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-OK ".env dosyasi olusturuldu"
}

$envContent = Get-Content ".env"

function Set-EnvLine($lines, $key, $val) {
    if ($lines -match "^${key}=") { return $lines -replace "^${key}=.*", "${key}=${val}" }
    return $lines + "${key}=${val}"
}

$envContent = Set-EnvLine $envContent $modelEnv $llmModel
$envContent | Set-Content ".env"
Write-OK ".env guncellendi: yerel Ollama / $llmModel"

Write-Info "Veritabani ve klasorler olusturuluyor..."
uv run hektor init
Write-OK "Veritabani hazir"

# ==========================================================================
# TAMAMLANDI
# ==========================================================================
Write-Host ""
Write-Host "  ====================================================" -ForegroundColor Green
Write-Host "    KURULUM TAMAMLANDI!" -ForegroundColor Green
Write-Host "  ====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Uygulamayi baslatmak icin:" -ForegroundColor White
Write-Host ""
Write-Host "    uv run hektor-web" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Tarayicinizda acin:" -ForegroundColor White
Write-Host "    http://127.0.0.1:8765" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Baglanti testi (opsiyonel):" -ForegroundColor DarkGray
Write-Host "    uv run hektor status" -ForegroundColor DarkGray
Write-Host ""
Write-Warn "NOT: LoRA egitim modlari -- macOS Apple Silicon: MLX (hizli), Windows/Linux: PEFT/CPU."
Write-Host "     Windows'ta tum ozellikler calismaktadir: RAG, backtest, formul cikarma, PEFT LoRA." -ForegroundColor White
Write-Host ""
