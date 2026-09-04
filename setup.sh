#!/usr/bin/env bash
# Hektor Trader AI — tek komutla kurulum (macOS + Linux)
# Kullanim: bash setup.sh
set -euo pipefail

OS="$(uname -s)"
ARCH="$(uname -m)"
COLS=70

hr()   { printf '  +'; printf -- '-%.0s' $(seq 1 $COLS); printf '+\n'; }
info() { echo "  >>   $1"; }
ok()   { echo "  [OK] $1"; }
warn() { echo "  [!]  $1"; }
link() { echo "       $1"; }

echo ""
echo "  ======================================================"
echo "    Hektor Trader AI  -  Kurulum"
echo "    Platform: $OS / $ARCH"
echo "  ======================================================"
echo ""

# ==========================================================================
# MODEL SECIM MENUSU
# ==========================================================================
echo "  +--------------------------------------------------------------------+"
echo "  |  Hangi yerel yapay zeka modelini kullanmak istiyorsunuz?           |"
echo "  |  (Secimden sonra adim adim yol gosterilecektir)                    |"
echo "  +--------------------------------------------------------------------+"
echo "  |                                                                    |"
echo "  |  YEREL MODELLER  (internetsiz, ucretsiz, bilgisayarda calisir)     |"
echo "  |  (Ollama programi + model dosyasi otomatik indirilir)              |"
echo "  |  Bu proje API anahtari istemez; bulut modeli yoktur.               |"
echo "  |                                                                    |"
echo "  |  -- Qwen3 (Alibaba) --                                             |"
echo "  |  [1] qwen3:4b     ~2.5 GB disk    8 GB+ RAM   Hizli   [ONERILEN]   |"
echo "  |  [2] qwen3:8b     ~5 GB disk     16 GB+ RAM   Dengeli              |"
echo "  |  [3] qwen3:14b    ~9 GB disk     32 GB+ RAM   Guclu                |"
echo "  |  [4] qwen3:30b    ~20 GB disk    32 GB+ RAM   Cok guclu            |"
echo "  |                                                                    |"
echo "  |  -- Llama 3.1 (Meta) --                                            |"
echo "  |  [5] llama3.1:8b     ~5 GB disk  16 GB+ RAM                        |"
echo "  |  [6] llama3.1:70b   ~40 GB disk  80 GB+ RAM   Cok guclu            |"
echo "  |                                                                    |"
echo "  |  -- Mistral --                                                     |"
echo "  |  [7] mistral:7b      ~4 GB disk   8 GB+ RAM   Hizli                |"
echo "  |                                                                    |"
echo "  |  -- DeepSeek --                                                    |"
echo "  |  [8] deepseek-r1:8b     ~5 GB disk  16 GB+ RAM  Akil yurutme       |"
echo "  |  [9] deepseek-r1:14b    ~9 GB disk  32 GB+ RAM  Guclu              |"
echo "  |                                                                    |"
echo "  +--------------------------------------------------------------------+"
echo ""

read -r -p "  Seciminiz [1-9] (Enter = 1 / qwen3:4b): " CHOICE
CHOICE="${CHOICE:-1}"

# Bu proje YALNIZ yerel Ollama kullanir; API anahtari istemez.
LLM_MODEL="qwen3:4b"
MODEL_ENV="HEKTOR_LLM_MODEL"
NEED_OLLAMA=true
OLLAMA_RAM=8
OLLAMA_DSK=3

case "$CHOICE" in
  1) LLM_MODEL="qwen3:4b";        OLLAMA_RAM=8;  OLLAMA_DSK=3  ;;
  2) LLM_MODEL="qwen3:8b";        OLLAMA_RAM=16; OLLAMA_DSK=5  ;;
  3) LLM_MODEL="qwen3:14b";       OLLAMA_RAM=32; OLLAMA_DSK=9  ;;
  4) LLM_MODEL="qwen3:30b";       OLLAMA_RAM=32; OLLAMA_DSK=20 ;;
  5) LLM_MODEL="llama3.1:8b";     OLLAMA_RAM=16; OLLAMA_DSK=5  ;;
  6) LLM_MODEL="llama3.1:70b";    OLLAMA_RAM=80; OLLAMA_DSK=40 ;;
  7) LLM_MODEL="mistral:7b";      OLLAMA_RAM=8;  OLLAMA_DSK=4  ;;
  8) LLM_MODEL="deepseek-r1:8b";  OLLAMA_RAM=16; OLLAMA_DSK=5  ;;
  9) LLM_MODEL="deepseek-r1:14b"; OLLAMA_RAM=32; OLLAMA_DSK=9  ;;
  *) LLM_MODEL="qwen3:4b";        OLLAMA_RAM=8;  OLLAMA_DSK=3  ;;
esac

# ==========================================================================
# YEREL MODELLER: SISTEM KONTROLU
# ==========================================================================
if [ "$NEED_OLLAMA" = true ]; then
    echo ""
    echo "  ======================================================"
    echo "    SISTEM KONTROLU  —  $LLM_MODEL"
    echo "  ======================================================"
    echo ""

    # RAM kontrolu
    RAM_GB=0
    if [ "$OS" = "Darwin" ]; then
        RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    elif [ -f /proc/meminfo ]; then
        RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
    fi

    # Disk kontrolu
    DSK_GB=0
    DSK_GB=$(df -BG "$HOME" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo 0)

    info "Bilgisayariniz:  RAM = ${RAM_GB} GB  |  Bos disk = ~${DSK_GB} GB"
    info "Bu model icin:   RAM = ${OLLAMA_RAM} GB+  |  Disk = ~${OLLAMA_DSK} GB"

    if [ "$RAM_GB" -gt 0 ] && [ "$RAM_GB" -lt "$OLLAMA_RAM" ] 2>/dev/null; then
        warn "RAM yetersiz olabilir (${RAM_GB} GB < ${OLLAMA_RAM} GB). Model yavas calisabilir."
        read -r -p "  Devam etmek istiyor musunuz? [e/h]: " CONT
        [ "$CONT" != "e" ] && [ "$CONT" != "E" ] && exit 0
    else
        ok "RAM: ${RAM_GB} GB"
    fi

    if [ "$DSK_GB" -gt 0 ] && [ "$DSK_GB" -lt "$OLLAMA_DSK" ] 2>/dev/null; then
        warn "Disk alani yetersiz olabilir (${DSK_GB} GB bos, gerekli ~${OLLAMA_DSK} GB)."
        read -r -p "  Devam etmek istiyor musunuz? [e/h]: " CONT
        [ "$CONT" != "e" ] && [ "$CONT" != "E" ] && exit 0
    else
        ok "Disk: ${DSK_GB} GB bos"
    fi
fi

# ==========================================================================
# [1/3] UV
# ==========================================================================
echo ""
echo "[1/3] uv paket yoneticisi..."
if ! command -v uv &>/dev/null; then
    info "uv bulunamadi, kuruluyor..."
    info "Kaynak: https://astral.sh/uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
else
    ok "uv: $(uv --version)"
fi

# ==========================================================================
# [2/3] BAGIMLILIKLAR + OLLAMA
# ==========================================================================
echo "[2/3] Python kutuphaneleri yukleniyor..."
uv sync
ok "Kutuphaneler tamam"

if [ "$NEED_OLLAMA" = true ]; then
    echo ""
    echo "  ---- Ollama Kurulumu ----"
    echo ""
    info "Ollama: bilgisayarda yapay zeka modeli calistiran ucretsiz program"
    info "Resmi site: https://ollama.com"
    echo ""

    if ! command -v ollama &>/dev/null; then

        if [ "$OS" = "Darwin" ]; then
            # Homebrew yoksa kur
            if ! command -v brew &>/dev/null; then
                echo ""
                info "Homebrew bulunamadi — once Homebrew kuruluyor..."
                info "Homebrew: macOS icin ucretsiz paket yoneticisi (brew.sh)"
                info "Sudo sifreniz istenebilir, bu normaldir."
                echo ""
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                if [ "$ARCH" = "arm64" ]; then
                    eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
                    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile" 2>/dev/null || true
                else
                    eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null || true
                fi
                ok "Homebrew kuruldu"
            fi
            info "brew install ollama calistiriliyor..."
            brew install ollama
            info "Ollama servisi baslatiliyor (brew services start ollama)..."
            brew services start ollama

        elif [ "$OS" = "Linux" ]; then
            info "Resmi Linux kurulum scripti calistiriliyor..."
            info "Kaynak: https://ollama.com/install.sh"
            info "(sudo sifreniz istenebilir)"
            curl -fsSL https://ollama.com/install.sh | sh
            if command -v systemctl &>/dev/null; then
                sudo systemctl enable --now ollama 2>/dev/null || true
                ok "Ollama systemd servisi etkinlestirildi"
            else
                info "Ollama arka planda baslatiliyor..."
                ollama serve &>/tmp/ollama.log &
                sleep 3
            fi
        else
            warn "Desteklenmeyen platform: $OS"
            info "Ollama'yi elle kurun: https://ollama.com/download"
            read -r -p "  Kurulduktan sonra Enter'a basin: "
        fi
    else
        ok "Ollama zaten yuklu: $(ollama --version 2>/dev/null || echo '?')"
        if [ "$OS" = "Darwin" ]; then
            brew services start ollama 2>/dev/null || true
        fi
    fi

    # Servis hazir mi?
    info "Ollama servisi bekleniyor (max 30 saniye)..."
    READY=false
    for i in $(seq 1 15); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            READY=true; break
        fi
        echo "  ... ($i/15)"
        sleep 2
    done

    if [ "$READY" = false ]; then
        warn "Ollama servisi yanit vermiyor."
        info "Yeni bir terminal acin ve 'ollama serve' yazin."
        read -r -p "  Ollama calisinca Enter'a basin: "
    else
        ok "Ollama servisi aktif (http://localhost:11434)"
    fi

    # Model indir
    echo ""
    echo "  ---- Model Indirme ----"
    echo ""
    info "Model: $LLM_MODEL  (boyut: ~${OLLAMA_DSK} GB)"
    info "Bu islem internet hizinize gore 5-30 dakika surebilir."
    info "Bilgisayari kapatmayin, interneti kesmeyin."
    echo ""
    ollama pull "$LLM_MODEL"
    echo ""
    info "Yaziya donusturme modeli: nomic-embed-text (~270 MB)..."
    ollama pull nomic-embed-text
    ok "Tum modeller indirildi ve hazir"
fi

# ==========================================================================
# [3/3] .ENV + VERITABANI
# ==========================================================================
echo "[3/5] .env ve veritabani..."
cp -n .env.example .env 2>/dev/null || true

set_env() {
    local key="$1" val="$2"
    if grep -q "^${key}=" .env; then
        sed -i.bak "s|^${key}=.*|${key}=${val}|" .env && rm -f .env.bak
    else
        echo "${key}=${val}" >> .env
    fi
}

set_env "$MODEL_ENV" "$LLM_MODEL"

ok ".env guncellendi: yerel Ollama / $LLM_MODEL"
uv run hektor init
ok "Veritabani hazir"

# ==========================================================================
# [4/5] ERISIM MODU + GUVENLIK  (yerel mi, uzaktan/kiralik sunucu mu?)
# ==========================================================================
echo ""
echo "  ======================================================"
echo "    BU MAKINEYE NASIL ERISECEKSINIZ?"
echo "  ======================================================"
echo ""
echo "    [1] Bu bilgisayardan (yerel)        [ONERILEN, en guvenli]"
echo "        -> Web UI yalniz 127.0.0.1; sifre gerekmez."
echo ""
echo "    [2] Uzaktan (kiralik / bulut sunucu)"
echo "        -> Tarayicidan sunucu IP'si ile baglanirsiniz."
echo "        -> Web 0.0.0.0'a acilir + otomatik API TOKEN (sifre) uretilir."
echo ""
read -r -p "  Seciminiz [1-2] (Enter = 1 / yerel): " ACCESS
ACCESS="${ACCESS:-1}"

API_TOKEN=""
if [ "$ACCESS" = "2" ]; then
    API_TOKEN="$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    set_env "HEKTOR_WEB_HOST" "0.0.0.0"
    set_env "HEKTOR_API_TOKEN" "$API_TOKEN"
    ok "Uzaktan erisim: WEB_HOST=0.0.0.0 + API token uretildi (.env'e yazildi)"
    warn "Token olmadan aga acmak TEHLIKELIDIR — token uretildi, sakli tutun."
else
    set_env "HEKTOR_WEB_HOST" "127.0.0.1"
    ok "Yerel erisim: yalniz 127.0.0.1 (guvenli varsayilan)"
fi

# ==========================================================================
# [5/5] DOGRULAMA  (kullanima/autostart'tan ONCE 'gercekten kalkiyor mu')
# ==========================================================================
echo ""
echo "[5/5] Kurulum dogrulaniyor (offline duman testi: init->status->gen-data->backtest->pytest)..."
VERIFY_OK=false
if bash scripts/verify-install.sh --skip-sync; then
    VERIFY_OK=true
else
    warn "Dogrulama KALDI — yukaridaki hatayi inceleyin. Sistem 'hazir' degil."
fi

# ==========================================================================
# OPSIYONEL: ACILISTA OTOMATIK BASLATMA (autostart)
# ==========================================================================
if [ "$VERIFY_OK" = true ]; then
    echo ""
    read -r -p "  Sunucu acilista kendiliginden kalksin mi? [e/h] (Enter = h): " AS
    if [ "$AS" = "e" ] || [ "$AS" = "E" ]; then
        bash scripts/install-autostart.sh || warn "Autostart kurulamadi (elle baslatabilirsiniz)."
    fi
fi

# ==========================================================================
# TAMAMLANDI
# ==========================================================================
echo ""
echo "  ======================================================"
if [ "$VERIFY_OK" = true ]; then
    echo "    KURULUM TAMAMLANDI ve DOGRULANDI!"
else
    echo "    KURULUM BITTI (dogrulama KALDI — once onu cozun)"
fi
echo "  ======================================================"
echo ""
echo "  Uygulamayi baslatmak icin:"
echo "    uv run hektor-web"
echo ""
if [ "$ACCESS" = "2" ]; then
    SRV_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [ -z "$SRV_IP" ] && SRV_IP="<SUNUCU-IP>"
    echo "  Tarayicidan acin (baska bilgisayardan):"
    echo "    http://$SRV_IP:8765"
    echo ""
    echo "  Giris icin API token (sifre) — guvenli sakla:"
    echo "    $API_TOKEN"
    echo ""
    warn "Bulut saglayicinizda 8765 portunu (guvenlik grubu/firewall) acin;"
    warn "mumkunse yalniz kendi IP'nize izin verin. Detay: SECURITY.md"
else
    echo "  Tarayicinizda acin:"
    echo "    http://127.0.0.1:8765"
fi
echo ""
echo "  Baglanti testi (opsiyonel):  uv run hektor status"
echo ""

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    info "LoRA egitimi: Apple Silicon MLX ile destekleniyor (hizli)"
else
    info "LoRA egitimi: PEFT/CPU ile destekleniyor (yavas; bulut-GPU onerilir)"
    info "LoRA icin:  uv pip install -e \".[train-cpu]\""
    info "RAG, backtest ve formul cikarma tam olarak calisir."
fi
echo ""
