#!/usr/bin/env bash
# Achilles Trader AI -- Linux/macOS autostart kurucu (web sunucusu acilista kalksin).
#
# Windows'ta autostart .vbs + Task Scheduler ile (setup.ps1/start-server.ps1) yapilir;
# bu script onun Linux/macOS karsiligidir. Kiralik/bulut CPU sunucusu yeniden
# baslatildiginda `achilles-web`'in kendiliginden ayaga kalkmasini saglar.
#
# Strateji (ilk uygun olan secilir):
#   Linux : systemd KULLANICI servisi (enable-linger ile reboot'ta da calisir)
#           -> yoksa cron @reboot
#   macOS : launchd (Makefile web-start ile uyumlu) -> yoksa cron @reboot
#   diger : yalnizca elle baslatma komutunu yazdirir
#
# Kullanim:  bash scripts/install-autostart.sh
#            bash scripts/install-autostart.sh --uninstall
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OS="$(uname -s)"
ACTION="${1:-install}"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
SVC_NAME="achilles-web"

info() { echo "  >>   $1"; }
ok()   { echo "  [OK] $1"; }
warn() { echo "  [!]  $1"; }

install_systemd_user() {
    local unit_dir="$HOME/.config/systemd/user"
    mkdir -p "$unit_dir"
    cat > "$unit_dir/$SVC_NAME.service" <<EOF
[Unit]
Description=Achilles Trader AI web sunucusu
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$UV run --project $PROJECT_DIR achilles-web
Restart=on-failure
RestartSec=5
Environment=UV_NO_SYNC=1

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now "$SVC_NAME.service"
    # reboot'ta kullanici oturumu olmadan da calissin (kiralik sunucu icin sart):
    if command -v loginctl >/dev/null 2>&1; then
        loginctl enable-linger "$(whoami)" 2>/dev/null && ok "enable-linger acik (reboot'ta calisir)" \
            || warn "enable-linger acilamadi (sudo gerekebilir; oturum acikken calisir)"
    fi
    ok "systemd kullanici servisi kuruldu: $SVC_NAME"
    info "Durum:  systemctl --user status $SVC_NAME"
    info "Durdur: systemctl --user stop $SVC_NAME"
}

install_cron_reboot() {
    local cmd="cd $PROJECT_DIR && UV_NO_SYNC=1 $UV run achilles-web >> $PROJECT_DIR/logs/web-autostart.log 2>&1"
    local line="@reboot $cmd"
    ( crontab -l 2>/dev/null | grep -v "achilles-web" ; echo "$line" ) | crontab -
    ok "cron @reboot kuruldu (autostart)"
    info "Kaldir: crontab -e -> 'achilles-web' satirini sil"
}

uninstall_all() {
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user disable --now "$SVC_NAME.service" 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/$SVC_NAME.service"
        systemctl --user daemon-reload 2>/dev/null || true
    fi
    crontab -l 2>/dev/null | grep -v "achilles-web" | crontab - 2>/dev/null || true
    ok "Autostart kaldirildi (systemd + cron)"
}

if [ "$ACTION" = "--uninstall" ]; then
    uninstall_all
    exit 0
fi

mkdir -p "$PROJECT_DIR/logs"
echo ""
echo "  Achilles autostart kurulumu ($OS)"

case "$OS" in
  Linux)
    if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
        install_systemd_user
    elif command -v crontab >/dev/null 2>&1; then
        warn "systemd kullanici servisi yok -> cron @reboot kullaniliyor"
        install_cron_reboot
    else
        warn "Ne systemd ne cron bulundu. Elle baslat:  $UV run achilles-web"
    fi
    ;;
  Darwin)
    if command -v crontab >/dev/null 2>&1; then
        install_cron_reboot
        info "macOS launchd alternatifi:  make web-start  (com.achilles.web.plist)"
    else
        warn "Elle baslat:  $UV run achilles-web   (veya: make web-start)"
    fi
    ;;
  *)
    warn "Bu platformda otomatik autostart desteklenmiyor."
    info "Elle baslat:  $UV run achilles-web"
    ;;
esac
