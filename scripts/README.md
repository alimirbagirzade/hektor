# scripts/ — operasyon scriptleri

Achilles'i kurma, doğrulama, çalıştırma ve PR akışı scriptleri.
Linux/macOS = `.sh`, Windows = `.ps1`.

## Kurulum & çalıştırma
| Script | Ne yapar |
|--------|----------|
| `../setup.sh` / `../setup.ps1` | Tek komutla kurulum sihirbazı (model, .env, erişim modu, doğrulama, autostart). |
| `verify-install.sh` / `verify-install.ps1` | Offline doğrulama kapısı (init→status→gen-data→backtest→pytest; exit 0/1/2). |
| `install-autostart.sh` | Linux/macOS açılışta otomatik başlatma (systemd→cron→manuel). |
| `../update.sh` | Hedefte güncelleme: web durdur → git pull → uv sync → web başlat → sağlık. |

## Otomatik PR (elle GitHub web'e son)
| Script | Ne yapar |
|--------|----------|
| `open-pr.sh` / `open-pr.ps1` | push + PR + CI geçince oto squash-merge (varsayılan). `--no-merge` ile yalnız PR. |
| `setup-pr-automation.sh` | BİR KERELİK: allow_auto_merge + main branch koruması (CI zorunlu, owner muaf). `--undo` ile geri al. |

Ön koşul (bir kerelik): `gh auth login`.

## Sürekli öğrenme / araştırma
| Script | Ne yapar |
|--------|----------|
| `continuous-learning.sh` | Sürekli öğrenme döngüsü (kavra→sentezle→veri-üret). Taşınabilir süreç kontrolü (pgrep). |
| `rag-research-scan.md` / `rag-research-cycle.md` | RAG güncel-araştırma tarama/entegrasyon yordamları. |

> Detaylı kullanıcı kılavuzu: `Desktop\RAG Kaynak\Kulanım talimatı\`.
