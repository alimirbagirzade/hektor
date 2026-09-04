# Güvenlik (SECURITY.md)

Achilles Trader AI **yerel-öncelikli bir araştırma aracıdır**. Web arayüzü,
çekirdek motoru saran ince bir katmandır. Tasarım ilkesi: **varsayılan olarak
dışarı kapalı, katmanlı savunma.**

## Tehdit modeli

| Varlık | Tehdit | Savunma |
|--------|--------|---------|
| Yerel makine | İstemeden ağa açılma | Varsayılan bind `127.0.0.1`; `0.0.0.0` değil |
| API uçları | Yetkisiz erişim (ağa açılırsa) | İsteğe bağlı bearer token (`ACHILLES_API_TOKEN`), sabit-zamanlı karşılaştırma |
| Upload (PDF) | Kötü amaçlı/sahte dosya | Uzantı + `%PDF-` sihirli bayt + boyut limiti |
| Upload (CSV) | Sahte/ikili veri, enjeksiyon | Uzantı + metin çözme + başlık sniff (open/high/low/close) + boyut limiti; kural çalıştırma YOK |
| Dosya sistemi | Path traversal (`../`) | `sanitize_filename` + `safe_destination` (hedef dizin doğrulaması) |
| Tarayıcı | XSS / clickjacking | CSP (`script-src 'self'`, inline yok), `X-Frame-Options: DENY`, `nosniff` |
| Servis | DoS / brute force | IP başına hız sınırı (kayan pencere) |
| Veritabanı | SQL injection | SQLAlchemy ORM (parametreli sorgular) |
| Strateji girdisi | Kod enjeksiyonu | `eval`/`exec` YOK; kurallar yalnız güvenli regex ile parse edilir |
| Sırlar | Kod içinde sızıntı | `.env` (Git-ignore); kodda hardcoded sır yok |
| Host başlığı | Host-header / DNS-rebinding | `TrustedHostMiddleware` — `ACHILLES_TRUSTED_HOSTS` ayarlıysa zorunlu |
| Yükleme uçları | Disk doldurma / DoS | Boyut limiti + ayrı sıkı hız sınırı (`ACHILLES_UPLOAD_RATE_LIMIT_PER_MIN`) |
| Bağımlılıklar | Bilinen CVE'li paket | `make audit` (pip-audit) + düzenli güncelleme |
| Git geçmişi | Kazara sır commit'i | pre-commit: gitleaks + detect-private-key |
| Aktarım | Token'ın düz HTTP'de açık gitmesi | TLS (reverse proxy) + `ACHILLES_HSTS_ENABLED=true` |

## Varsayılan davranış (güvenli)

- Sunucu yalnız **localhost**'a bağlanır (`ACHILLES_WEB_HOST=127.0.0.1`).
- Token boşsa kimlik doğrulama atlanır — **çünkü dışarıdan erişilemez.**
- Tüm yanıtlara güvenlik başlıkları eklenir.
- Yüklenen PDF'ler `data/papers/raw_pdf/`, CSV'ler `data/market/raw/` içinde temizlenmiş adla saklanır (path-traversal korumalı).

## Sunucuyu ağa / uzağa (kiralık CPU) açacaksan — sertleştirme checklist

> Lokal kullanımda bunların hiçbiri gerekmez. **Yalnız ağa açarken** (örn. uzaktan
> kiralanan CPU) sırayla uygula. En güvenlisi: sunucuyu hiç internete açmamak.

**P0 — açmadan ÖNCE (zorunlu):**

1. **Güçlü token ata** (yoksa auth tamamen KAPALIDIR):
   ```bash
   # .env
   ACHILLES_API_TOKEN=$(openssl rand -hex 32)
   ```
   Tüm `/api/*` istekleri `Authorization: Bearer <token>` gerektirir.
2. **Doğrudan internete AÇMA.** Tercih sırası:
   - **VPN / SSH tüneli** (Tailscale, WireGuard, `ssh -L 8765:127.0.0.1:8765 …`) —
     `ACHILLES_WEB_HOST=127.0.0.1` kalır, dışarıdan yalnız tünelle erişilir. **En iyisi.**
   - Olmazsa **reverse proxy (Caddy/nginx) + TLS** + güvenlik duvarında **tek IP allowlist**.
   - `ACHILLES_WEB_HOST=0.0.0.0` + public IP = **en kötü senaryo**, yapma.
3. **TLS (HTTPS) şart** — token düz HTTP'de açık gider. Caddy otomatik TLS en kolayı.
   TLS varsa: `ACHILLES_HSTS_ENABLED=true`.

**P1 — uygulama knobları:**

4. **Host-header koruması:** `ACHILLES_TRUSTED_HOSTS=alanadi.com,127.0.0.1`
   (boşken kısıt yok; ayarlanınca `TrustedHostMiddleware` devreye girer).
5. **Hız sınırlarını sıkılaştır:** `ACHILLES_RATE_LIMIT_PER_MIN` (ağda 120 yüksek),
   yükleme için ayrıca `ACHILLES_UPLOAD_RATE_LIMIT_PER_MIN`.
6. `ACHILLES_CORS_ORIGINS`'i kendi alan adına daralt; `ACHILLES_MAX_UPLOAD_MB`'yi makul tut.

**P2 — operasyon / hijyen:**

7. **Bağımlılık taraması:** `make audit` (pip-audit) — bilinen CVE'leri yakala, düzenli güncelle.
8. **Sır taraması:** `uv run pre-commit install` → her commit'te gitleaks + private-key kontrolü.
9. `storage/` (DB + adapter registry) **yedeği**; süreç en az yetkili kullanıcıyla; erişim logu izle.

## Bilinçli kapsam dışı (MVP)

- Canlı borsa bağlantısı / emir gönderme **yok**.
- Çoklu kullanıcı / rol yönetimi yok (tek kullanıcı yerel araç).
- Kullanıcı hesapları / parola saklama yok (yalnız tek paylaşılan token).

## İlgili kod

- `app/web/security.py` — auth, hız sınırı, başlıklar, upload doğrulama
- `app/web/server.py` — middleware + uçlar
- `app/trading/strategy_ir.py` — güvenli kural ayrıştırma (kod yürütme yok)

## Açık bildirimi

Bu yazılım **eğitim/araştırma** amaçlıdır, yatırım tavsiyesi değildir.
Bir güvenlik açığı bulursan lütfen sorumlu biçimde bildir.
