"""Eğitim dokümanlarını (RAG / LoRA detaylı anlatım) Markdown'dan PDF'e çevirir.

Kaynak Markdown dosyaları ``docs/egitim/`` altında tutulur (sürümlenebilir, git-dostu).
Bu betik onları şık CSS ile PDF'e render eder ve hem repoya hem (varsa) masaüstündeki
"RAG Kaynak" hedef klasörlerine kopyalar.

Sürüm numarası dokümanın İÇİNDE ("## Sürüm Geçmişi" tablosu) tutulur — dosya adı sabittir,
git geçmişi tüm sürümleri korur. Yeni eğitim geliştirmesinde: Markdown'ı güncelle, sürüm
satırı ekle, bu betiği çalıştır → PDF'ler her yerde yenilenir.

Gereksinim: ``uv pip install -e ".[docs]"`` (markdown-pdf).
Kullanım:    ``uv run python scripts/gen_egitim_pdf.py``
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

_REPO = Path(__file__).resolve().parent.parent
_DOCS = _REPO / "docs" / "egitim"

# Masaüstü hedef klasörleri (makineye göre; yalnızca var olana kopyalanır).
_DESKTOP_CANDIDATES = [
    Path.home() / "Desktop" / "RAG Kaynak",
    Path.home() / "Masaüstü" / "RAG Kaynak",
]

# doküman: (markdown kaynağı, çıktı PDF adı, masaüstü alt klasörü)
_DOCS_SPEC = [
    ("RAG_EGITIM_DETAYLI_ANLATIM", "RAG EĞİTİM DETAYLIANLATIMI"),
    ("LORA_EGITIM_DETAYLI_ANLATIM", "LORA EĞİTİM DETAYLIANLATIMI"),
]

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { color: #1d4ed8; border-bottom: 2px solid #1d4ed8; padding-bottom: 4px; }
h2 { color: #0369a1; border-bottom: 1px solid #cbd5e1; padding-bottom: 2px; }
h3 { color: #334155; }
h4 { color: #475569; }
a { color: #2563eb; }
code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px;
       font-family: 'Consolas', 'Courier New', monospace; font-size: 9.5pt; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;
      padding: 8px; font-size: 9pt; }
table { border-collapse: collapse; }
th, td { border: 1px solid #cbd5e1; padding: 4px 8px; text-align: left; font-size: 10pt; }
th { background: #eff6ff; }
blockquote { border-left: 3px solid #94a3b8; color: #475569; padding-left: 10px; }
"""


def render(md_path: Path, pdf_path: Path) -> None:
    """Tek bir Markdown dosyasını PDF'e render eder (TOC + CSS)."""
    text = md_path.read_text(encoding="utf-8")
    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.add_section(Section(text, toc=True), user_css=_CSS)
    pdf.meta["title"] = md_path.stem.replace("_", " ")
    pdf.meta["author"] = "Achilles Trader AI"
    pdf.save(str(pdf_path))


def main() -> int:
    desktop_root = next((p for p in _DESKTOP_CANDIDATES if p.is_dir()), None)
    made = 0
    for stem, desktop_subdir in _DOCS_SPEC:
        md_path = _DOCS / f"{stem}.md"
        if not md_path.exists():
            print(f"ATLA  {md_path} bulunamadı")
            continue
        pdf_path = _DOCS / f"{stem}.pdf"
        render(md_path, pdf_path)
        size = pdf_path.stat().st_size
        print(f"PDF   {pdf_path}  ({size:,} bayt)")
        made += 1

        # Masaüstü hedef klasörüne kopyala (varsa).
        if desktop_root is not None:
            target_dir = desktop_root / desktop_subdir
            if target_dir.is_dir():
                dest = target_dir / pdf_path.name
                shutil.copy2(pdf_path, dest)
                print(f"KOPYA {dest}")
            else:
                print(f"NOT   masaüstü klasörü yok, atlandı: {target_dir}")

    if made == 0:
        print("Hiç doküman render edilmedi.")
        return 1
    print(f"Tamamlandı: {made} doküman.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
