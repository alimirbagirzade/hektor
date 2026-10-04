"""Araştırma planı inceleyicileri; eğitim sürücüsünün yetkilerinden bağımsızdır."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.orchestration import engines
from app.orchestration.executable import resolve_cli

# Bu profil bu sürümün sistem ayarları/araç kapıları sözleşmesine göre hazırlanmıştır.
# Yeni sürümler sessizce güvenilir sayılmaz; profil yeniden doğrulanmalıdır.
GEMINI_VERSION = "0.62.0"
NAMES = ("codex", "claude", "gemini", "ollama")


def gemini_entry() -> Path | None:
    """npm sarmalayıcısını kabukta çalıştırmadan, sürümlü Node girişini bul."""
    binary = resolve_cli("gemini")
    if not binary or not resolve_cli("node"):
        return None
    path = Path(binary)
    candidates = [path.parent / "node_modules" / "@google" / "gemini-cli"]
    candidates.extend(path.parents)
    for directory in candidates:
        try:
            meta = json.loads((directory / "package.json").read_text(encoding="utf-8"))
            if meta.get("name") != "@google/gemini-cli":
                continue
            if meta.get("version") != GEMINI_VERSION:
                return None
            entry = (directory / "dist" / "index.js").resolve()
            if entry.is_file() and entry.is_relative_to(directory.resolve()):
                return entry
        except (OSError, ValueError, AttributeError):
            continue
    return None


def blocked_reason(name: str) -> str:
    """Salt-okunur kurulum/profil kontrolü; model, CLI veya oturum başlatmaz."""
    if name not in NAMES:
        return f"Bilinmeyen araştırma motoru: {name}"
    if name == "ollama":
        return ""
    if name == "gemini":
        if not resolve_cli("gemini"):
            return "Gemini CLI kurulu değil veya sunucunun PATH'inde bulunamadı."
        if gemini_entry() is None:
            return f"Gemini araştırma profili Node.js ve @google/gemini-cli {GEMINI_VERSION} ister."
        return ""
    return engines.run_blocked_reason(name)


def describe_all() -> list[dict[str, Any]]:
    """Kurulum, seçilebilirlik ve oturum belirsizliği ayrı gösterilir."""
    rows: list[dict[str, Any]] = []
    for name in NAMES:
        reason = blocked_reason(name)
        local = name == "ollama"
        rows.append(
            {
                "name": name,
                "label": "Ollama (yerel plan incelemesi)"
                if local
                else engines.get_engine(name).label,
                "installed": None if local else bool(resolve_cli(name)),
                "selectable": not reason,
                "blocked_reason": reason,
                "connection_note": (
                    "Yerel sunucu/model erişimi ilk gerçek turda doğrulanır."
                    if local
                    else "CLI kurulumu abonelik oturumunun açık olduğunu kanıtlamaz."
                ),
                "install_hint": (
                    "Reçetedeki temel modeli Ollama'ya kur; API anahtarı gerekmez."
                    if local
                    else (
                        f"npm install -g @google/gemini-cli@{GEMINI_VERSION}; "
                        "ardından gemini ile Google hesabına giriş yap."
                        if name == "gemini"
                        else engines.get_engine(name).install_hint
                    )
                ),
            }
        )
    return rows


def prepare(
    name: str, prompt: str, model: str, seed: int, env: dict[str, str], directory: Path
) -> tuple[list[str], dict[str, str], Path | None]:
    """Yalnız bu çağrı için inceleme profili kur; kullanıcı ayarlarını değiştirme."""
    if name == "ollama":
        return (
            [sys.executable, "-m", "app.orchestration.research_review", model, str(seed), prompt],
            env,
            None,
        )
    if name != "gemini":
        return engines.build_command(name, prompt), env, None
    entry = gemini_entry()
    node = resolve_cli("node")
    if entry is None or node is None:
        raise ValueError("Gemini sürümü veya Node.js doğrulanamadı.")
    policy = directory / "deny-tools.toml"
    policy.write_text(
        '[[rule]]\ntoolName = "*"\ndecision = "deny"\npriority = 999\n', encoding="utf-8"
    )
    settings = {
        "adminPolicyPaths": [str(policy)],
        "admin": {
            "secureModeEnabled": True,
            "extensions": {"enabled": False},
            "mcp": {"enabled": False},
            "skills": {"enabled": False},
        },
        "tools": {"core": ["__hektor_no_tools__"], "discoveryCommand": "", "callCommand": ""},
        "hooksConfig": {"enabled": False},
        "skills": {"enabled": False},
        "security": {"auth": {"selectedType": "oauth-personal", "enforcedType": "oauth-personal"}},
    }
    config = directory / "settings.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    child = dict(env)
    for key in list(child):
        if key.startswith("GEMINI_") or key in {
            "GOOGLE_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "NODE_OPTIONS",
        }:
            child.pop(key)
    child["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(config)
    return (
        [node, str(entry), "-p", prompt, "--extensions", "none", "--output-format", "text"],
        child,
        directory,
    )
