"""Detached launch profil-güvenliği testleri — Kademe-2 av bulgusu (profile-drop).

v5 disiplin-regresyonunun kökü: web butonu / auto_pipeline / start-train.ps1 ile
başlatılan eğitim `--profile` geçmediği için `train --run` VANİLYA varsayılanla
(assistant_only_loss=False, NEFTune=0, lr=2e-4) koşuyordu → prompt maskeleme fiilen
kapalı. Bu testler `launch()`'un güvenli-varsayılan profili ve komut kurulumunu korur.
Gerçek eğitim BAŞLATILMAZ (yalnız saf komut-kurucu ve imza denetlenir).
"""

from __future__ import annotations

import inspect

from app.training.detached_launch import _build_train_cmd, launch


def test_launch_default_profile_is_discipline_safe_local() -> None:
    """launch() varsayılan profili güvenli (maskeli) olmalı — vanilya DEĞİL."""
    default = inspect.signature(launch).parameters["profile"].default
    assert default == "discipline_safe_local"


def test_build_train_cmd_includes_profile_when_given() -> None:
    base = ["achilles"]
    cmd = _build_train_cmd(base, "a15b", 600, None, "discipline_safe_local", 0)
    assert "--profile" in cmd
    assert cmd[cmd.index("--profile") + 1] == "discipline_safe_local"
    # Temel bayraklar da bulunmalı
    assert cmd[:6] == ["achilles", "train", "--run", "--backend", "peft", "--adapter-name"]
    assert "--iterations" in cmd and cmd[cmd.index("--iterations") + 1] == "600"


def test_build_train_cmd_omits_profile_when_empty() -> None:
    """profile boş/None ise --profile EKLENMEZ (bilinçli vanilya kaçış yolu)."""
    assert "--profile" not in _build_train_cmd(["achilles"], "a", 100, None, "", 0)
    assert "--profile" not in _build_train_cmd(["achilles"], "a", 100, None, None, 0)


def test_build_train_cmd_optional_flags() -> None:
    cmd = _build_train_cmd(["achilles"], "a", 100, "Qwen/Q", "discipline_safe_local", 300)
    assert cmd[cmd.index("--base-model") + 1] == "Qwen/Q"
    assert cmd[cmd.index("--max-examples") + 1] == "300"
