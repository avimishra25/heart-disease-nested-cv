"""Configuration loading: one YAML file, one named profile."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None, profile: str = "smoke") -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if profile not in cfg["profiles"]:
        raise KeyError(f"unknown profile {profile!r}; choose from {list(cfg['profiles'])}")
    run = copy.deepcopy(cfg["profiles"][profile])
    run.setdefault("models", cfg["models"])
    run.setdefault("selectors", cfg["selectors"])
    cfg["run"] = run
    cfg["profile"] = profile
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
