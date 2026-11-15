"""Shared utilities for the recommendation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any] | None = None) -> dict[str, Path]:
    config = config or load_config()
    paths = {}
    for key, rel in config["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def project_root() -> Path:
    return ROOT
