"""Persistência do modelo treinado e de seus metadados."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def build_version(now: datetime | None = None) -> str:
    """Gera uma versão de modelo baseada em timestamp UTC (ex.: 20261101T201500Z)."""
    moment = now or datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def save_model(model: Pipeline, path: Path) -> Path:
    """Serializa o pipeline com joblib (compressão leve)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def save_json(content: dict[str, Any], path: Path) -> Path:
    """Grava um dicionário em JSON legível (metadados e relatórios)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(content, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", "utf-8")
    return path


def load_metadata(path: Path) -> dict[str, Any]:
    """Lê os metadados do modelo; retorna dicionário vazio se ausentes."""
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))
