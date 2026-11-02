"""Persistência dos artefatos intermediários de uma execução do pipeline.

Cada etapa do pipeline roda isolada (por exemplo, como task do Airflow, em
outro processo), então os resultados trafegam por arquivos dentro do
diretório da execução — nunca por memória compartilhada.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from triage.data.dataset import TextDataset

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class RunPaths:
    """Arquivos produzidos por uma execução do pipeline."""

    root: Path

    @property
    def splits_dir(self) -> Path:
        """Diretório com os conjuntos de treino, validação e teste."""
        return self.root / "splits"

    @property
    def data_summary(self) -> Path:
        """Resumo da deduplicação e da divisão."""
        return self.root / "data_summary.json"

    @property
    def decision(self) -> Path:
        """Regra de decisão calibrada e métricas de validação cruzada."""
        return self.root / "decision.json"

    @property
    def model(self) -> Path:
        """Modelo treinado (ainda não publicado)."""
        return self.root / "model.joblib"

    @property
    def onnx(self) -> Path:
        """Modelo exportado para ONNX (ainda não publicado)."""
        return self.root / "model.onnx"

    @property
    def onnx_report(self) -> Path:
        """Resultado da verificação de paridade da exportação ONNX."""
        return self.root / "onnx_export.json"

    @property
    def metrics(self) -> Path:
        """Métricas de teste do modelo treinado."""
        return self.root / "test_metrics.json"

    @property
    def metadata(self) -> Path:
        """Metadados completos, no formato consumido pela API."""
        return self.root / "metadata.json"

    def split_file(self, name: str) -> Path:
        """Arquivo JSON Lines de um conjunto."""
        return self.splits_dir / f"{name}.jsonl"


def safe_run_id(run_id: str) -> str:
    """Converte um identificador de execução em nome de diretório seguro.

    Raises:
        ValueError: Se não sobrar nenhum caractere válido.
    """
    cleaned = _UNSAFE_CHARS.sub("_", run_id).strip("._")
    if not cleaned:
        raise ValueError(f"Identificador de execução inválido: {run_id!r}")
    return cleaned


def run_paths(artifacts_dir: Path, run_id: str) -> RunPaths:
    """Caminhos da execução `run_id` dentro do diretório de artefatos."""
    return RunPaths(root=artifacts_dir / "runs" / safe_run_id(run_id))


def write_dataset(dataset: TextDataset, path: Path) -> Path:
    """Grava um dataset em JSON Lines (um laudo por linha)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text, label in zip(dataset.texts, dataset.labels, strict=True):
            handle.write(json.dumps({"text": text, "label": label}, ensure_ascii=False))
            handle.write("\n")
    return path


def read_dataset(path: Path) -> TextDataset:
    """Lê um dataset gravado por `write_dataset`.

    Raises:
        FileNotFoundError: Se a etapa que produz o arquivo não tiver rodado.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artefato ausente: {path}. Rode a etapa anterior.")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return TextDataset([r["text"] for r in rows], [r["label"] for r in rows])
