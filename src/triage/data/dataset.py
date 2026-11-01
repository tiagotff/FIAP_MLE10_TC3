"""Carregamento do corpus e conversão de condição clínica em urgência."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from triage.data.schema import DataConfig


@dataclass(frozen=True)
class TextDataset:
    """Par de textos e rótulos de urgência prontos para o modelo."""

    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)


def map_conditions(conditions: pd.Series, mapping: dict[int, str]) -> pd.Series:
    """Converte códigos de condição em rótulos de urgência.

    Raises:
        ValueError: Se houver códigos sem mapeamento definido.
    """
    mapped = conditions.map(mapping)
    missing = sorted(conditions[mapped.isna()].unique().tolist())
    if missing:
        raise ValueError(f"Condições sem mapeamento de urgência: {missing}")
    return mapped


def build_dataset(frame: pd.DataFrame, config: DataConfig) -> TextDataset:
    """Limpa o DataFrame bruto e aplica o mapeamento de urgência."""
    text_col, cond_col = config.columns.text, config.columns.condition
    clean = frame.dropna(subset=[text_col, cond_col])
    clean = clean[clean[text_col].str.strip().astype(bool)]
    labels = map_conditions(clean[cond_col].astype(int), config.urgency_mapping)
    return TextDataset(texts=clean[text_col].tolist(), labels=labels.tolist())


def load_split(raw_dir: Path, split: str, config: DataConfig) -> TextDataset:
    """Lê um split (train/test) do disco e o converte em `TextDataset`.

    Raises:
        FileNotFoundError: Se o CSV do split não tiver sido baixado.
    """
    path = raw_dir / config.source.files[split].name
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo {path} não encontrado. Execute: python -m triage.data.download"
        )
    return build_dataset(pd.read_csv(path), config)
