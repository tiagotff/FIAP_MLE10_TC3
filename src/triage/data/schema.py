"""Modelo tipado da configuração de dados (configs/data.yaml)."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from triage.config.loader import load_yaml


class SourceFile(BaseModel):
    """Arquivo remoto do dataset com checksum esperado."""

    name: str
    sha256: str = Field(min_length=64, max_length=64)


class DataSource(BaseModel):
    """Origem remota do dataset."""

    base_url: str
    files: dict[str, SourceFile]


class ColumnConfig(BaseModel):
    """Nomes das colunas de texto e de condição no CSV."""

    text: str
    condition: str


class DataConfig(BaseModel):
    """Configuração completa de dados e do mapeamento de urgência."""

    source: DataSource
    columns: ColumnConfig
    urgency_mapping: dict[int, str]
    labels: list[str]

    @model_validator(mode="after")
    def _check_mapping(self) -> "DataConfig":
        unknown = set(self.urgency_mapping.values()) - set(self.labels)
        if unknown:
            raise ValueError(f"Rótulos de urgência não declarados: {sorted(unknown)}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "DataConfig":
        """Cria a configuração a partir de um arquivo YAML."""
        return cls.model_validate(load_yaml(path))
