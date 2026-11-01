"""Configurações de runtime lidas de variáveis de ambiente / arquivo .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros de execução compartilhados por treino e inferência.

    Os valores padrão permitem rodar o projeto localmente sem `.env`;
    em container, as variáveis de ambiente sobrescrevem os padrões.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_raw_dir: Path = Path("data/raw")
    models_dir: Path = Path("models")
    model_filename: str = "model.joblib"
    metadata_filename: str = "metadata.json"
    reports_dir: Path = Path("reports")
    data_config_path: Path = Path("configs/data.yaml")
    model_config_path: Path = Path("configs/model.yaml")
    training_config_path: Path = Path("configs/training.yaml")
    tuning_config_path: Path = Path("configs/tuning.yaml")
    random_seed: int = Field(default=42, ge=0)
    inference_backend: str = "sklearn"
    log_level: str = "INFO"

    @property
    def model_path(self) -> Path:
        """Caminho completo do artefato serializado do modelo."""
        return self.models_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        """Caminho do JSON com metadados e métricas do modelo."""
        return self.models_dir / self.metadata_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna uma instância única (cacheada) das configurações."""
    return Settings()
