"""Download do dataset com verificação de integridade (SHA-256).

Uso:
    python -m triage.data.download
"""

import hashlib
import logging
import urllib.request
from pathlib import Path

from triage.config.settings import get_settings
from triage.data.schema import DataConfig, SourceFile

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20


def sha256_of(path: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em blocos."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, expected_sha256: str) -> bool:
    """Indica se o arquivo existe e possui o checksum esperado."""
    return path.exists() and sha256_of(path) == expected_sha256


def fetch_file(base_url: str, source: SourceFile, target_dir: Path) -> Path:
    """Baixa um arquivo (se necessário) e valida seu checksum.

    Raises:
        ValueError: Se o checksum do arquivo baixado não conferir.
    """
    target = target_dir / source.name
    if is_valid(target, source.sha256):
        logger.info("Arquivo já presente e íntegro: %s", target)
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{base_url}/{source.name}", target)  # noqa: S310
    if not is_valid(target, source.sha256):
        target.unlink(missing_ok=True)
        raise ValueError(f"Checksum inválido para {source.name}; download descartado.")
    logger.info("Download concluído: %s", target)
    return target


def download_dataset(config: DataConfig, target_dir: Path) -> dict[str, Path]:
    """Baixa todos os splits declarados na configuração."""
    base_url = config.source.base_url
    return {
        split: fetch_file(base_url, source, target_dir)
        for split, source in config.source.files.items()
    }


def main() -> None:
    """Ponto de entrada de linha de comando."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    config = DataConfig.from_yaml(settings.data_config_path)
    download_dataset(config, settings.data_raw_dir)


if __name__ == "__main__":
    main()
