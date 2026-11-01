"""Leitura de arquivos de configuração YAML."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Carrega um arquivo YAML e garante que a raiz seja um dicionário.

    Args:
        path: Caminho do arquivo YAML.

    Returns:
        Conteúdo do arquivo como dicionário.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se a raiz do YAML não for um mapeamento.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"O arquivo {path} deve conter um mapeamento na raiz.")
    return content
