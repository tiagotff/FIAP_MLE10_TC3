"""Configuração de logging do serviço."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configura o logger raiz uma única vez (idempotente).

    Raises:
        ValueError: Se o nível de log for inválido.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Nível de log inválido: {level}")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format=LOG_FORMAT)
    root.setLevel(numeric)
