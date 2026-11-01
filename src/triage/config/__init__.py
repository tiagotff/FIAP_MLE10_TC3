"""Configurações da aplicação (variáveis de ambiente e arquivos YAML)."""

from triage.config.loader import load_yaml
from triage.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "load_yaml"]
