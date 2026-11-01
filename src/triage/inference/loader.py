"""Seleção e carregamento do backend de inferência (Factory)."""

from collections.abc import Callable
from pathlib import Path

from triage.config.settings import Settings
from triage.inference.predictor import SklearnPredictor, UrgencyPredictor
from triage.training.artifacts import load_metadata

PredictorBuilder = Callable[[Path, str], UrgencyPredictor]

PREDICTOR_REGISTRY: dict[str, PredictorBuilder] = {
    "sklearn": SklearnPredictor.from_path,
}


def load_predictor(settings: Settings, backend: str = "sklearn") -> UrgencyPredictor:
    """Instancia o preditor do backend escolhido com a versão do modelo.

    Raises:
        ValueError: Se o backend não estiver registrado.
    """
    builder = PREDICTOR_REGISTRY.get(backend)
    if builder is None:
        available = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise ValueError(f"Backend '{backend}' desconhecido. Opções: {available}")
    version = load_metadata(settings.metadata_path).get("model_version", "unknown")
    return builder(settings.model_path, version)
