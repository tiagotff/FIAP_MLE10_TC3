"""Seleção e carregamento do backend de inferência (Factory)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from triage.config.settings import Settings
from triage.inference.policy import DecisionPolicy, build_policy
from triage.inference.predictor import SklearnPredictor, UrgencyPredictor
from triage.training.artifacts import load_metadata

PredictorBuilder = Callable[[Path, DecisionPolicy, dict[str, Any]], UrgencyPredictor]

PREDICTOR_REGISTRY: dict[str, PredictorBuilder] = {
    "sklearn": SklearnPredictor.from_path,
}


def load_predictor(settings: Settings, backend: str = "sklearn") -> UrgencyPredictor:
    """Instancia o preditor do backend escolhido, com política e metadados.

    A política de decisão (ex.: limiar calibrado para `urgente`) é lida dos
    metadados gravados no treino; sem metadados, usa-se argmax.

    Raises:
        ValueError: Se o backend não estiver registrado.
    """
    builder = PREDICTOR_REGISTRY.get(backend)
    if builder is None:
        available = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise ValueError(f"Backend '{backend}' desconhecido. Opções: {available}")
    metadata = load_metadata(settings.metadata_path)
    policy = build_policy(metadata.get("decision"))
    return builder(settings.model_path, policy, metadata)
