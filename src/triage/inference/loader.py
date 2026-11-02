"""Seleção e carregamento do backend de inferência (Factory).

Cada backend declara como se constrói e qual artefato lê. Para adicionar um
novo (ex.: TensorRT), basta registrá-lo aqui — a API não muda.

    sklearn -> models/model.joblib  (pipeline scikit-learn original)
    onnx    -> models/model.onnx    (mesmo modelo exportado para ONNX Runtime)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage.config.settings import Settings
from triage.inference.onnx_predictor import OnnxPredictor
from triage.inference.policy import DecisionPolicy, build_policy
from triage.inference.predictor import SklearnPredictor, UrgencyPredictor
from triage.training.artifacts import load_metadata

PredictorBuilder = Callable[[Path, DecisionPolicy, dict[str, Any]], UrgencyPredictor]


@dataclass(frozen=True)
class PredictorBackend:
    """Como construir um backend e onde está o artefato que ele carrega."""

    build: PredictorBuilder
    artifact: Callable[[Settings], Path]


PREDICTOR_REGISTRY: dict[str, PredictorBackend] = {
    "sklearn": PredictorBackend(
        SklearnPredictor.from_path, lambda settings: settings.model_path
    ),
    "onnx": PredictorBackend(
        OnnxPredictor.from_path, lambda settings: settings.onnx_path
    ),
}


def load_predictor(settings: Settings, backend: str = "sklearn") -> UrgencyPredictor:
    """Instancia o preditor do backend escolhido, com política e metadados.

    A política de decisão (ex.: limiar calibrado para `urgente`) é lida dos
    metadados gravados no treino; sem metadados, usa-se argmax. Os dois
    backends compartilham os mesmos metadados e a mesma política.

    Raises:
        ValueError: Se o backend não estiver registrado.
    """
    entry = PREDICTOR_REGISTRY.get(backend)
    if entry is None:
        available = ", ".join(sorted(PREDICTOR_REGISTRY))
        raise ValueError(f"Backend '{backend}' desconhecido. Opções: {available}")
    metadata = load_metadata(settings.metadata_path)
    policy = build_policy(metadata.get("decision"))
    return entry.build(entry.artifact(settings), policy, metadata)
