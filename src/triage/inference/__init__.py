"""Backends de inferência e regras de decisão do classificador de urgência."""

from triage.inference.loader import PREDICTOR_REGISTRY, load_predictor
from triage.inference.policy import (
    ArgmaxPolicy,
    DecisionPolicy,
    ThresholdPolicy,
    build_policy,
)
from triage.inference.predictor import Prediction, SklearnPredictor, UrgencyPredictor

__all__ = [
    "PREDICTOR_REGISTRY",
    "ArgmaxPolicy",
    "DecisionPolicy",
    "Prediction",
    "SklearnPredictor",
    "ThresholdPolicy",
    "UrgencyPredictor",
    "build_policy",
    "load_predictor",
]
