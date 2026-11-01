"""Backends de inferência do classificador de urgência."""

from triage.inference.loader import PREDICTOR_REGISTRY, load_predictor
from triage.inference.predictor import Prediction, SklearnPredictor, UrgencyPredictor

__all__ = [
    "PREDICTOR_REGISTRY",
    "Prediction",
    "SklearnPredictor",
    "UrgencyPredictor",
    "load_predictor",
]
