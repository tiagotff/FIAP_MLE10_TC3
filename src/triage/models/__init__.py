"""Construção dos pipelines de classificação de texto."""

from triage.models.factory import (
    CLASSIFIER_REGISTRY,
    ModelConfig,
    build_classifier,
    build_pipeline,
)

__all__ = ["CLASSIFIER_REGISTRY", "ModelConfig", "build_classifier", "build_pipeline"]
