"""Métricas Prometheus do serviço de triagem.

Três grupos de sinais, expostos em `GET /metrics`:

- **Serviço** — requisições por rota e status, latência ponta a ponta e
  requisições em andamento. A latência é um histograma; o p99 é calculado
  no Prometheus com `histogram_quantile`.
- **Modelo** — tempo de inferência, tamanho dos lotes e versão em uso.
- **Dados e predições** — distribuição dos rótulos, confiança, tamanho dos
  textos e cobertura de vocabulário. Sem rótulos reais em produção, são
  esses sinais que denunciam drift (ex.: laudos em outro idioma derrubam a
  cobertura de vocabulário).

Cada instância da aplicação usa o próprio `CollectorRegistry`, o que
mantém os testes isolados. Nenhuma métrica carrega o texto dos laudos
(LGPD): só contagens e distribuições.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    disable_created_metrics,
    generate_latest,
)

from triage.inference.predictor import Prediction

NAMESPACE = "triage"

# Séries `*_created` (timestamp de criação de cada contador) não são usadas
# nos painéis nem nos alertas e dobrariam o volume exposto.
disable_created_metrics()

# Faixa de milissegundos com resolução fina: a inferência leva ~1-3 ms e a
# requisição completa ~3-30 ms. O p99 interpolado depende dessas fronteiras.
LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)
RATIO_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
LENGTH_BUCKETS = (100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000, 20000)


@dataclass(frozen=True)
class ApiMetrics:
    """Coletores de métricas da API, registrados num registry próprio."""

    registry: CollectorRegistry
    requests: Counter
    request_duration: Histogram
    in_progress: Gauge
    inference_duration: Histogram
    batch_size: Histogram
    predictions: Counter
    confidence: Histogram
    vocabulary_coverage: Histogram
    text_length: Histogram
    model_info: Gauge

    @classmethod
    def create(cls, registry: CollectorRegistry | None = None) -> "ApiMetrics":
        """Cria e registra todos os coletores (inclui CPU/memória do processo)."""
        registry = registry or CollectorRegistry()
        ProcessCollector(registry=registry)
        common = {"namespace": NAMESPACE, "registry": registry}
        return cls(
            registry=registry,
            requests=Counter(
                "http_requests",
                "Requisições HTTP atendidas, por método, rota e status.",
                ["method", "route", "status"],
                **common,
            ),
            request_duration=Histogram(
                "http_request_duration_seconds",
                "Latência ponta a ponta das requisições HTTP.",
                ["method", "route"],
                buckets=LATENCY_BUCKETS,
                **common,
            ),
            in_progress=Gauge(
                "http_requests_in_progress",
                "Requisições HTTP em processamento.",
                **common,
            ),
            inference_duration=Histogram(
                "inference_duration_seconds",
                "Tempo de inferência do modelo (sem rede e serialização).",
                ["endpoint"],
                buckets=LATENCY_BUCKETS,
                **common,
            ),
            batch_size=Histogram(
                "batch_size",
                "Quantidade de laudos por chamada de inferência.",
                ["endpoint"],
                buckets=BATCH_BUCKETS,
                **common,
            ),
            predictions=Counter(
                "predictions",
                "Laudos classificados, por rótulo previsto.",
                ["label"],
                **common,
            ),
            confidence=Histogram(
                "prediction_confidence",
                "Probabilidade do rótulo previsto.",
                ["label"],
                buckets=RATIO_BUCKETS,
                **common,
            ),
            vocabulary_coverage=Histogram(
                "vocabulary_coverage_ratio",
                "Fração dos termos do laudo presentes no vocabulário do modelo.",
                buckets=RATIO_BUCKETS,
                **common,
            ),
            text_length=Histogram(
                "input_text_length_chars",
                "Tamanho dos laudos recebidos, em caracteres.",
                buckets=LENGTH_BUCKETS,
                **common,
            ),
            model_info=Gauge(
                "model_info",
                "Modelo em uso (valor 1 para a versão carregada).",
                ["model_version", "backend"],
                **common,
            ),
        )

    def set_model(self, model_version: str, backend: str) -> None:
        """Registra a versão do modelo carregado (substitui a anterior)."""
        self.model_info.clear()
        self.model_info.labels(model_version=model_version, backend=backend).set(1)

    def observe_request(
        self, method: str, route: str, status: int, seconds: float
    ) -> None:
        """Contabiliza uma requisição HTTP concluída."""
        self.requests.labels(method=method, route=route, status=str(status)).inc()
        self.request_duration.labels(method=method, route=route).observe(seconds)

    def observe_inference(self, endpoint: str, seconds: float, size: int) -> None:
        """Registra o tempo de uma chamada ao modelo e o tamanho do lote."""
        self.inference_duration.labels(endpoint=endpoint).observe(seconds)
        self.batch_size.labels(endpoint=endpoint).observe(size)

    def observe_predictions(self, predictions: Sequence[Prediction]) -> None:
        """Atualiza a distribuição de rótulos e de confiança."""
        for prediction in predictions:
            self.predictions.labels(label=prediction.label).inc()
            self.confidence.labels(label=prediction.label).observe(
                prediction.confidence
            )

    def observe_inputs(
        self, texts: Sequence[str], coverages: Sequence[float] | None = None
    ) -> None:
        """Registra tamanho dos textos e, se disponível, a cobertura."""
        for text in texts:
            self.text_length.observe(len(text))
        for coverage in coverages or ():
            self.vocabulary_coverage.observe(coverage)

    def render(self) -> bytes:
        """Serializa as métricas no formato de exposição do Prometheus."""
        return generate_latest(self.registry)
