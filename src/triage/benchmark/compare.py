"""Compara a latência dos backends de inferência (scikit-learn x ONNX).

Uso:
    python -m triage.benchmark.compare
    python -m triage.benchmark.compare --requests 2000 --output reports/latency/x.json

Mede o modelo isolado (sem HTTP), pelo mesmo caminho que a API usa
(`load_predictor` + `predict`, com a regra de decisão), em dois cenários:

  - unitário: um laudo por chamada, o padrão de `/predict`;
  - lote: lotes de `--batch-size` laudos, o padrão de `/predict/batch`.

Os laudos são sorteados do corpus (seed fixa), os backends são medidos
alternadamente em rodadas para que oscilações da máquina afetem os dois
igualmente, e o p99 é a métrica de referência. Também registra o tamanho
do artefato, o tempo de carga e a concordância de rótulos entre backends.
"""

import argparse
import json
import logging
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triage.benchmark.latency import LatencySummary, measure, summarize
from triage.config.settings import Settings, get_settings
from triage.data.dataset import load_corpus
from triage.data.schema import DataConfig
from triage.inference.loader import PREDICTOR_REGISTRY, load_predictor
from triage.inference.predictor import UrgencyPredictor

logger = logging.getLogger(__name__)

BACKENDS = ("sklearn", "onnx")
DEFAULT_OUTPUT = Path("reports/latency/backends_comparison.json")


def sample_texts(settings: Settings, size: int, seed: int) -> list[str]:
    """Sorteia `size` laudos do corpus (com reposição, seed fixa)."""
    config = DataConfig.from_yaml(settings.data_config_path)
    texts = load_corpus(settings.data_raw_dir, config).texts
    return random.Random(seed).choices(texts, k=size)


def timed_load(settings: Settings, backend: str) -> tuple[UrgencyPredictor, float]:
    """Carrega o backend e mede o tempo de carga (ms)."""
    start = time.perf_counter()
    predictor = load_predictor(settings, backend)
    return predictor, (time.perf_counter() - start) * 1000


def batches(texts: list[str], size: int) -> list[list[str]]:
    """Divide os textos em lotes de `size` (o último pode ser menor)."""
    return [texts[i : i + size] for i in range(0, len(texts), size)]


def label_agreement(predictors: dict[str, UrgencyPredictor], texts: list[str]) -> float:
    """Fração de laudos com o mesmo rótulo em todos os backends."""
    labels = [[p.label for p in pred.predict(texts)] for pred in predictors.values()]
    same = [len(set(column)) == 1 for column in zip(*labels, strict=True)]
    return sum(same) / len(same)


def benchmark(
    predictors: dict[str, UrgencyPredictor],
    texts: list[str],
    batch_size: int,
    rounds: int,
    warmup: int,
) -> dict[str, dict[str, LatencySummary]]:
    """Mede unitário e lote, alternando os backends em `rounds` rodadas."""
    for predictor in predictors.values():
        for text in texts[:warmup]:
            predictor.predict([text])
    single: dict[str, list[float]] = {name: [] for name in predictors}
    batched: dict[str, list[float]] = {name: [] for name in predictors}
    chunks = [texts[i::rounds] for i in range(rounds)]
    for chunk in chunks:
        for name, predictor in predictors.items():
            single[name] += measure(lambda t, p=predictor: p.predict([t]), chunk)
            batched[name] += measure(predictor.predict, batches(chunk, batch_size))
    return {
        name: {"single": summarize(single[name]), "batch": summarize(batched[name])}
        for name in predictors
    }


def speedup(baseline: LatencySummary, optimized: LatencySummary) -> dict[str, float]:
    """Quantas vezes o otimizado é mais rápido, no p50 e no p99."""
    return {
        "p50": round(baseline.p50_ms / optimized.p50_ms, 2),
        "p99": round(baseline.p99_ms / optimized.p99_ms, 2),
    }


def build_report(
    settings: Settings,
    args: argparse.Namespace,
    results: dict[str, dict[str, LatencySummary]],
    load_ms: dict[str, float],
    agreement: float,
) -> dict[str, Any]:
    """Relatório JSON com contexto, medições e ganhos."""
    artifacts = {name: PREDICTOR_REGISTRY[name].artifact(settings) for name in results}
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "requests": args.requests,
        "batch_size": args.batch_size,
        "label_agreement": agreement,
        "backends": {
            name: {
                "artifact": artifacts[name].name,
                "artifact_bytes": artifacts[name].stat().st_size,
                "load_ms": round(load_ms[name], 1),
                "single": summaries["single"].to_dict(),
                "batch": summaries["batch"].to_dict(),
            }
            for name, summaries in results.items()
        },
        "speedup_onnx_vs_sklearn": {
            scenario: speedup(results["sklearn"][scenario], results["onnx"][scenario])
            for scenario in ("single", "batch")
        },
    }


def markdown_table(report: dict[str, Any]) -> str:
    """Tabela-resumo (p99 em destaque) para colar no README."""
    lines = [
        "| Backend | Unitário p99 (p50) | Lote de "
        f"{report['batch_size']} p99 (p50) | Carga | Artefato |",
        "|---|---|---|---|---|",
    ]
    for name, data in report["backends"].items():
        single, batch = data["single"], data["batch"]
        lines.append(
            f"| {name} | **{single['p99_ms']:.3f} ms** ({single['p50_ms']:.3f} ms) "
            f"| **{batch['p99_ms']:.2f} ms** ({batch['p50_ms']:.2f} ms) "
            f"| {data['load_ms']:.0f} ms | {data['artifact_bytes'] / 1e6:.2f} MB |"
        )
    gains = report["speedup_onnx_vs_sklearn"]
    lines.append(
        f"\nGanho do ONNX: unitário {gains['single']['p99']}x no p99 "
        f"({gains['single']['p50']}x no p50); lote {gains['batch']['p99']}x no p99. "
        f"Concordância de rótulos: {report['label_agreement']:.2%}."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada de linha de comando."""
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    texts = sample_texts(settings, args.requests, args.seed)
    loaded = {name: timed_load(settings, name) for name in BACKENDS}
    predictors = {name: predictor for name, (predictor, _) in loaded.items()}
    load_ms = {name: ms for name, (_, ms) in loaded.items()}
    agreement = label_agreement(predictors, texts)
    results = benchmark(predictors, texts, args.batch_size, args.rounds, args.warmup)
    report = build_report(settings, args, results, load_ms, agreement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(markdown_table(report))
    logger.info("Relatório gravado em %s", args.output)


if __name__ == "__main__":
    main()
