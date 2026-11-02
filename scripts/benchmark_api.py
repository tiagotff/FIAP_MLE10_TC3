"""Mede a latência ponta a ponta do endpoint /predict.

Envia laudos reais do split de teste, um por vez (ou com N clientes
simultâneos, via `--concurrency`), e registra:
  - latência no cliente (rede + serialização + inferência);
  - tempo de inferência informado pela própria API (`inference_ms`).

Uso:
    python scripts/benchmark_api.py --url http://localhost:8000 \
        --requests 500 --output reports/latency/baseline_api.json
    python scripts/benchmark_api.py --concurrency 8 --requests 2000 \
        --label onnx-concorrente --output reports/latency/api_onnx_c8.json
"""

import argparse
import csv
import json
import platform
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triage.benchmark.latency import LatencySummary, measure, summarize

DEFAULT_DATA = Path("data/raw/medical_tc_test.csv")


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--text-column", default="medical_abstract")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="clientes simultâneos (1 = sequencial, como no baseline)",
    )
    parser.add_argument("--label", default="baseline", help="Rótulo do cenário")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sample_texts(path: Path, column: str, size: int, seed: int) -> list[str]:
    """Sorteia `size` textos (com reposição) do CSV de teste."""
    with path.open(encoding="utf-8", newline="") as handle:
        texts = [row[column] for row in csv.DictReader(handle) if row[column]]
    return random.Random(seed).choices(texts, k=size)


def post_predict(url: str, text: str) -> dict[str, Any]:
    """Chama POST /predict e devolve o JSON de resposta."""
    request = urllib.request.Request(  # noqa: S310
        f"{url}/predict",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        return json.loads(response.read())


def run(args: argparse.Namespace) -> dict[str, LatencySummary]:
    """Executa aquecimento + medição; retorna os resumos por medida.

    Com `--concurrency N`, N clientes enviam requisições ao mesmo tempo
    (cada um com sua fatia dos textos). A latência é medida por requisição;
    a vazão real (`wall_throughput_rps`) vem do tempo total de parede.
    """
    texts = sample_texts(args.data, args.text_column, args.requests, args.seed)
    for text in texts[: args.warmup]:
        post_predict(args.url, text)
    server_ms: list[float] = []

    def call(text: str) -> None:
        server_ms.append(post_predict(args.url, text)["inference_ms"])

    slices = [texts[i :: args.concurrency] for i in range(args.concurrency)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        client_ms = [
            ms for part in pool.map(lambda t: measure(call, t), slices) for ms in part
        ]
    args.wall_throughput_rps = len(texts) / (time.perf_counter() - start)
    return {
        "client_latency": summarize(client_ms),
        "server_inference": summarize(server_ms),
    }


def build_report(
    args: argparse.Namespace, summaries: dict[str, LatencySummary]
) -> dict[str, Any]:
    """Monta o relatório JSON com o contexto da medição."""
    return {
        "scenario": args.label,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": args.url,
        "concurrency": args.concurrency,
        "wall_throughput_rps": round(args.wall_throughput_rps, 1),
        "machine": f"{platform.system()} {platform.machine()}",
        **{name: summary.to_dict() for name, summary in summaries.items()},
    }


def print_headlines(summaries: dict[str, LatencySummary]) -> None:
    """Mostra o p99 de cada medida no stderr (o stdout fica só com o JSON)."""
    labels = {"client_latency": "Ponta a ponta", "server_inference": "Inferência"}
    for name, summary in summaries.items():
        print(f"{labels[name]}: {summary.headline()}", file=sys.stderr)


def main() -> None:
    """Ponto de entrada de linha de comando."""
    args = parse_args()
    summaries = run(args)
    content = json.dumps(build_report(args, summaries), indent=2, ensure_ascii=False)
    print(content)
    print_headlines(summaries)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", "utf-8")


if __name__ == "__main__":
    main()
