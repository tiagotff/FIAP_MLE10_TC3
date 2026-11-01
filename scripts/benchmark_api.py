"""Mede a latência ponta a ponta do endpoint /predict.

Envia laudos reais do split de teste, um por vez, e registra:
  - latência no cliente (rede + serialização + inferência);
  - tempo de inferência informado pela própria API (`inference_ms`).

Uso:
    python scripts/benchmark_api.py --url http://localhost:8000 \
        --requests 500 --output reports/latency/baseline_api.json
"""

import argparse
import csv
import json
import platform
import random
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triage.benchmark.latency import measure, summarize

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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Executa aquecimento + medição e monta o relatório."""
    texts = sample_texts(args.data, args.text_column, args.requests, args.seed)
    for text in texts[: args.warmup]:
        post_predict(args.url, text)
    server_ms: list[float] = []

    def call(text: str) -> None:
        server_ms.append(post_predict(args.url, text)["inference_ms"])

    client_ms = measure(call, texts)
    return {
        "scenario": args.label,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": args.url,
        "machine": f"{platform.system()} {platform.machine()}",
        "client_latency": summarize(client_ms).to_dict(),
        "server_inference": summarize(server_ms).to_dict(),
    }


def main() -> None:
    """Ponto de entrada de linha de comando."""
    args = parse_args()
    content = json.dumps(run(args), indent=2, ensure_ascii=False)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", "utf-8")


if __name__ == "__main__":
    main()
