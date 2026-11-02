"""Gera tráfego sintético contra a API para alimentar o monitoramento.

Usa apenas a biblioteca padrão, para rodar tanto no ambiente do projeto
quanto num container `python:3.12-slim` (serviço `traffic` do compose).

O tráfego segue um plano de fases `cenário:segundos`, executadas em ordem:

    normal  laudos em inglês (domínio do modelo); 20% via /predict/batch
    drift   laudos em português: a cobertura de vocabulário despenca e o
            alerta TriageVocabularyDrift dispara após alguns minutos
    errors  80% de laudos normais e 20% de requisições inválidas (422)

Exemplos:
    python scripts/generate_traffic.py                       # normal por 2 min
    python scripts/generate_traffic.py --plan normal:180,drift:300,normal:120
    python scripts/generate_traffic.py --rps 30 --workers 8 --plan errors:60

Se o CSV de teste (`data/raw/medical_tc_test.csv`) existir, os laudos em
inglês são sorteados dele; caso contrário, usa amostras embutidas.
"""

import argparse
import csv
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA = Path("data/raw/medical_tc_test.csv")

ENGLISH_SAMPLES = [
    "Patient admitted with acute chest pain radiating to the left arm, "
    "ST-segment elevation and elevated troponin levels.",
    "Acute ischemic stroke with right hemiparesis and aphasia; thrombolysis "
    "was started within the therapeutic window.",
    "Subarachnoid hemorrhage presenting with sudden severe headache and "
    "neck stiffness.",
    "Atrial fibrillation with rapid ventricular response and signs of "
    "congestive heart failure.",
    "Gastric ulcer identified at endoscopy without active bleeding; proton "
    "pump inhibitor therapy recommended.",
    "Breast carcinoma confirmed by biopsy; staging and oncology referral are pending.",
    "Colorectal adenoma removed during colonoscopy, histology shows "
    "low-grade dysplasia.",
    "Chronic hepatitis with mild elevation of liver enzymes and no "
    "evidence of cirrhosis.",
    "Routine follow-up visit with mild inflammation and stable general condition.",
    "Benign skin lesion with no signs of infection; no further treatment required.",
]

PORTUGUESE_SAMPLES = [
    "Paciente admitido com dor torácica aguda irradiando para o braço "
    "esquerdo, supradesnivelamento do segmento ST e troponina elevada.",
    "Acidente vascular cerebral isquêmico com hemiparesia à direita e "
    "afasia; trombólise iniciada dentro da janela terapêutica.",
    "Úlcera gástrica identificada na endoscopia, sem sangramento ativo; "
    "recomendado inibidor de bomba de prótons.",
    "Carcinoma de mama confirmado por biópsia; estadiamento e "
    "encaminhamento à oncologia pendentes.",
    "Consulta de retorno de rotina, inflamação leve e estado geral estável.",
    "Fibrilação atrial com resposta ventricular rápida e sinais de "
    "insuficiência cardíaca congestiva.",
    "Pólipo adenomatoso removido na colonoscopia, histologia com displasia "
    "de baixo grau.",
    "Lesão de pele benigna sem sinais de infecção; não há necessidade de "
    "tratamento adicional.",
]

SCENARIOS = ("normal", "drift", "errors")

INVALID_PAYLOADS = [
    {"text": ""},
    {"text": "   "},
    {"texto": "campo com nome errado"},
    {"texts": []},
]


@dataclass
class Request:
    """Chamada HTTP a ser feita contra a API."""

    path: str
    payload: dict


@dataclass
class Stats:
    """Resultados agregados (thread-safe) do tráfego gerado."""

    statuses: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, status: int, elapsed_ms: float, labels: list[str]) -> None:
        """Contabiliza uma resposta."""
        with self.lock:
            self.statuses[status] += 1
            self.latencies_ms.append(elapsed_ms)
            self.labels.update(labels)

    def summary(self) -> str:
        """Resumo legível: volume, status, rótulos e percentis de latência."""
        total = sum(self.statuses.values())
        if not total:
            return "Nenhuma requisição concluída."
        ordered = sorted(self.latencies_ms)
        p99 = ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))]
        return "\n".join(
            [
                f"Requisições: {total}",
                f"Status HTTP: {dict(sorted(self.statuses.items()))}",
                f"Rótulos previstos: {dict(self.labels.most_common())}",
                f"Latência no cliente: p99={p99:.1f} ms "
                f"p50={statistics.median(ordered):.1f} ms",
            ]
        )


def load_english_texts(path: Path, column: str = "medical_abstract") -> list[str]:
    """Laudos do CSV de teste, ou as amostras embutidas se ele não existir."""
    if not path.exists():
        print(f"{path} não encontrado: usando amostras embutidas.")
        return ENGLISH_SAMPLES
    with path.open(encoding="utf-8", newline="") as handle:
        texts = [row[column] for row in csv.DictReader(handle) if row.get(column)]
    print(f"{len(texts)} laudos carregados de {path}.")
    return texts or ENGLISH_SAMPLES


def make_request(texts: list[str], rng: random.Random, batch_share: float) -> Request:
    """Monta um /predict (ou /predict/batch, na fração `batch_share`)."""
    if rng.random() < batch_share:
        return Request(
            "/predict/batch", {"texts": rng.sample(texts, k=min(8, len(texts)))}
        )
    return Request("/predict", {"text": rng.choice(texts)})


def scenario_factory(
    name: str, english: list[str], batch_share: float
) -> Callable[[random.Random], Request]:
    """Função que sorteia a próxima requisição de um cenário.

    Raises:
        ValueError: Se o cenário não existir.
    """
    if name == "normal":
        return lambda rng: make_request(english, rng, batch_share)
    if name == "drift":
        return lambda rng: make_request(PORTUGUESE_SAMPLES, rng, batch_share)
    if name == "errors":

        def errors(rng: random.Random) -> Request:
            if rng.random() < 0.2:
                return Request("/predict", rng.choice(INVALID_PAYLOADS))
            return make_request(english, rng, batch_share)

        return errors
    raise ValueError(f"Cenário desconhecido: {name!r} (use normal, drift ou errors)")


def parse_plan(plan: str) -> list[tuple[str, float]]:
    """Converte `normal:120,drift:60` em [("normal", 120.0), ("drift", 60.0)].

    Raises:
        ValueError: Se alguma fase estiver mal formatada.
    """
    phases = []
    for item in plan.split(","):
        name, _, seconds = item.strip().partition(":")
        if not name or not seconds:
            raise ValueError(f"Fase inválida: {item!r} (formato cenário:segundos)")
        if name not in SCENARIOS:
            raise ValueError(f"Cenário desconhecido: {name!r} (use {SCENARIOS})")
        phases.append((name, float(seconds)))
    return phases


def send(base_url: str, request: Request, timeout: float) -> tuple[int, dict]:
    """Envia a requisição e devolve (status, corpo JSON)."""
    http_request = urllib.request.Request(  # noqa: S310
        base_url + request.path,
        data=json.dumps(request.payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {}


def predicted_labels(body: dict) -> list[str]:
    """Rótulos de uma resposta de /predict ou /predict/batch."""
    if "label" in body:
        return [body["label"]]
    return [item["label"] for item in body.get("predictions", [])]


def worker(
    base_url: str,
    next_request: Callable[[random.Random], Request],
    interval: float,
    deadline: float,
    seed: int,
    stats: Stats,
) -> None:
    """Envia requisições em ritmo constante até o fim da fase."""
    rng = random.Random(seed)
    next_send = time.monotonic()
    while next_send < deadline:
        start = time.perf_counter()
        try:
            status, body = send(base_url, next_request(rng), timeout=10)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            status, body = 0, {}
        stats.record(
            status, (time.perf_counter() - start) * 1000, predicted_labels(body)
        )
        next_send += interval
        time.sleep(max(0.0, next_send - time.monotonic()))


def wait_until_ready(base_url: str, attempts: int = 60) -> None:
    """Aguarda /ready responder 200.

    Raises:
        SystemExit: Se a API não ficar pronta a tempo.
    """
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(base_url + "/ready", timeout=2):  # noqa: S310
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    raise SystemExit(f"API indisponível em {base_url}")


def run_phase(
    args: argparse.Namespace,
    name: str,
    seconds: float,
    english: list[str],
    stats: Stats,
) -> None:
    """Executa uma fase do plano com `args.workers` threads."""
    next_request = scenario_factory(name, english, args.batch_share)
    interval = args.workers / args.rps
    deadline = time.monotonic() + seconds
    print(f"Fase '{name}': {seconds:.0f}s a ~{args.rps} req/s")
    threads = [
        threading.Thread(
            target=worker,
            args=(args.url, next_request, interval, deadline, args.seed + i, stats),
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    # join com timeout: mantém Ctrl+C responsivo também no Windows.
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=0.5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--plan", default="normal:120", help="ex.: normal:120,drift:60")
    parser.add_argument("--rps", type=float, default=10.0, help="requisições/s")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-share", type=float, default=0.2)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.rps <= 0 or args.workers < 1:
        parser.error("--rps deve ser positivo e --workers, ao menos 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Executa o plano de tráfego e imprime o resumo."""
    args = parse_args(argv)
    args.url = args.url.rstrip("/")
    try:
        plan = parse_plan(args.plan)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    english = load_english_texts(args.data)
    wait_until_ready(args.url)
    stats = Stats()
    try:
        for name, seconds in plan:
            run_phase(args, name, seconds, english, stats)
    except KeyboardInterrupt:
        print("\nInterrompido.")
    print(stats.summary())


if __name__ == "__main__":
    sys.exit(main())
