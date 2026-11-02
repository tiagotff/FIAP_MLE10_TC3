"""Verifica a stack de monitoramento de ponta a ponta (API, Prometheus, Grafana).

Pensado para rodar logo após `docker compose up` e algum tráfego (CI ou
local). Usa apenas a biblioteca padrão.

Checagens:
  1. API pronta e expondo métricas em /metrics;
  2. Prometheus coletando a API, com as regras carregadas e as recording
     rules (p99, distribuição das predições, cobertura) produzindo dados;
  3. Grafana com o datasource saudável, o dashboard provisionado e todas
     as consultas dos painéis executando sem erro.

Uso:
    python scripts/smoke_monitoring.py
    python scripts/smoke_monitoring.py --timeout 120
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

DASHBOARD = Path("monitoring/grafana/dashboards/triage.json")
EXPECTED_RULE_GROUPS = {"triage-recording", "triage-alerts"}
REQUIRED_SERIES = [
    "triage:http_request_duration_seconds:p99_1m",
    "triage:inference_duration_seconds:p99_1m",
    "triage:predictions:share_5m",
    "triage:vocabulary_coverage:mean_5m",
]


class CheckError(Exception):
    """Falha de uma checagem."""


def http_json(
    url: str,
    payload: dict | None = None,
    auth: tuple[str, str] | None = None,
) -> Any:  # noqa: ANN401
    """GET (ou POST, se houver payload) que devolve o corpo JSON."""
    headers = {"Content-Type": "application/json"}
    if auth:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        return json.loads(response.read())


def http_text(url: str) -> str:
    """GET que devolve o corpo como texto."""
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
        return response.read().decode()


def eventually(check: Callable[[], str], timeout: float, interval: float = 3) -> str:
    """Repete `check` até passar ou estourar o tempo (devolve a mensagem)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check()
        except (CheckError, urllib.error.URLError, ConnectionError, OSError) as error:
            if time.monotonic() >= deadline:
                raise CheckError(str(error)) from error
            time.sleep(interval)


def prom_query(prometheus: str, query: str) -> list[dict]:
    """Executa uma consulta instantânea no Prometheus."""
    url = f"{prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": query})
    return http_json(url)["data"]["result"]


def check_api(api: str) -> str:
    """API pronta e expondo as métricas do modelo."""
    ready = http_json(f"{api}/ready")
    metrics = http_text(f"{api}/metrics")
    if "triage_model_info" not in metrics:
        raise CheckError("/metrics sem triage_model_info")
    return f"API pronta (modelo {ready['model_version']})"


def check_prometheus_target(prometheus: str) -> str:
    """Prometheus coletando a API com sucesso."""
    result = prom_query(prometheus, 'up{job="triage-api"}')
    if not result or result[0]["value"][1] != "1":
        raise CheckError("alvo triage-api não está 'up' no Prometheus")
    return "Prometheus coletando a API"


def check_prometheus_rules(prometheus: str) -> str:
    """Grupos de regras carregados e sem erro de avaliação."""
    groups = http_json(f"{prometheus}/api/v1/rules")["data"]["groups"]
    names = {group["name"] for group in groups}
    if not names >= EXPECTED_RULE_GROUPS:
        raise CheckError(f"grupos de regras ausentes: {EXPECTED_RULE_GROUPS - names}")
    unhealthy = [
        rule["name"]
        for group in groups
        for rule in group["rules"]
        if rule.get("health") == "err"
    ]
    if unhealthy:
        raise CheckError(f"regras com erro: {unhealthy}")
    total = sum(len(group["rules"]) for group in groups)
    return f"{total} regras carregadas, nenhuma com erro"


def check_recording_series(prometheus: str) -> str:
    """Recording rules usadas pelo dashboard já produzem dados."""
    missing = [name for name in REQUIRED_SERIES if not prom_query(prometheus, name)]
    if missing:
        raise CheckError(f"séries sem dados: {missing}")
    p99 = prom_query(
        prometheus, 'triage:http_request_duration_seconds:p99_1m{route="/predict"}'
    )
    p99_ms = float(p99[0]["value"][1]) * 1000 if p99 else float("nan")
    return f"recording rules com dados (p99 de /predict = {p99_ms:.1f} ms)"


def dashboard_queries(path: Path) -> Iterator[tuple[str, str]]:
    """(título do painel, expressão) de todas as consultas do dashboard."""
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            yield panel["title"], target["expr"]


def check_grafana(grafana: str, auth: tuple[str, str], dashboard: Path) -> str:
    """Datasource saudável, dashboard provisionado e consultas sem erro."""
    health = http_json(f"{grafana}/api/datasources/uid/prometheus/health", auth=auth)
    if health.get("status") != "OK":
        raise CheckError(f"datasource Prometheus: {health}")
    uid = json.loads(dashboard.read_text(encoding="utf-8"))["uid"]
    provisioned = http_json(f"{grafana}/api/dashboards/uid/{uid}", auth=auth)
    panels = [p for p in provisioned["dashboard"]["panels"] if p["type"] != "row"]
    failures = []
    queries = list(dashboard_queries(dashboard))
    for title, expr in queries:
        body = {
            "from": "now-15m",
            "to": "now",
            "queries": [
                {
                    "refId": "A",
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "expr": expr,
                    "instant": True,
                    "range": False,
                }
            ],
        }
        try:
            result = http_json(f"{grafana}/api/ds/query", payload=body, auth=auth)
        except urllib.error.HTTPError as error:
            failures.append(f"{title}: HTTP {error.code}")
            continue
        if result["results"]["A"].get("error"):
            failures.append(f"{title}: {result['results']['A']['error']}")
    if failures:
        raise CheckError("consultas com erro no Grafana: " + "; ".join(failures))
    return (
        f"Grafana com o dashboard '{provisioned['dashboard']['title']}' "
        f"({len(panels)} painéis, {len(queries)} consultas OK)"
    )


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--grafana", default="http://localhost:3000")
    parser.add_argument("--grafana-user", default="admin")
    parser.add_argument("--grafana-password", default="admin")
    parser.add_argument("--dashboard", type=Path, default=DASHBOARD)
    parser.add_argument("--timeout", type=float, default=90)
    return parser.parse_args()


def main() -> int:
    """Executa as checagens em ordem; retorna 1 na primeira falha."""
    args = parse_args()
    auth = (args.grafana_user, args.grafana_password)
    checks = [
        ("API", lambda: check_api(args.api)),
        ("Alvo", lambda: check_prometheus_target(args.prometheus)),
        ("Regras", lambda: check_prometheus_rules(args.prometheus)),
        ("Séries", lambda: check_recording_series(args.prometheus)),
        ("Grafana", lambda: check_grafana(args.grafana, auth, args.dashboard)),
    ]
    for name, check in checks:
        try:
            print(f"[ok]    {name}: {eventually(check, args.timeout)}")
        except CheckError as error:
            print(f"[falha] {name}: {error}")
            return 1
    print("Stack de monitoramento OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
