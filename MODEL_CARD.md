# Model Card — Triagem de Laudos por Urgência

**Versão do modelo:** `20261101T221821Z` (registrada em `models/metadata.json` e em
[`reports/model/metrics.json`](reports/model/metrics.json))
**Data:** Novembro de 2026
**Framework:** scikit-learn 1.7
**Tipo:** Classificação de texto em 3 classes — TF-IDF + Regressão Logística, com regra
de decisão por limiar para a classe crítica
**Repositório:** [github.com/tiagotff/FIAP_MLE10_TC3](https://github.com/tiagotff/FIAP_MLE10_TC3)
**Licença:** MIT (código); dataset sob CC BY-SA 3.0 (ver [Dados](#dados))

## Sumário

Classificador leve que atribui a um texto clínico um nível de urgência — `normal`,
`atencao` ou `urgente` — para **reordenar a fila de leitura de laudos**, de modo que a
equipe clínica veja primeiro os casos que não podem esperar. É servido por uma API REST
(FastAPI) em container Docker, com latência de inferência de ~1 ms em CPU.

## Detalhes do modelo

| | |
|---|---|
| Representação | TF-IDF de unigramas, 30 mil termos, `min_df=1`, `max_df≈0,55`, stop words em inglês, sem `sublinear_tf` |
| Classificador | Regressão Logística multinomial, `C≈2,31`, `class_weight="balanced"`, `max_iter=3000` |
| Regra de decisão | `urgente` sempre que `P(urgente) ≥ 0,376`; caso contrário, a classe mais provável entre `normal` e `atencao` |
| Seleção de hiperparâmetros | Optuna (TPE, 100 trials, seed 42), maximizando F1 macro na validação |
| Calibração do limiar | Probabilidades out-of-fold (validação cruzada estratificada de 5 folds em treino + validação), meta de recall ≥ 0,85 para `urgente` |
| Treino final | Treino + validação (85% do corpus deduplicado) |
| Tamanho do artefato | ~1 MB (`joblib`) |
| Compatibilidade ONNX | Conversão verificada: 100% de concordância de rótulos e diferença máxima de probabilidade de ~2·10⁻⁷ |
| Seed | 42 (divisão, validação cruzada, Optuna e modelo) |

## Uso pretendido

**Uso primário:** apoio à **priorização** de uma fila de laudos em texto — o sistema de
laudos consulta a API ao emitir o exame e usa o rótulo e as probabilidades para ordenar a
fila de leitura.

**Usuários pretendidos:** equipes de engenharia e de informática em saúde que integram
triagem automática a sistemas de laudos; neste projeto, avaliação acadêmica (FIAP MLE10
Tech Challenge Fase 03).

**Fora do escopo:** o modelo **não** deve ser usado para:

- diagnóstico, conduta clínica ou qualquer decisão sem revisão de um profissional;
- **descartar** ou adiar indefinidamente laudos classificados como `normal` — a
  classificação só altera a **ordem** de leitura, nunca a necessidade de leitura;
- textos em **português** ou em outros idiomas sem retreino (ver
  [Limitações](#limitações));
- laudos reais de produção sem validação prévia em dados do próprio hospital.

## Dados

**Medical Abstracts Text Classification Corpus** (Schopf, Braun & Matthes, 2023), CC BY-SA
3.0 — resumos de artigos científicos da área médica, em inglês, rotulados em 5 condições
clínicas.

| Etapa | Registros |
|---|---|
| Arquivos originais (treino + teste) | 14.438 |
| Duplicatas removidas | 3.211 |
| Textos únicos | **11.227** |
| Textos com rótulos de urgência conflitantes (resolvidos para o mais urgente) | 2.681 |

**Mapeamento condição → urgência:** doenças cardiovasculares e do sistema nervoso →
`urgente`; neoplasias e doenças do aparelho digestivo → `atencao`; condições patológicas
gerais → `normal`. Critério: tempo até dano irreversível caso o laudo não seja priorizado.

**Divisão (estratificada, seed 42):**

| Conjunto | Laudos | normal | atencao | urgente |
|---|---|---|---|---|
| Treino (70%) | 7.859 | 1.676 | 2.838 | 3.345 |
| Validação (15%) | 1.684 | 359 | 608 | 717 |
| Teste (15%) | 1.684 | 359 | 608 | 717 |

A divisão oficial do corpus foi descartada porque 1.010 textos do arquivo de teste
também estão no arquivo de treino (vazamento).

## Métricas

### Teste (1.684 laudos nunca vistos, avaliação única)

| Métrica | Argmax | **Limiar (produção)** |
|---|---|---|
| Acurácia | 0,795 | **0,804** |
| F1 macro | 0,772 | **0,776** |
| Recall de `urgente` | 0,796 | **0,847** |
| Precisão de `urgente` | 0,865 | 0,848 |
| Laudos urgentes não sinalizados | 146 | **110** |

| Classe (limiar) | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| normal | 0,625 | 0,618 | 0,622 | 359 |
| atencao | 0,856 | 0,863 | 0,860 | 608 |
| urgente | 0,848 | 0,847 | 0,847 | 717 |

### Validação cruzada (5 folds, out-of-fold em treino + validação, com o limiar)

Acurácia 0,809 · F1 macro 0,785 · recall de `urgente` 0,850 — coerente com o teste, o que
indica que a calibração do limiar generaliza.

### Desempenho computacional

Latência medida pelo **p99** (o tempo que cobre 99 de cada 100 requisições), a métrica de
referência do projeto; o p50 aparece como contexto.

| Ambiente | Inferência do modelo — **p99** (p50) | Ponta a ponta — **p99** (p50) |
|---|---|---|
| Local, uvicorn 1 worker | **2,02 ms** (1,13 ms) | **6,79 ms** (3,42 ms) |
| Docker Desktop (Windows) | **3,43 ms** (1,01 ms) | **31,08 ms** (15,99 ms) |

## Regra de decisão e trade-off

O limiar de 0,376 foi escolhido para priorizar a sensibilidade na classe crítica. Em
relação ao argmax, no teste ele:

- **reduz em 25% os laudos urgentes não sinalizados** (146 → 110);
- adiciona **20 alarmes falsos** (laudos `normal`/`atencao` antecipados na fila);
- **melhora** acurácia e F1 macro.

Um alarme falso custa uma leitura antecipada; um urgente não sinalizado custa tempo de
tratamento. O limiar fica nos metadados do modelo e pode ser ajustado sem retreinar (por
exemplo, elevando a meta de recall em `configs/training.yaml`), desde que o novo valor seja
calibrado e avaliado pelo mesmo protocolo.

## Limitações

### Idioma: o modelo só entende inglês

O vocabulário vem de resumos em inglês. Em textos em português, a maior parte dos termos
não existe no vocabulário e o modelo passa a decidir com base em poucos cognatos (como
"troponina" ou "segmento"). Teste com o mesmo conteúdo clínico nos dois idiomas:

| Caso | Idioma | Termos reconhecidos | Predição | P(rótulo previsto) |
|---|---|---|---|---|
| Dor torácica, supra de ST, troponina elevada | EN | 100% (14/14) | `urgente` | 0,70 |
| | PT | 25% (4/16) | `urgente` | 0,40 |
| AVC isquêmico com hemiparesia e afasia | EN | 100% (14/14) | `urgente` | 0,98 |
| | PT | 29% (5/17) | `urgente` | 0,86 |
| Úlcera gástrica sem sangramento ativo | EN | 100% (9/9) | `atencao` ✓ | 0,90 |
| | PT | 18% (2/11) | `normal` ✗ | 0,48 |

**Consequência:** para um hospital brasileiro, o modelo **não deve ser usado com laudos em
português** sem retreino. O caminho é retreinar com laudos em português rotulados por
especialistas, mantendo a mesma arquitetura e o mesmo protocolo; alternativamente, usar um
modelo multilíngue, reavaliando a latência.

### Domínio: resumos científicos não são laudos

O corpus é formado por resumos de artigos, que descrevem estudos, e não casos individuais.
Laudos reais têm vocabulário, estrutura e tamanho diferentes, então o desempenho em
produção precisa ser medido em dados do próprio hospital antes de qualquer uso.

### Rótulo de urgência é uma *proxy*

A urgência é derivada da **doença**, não da gravidade do caso: um laudo cardiológico de
rotina recebe `urgente`, e uma complicação grave classificada como "condição patológica
geral" recebe `normal`. Rótulos de triagem feitos por especialistas, caso a caso,
eliminariam essa limitação.

### Classe `normal` é a mais difícil

`normal` vem da categoria heterogênea *general pathological conditions*, que se sobrepõe às
demais; ela tem o menor F1 (0,622) e é a classe que mais recebe alarmes falsos de
`urgente`.

## Considerações éticas e de segurança

- **Humano no circuito:** o modelo apenas reordena a fila. Toda decisão clínica é humana,
  e todo laudo continua sendo lido.
- **Falsos negativos são o risco principal:** cerca de 15% dos laudos urgentes não são
  sinalizados como `urgente` no teste. Um processo seguro deve manter um tempo máximo de
  espera para qualquer laudo, independentemente da classificação.
- **Privacidade (LGPD):** laudos são dados pessoais sensíveis. A API não registra o texto
  dos laudos em logs, não o ecoa em mensagens de erro e foi desenhada para rodar em região
  brasileira (`southamerica-east1`), sem acesso público.
- **Viés:** o corpus não traz dados demográficos, então não foi possível avaliar diferenças
  de desempenho entre grupos de pacientes. Essa avaliação é necessária antes de uso real.

## Monitoramento

A API expõe métricas Prometheus e a stack local (API + Prometheus + Grafana) traz
dashboard e alertas (ver [README](README.md#monitoramento-prometheus--grafana)). Sinais
monitorados e limites de alerta:

| Sinal | Referência (teste) | Alerta |
|---|---|---|
| **Cobertura de vocabulário** (fração dos termos do laudo no vocabulário do TF-IDF) | média ~97%; 1% dos laudos abaixo de 87,5% | média abaixo de 80% por 2 min — textos fora do domínio, como laudos em português |
| **Distribuição das predições** | ~43% `urgente` | `urgente` fora de 28%–58% por 5 min |
| **Confiança média** por rótulo | ~0,74–0,81 | acompanhada no dashboard |
| **Latência p99** de `/predict` | 2–7 ms local; ~31 ms no Docker Desktop | acima de 100 ms por 2 min |
| **Taxa de erro 5xx** | 0% | acima de 5% por 2 min |

Esses sinais não precisam de rótulos, então funcionam em tempo real. Eles indicam
**quando** desconfiar do modelo, mas não medem o acerto: para isso, continua necessário
acompanhar o **recall de `urgente` em amostras revisadas** por especialistas.

## Reprodutibilidade

```bash
poetry install
poetry run python -m triage.data.download     # commit e SHA-256 fixados
poetry run python -m triage.training.tune     # opcional: refaz a busca do Optuna
poetry run python -m triage.training.train    # gera models/ e reports/model/metrics.json
poetry run python -m triage.pipeline.run      # alternativa: pipeline com quality gate
```

Em produção, o retreino é orquestrado pela DAG `triage_retraining` do Airflow, que só
publica um novo modelo se ele passar no quality gate (F1 macro ≥ 0,74 e recall de
`urgente` ≥ 0,80 no teste) e mantém cada versão publicada em `models/archive/`.

Configurações: [`configs/data.yaml`](configs/data.yaml),
[`configs/model.yaml`](configs/model.yaml), [`configs/training.yaml`](configs/training.yaml)
e [`configs/tuning.yaml`](configs/tuning.yaml).

## Citação do dataset

Schopf, T.; Braun, D.; Matthes, F. *Evaluating Unsupervised Text Classification:
Zero-shot and Similarity-based Approaches.* NLPIR 2022. doi:
[10.1145/3582768.3582795](https://doi.org/10.1145/3582768.3582795).
