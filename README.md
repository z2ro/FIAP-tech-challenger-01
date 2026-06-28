# Churn Prediction — IBM Telco

Pipeline end-to-end para classificação binária de churn com Scikit-Learn, PyTorch, MLflow e FastAPI.

## Problema e objetivo
Churn reduz receita recorrente. O objetivo é estimar probabilidade de churn para priorizar campanhas de retenção, considerando métricas técnicas e custo de falsos positivos/falsos negativos.

## Dataset
Use o IBM Telco Customer Churn. Baixe de uma fonte confiável, como Kaggle/IBM, e salve manualmente em:

```bash
mkdir -p data/raw
# coloque o CSV em data/raw/telco_customer_churn.csv
```

O projeto não baixa dados automaticamente.

## Arquitetura
CSV → limpeza/EDA → split estratificado → ColumnTransformer → baselines → MLP PyTorch → MLflow → artefatos em `models/` → FastAPI.

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Execução
```bash
make lint
make test
make train-baselines
make train-mlp
make mlflow
make run-api
```

## API
```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No","tenure":12,"phone_service":"Yes","multiple_lines":"No","internet_service":"DSL","online_security":"No","online_backup":"Yes","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":70.0,"total_charges":840.0}'
```

Níveis de risco: low < 0.30; medium entre 0.30 e 0.60; high >= 0.60.

## Resultados e custo
Resultados, threshold e comparação devem ser preenchidos após `make train-baselines` e `make train-mlp` com o dataset real. Não há métricas inventadas neste README.

## Estrutura
Código em `src/churn_prediction`, testes em `tests`, documentação em `docs`, artefatos em `models`.

## Limitações e próximos passos
O projeto depende do CSV local e não inclui infraestrutura cloud. Próximos passos: treinar com dados reais, preencher métricas, validar custo com negócio e monitorar drift.

## Roteiro STAR — vídeo (~5 min)
Situation: explicar churn e impacto financeiro. Task: construir pipeline reprodutível. Action: mostrar EDA, pré-processamento, baselines, MLP, MLflow, API, testes e custo. Result: preencher comparação, métricas, threshold escolhido, limitações e aprendizados após treinamento real.
