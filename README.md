# Churn Prediction — IBM Telco

Pipeline end-to-end para classificação binária de churn com Scikit-Learn, PyTorch, MLflow e FastAPI.

## Problema de negócio
Churn reduz receita recorrente e aumenta custo de aquisição. Este projeto estima risco de churn para apoiar priorização de ações de retenção. O modelo não prova causalidade e não deve tomar decisões comerciais automaticamente.

## Dataset
Dataset: IBM Telco Customer Churn.

- Caminho esperado: `data/raw/telco_customer_churn.csv`.
- Tamanho real: 7.043 clientes.
- Distribuição da classe alvo: aproximadamente 5.174 não churn (73,46%) e 1.869 churn (26,54%).
- O CSV deve ser baixado manualmente de fonte confiável; o projeto não baixa dados automaticamente.

```bash
mkdir -p data/raw
# coloque o CSV em data/raw/telco_customer_churn.csv
```

## Arquitetura
CSV → limpeza/EDA → split estratificado → `ColumnTransformer` → baselines → MLP PyTorch → MLflow → artefatos em `models/` → FastAPI.

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Comandos principais
```bash
make lint
make test
make train-baselines
make train-mlp
make mlflow
make run-api
```

## Treinamento e threshold
O threshold `0.2` foi selecionado no conjunto de validação por análise de custo. As métricas finais da MLP foram calculadas uma única vez no conjunto de teste após o threshold estar fixado.

```text
Threshold selecionado no conjunto: validação
Métricas finais calculadas no conjunto: teste
```

## Comparação de modelos
A tabela abaixo consolida os resultados disponíveis. Os baselines estão marcados como média de validação cruzada; a MLP está marcada como teste. Não compare esses números como se fossem exatamente o mesmo protocolo sem considerar a coluna `Escopo`.

| Modelo | Escopo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| ------ | ------ | -------: | --------: | -----: | -: | ------: | -----: |
| DummyClassifier | CV média | 0.734630 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.265370 |
| LogisticRegression | CV média | 0.745704 | 0.513362 | 0.801484 | 0.625824 | 0.844875 | 0.655381 |
| RandomForestClassifier | CV média | 0.771118 | 0.560268 | 0.637214 | 0.596175 | 0.822030 | 0.600438 |
| PyTorch MLP | Teste | 0.591296 | 0.389458 | 0.946619 | 0.551867 | 0.843192 | 0.650842 |

A versão CSV está em `models/final_model_comparison.csv`.

## Interpretação técnica
Logistic Regression apresenta melhor equilíbrio geral nos resultados atuais: maior F1, PR-AUC ligeiramente superior e ROC-AUC semelhante/superior. Random Forest tem maior accuracy, mas menor recall de churn. A MLP com threshold `0.2` entrega recall muito alto, aproximadamente 94,66%, ao custo de menor precision e mais falsos positivos.

A MLP é adequada quando o custo de perder um cliente em churn for muito maior que o custo de abordar um cliente que não cancelaria. Sem parâmetros reais de negócio, a escolha final deve ser tratada como decisão a validar com stakeholders.

## Análise de custo
Cenário acadêmico hipotético usado no código:

- Custo de abordagem: `10.0`.
- Perda estimada por churn: `100.0`.
- Taxa de sucesso de retenção: `0.2`.
- Custo de falso positivo: custo de abordar um cliente que não cancelaria.
- Custo de falso negativo: perda estimada por churn não abordado.
- Benefício líquido esperado: benefício dos verdadeiros positivos retidos menos custos de abordagem e falsos negativos.

Esses valores não são dados reais da IBM nem de uma operadora.

## MLflow
```bash
make mlflow
```

Os scripts de treino registram parâmetros, métricas e artefatos. Use a UI do MLflow para inspecionar runs após `make train-baselines` e `make train-mlp`.

## API
Inicie a API:

```bash
make run-api
```

Health real validado:

```bash
curl -s http://127.0.0.1:8000/health
```

Resposta:

```json
{"ok": true}
```

Predição real validada:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No","tenure":12,"phone_service":"Yes","multiple_lines":"No","internet_service":"DSL","online_security":"No","online_backup":"Yes","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":70.0,"total_charges":840.0}'
```

Resposta HTTP 200 validada:

```json
{
  "churn_probability": 0.7876151204109192,
  "prediction": 1,
  "threshold": 0.2,
  "risk_level": "high",
  "model_version": "1.0.0"
}
```

Payload inválido retorna `HTTP 422 Unprocessable Entity`.

Níveis de risco: low < 0.30; medium entre 0.30 e 0.60; high >= 0.60.

## Qualidade
Resultados validados:

```text
ruff check .: aprovado
pytest: 10 passed, 0 failed, 0 skipped
```

Há um warning de depreciação entre Starlette TestClient e httpx; ele não afeta os testes atuais.

## Limitações
- Baselines consolidados no README estão como médias de validação cruzada; para comparação final estritamente justa, gere também métricas dos baselines no mesmo conjunto de teste.
- Parâmetros de custo são hipotéticos e acadêmicos.
- O modelo depende da distribuição do IBM Telco e precisa de monitoramento de drift.
- O modelo estima risco, não causalidade.
- Não deve definir automaticamente preço, tratamento comercial ou elegibilidade.

## Roteiro STAR — aproximadamente 5 minutos

### Situation
Churn representa perda de receita recorrente e pressiona custos de aquisição. Usei o IBM Telco Customer Churn, com 7.043 clientes e classe positiva minoritária de aproximadamente 26,54%, para simular um problema de priorização de retenção.

### Task
A tarefa foi criar um pipeline end-to-end reprodutível: limpar dados, fazer EDA, treinar baselines e uma MLP PyTorch, rastrear experimentos com MLflow, analisar custo e servir inferência via API FastAPI.

### Action
Normalizei colunas, removi identificador sem valor preditivo, converti `total_charges`, apliquei split estratificado e usei `ColumnTransformer` com imputação, scaler e OneHotEncoder. Comparei DummyClassifier, Logistic Regression e Random Forest com validação cruzada estratificada. Treinei uma MLP com BCEWithLogitsLoss, mini-batches, Adam e early stopping. O threshold foi escolhido no conjunto de validação por cenário acadêmico de custo e depois avaliado no teste. Também implementei API, logging, testes automatizados e documentação.

### Result
Os testes passaram com 10 aprovados e o ruff passou sem erros. A API respondeu `/health` com `ok: true`, `/predict` com probabilidade de churn e payload inválido com 422. Logistic Regression mostrou melhor equilíbrio geral nos resultados atuais, enquanto a MLP com threshold 0.2 atingiu recall de aproximadamente 94,66%, com trade-off de mais falsos positivos. O modelo apoia priorização de ações de retenção, mas não evita churn por si só e precisa de validação de negócio e monitoramento.
