# Model Card — Churn Prediction

## Visão geral
Modelo principal: MLP PyTorch para classificação binária de churn no IBM Telco Customer Churn. O modelo estima risco de churn; ele não prova causalidade.

## Treinamento
- Data de treinamento: não versionada neste repositório; consultar `models/metadata.json` do ambiente treinado.
- Dataset hash: não versionado neste repositório; consultar `models/metadata.json` do ambiente treinado.
- Dataset: IBM Telco Customer Churn, 7.043 clientes.
- Distribuição alvo: aproximadamente 73,46% não churn e 26,54% churn.
- Seed: 42.
- Feature count após pré-processamento: 45.
- Model version: 1.0.0.

## Features e pré-processamento
O pipeline remove `customer_id`, converte `total_charges`, imputa numéricas por mediana, escala com `StandardScaler`, imputa categóricas por moda e codifica categorias com `OneHotEncoder(handle_unknown="ignore")`.

## Arquitetura
MLP com camadas `Linear(input_size, 64)`, ReLU, Dropout 0.3, `Linear(64, 32)`, ReLU, Dropout 0.2 e `Linear(32, 1)`. O treinamento usa `BCEWithLogitsLoss`; sigmoid é aplicado apenas na inferência.

## Threshold
- Threshold final: 0.2.
- Threshold selecionado no conjunto: validação.
- Métricas finais calculadas no conjunto: teste.
- Critério: cenário acadêmico hipotético de custo.

## Métricas finais da MLP
| Métrica | Valor |
| ------- | ----: |
| Accuracy | 0.591296 |
| Precision | 0.389458 |
| Recall | 0.946619 |
| F1 | 0.551867 |
| ROC-AUC | 0.843192 |
| PR-AUC | 0.650842 |

## Comparação com baselines
Durante o desenvolvimento, `models/baseline_comparison.csv` registra médias de validação cruzada dos baselines. A comparação final equivalente deve usar `models/final_model_comparison.csv`, gerado por `make train-mlp`, com DummyClassifier, LogisticRegression, RandomForestClassifier e PyTorch MLP avaliados no mesmo conjunto de teste.

### Validação cruzada dos baselines

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| ------ | -------: | --------: | -----: | -: | ------: | -----: |
| DummyClassifier | 0.734630 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.265370 |
| LogisticRegression | 0.745704 | 0.513362 | 0.801484 | 0.625824 | 0.844875 | 0.655381 |
| RandomForestClassifier | 0.771118 | 0.560268 | 0.637214 | 0.596175 | 0.822030 | 0.600438 |

A conclusão sobre melhor equilíbrio deve ser revisada a partir da tabela final de teste, não das médias de validação cruzada. A MLP prioriza cobertura de churn com recall alto, aceitando mais falsos positivos.

## Falsos positivos e falsos negativos
- Falso positivo: cliente abordado apesar de não churnar; gera custo de campanha e possível incômodo.
- Falso negativo: cliente em churn não priorizado; representa perda potencial de receita.
- O threshold baixo reduz falsos negativos e aumenta falsos positivos.

## Usos recomendados
- Priorização de campanhas de retenção.
- Apoio à análise operacional de risco de churn.
- Simulações de threshold e custo com stakeholders.

## Usos não recomendados
- Decidir automaticamente preço, tratamento, elegibilidade ou concessão de benefícios.
- Usar como prova causal de churn.
- Aplicar sem monitoramento em população diferente do dataset.

## Limitações, vieses e riscos
O modelo depende da distribuição do IBM Telco, pode refletir vieses históricos e pode degradar com drift de dados, mudanças de produto, preço ou atendimento. Deve ser monitorado e reavaliado periodicamente.

## Plano de reavaliação
Reavaliar mensalmente ou após mudanças relevantes de oferta, política comercial ou distribuição de clientes. Monitorar data drift, concept drift, precision, recall, PR-AUC, taxa prevista de churn e distribuição das probabilidades.
