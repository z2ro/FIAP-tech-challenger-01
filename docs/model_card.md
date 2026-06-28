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
Durante o desenvolvimento, `models/baseline_comparison.csv` registra médias de validação cruzada dos baselines. A comparação final abaixo usa `models/final_model_comparison.csv`, com DummyClassifier, LogisticRegression, RandomForestClassifier e PyTorch MLP avaliados no mesmo conjunto de teste.

### Resultados finais — mesmo conjunto de teste

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| ------ | -------: | --------: | -----: | -: | ------: | -----: |
| DummyClassifier | 0.734153 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.265847 |
| LogisticRegression | 0.747398 | 0.516509 | 0.779359 | 0.621277 | 0.844627 | 0.658615 |
| RandomForestClassifier | 0.773888 | 0.568182 | 0.622776 | 0.594228 | 0.820746 | 0.611227 |
| PyTorch MLP | 0.591296 | 0.389458 | 0.946619 | 0.551867 | 0.843192 | 0.650842 |

Logistic Regression apresentou melhor equilíbrio geral no teste, com melhor F1, ROC-AUC e PR-AUC. Random Forest teve maior accuracy e precision. A MLP PyTorch é o modelo servido pela API por ser o modelo principal da atividade e por apresentar o maior recall. Com threshold `0.2`, ela identifica aproximadamente 94,66% dos churns, mas gera mais falsos positivos.

A escolha de servir a MLP está relacionada ao objetivo acadêmico e ao alto recall; em produção, os parâmetros de custo são hipotéticos e precisam ser recalibrados com dados reais do negócio antes de decidir entre maior cobertura de churn e melhor equilíbrio geral.

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
