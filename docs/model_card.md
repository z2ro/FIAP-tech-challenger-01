# Model Card — Churn Prediction

## Visão geral
Modelo principal planejado: MLP PyTorch para classificação binária de churn no IBM Telco Customer Churn.

## Objetivo e público-alvo
Apoiar times de retenção a priorizar contatos com clientes sob risco de churn.

## Dataset
Esperado em `data/raw/telco_customer_churn.csv`. Métricas devem ser preenchidas após treinamento real.

## Features e pré-processamento
Remove `customer_id`, converte `total_charges`, imputa numéricas por mediana, escala com `StandardScaler`, imputa categóricas por moda e codifica com OneHotEncoder.

## Modelo, métricas e threshold
Modelo: MLP 64-32 com dropout. Métricas: **PENDENTE_EXECUCAO**. Threshold: **PENDENTE_EXECUCAO**.

## Limitações, vieses e riscos
Pode refletir vieses históricos de atendimento, preços e cobertura. Não deve ser usado para decisões discriminatórias ou sem revisão humana.

## Cenários de falha
Drift de ofertas, categorias novas, dados ausentes excessivos, mudança de política comercial.

## Uso recomendado / não recomendado
Recomendado para priorização de campanhas. Não recomendado como única base para negar serviços ou benefícios.

## Plano de reavaliação
Reavaliar mensalmente ou após mudança relevante de produto/preço.
