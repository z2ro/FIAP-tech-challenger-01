# ML Canvas

- Problema: churn de clientes reduz receita recorrente.
- Stakeholders: retenção, marketing, atendimento, dados.
- Usuários: analistas e sistemas de campanha.
- Decisão suportada: priorizar contato de retenção.
- Fonte de dados: IBM Telco Customer Churn.
- Variável alvo: `churn`.
- Métricas técnicas: recall, precision, F1, ROC-AUC, PR-AUC.
- Métricas de negócio: benefício líquido, falsos negativos, custo de abordagem.
- Riscos: vieses históricos, drift, dados incompletos.
- Hipóteses: clientes com maior probabilidade podem ser retidos com ação adequada.
- SLOs: API p95 < 300 ms em carga nominal; erro 5xx < 1%.
- Critérios de sucesso: ganho de benefício líquido validado contra baseline operacional.
