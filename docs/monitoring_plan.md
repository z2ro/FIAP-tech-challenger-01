# Plano de Monitoramento

Monitorar latência, taxa de erro, volume de requisições, distribuição das probabilidades, taxa prevista de churn, data drift, concept drift, valores ausentes, categorias desconhecidas, precision, recall e PR-AUC quando rótulos chegarem.

Frequência: operacional diária; qualidade mensal. Alertas para p95 acima do SLO, erro 5xx > 1%, drift relevante ou queda de recall. Responsáveis: dados e retenção. Playbook: pausar automações críticas, investigar dados, reavaliar threshold e retreinar se necessário.
