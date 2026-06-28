# Arquitetura

A inferência em tempo real via FastAPI é adequada porque campanhas e telas de atendimento podem solicitar risco de churn sob demanda com baixa latência.

Fluxo simples de produção:

Cliente → Load Balancer/API Gateway → container FastAPI → artefatos do modelo → logs e métricas → monitoramento.

Esta versão não implementa cloud, Kubernetes, Terraform ou CI/CD. O deploy é opcional e deve evoluir apenas quando houver necessidade operacional.
