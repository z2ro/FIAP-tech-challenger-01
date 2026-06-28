.PHONY: install lint test train-baselines train-mlp mlflow run-api all
install:
	python -m pip install -e .
lint:
	ruff check .
test:
	PYTHONPATH=src pytest
train-baselines:
	PYTHONPATH=src python -m churn_prediction.train_baselines
train-mlp:
	PYTHONPATH=src python -m churn_prediction.train_mlp
mlflow:
	mlflow ui
run-api:
	PYTHONPATH=src uvicorn churn_prediction.api:app --reload
all: lint test
