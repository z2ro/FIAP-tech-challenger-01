.PHONY: install lint test train-baselines train-mlp mlflow run-api all
install:
	python -m pip install -e .
lint:
	ruff check .
test:
	pytest
train-baselines:
	python -m churn_prediction.train_baselines
train-mlp:
	python -m churn_prediction.train_mlp
mlflow:
	mlflow ui
run-api:
	uvicorn churn_prediction.api:app --reload
all: lint test
