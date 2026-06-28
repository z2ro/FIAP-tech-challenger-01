from fastapi.testclient import TestClient

from churn_prediction import api


def test_health_returns_200(test_artifacts):
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_predict_returns_200(test_artifacts, sample_payload):
    api._predictor = None
    client = TestClient(api.app)
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


def test_predict_invalid_returns_422(test_artifacts, sample_payload):
    client = TestClient(api.app)
    response = client.post("/predict", json=sample_payload | {"senior_citizen": 9})
    assert response.status_code == 422
