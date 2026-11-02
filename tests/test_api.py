import json
from datetime import datetime

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app


class FakeModel:
    def predict_proba(self, frame):
        assert len(frame) == 1
        return np.array([[0.25, 0.75]])


def valid_payload():
    return {
        "items_purchased": 2,
        "transaction_amount": 24.5,
        "avg_7day_spend": 18,
        "account_age_months": 8,
        "mcc_code": "Retail_A",
        "product_category": "Clothing",
        "last_login_time": "2023-01-01T10:00:00",
        "transaction_datetime": "2023-01-01T10:30:00",
    }


def test_predict_validates_and_returns_latency():
    client = TestClient(create_app(FakeModel(), "test-model"))
    response = client.post("/predict", json=valid_payload(), headers={"x-request-id": "test-1"})
    assert response.status_code == 200
    assert response.json()["fraud_probability"] == 0.75
    assert response.json()["predicted_fraud"] is True
    assert response.json()["latency_ms"] >= 0
    assert response.json()["request_id"] == "test-1"


def test_predict_logs_features_and_timestamp(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    with TestClient(create_app(FakeModel(), "test-model")) as client:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["event"] == "fraud_prediction"
    assert record["model_version"] == "test-model"
    assert record["features"]["transaction_amount"] == 24.5
    assert record["timestamp"]


def test_predict_rejects_unknown_or_invalid_features():
    client = TestClient(create_app(FakeModel(), "test-model"))
    payload = valid_payload() | {"transaction_amount": "Infinity"}
    assert client.post("/predict", json=payload).status_code == 422
    assert client.post("/predict", json=valid_payload() | {"unexpected": 1}).status_code == 422


def test_predict_allows_missing_features_for_pipeline_imputation():
    client = TestClient(create_app(FakeModel(), "test-model"))
    assert client.post("/predict", json={}).status_code == 200


def test_homepage_serves_prediction_form():
    client = TestClient(create_app(FakeModel(), "test-model"))
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="prediction-form"' in response.text
