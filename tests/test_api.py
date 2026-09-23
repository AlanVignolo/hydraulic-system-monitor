from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"cooler", "valve", "pump", "accumulator"}


def test_predict_returns_all_components(sample_sensor_data):
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_sensor_data)

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert set(predictions.keys()) == {"cooler", "valve", "pump", "accumulator"}
    for component in predictions.values():
        assert isinstance(component["condition"], str)
        assert 0.0 <= component["confidence"] <= 1.0


def test_predict_missing_sensor_returns_422(sample_sensor_data):
    incomplete_data = dict(sample_sensor_data)
    del incomplete_data["PS1"]

    with TestClient(app) as client:
        response = client.post("/predict", json=incomplete_data)

    assert response.status_code == 422
