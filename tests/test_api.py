from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_schema():
    payload = {
        "transaction_id": "TXN_TEST_01",
        "user_id": 1012,
        "amount": 250.0,
        "merchant_id": 2005,
        "location": "Delhi",
        "device_type": "iOS",
        "transaction_type": "purchase",
        "location_deviation": 0,
        "mins_since_last_txn": 120.0,
        "hour": 14,
        "dayofweek": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
