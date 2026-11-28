from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "SAHAAY AI" in data["message"]

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "SAHAAY AI"
    assert "drishti_vision" in data["services"]
    assert "samvaad_sign" in data["services"]


def test_cors_restricts_unknown_origins():
    disallowed = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert disallowed.status_code == 200
    assert "access-control-allow-origin" not in disallowed.headers

    allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert allowed.status_code == 200
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert allowed.headers.get("access-control-allow-credentials") != "true"
