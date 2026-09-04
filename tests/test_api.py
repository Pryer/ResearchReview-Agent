"""FastAPI 路由冒烟测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["status"] == "healthy"


def test_openapi_schema_contains_core_routes():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/papers/search" in paths
    assert "/api/reviews/jobs" in paths
    assert "/api/library/search" in paths
    # 已废弃的同步入口不应再出现在 OpenAPI 中
    assert "/api/reviews/agent" not in paths
    assert "/api/reviews/generate" not in paths


def test_optional_api_key_protects_metrics_and_business_routes():
    settings = get_settings()
    previous = settings.app_api_key
    settings.app_api_key = "test-secret"
    try:
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 401
        assert client.get(
            "/metrics", headers={"X-API-Key": "test-secret"}
        ).status_code == 200
        assert client.get("/api/reviews/sessions/missing").status_code == 401
    finally:
        settings.app_api_key = previous
