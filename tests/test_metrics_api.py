from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.metrics_service import MetricsService


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_summary_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    last_request_at = datetime(2026, 6, 18, 21, 33, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(
        MetricsService,
        "get_summary",
        lambda self: {
            "total_requests": 3,
            "avg_latency_ms": 50.5,
            "avg_recommendation_count": 10.0,
            "last_request_at": last_request_at,
        },
    )

    response = client.get("/metrics/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_requests": 3,
        "avg_latency_ms": 50.5,
        "avg_recommendation_count": 10.0,
        "last_request_at": "2026-06-18T21:33:02Z",
    }


def test_metrics_summary_empty(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MetricsService,
        "get_summary",
        lambda self: {
            "total_requests": 0,
            "avg_latency_ms": None,
            "avg_recommendation_count": None,
            "last_request_at": None,
        },
    )

    response = client.get("/metrics/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_requests": 0,
        "avg_latency_ms": None,
        "avg_recommendation_count": None,
        "last_request_at": None,
    }


def test_metrics_summary_database_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_query(self: MetricsService) -> dict[str, object]:
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(MetricsService, "get_summary", fail_query)

    response = client.get("/metrics/summary")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Recommendation metrics are temporarily unavailable"
    }
