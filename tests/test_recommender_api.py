import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.recommendation_log_service import RecommendationLogService


SAMPLE_CART_PRODUCT_IDS = [24852, 21137, 47766]


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def disable_database_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RecommendationLogService,
        "log_recommendation",
        lambda self, **log_data: None,
    )


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "retail-recommender-api",
    }


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    model_info = response.json()
    assert model_info["model_type"] == "item_item_cooccurrence_recommender"
    assert model_info["serving_mode"] == "local"
    assert model_info["total_rows"] > 0
    assert model_info["total_source_products"] > 0
    assert model_info["total_recommended_products"] > 0


def test_recommend_endpoint_success(client: TestClient) -> None:
    response = client.post(
        "/recommend",
        json={
            "cart_product_ids": SAMPLE_CART_PRODUCT_IDS,
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "cart_products" in body
    assert "recommendations" in body
    assert len(body["cart_products"]) == 3
    assert len(body["recommendations"]) > 0

    cart_product_ids = set(SAMPLE_CART_PRODUCT_IDS)
    for recommendation in body["recommendations"]:
        assert recommendation["recommended_product_id"] not in cart_product_ids
        assert "matched_cart_products" in recommendation


def test_recommend_empty_cart_validation(client: TestClient) -> None:
    response = client.post(
        "/recommend",
        json={
            "cart_product_ids": [],
            "top_k": 10,
        },
    )

    assert response.status_code == 422


def test_recommend_invalid_top_k_validation(client: TestClient) -> None:
    response = client.post(
        "/recommend",
        json={
            "cart_product_ids": [24852],
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_recommend_logs_successful_request(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged_requests: list[dict[str, object]] = []

    def capture_log(
        self: RecommendationLogService,
        **log_data: object,
    ) -> None:
        logged_requests.append(log_data)

    monkeypatch.setattr(RecommendationLogService, "log_recommendation", capture_log)

    response = client.post(
        "/recommend",
        json={
            "cart_product_ids": SAMPLE_CART_PRODUCT_IDS,
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    assert len(logged_requests) == 1

    logged_request = logged_requests[0]
    recommendations = response.json()["recommendations"]
    assert logged_request["cart_product_ids"] == SAMPLE_CART_PRODUCT_IDS
    assert logged_request["top_k"] == 10
    assert logged_request["recommendation_count"] == len(recommendations)
    assert logged_request["recommended_product_ids"] == [
        item["recommended_product_id"] for item in recommendations
    ]
    assert logged_request["model_type"] == "item_item_cooccurrence_recommender"
    assert logged_request["serving_mode"] == "local"
    assert isinstance(logged_request["latency_ms"], float)
    assert logged_request["latency_ms"] >= 0


def test_recommend_succeeds_when_logging_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_logging(
        self: RecommendationLogService,
        **log_data: object,
    ) -> None:
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr(RecommendationLogService, "log_recommendation", fail_logging)

    response = client.post(
        "/recommend",
        json={
            "cart_product_ids": SAMPLE_CART_PRODUCT_IDS,
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) > 0
