import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


SAMPLE_CART_PRODUCT_IDS = [24852, 21137, 47766]


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


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
