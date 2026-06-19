import argparse
import sys
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v1.parquet"
)
PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "instacart" / "products.csv"
SAMPLE_CART_PRODUCT_IDS = [24852, 21137, 47766]
REQUIRED_METRICS_FIELDS = {
    "total_requests",
    "avg_latency_ms",
    "avg_recommendation_count",
    "last_request_at",
}


class CheckFailure(RuntimeError):
    pass


def print_ok(name: str) -> None:
    print(f"[OK] {name}")


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise CheckFailure(detail)


def require_json_object(response: httpx.Response, endpoint: str) -> dict[str, Any]:
    require(
        response.status_code == 200,
        f"{endpoint} returned HTTP {response.status_code}: {response.text}",
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CheckFailure(f"{endpoint} returned invalid JSON") from exc

    require(isinstance(payload, dict), f"{endpoint} did not return a JSON object")
    return payload


def check_required_files() -> None:
    require(MODEL_PATH.is_file(), f"missing required artifact: {MODEL_PATH}")
    print_ok("artifact exists")

    require(PRODUCTS_PATH.is_file(), f"missing required catalog: {PRODUCTS_PATH}")
    print_ok("products.csv exists")


def check_api(base_url: str) -> None:
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        health = require_json_object(client.get("/health"), "/health")
        require(health.get("status") == "ok", "/health status is not 'ok'")
        print_ok("/health")

        model_info = require_json_object(client.get("/model-info"), "/model-info")
        require(
            model_info.get("model_type") == "item_item_cooccurrence_recommender",
            "/model-info returned an unexpected model_type",
        )
        require(
            isinstance(model_info.get("total_rows"), int)
            and model_info["total_rows"] > 0,
            "/model-info reports an empty model artifact",
        )
        print_ok("/model-info")

        recommendation = require_json_object(
            client.post(
                "/recommend",
                json={
                    "cart_product_ids": SAMPLE_CART_PRODUCT_IDS,
                    "top_k": 10,
                },
            ),
            "/recommend",
        )
        validate_recommendation(recommendation)
        print_ok("/recommend")

        metrics = require_json_object(
            client.get("/metrics/summary"),
            "/metrics/summary",
        )
        validate_metrics(metrics)
        print_ok("/metrics/summary")


def validate_recommendation(payload: dict[str, Any]) -> None:
    cart_products = payload.get("cart_products")
    recommendations = payload.get("recommendations")

    require(isinstance(cart_products, list), "/recommend is missing cart_products")
    require(
        isinstance(recommendations, list),
        "/recommend is missing recommendations",
    )
    require(len(recommendations) > 0, "/recommend returned no recommendations")

    cart_ids = set(SAMPLE_CART_PRODUCT_IDS)
    for item in recommendations:
        require(isinstance(item, dict), "/recommend contains an invalid item")
        require(
            item.get("recommended_product_id") not in cart_ids,
            "/recommend returned a product already present in the cart",
        )
        require(
            "matched_cart_products" in item,
            "/recommend item is missing matched_cart_products",
        )


def validate_metrics(payload: dict[str, Any]) -> None:
    missing_fields = REQUIRED_METRICS_FIELDS.difference(payload)
    require(
        not missing_fields,
        f"/metrics/summary is missing fields: {sorted(missing_fields)}",
    )
    require(
        isinstance(payload["total_requests"], int),
        "/metrics/summary total_requests is not an integer",
    )
    require(
        payload["total_requests"] > 0,
        "/metrics/summary reports no recommendation logs",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the local retail recommender stack end to end."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL (default: http://127.0.0.1:8000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        check_required_files()
        check_api(base_url)
    except CheckFailure as exc:
        print(f"[FAIL] {exc}")
        return 1
    except httpx.HTTPError as exc:
        print(f"[FAIL] unable to reach API at {base_url}: {exc}")
        return 1

    print("[OK] local stack validation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
