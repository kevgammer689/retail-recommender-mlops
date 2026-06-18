import json
from functools import lru_cache
from uuid import UUID

from sqlalchemy import text

from backend.app.core.database import get_database_engine


INSERT_RECOMMENDATION_LOG = text(
    """
    INSERT INTO recommendation_logs (
        request_id,
        cart_product_ids,
        top_k,
        recommendation_count,
        recommended_product_ids,
        model_type,
        serving_mode,
        latency_ms
    )
    VALUES (
        :request_id,
        CAST(:cart_product_ids AS JSONB),
        :top_k,
        :recommendation_count,
        CAST(:recommended_product_ids AS JSONB),
        :model_type,
        :serving_mode,
        :latency_ms
    )
    """
)


class RecommendationLogService:
    def log_recommendation(
        self,
        *,
        request_id: UUID,
        cart_product_ids: list[int],
        top_k: int,
        recommendation_count: int,
        recommended_product_ids: list[int],
        model_type: str,
        serving_mode: str,
        latency_ms: float,
    ) -> None:
        parameters = {
            "request_id": str(request_id),
            "cart_product_ids": json.dumps([int(value) for value in cart_product_ids]),
            "top_k": int(top_k),
            "recommendation_count": int(recommendation_count),
            "recommended_product_ids": json.dumps(
                [int(value) for value in recommended_product_ids]
            ),
            "model_type": str(model_type),
            "serving_mode": str(serving_mode),
            "latency_ms": float(latency_ms),
        }

        with get_database_engine().begin() as connection:
            connection.execute(INSERT_RECOMMENDATION_LOG, parameters)


@lru_cache(maxsize=1)
def get_recommendation_log_service() -> RecommendationLogService:
    return RecommendationLogService()
