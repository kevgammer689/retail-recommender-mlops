from datetime import datetime
from decimal import Decimal
from functools import lru_cache

from sqlalchemy import text

from backend.app.core.database import get_database_engine


METRICS_SUMMARY_QUERY = text(
    """
    SELECT
        COUNT(*) AS total_requests,
        AVG(latency_ms) AS avg_latency_ms,
        AVG(recommendation_count) AS avg_recommendation_count,
        MAX(created_at) AS last_request_at
    FROM recommendation_logs
    """
)


class MetricsService:
    def get_summary(self) -> dict[str, int | float | datetime | None]:
        with get_database_engine().connect() as connection:
            result = connection.execute(METRICS_SUMMARY_QUERY).mappings().one()

        return {
            "total_requests": int(result["total_requests"]),
            "avg_latency_ms": self._to_float(result["avg_latency_ms"]),
            "avg_recommendation_count": self._to_float(
                result["avg_recommendation_count"]
            ),
            "last_request_at": result["last_request_at"],
        }

    @staticmethod
    def _to_float(value: Decimal | float | None) -> float | None:
        return None if value is None else float(value)


@lru_cache(maxsize=1)
def get_metrics_service() -> MetricsService:
    return MetricsService()
