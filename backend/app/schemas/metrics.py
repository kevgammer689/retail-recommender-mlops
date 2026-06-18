from datetime import datetime

from pydantic import BaseModel


class MetricsSummaryResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float | None
    avg_recommendation_count: float | None
    last_request_at: datetime | None
