import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.schemas.metrics import MetricsSummaryResponse
from backend.app.services.metrics_service import MetricsService, get_metrics_service


router = APIRouter(prefix="/metrics", tags=["metrics"])
logger = logging.getLogger(__name__)
MetricsServiceDependency = Annotated[
    MetricsService,
    Depends(get_metrics_service),
]


@router.get("/summary", response_model=MetricsSummaryResponse)
def metrics_summary(
    service: MetricsServiceDependency,
) -> MetricsSummaryResponse:
    try:
        return MetricsSummaryResponse(**service.get_summary())
    except Exception as exc:
        logger.exception("Failed to load recommendation metrics")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation metrics are temporarily unavailable",
        ) from exc
