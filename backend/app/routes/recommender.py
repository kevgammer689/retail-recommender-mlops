import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.schemas.recommender import (
    HealthResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)
from backend.app.services.recommender_service import (
    RecommenderService,
    get_recommender_service,
)
from backend.app.services.recommendation_log_service import (
    RecommendationLogService,
    get_recommendation_log_service,
)


router = APIRouter()
logger = logging.getLogger(__name__)
RecommenderServiceDependency = Annotated[
    RecommenderService,
    Depends(get_recommender_service),
]
RecommendationLogServiceDependency = Annotated[
    RecommendationLogService,
    Depends(get_recommendation_log_service),
]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="retail-recommender-api")


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    service: RecommenderServiceDependency,
) -> ModelInfoResponse:
    return ModelInfoResponse(**service.get_model_info())


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    payload: RecommendRequest,
    service: RecommenderServiceDependency,
    log_service: RecommendationLogServiceDependency,
) -> RecommendResponse:
    started_at = time.perf_counter()
    result = service.recommend_for_cart(
        cart_product_ids=payload.cart_product_ids,
        top_k=payload.top_k,
    )
    response = RecommendResponse(**result)
    latency_ms = (time.perf_counter() - started_at) * 1000
    model_info = service.get_model_info()

    try:
        log_service.log_recommendation(
            request_id=uuid.uuid4(),
            cart_product_ids=payload.cart_product_ids,
            top_k=payload.top_k,
            recommendation_count=len(response.recommendations),
            recommended_product_ids=[
                item.recommended_product_id for item in response.recommendations
            ],
            model_type=str(model_info["model_type"]),
            serving_mode=str(model_info["serving_mode"]),
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("Failed to persist recommendation request log")

    return response
