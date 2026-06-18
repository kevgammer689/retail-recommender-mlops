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


router = APIRouter()
RecommenderServiceDependency = Annotated[
    RecommenderService,
    Depends(get_recommender_service),
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
) -> RecommendResponse:
    result = service.recommend_for_cart(
        cart_product_ids=payload.cart_product_ids,
        top_k=payload.top_k,
    )
    return RecommendResponse(**result)
