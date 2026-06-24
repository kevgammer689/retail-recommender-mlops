from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelInfoResponse(BaseModel):
    model_type: str
    serving_mode: str
    model_version: str
    scoring: str
    total_rows: int
    total_source_products: int
    total_recommended_products: int


class RecommendRequest(BaseModel):
    cart_product_ids: list[int] = Field(min_length=1)
    top_k: int = Field(default=10, gt=0)


class CartProduct(BaseModel):
    product_id: int
    product_name: str
    aisle_id: int
    department_id: int


class Recommendation(BaseModel):
    recommended_product_id: int
    product_name: str
    aisle_id: int
    department_id: int
    score: float
    cooccurrence_count: int
    matched_cart_products: int


class RecommendResponse(BaseModel):
    cart_products: list[CartProduct]
    recommendations: list[Recommendation]
