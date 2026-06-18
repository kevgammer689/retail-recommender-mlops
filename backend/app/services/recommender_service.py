from functools import lru_cache
from typing import Any

import pandas as pd

from ml.serving.local_recommender import LocalItemItemRecommender


class RecommenderService:
    """Adapts the pandas recommender output to API-friendly native values."""

    def __init__(self) -> None:
        self.recommender = LocalItemItemRecommender()

    def get_model_info(self) -> dict[str, str | int]:
        model = self.recommender.model
        return {
            "model_type": "item_item_cooccurrence_recommender",
            "serving_mode": "local",
            "total_rows": int(len(model)),
            "total_source_products": int(model["source_product_id"].nunique()),
            "total_recommended_products": int(
                model["recommended_product_id"].nunique()
            ),
        }

    def recommend_for_cart(
        self,
        cart_product_ids: list[int],
        top_k: int,
    ) -> dict[str, list[dict[str, Any]]]:
        cart_products = self.recommender.get_cart_products(cart_product_ids)
        recommendations = self.recommender.recommend_for_cart(
            cart_product_ids,
            top_k=top_k,
        )
        return {
            "cart_products": self._to_native_records(cart_products),
            "recommendations": self._to_native_records(recommendations),
        }

    @staticmethod
    def _to_native_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in dataframe.to_dict(orient="records"):
            records.append(
                {
                    key: value.item() if hasattr(value, "item") else value
                    for key, value in row.items()
                }
            )
        return records


@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    return RecommenderService()
