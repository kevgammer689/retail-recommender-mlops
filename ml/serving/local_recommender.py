from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v1.parquet"
)
DEFAULT_PRODUCTS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "instacart" / "products.csv"
)

RECOMMENDATION_COLUMNS = [
    "recommended_product_id",
    "product_name",
    "aisle_id",
    "department_id",
    "score",
    "cooccurrence_count",
    "matched_cart_products",
]
PRODUCT_COLUMNS = ["product_id", "product_name", "aisle_id", "department_id"]


class LocalItemItemRecommender:
    """Local item-item recommender backed by a precomputed parquet artifact."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        products_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.products_path = (
            Path(products_path) if products_path else DEFAULT_PRODUCTS_PATH
        )
        self.model = self._load_model()
        self.products = self._load_products()

    def _load_model(self) -> pd.DataFrame:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        model = pd.read_parquet(self.model_path)
        required_columns = {
            "source_product_id",
            "recommended_product_id",
            "score",
            "cooccurrence_count",
        }
        missing_columns = required_columns.difference(model.columns)
        if missing_columns:
            raise ValueError(
                "Model artifact is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        return model

    def _load_products(self) -> pd.DataFrame:
        if not self.products_path.exists():
            raise FileNotFoundError(f"Products file not found: {self.products_path}")

        products = pd.read_csv(self.products_path)
        missing_columns = set(PRODUCT_COLUMNS).difference(products.columns)
        if missing_columns:
            raise ValueError(
                "Products catalog is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        return products[PRODUCT_COLUMNS]

    def recommend_for_cart(
        self,
        cart_product_ids: Iterable[int],
        top_k: int = 10,
    ) -> pd.DataFrame:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Deduplication avoids counting the same cart product more than once.
        cart_ids = list(dict.fromkeys(cart_product_ids))
        if not cart_ids:
            return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

        candidates = self.model[
            self.model["source_product_id"].isin(cart_ids)
            & ~self.model["recommended_product_id"].isin(cart_ids)
        ]
        if candidates.empty:
            return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

        recommendations = (
            candidates.groupby("recommended_product_id", as_index=False)
            .agg(
                score=("score", "sum"),
                cooccurrence_count=("cooccurrence_count", "sum"),
                matched_cart_products=("source_product_id", "nunique"),
            )
            .sort_values(
                ["score", "cooccurrence_count"],
                ascending=[False, False],
            )
            .head(top_k)
        )

        recommendations = recommendations.merge(
            self.products,
            left_on="recommended_product_id",
            right_on="product_id",
            how="left",
        )

        return recommendations[RECOMMENDATION_COLUMNS].reset_index(drop=True)

    def get_cart_products(
        self,
        cart_product_ids: Iterable[int],
    ) -> pd.DataFrame:
        cart_ids = list(dict.fromkeys(cart_product_ids))
        if not cart_ids:
            return pd.DataFrame(columns=PRODUCT_COLUMNS)

        return (
            self.products[self.products["product_id"].isin(cart_ids)][PRODUCT_COLUMNS]
            .copy()
            .reset_index(drop=True)
        )
