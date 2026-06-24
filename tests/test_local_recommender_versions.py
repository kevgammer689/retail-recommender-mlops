from pathlib import Path

from ml.serving.local_recommender import LocalItemItemRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v1.parquet"
)
V2_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v2.parquet"
)
PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "instacart" / "products.csv"


def test_default_recommender_prefers_v2() -> None:
    recommender = LocalItemItemRecommender()

    assert recommender.model_path == V2_MODEL_PATH
    assert recommender.model_version == "v2"
    assert recommender.scoring == "lift_log_cooccurrence"
    assert "final_score" in recommender.model.columns
    assert recommender.model["score"].equals(recommender.model["final_score"])


def test_explicit_v1_path_remains_supported() -> None:
    recommender = LocalItemItemRecommender(V1_MODEL_PATH, PRODUCTS_PATH)
    recommendations = recommender.recommend_for_cart([24852], top_k=3)

    assert recommender.model_version == "v1"
    assert recommender.scoring == "confidence"
    assert len(recommendations) == 3
    assert list(recommendations.columns) == [
        "recommended_product_id",
        "product_name",
        "aisle_id",
        "department_id",
        "score",
        "cooccurrence_count",
        "matched_cart_products",
    ]
