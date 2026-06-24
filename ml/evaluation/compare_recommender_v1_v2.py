import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.serving.local_recommender import LocalItemItemRecommender


V1_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v1.parquet"
)
V2_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v2.parquet"
)
PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "instacart" / "products.csv"

TEST_CARTS = [
    [24852, 21137, 47766],
    [47626, 26209],
    [27845, 21903],
    [49683, 28204],
    [16797, 27966],
]
TOP_K = 5


def print_recommendations(
    label: str,
    recommender: LocalItemItemRecommender,
    cart_product_ids: list[int],
) -> list[str]:
    recommendations = recommender.recommend_for_cart(
        cart_product_ids,
        top_k=TOP_K,
    )
    names = recommendations["product_name"].fillna("Unknown product").tolist()

    print(label)
    if recommendations.empty:
        print("  No recommendations")
        return names

    for rank, row in enumerate(recommendations.itertuples(), start=1):
        print(
            f"  {rank}. {row.product_name} "
            f"(id={row.recommended_product_id}, score={row.score:.6f}, "
            f"cooccurrence={row.cooccurrence_count})"
        )
    return names


def main() -> None:
    for path in (V1_MODEL_PATH, V2_MODEL_PATH, PRODUCTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    v1 = LocalItemItemRecommender(V1_MODEL_PATH, PRODUCTS_PATH)
    v2 = LocalItemItemRecommender(V2_MODEL_PATH, PRODUCTS_PATH)

    repeated_products = {
        "Banana",
        "Bag of Organic Bananas",
        "Organic Baby Spinach",
    }
    v1_repeated_count = 0
    v2_repeated_count = 0

    for index, cart in enumerate(TEST_CARTS, start=1):
        print("\n" + "=" * 88)
        print(f"Cart {index}: {cart}")
        cart_products = v1.get_cart_products(cart)
        print(
            "Cart products: "
            + ", ".join(cart_products["product_name"].fillna("Unknown product"))
        )

        v1_names = print_recommendations("Top 5 v1:", v1, cart)
        v2_names = print_recommendations("Top 5 v2:", v2, cart)
        v1_repeated_count += sum(name in repeated_products for name in v1_names)
        v2_repeated_count += sum(name in repeated_products for name in v2_names)

    print("\n" + "=" * 88)
    print("Popularity-prone products in all displayed top-5 lists:")
    print(f"  v1: {v1_repeated_count}")
    print(f"  v2: {v2_repeated_count}")


if __name__ == "__main__":
    main()
