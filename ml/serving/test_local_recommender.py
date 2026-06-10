from pathlib import Path

import polars as pl


# Raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Artifact entrenado.
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "item_item_recommender_v1.parquet"

# Catálogo de productos para traducir product_id a nombres.
PRODUCTS_PATH = PROJECT_ROOT / "data" / "raw" / "instacart" / "products.csv"

# Productos de prueba.
# 24852 suele ser Banana en Instacart.
SAMPLE_CART_PRODUCT_IDS = [24852]

TOP_K = 10


def load_model() -> pl.DataFrame:
    """Carga el artifact de recomendaciones item-item."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    return pl.read_parquet(MODEL_PATH)


def load_products() -> pl.DataFrame:
    """Carga catálogo de productos para mostrar nombres legibles."""
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Products file not found: {PRODUCTS_PATH}")

    return pl.read_csv(PRODUCTS_PATH)


def recommend_for_cart(
    model: pl.DataFrame,
    cart_product_ids: list[int],
    top_k: int,
) -> pl.DataFrame:
    """
    Genera recomendaciones para un carrito.

    Lógica v1:
    - Busca recomendaciones asociadas a cada producto del carrito.
    - Excluye productos que ya están en el carrito.
    - Agrupa recomendaciones repetidas.
    - Suma scores cuando un producto recomendado aparece asociado a varios productos del carrito.
    """

    recommendations = (
        model
        .filter(pl.col("source_product_id").is_in(cart_product_ids))
        .filter(~pl.col("recommended_product_id").is_in(cart_product_ids))
        .group_by("recommended_product_id")
        .agg(
            pl.sum("score").alias("score"),
            pl.sum("cooccurrence_count").alias("cooccurrence_count"),
            pl.len().alias("matched_cart_products"),
        )
        .sort(["score", "cooccurrence_count"], descending=[True, True])
        .head(top_k)
    )

    return recommendations


def main() -> None:
    print("Loading local recommender artifact...")
    model = load_model()

    print("Loading product catalog...")
    products = load_products().select("product_id", "product_name")

    print(f"Input cart product IDs: {SAMPLE_CART_PRODUCT_IDS}")

    cart_products = (
        products
        .filter(pl.col("product_id").is_in(SAMPLE_CART_PRODUCT_IDS))
    )

    print("\nCart products:")
    print(cart_products)

    recommendations = recommend_for_cart(
        model=model,
        cart_product_ids=SAMPLE_CART_PRODUCT_IDS,
        top_k=TOP_K,
    )

    recommendations_with_names = (
        recommendations
        .join(
            products,
            left_on="recommended_product_id",
            right_on="product_id",
            how="left",
        )
        .select(
            "recommended_product_id",
            "product_name",
            "score",
            "cooccurrence_count",
            "matched_cart_products",
        )
    )

    print("\nRecommendations:")
    print(recommendations_with_names)


if __name__ == "__main__":
    main()