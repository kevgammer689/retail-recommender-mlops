from pathlib import Path

import polars as pl


# Raíz del proyecto:
# retail-recommender-mlops/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Archivo original grande de Instacart.
ORDER_PRODUCTS_PRIOR_PATH = (
    PROJECT_ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
)

# Carpeta donde guardaremos datasets procesados.
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Dataset v1 para entrenamiento inicial del recomendador.
OUTPUT_PATH = PROCESSED_DATA_DIR / "training_baskets_v1.parquet"


# Criterios v1:
# - product_purchase_count >= 100: productos con evidencia suficiente.
# - basket_size entre 2 y 25: canastas útiles y dentro del p95 observado.
MIN_PRODUCT_PURCHASES = 100
MIN_BASKET_SIZE = 2
MAX_BASKET_SIZE = 25


def main() -> None:
    """
    Construye el dataset v1 de entrenamiento del recomendador.

    Este dataset es más pequeño que training_baskets_filtered.parquet
    para permitir entrenar el primer modelo en una máquina local sin exigir demasiada RAM.

    No es el dataset final del proyecto. Es una versión práctica para:
    1. Entrenar el recomendador item-item inicial.
    2. Validar lógica de inferencia.
    3. Conectar FastAPI.
    4. Luego desplegar en Vertex AI.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading source file with Polars lazy scan:")
    print(ORDER_PRODUCTS_PRIOR_PATH)

    order_products = pl.scan_csv(ORDER_PRODUCTS_PRIOR_PATH)

    print("Computing valid basket sizes...")

    basket_sizes = (
        order_products
        .group_by("order_id")
        .agg(pl.len().alias("basket_size"))
        .filter(
            (pl.col("basket_size") >= MIN_BASKET_SIZE)
            & (pl.col("basket_size") <= MAX_BASKET_SIZE)
        )
    )

    print("Computing valid product frequencies...")

    product_frequency = (
        order_products
        .group_by("product_id")
        .agg(pl.len().alias("product_purchase_count"))
        .filter(pl.col("product_purchase_count") >= MIN_PRODUCT_PURCHASES)
    )

    print("Building v1 training dataset...")

    training_baskets_v1 = (
        order_products
        .join(basket_sizes, on="order_id", how="inner")
        .join(product_frequency, on="product_id", how="inner")
        .select(
            "order_id",
            "product_id",
            "add_to_cart_order",
            "reordered",
            "basket_size",
            "product_purchase_count",
        )
    )

    print("Writing parquet output:")
    print(OUTPUT_PATH)

    training_baskets_v1.sink_parquet(OUTPUT_PATH)

    print("Training v1 parquet created successfully.")

    validation_summary = (
        pl.scan_parquet(OUTPUT_PATH)
        .select(
            pl.len().alias("rows"),
            pl.col("order_id").n_unique().alias("orders"),
            pl.col("product_id").n_unique().alias("products"),
            pl.col("basket_size").min().alias("min_basket_size"),
            pl.col("basket_size").max().alias("max_basket_size"),
            pl.col("product_purchase_count").min().alias("min_product_purchase_count"),
        )
        .collect()
    )

    print("\nValidation summary:")
    print(validation_summary)

    pair_workload = (
        pl.scan_parquet(OUTPUT_PATH)
        .select("order_id", "basket_size")
        .unique()
        .with_columns(
            (
                pl.col("basket_size")
                * (pl.col("basket_size") - 1)
                / 2
            ).alias("pairs_per_order")
        )
        .select(
            pl.col("pairs_per_order").sum().alias("estimated_total_pairs"),
            pl.col("pairs_per_order").mean().alias("mean_pairs_per_order"),
            pl.col("pairs_per_order").max().alias("max_pairs_per_order"),
        )
        .collect()
    )

    print("\nEstimated product-pair workload:")
    print(pair_workload)


if __name__ == "__main__":
    main()