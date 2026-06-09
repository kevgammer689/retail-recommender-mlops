from pathlib import Path

import polars as pl


# Raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset original de productos por orden.
ORDER_PRODUCTS_PRIOR_PATH = (
    PROJECT_ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
)

# Carpeta de salida para datasets procesados.
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Dataset compacto para el primer entrenamiento local.
OUTPUT_PATH = PROCESSED_DATA_DIR / "training_baskets_v1_compact.parquet"


# Criterios compactos:
# - Productos comprados al menos 1000 veces.
# - Órdenes con 2 a 15 productos.
#
# Objetivo:
# Reducir el número de pares producto-producto para que el primer entrenamiento
# sea viable en local y podamos avanzar hacia FastAPI + Vertex AI.
MIN_PRODUCT_PURCHASES = 1000
MIN_BASKET_SIZE = 2
MAX_BASKET_SIZE = 15


def main() -> None:
    """
    Construye un dataset compacto para entrenar el primer recomendador item-item.

    Este dataset no busca maximizar performance final.
    Busca crear una primera versión estable, rápida y desplegable.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading source file with Polars lazy scan:")
    print(ORDER_PRODUCTS_PRIOR_PATH)

    # Lazy scan: Polars no carga todo el CSV en memoria de inmediato.
    order_products = pl.scan_csv(ORDER_PRODUCTS_PRIOR_PATH)

    print("Computing valid basket sizes...")

    # Calcula el tamaño de cada canasta y conserva solo órdenes útiles.
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

    # Calcula frecuencia histórica por producto.
    # Conservamos productos con suficiente evidencia estadística.
    product_frequency = (
        order_products
        .group_by("product_id")
        .agg(pl.len().alias("product_purchase_count"))
        .filter(pl.col("product_purchase_count") >= MIN_PRODUCT_PURCHASES)
    )

    print("Building compact v1 training dataset...")

    # Une las órdenes válidas y productos válidos.
    training_baskets = (
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

    # Escribe parquet de forma eficiente.
    training_baskets.sink_parquet(OUTPUT_PATH)

    print("Compact v1 training parquet created successfully.")

    # Validación básica del parquet generado.
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

    # Estima cuántos pares producto-producto tendría que procesar el modelo.
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