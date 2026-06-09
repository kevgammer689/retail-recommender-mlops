from pathlib import Path

import polars as pl


# Raíz del proyecto:
# retail-recommender-mlops/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Archivo grande original de Instacart.
ORDER_PRODUCTS_PRIOR_PATH = (
    PROJECT_ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
)

# Carpeta de salida para datasets procesados.
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Dataset filtrado que usaremos para entrenar el primer recomendador.
OUTPUT_PATH = PROCESSED_DATA_DIR / "training_baskets_filtered.parquet"


MIN_PRODUCT_PURCHASES = 25
MIN_BASKET_SIZE = 2
MAX_BASKET_SIZE = 30


def main() -> None:
    """
    Construye un dataset de entrenamiento filtrado usando Polars.

    Este script no usa PostgreSQL porque:
    1. El archivo order_products__prior tiene más de 32M filas.
    2. Para EDA y filtrado, Polars es más cómodo que SQL.
    3. El resultado se guarda como parquet para entrenar rápido después.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading source file with Polars lazy scan:")
    print(ORDER_PRODUCTS_PRIOR_PATH)

    # scan_csv crea un LazyFrame.
    # Eso significa que Polars no carga todo inmediatamente en memoria.
    order_products = pl.scan_csv(ORDER_PRODUCTS_PRIOR_PATH)

    print("Computing basket sizes...")

    # Cantidad de productos por orden.
    basket_sizes = (
        order_products
        .group_by("order_id")
        .agg(
            pl.len().alias("basket_size")
        )
        .filter(
            (pl.col("basket_size") >= MIN_BASKET_SIZE)
            & (pl.col("basket_size") <= MAX_BASKET_SIZE)
        )
    )

    print("Computing product frequencies...")

    # Frecuencia total de compra por producto.
    product_frequency = (
        order_products
        .group_by("product_id")
        .agg(
            pl.len().alias("product_purchase_count")
        )
        .filter(
            pl.col("product_purchase_count") >= MIN_PRODUCT_PURCHASES
        )
    )

    print("Building filtered training dataset...")

    # Filtramos:
    # 1. órdenes con tamaño razonable
    # 2. productos con suficiente frecuencia histórica
    # 3. columnas mínimas necesarias para entrenar recomendador item-item
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

    # sink_parquet permite escribir el resultado sin materializar todo como DataFrame
    # cuando Polars puede ejecutar el plan en modo eficiente.
    training_baskets.sink_parquet(OUTPUT_PATH)

    print("Filtered training parquet created successfully.")

    # Validación ligera del archivo generado.
    result = (
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
    print(result)


if __name__ == "__main__":
    main()