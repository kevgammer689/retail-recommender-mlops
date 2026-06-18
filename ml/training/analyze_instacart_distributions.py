from pathlib import Path

import polars as pl


# Ruta raíz del proyecto:
# retail-recommender-mlops/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpeta donde están los CSV originales de Instacart.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"

ORDER_PRODUCTS_PRIOR_PATH = RAW_DATA_DIR / "order_products__prior.csv"


def print_section(title: str) -> None:
    """Imprime separadores para que la salida sea fácil de leer."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("Loading order_products__prior.csv with Polars lazy scan")

    # scan_csv no carga inmediatamente todo el archivo en memoria.
    # Crea un plan de consulta lazy que Polars optimiza antes de ejecutar.
    order_products = pl.scan_csv(ORDER_PRODUCTS_PRIOR_PATH)

    print("File:", ORDER_PRODUCTS_PRIOR_PATH)

    print_section("Basic row count")

    # collect() ejecuta realmente la consulta lazy.
    row_count = order_products.select(pl.len().alias("rows")).collect()
    print(row_count)

    print_section("Basket size distribution")

    # Calculamos cuántos productos tiene cada orden.
    # Esto sirve para decidir si debemos eliminar canastas de tamaño 1
    # o canastas extremadamente grandes.
    basket_sizes = (
        order_products
        .group_by("order_id")
        .agg(pl.len().alias("basket_size"))
    )

    basket_summary = (
        basket_sizes
        .select(
            pl.len().alias("total_orders"),
            pl.col("basket_size").min().alias("min_basket_size"),
            pl.col("basket_size").quantile(0.25).alias("p25"),
            pl.col("basket_size").quantile(0.50).alias("p50"),
            pl.col("basket_size").quantile(0.75).alias("p75"),
            pl.col("basket_size").quantile(0.90).alias("p90"),
            pl.col("basket_size").quantile(0.95).alias("p95"),
            pl.col("basket_size").quantile(0.99).alias("p99"),
            pl.col("basket_size").max().alias("max_basket_size"),
            pl.col("basket_size").mean().alias("mean_basket_size"),
        )
        .collect()
    )

    print(basket_summary)

    print_section("Product frequency distribution")

    # Calculamos cuántas veces aparece cada producto en órdenes históricas.
    # Esto sirve para decidir si debemos eliminar productos muy raros.
    product_frequency = (
        order_products
        .group_by("product_id")
        .agg(pl.len().alias("product_purchase_count"))
    )

    product_summary = (
        product_frequency
        .select(
            pl.len().alias("total_products"),
            pl.col("product_purchase_count").min().alias("min_purchase_count"),
            pl.col("product_purchase_count").quantile(0.25).alias("p25"),
            pl.col("product_purchase_count").quantile(0.50).alias("p50"),
            pl.col("product_purchase_count").quantile(0.75).alias("p75"),
            pl.col("product_purchase_count").quantile(0.90).alias("p90"),
            pl.col("product_purchase_count").quantile(0.95).alias("p95"),
            pl.col("product_purchase_count").quantile(0.99).alias("p99"),
            pl.col("product_purchase_count").max().alias("max_purchase_count"),
            pl.col("product_purchase_count").mean().alias("mean_purchase_count"),
        )
        .collect()
    )

    print(product_summary)

    print_section("Coverage by product frequency threshold")

    # Probamos varios umbrales para ver cuántos productos quedarían.
    # Esto ayuda a no escoger arbitrariamente un mínimo de frecuencia.
    thresholds = [5, 10, 25, 50, 100, 250, 500, 1000]

    product_frequency_df = product_frequency.collect()

    for threshold in thresholds:
        kept_products = product_frequency_df.filter(
            pl.col("product_purchase_count") >= threshold
        )

        print(
            f"Products with >= {threshold:>4} purchases: "
            f"{kept_products.height:,}"
        )

    print_section("Coverage by basket size threshold")

    # Probamos límites superiores de tamaño de canasta.
    # Canastas demasiado grandes generan muchas combinaciones producto-producto.
    max_basket_sizes = [10, 20, 30, 40, 50, 75, 100]

    basket_sizes_df = basket_sizes.collect()

    for max_size in max_basket_sizes:
        kept_orders = basket_sizes_df.filter(
            (pl.col("basket_size") >= 2)
            & (pl.col("basket_size") <= max_size)
        )

        print(
            f"Orders with basket size between 2 and {max_size:>3}: "
            f"{kept_orders.height:,}"
        )


if __name__ == "__main__":
    main()