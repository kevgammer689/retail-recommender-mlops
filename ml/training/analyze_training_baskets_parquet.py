from pathlib import Path

import polars as pl


# Raíz del proyecto:
# retail-recommender-mlops/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset filtrado creado en el paso anterior.
TRAINING_BASKETS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "training_baskets_filtered.parquet"
)


def print_section(title: str) -> None:
    """Imprime un título legible para separar bloques de salida."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    """
    Analiza el parquet filtrado que se usará para entrenar el recomendador.

    Este script no entrena todavía.
    Su función es confirmar que el dataset procesado tiene una forma razonable
    antes de construir pares producto-producto.
    """

    if not TRAINING_BASKETS_PATH.exists():
        raise FileNotFoundError(
            f"Training parquet not found: {TRAINING_BASKETS_PATH}"
        )

    print_section("Loading filtered training parquet")
    print(TRAINING_BASKETS_PATH)

    # scan_parquet permite trabajar en modo lazy, sin cargar todo el archivo de una vez.
    baskets = pl.scan_parquet(TRAINING_BASKETS_PATH)

    print_section("Basic shape")

    shape_summary = (
        baskets
        .select(
            pl.len().alias("rows"),
            pl.col("order_id").n_unique().alias("orders"),
            pl.col("product_id").n_unique().alias("products"),
        )
        .collect()
    )

    print(shape_summary)

    print_section("Basket size distribution")

    # Como basket_size viene repetido por cada producto de una misma orden,
    # primero dejamos una fila por order_id.
    basket_size_summary = (
        baskets
        .select("order_id", "basket_size")
        .unique()
        .select(
            pl.col("basket_size").min().alias("min"),
            pl.col("basket_size").quantile(0.25).alias("p25"),
            pl.col("basket_size").quantile(0.50).alias("p50"),
            pl.col("basket_size").quantile(0.75).alias("p75"),
            pl.col("basket_size").quantile(0.90).alias("p90"),
            pl.col("basket_size").quantile(0.95).alias("p95"),
            pl.col("basket_size").quantile(0.99).alias("p99"),
            pl.col("basket_size").max().alias("max"),
            pl.col("basket_size").mean().alias("mean"),
        )
        .collect()
    )

    print(basket_size_summary)

    print_section("Top 20 products by frequency in filtered dataset")

    top_products = (
        baskets
        .group_by("product_id")
        .agg(
            pl.len().alias("rows")
        )
        .sort("rows", descending=True)
        .head(20)
        .collect()
    )

    print(top_products)

    print_section("Estimated product-pair workload")

    # Para un recomendador item-item por co-ocurrencia,
    # cada orden genera combinaciones de productos:
    # pairs = n * (n - 1) / 2
    # Esta métrica estima cuántos pares habrá que procesar.
    pair_workload = (
        baskets
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

    print(pair_workload)


if __name__ == "__main__":
    main()