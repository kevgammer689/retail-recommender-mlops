from pathlib import Path

import polars as pl


# Raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset compacto seleccionado para el primer entrenamiento local.
TRAINING_BASKETS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "training_baskets_v1_compact.parquet"
)

# Carpeta donde guardaremos el artifact del modelo.
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

# Artifact final: top recomendaciones por producto.
OUTPUT_MODEL_PATH = MODELS_DIR / "item_item_recommender_v1.parquet"


# Número máximo de recomendaciones que guardaremos por producto base.
TOP_N_RECOMMENDATIONS = 20


def main() -> None:
    """
    Entrena un recomendador item-item simple basado en co-ocurrencia.

    Idea del modelo:
    - Si dos productos aparecen juntos muchas veces en las mismas órdenes,
      asumimos que son buenos candidatos para recomendarse entre sí.

    El modelo final no guarda todas las combinaciones posibles.
    Solo guarda el top-N de productos recomendados por cada producto base.
    """

    if not TRAINING_BASKETS_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_BASKETS_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training baskets:")
    print(TRAINING_BASKETS_PATH)

    baskets = pl.scan_parquet(TRAINING_BASKETS_PATH).select(
        "order_id",
        "product_id",
        "product_purchase_count",
    )

    print("Building product pairs...")

    # Hacemos self-join por order_id:
    # Para cada orden, combinamos cada producto con los demás productos de la misma orden.
    #
    # Ejemplo:
    # Orden: [A, B, C]
    # Pares generados:
    # A-B, A-C, B-A, B-C, C-A, C-B
    #
    # Excluimos pares donde source_product_id == recommended_product_id.
    product_pairs = (
        baskets
        .join(baskets, on="order_id", how="inner", suffix="_recommended")
        .filter(
            pl.col("product_id") != pl.col("product_id_recommended")
        )
        .select(
            pl.col("product_id").alias("source_product_id"),
            pl.col("product_id_recommended").alias("recommended_product_id"),
            pl.col("product_purchase_count").alias("source_product_purchase_count"),
            pl.col("product_purchase_count_recommended").alias(
                "recommended_product_purchase_count"
            ),
        )
    )

    print("Counting co-occurrences...")

    # Contamos cuántas veces aparece cada par de productos en la misma orden.
    cooccurrence = (
        product_pairs
        .group_by(
            "source_product_id",
            "recommended_product_id",
            "source_product_purchase_count",
            "recommended_product_purchase_count",
        )
        .agg(
            pl.len().alias("cooccurrence_count")
        )
    )

    print("Computing recommendation score...")

    # Score simple:
    # coocurrencias normalizadas por la frecuencia del producto base.
    #
    # Interpretación:
    # entre las compras donde aparece source_product_id,
    # qué tan frecuente aparece recommended_product_id.
    scored = (
        cooccurrence
        .with_columns(
            (
                pl.col("cooccurrence_count")
                / pl.col("source_product_purchase_count")
            ).alias("score")
        )
    )

    print("Ranking recommendations per product...")

    # Ordenamos por producto base y score descendente.
    # Luego asignamos ranking dentro de cada source_product_id.
    ranked = (
        scored
        .sort(
            ["source_product_id", "score", "cooccurrence_count"],
            descending=[False, True, True],
        )
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over("source_product_id")
            .alias("rank")
        )
        .filter(pl.col("rank") <= TOP_N_RECOMMENDATIONS)
        .select(
            "source_product_id",
            "recommended_product_id",
            "cooccurrence_count",
            "score",
            "rank",
        )
    )

    print("Writing model artifact:")
    print(OUTPUT_MODEL_PATH)

    ranked.sink_parquet(OUTPUT_MODEL_PATH)

    print("Model artifact created successfully.")

    summary = (
        pl.scan_parquet(OUTPUT_MODEL_PATH)
        .select(
            pl.len().alias("rows"),
            pl.col("source_product_id").n_unique().alias("source_products"),
            pl.col("recommended_product_id").n_unique().alias("recommended_products"),
            pl.col("rank").max().alias("max_rank"),
        )
        .collect()
    )

    print("\nArtifact summary:")
    print(summary)


if __name__ == "__main__":
    main()