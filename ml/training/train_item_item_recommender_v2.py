from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_BASKETS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "training_baskets_v1_compact.parquet"
)
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_MODEL_PATH = MODELS_DIR / "item_item_recommender_v2.parquet"

TOP_N_RECOMMENDATIONS = 20
ORDER_ID_CHUNK_SIZE = 75_000
SOURCE_PRODUCTS_PER_CHUNK = 250


def main() -> None:
    """Train the popularity-adjusted item-item recommender v2."""

    if not TRAINING_BASKETS_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_BASKETS_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training baskets: {TRAINING_BASKETS_PATH}")
    baskets = (
        pl.scan_parquet(TRAINING_BASKETS_PATH)
        .select("order_id", "product_id")
        .unique()
    )

    dataset_summary = (
        baskets.select(
            pl.col("order_id").n_unique().alias("total_orders"),
            pl.col("order_id").min().alias("min_order_id"),
            pl.col("order_id").max().alias("max_order_id"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    total_orders = int(dataset_summary["total_orders"])
    min_order_id = int(dataset_summary["min_order_id"])
    max_order_id = int(dataset_summary["max_order_id"])
    print(f"Training orders: {total_orders:,}")

    product_frequencies = (
        baskets.group_by("product_id")
        .agg(pl.len().cast(pl.UInt32).alias("product_frequency"))
        .collect(engine="streaming")
    )
    source_product_ids = sorted(product_frequencies["product_id"].to_list())

    source_frequencies = product_frequencies.lazy().rename(
        {
            "product_id": "source_product_id",
            "product_frequency": "source_product_frequency",
        }
    )
    recommended_frequencies = product_frequencies.lazy().rename(
        {
            "product_id": "recommended_product_id",
            "product_frequency": "recommended_product_frequency",
        }
    )

    with TemporaryDirectory(
        prefix="item_item_v2_",
        dir=MODELS_DIR,
    ) as temporary_directory:
        work_dir = Path(temporary_directory)
        pair_parts_dir = work_dir / "pair_parts"
        ranked_parts_dir = work_dir / "ranked_parts"
        pair_parts_dir.mkdir()
        ranked_parts_dir.mkdir()

        print("Building directed product pairs in bounded order chunks...")
        pair_part = 0
        for start_order_id in range(
            min_order_id,
            max_order_id + 1,
            ORDER_ID_CHUNK_SIZE,
        ):
            end_order_id = start_order_id + ORDER_ID_CHUNK_SIZE
            chunk = baskets.filter(
                (pl.col("order_id") >= start_order_id)
                & (pl.col("order_id") < end_order_id)
            )
            chunk_pairs = (
                chunk.join(
                    chunk,
                    on="order_id",
                    how="inner",
                    suffix="_recommended",
                )
                .filter(pl.col("product_id") != pl.col("product_id_recommended"))
                .group_by("product_id", "product_id_recommended")
                .agg(pl.len().cast(pl.UInt32).alias("cooccurrence_count"))
                .rename(
                    {
                        "product_id": "source_product_id",
                        "product_id_recommended": "recommended_product_id",
                    }
                )
            )
            pair_path = pair_parts_dir / f"pairs_{pair_part:04d}.parquet"
            chunk_pairs.sink_parquet(pair_path)
            pair_part += 1
            print(
                f"  Pair chunk {pair_part}: orders "
                f"[{start_order_id:,}, {end_order_id:,})"
            )

        pair_parts_pattern = str(pair_parts_dir / "*.parquet")
        print("Aggregating, scoring and ranking source-product chunks...")
        ranked_part_paths: list[Path] = []
        for offset in range(0, len(source_product_ids), SOURCE_PRODUCTS_PER_CHUNK):
            source_chunk = source_product_ids[
                offset : offset + SOURCE_PRODUCTS_PER_CHUNK
            ]
            cooccurrence = (
                pl.scan_parquet(pair_parts_pattern)
                .filter(pl.col("source_product_id").is_in(source_chunk))
                .group_by("source_product_id", "recommended_product_id")
                .agg(
                    pl.col("cooccurrence_count")
                    .sum()
                    .cast(pl.UInt32)
                    .alias("cooccurrence_count")
                )
            )
            scored = (
                cooccurrence.join(source_frequencies, on="source_product_id")
                .join(recommended_frequencies, on="recommended_product_id")
                .with_columns(
                    (
                        pl.col("cooccurrence_count")
                        / pl.col("source_product_frequency")
                    ).alias("confidence_score"),
                    (
                        pl.col("recommended_product_frequency") / total_orders
                    ).alias("recommended_probability"),
                    (
                        pl.col("cooccurrence_count")
                        / (
                            pl.col("source_product_frequency").cast(pl.Float64)
                            * pl.col("recommended_product_frequency").cast(
                                pl.Float64
                            )
                        ).sqrt()
                    ).alias("cosine_score"),
                )
                .with_columns(
                    (
                        pl.col("confidence_score")
                        / pl.col("recommended_probability")
                    ).alias("lift_score")
                )
                .with_columns(
                    # Lift directly penalizes globally popular candidates.
                    # log1p(cooccurrence) retains a preference for well-supported
                    # pairs so rare associations do not dominate.
                    (
                        pl.col("lift_score")
                        * pl.col("cooccurrence_count").cast(pl.Float64).log1p()
                    ).alias("final_score")
                )
            )
            ranked = (
                scored.sort(
                    ["source_product_id", "final_score", "cooccurrence_count"],
                    descending=[False, True, True],
                )
                .with_columns(
                    pl.col("final_score")
                    .rank(method="ordinal", descending=True)
                    .over("source_product_id")
                    .cast(pl.UInt32)
                    .alias("rank")
                )
                .filter(pl.col("rank") <= TOP_N_RECOMMENDATIONS)
                .select(
                    "source_product_id",
                    "recommended_product_id",
                    "cooccurrence_count",
                    "source_product_frequency",
                    "recommended_product_frequency",
                    "confidence_score",
                    "lift_score",
                    "cosine_score",
                    "final_score",
                    "rank",
                )
            )
            ranked_path = ranked_parts_dir / f"ranked_{offset:05d}.parquet"
            ranked.sink_parquet(ranked_path)
            ranked_part_paths.append(ranked_path)
            print(
                f"  Ranked source products "
                f"{offset + 1}-{min(offset + len(source_chunk), len(source_product_ids))}"
            )

        print(f"Writing model artifact: {OUTPUT_MODEL_PATH}")
        (
            pl.scan_parquet([str(path) for path in ranked_part_paths])
            .sort(
                ["source_product_id", "final_score", "cooccurrence_count"],
                descending=[False, True, True],
            )
            .sink_parquet(OUTPUT_MODEL_PATH)
        )

    summary = (
        pl.scan_parquet(OUTPUT_MODEL_PATH)
        .select(
            pl.len().alias("rows"),
            pl.col("source_product_id").n_unique().alias("source_products"),
            pl.col("recommended_product_id")
            .n_unique()
            .alias("recommended_products"),
            pl.col("rank").max().alias("max_rank"),
        )
        .collect()
    )
    print("Model artifact created successfully.")
    print(summary)


if __name__ == "__main__":
    main()
