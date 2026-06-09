from pathlib import Path
import os
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"

TABLE_FILES = {
    "departments": "departments.csv",
    "aisles": "aisles.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
}


def get_connection():
    load_dotenv(PROJECT_ROOT / ".env")

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "retail_recommender"),
        user=os.getenv("POSTGRES_USER", "retail_user"),
        password=os.getenv("POSTGRES_PASSWORD", "retail_password"),
    )


def copy_csv_to_table(conn, table_name: str, csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table_name};")

        with csv_path.open("r", encoding="utf-8") as file:
            copy_sql = f"""
                COPY {table_name}
                FROM STDIN
                WITH (
                    FORMAT CSV,
                    HEADER TRUE,
                    DELIMITER ',',
                    NULL ''
                );
            """
            cur.copy_expert(copy_sql, file)

    conn.commit()


def count_rows(conn, table_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        return cur.fetchone()[0]


def main() -> None:
    print(f"Raw data directory: {RAW_DATA_DIR}")

    conn = get_connection()

    try:
        for table_name, file_name in TABLE_FILES.items():
            csv_path = RAW_DATA_DIR / file_name
            print(f"Loading {file_name} into {table_name}...")
            copy_csv_to_table(conn, table_name, csv_path)
            rows = count_rows(conn, table_name)
            print(f"Loaded {rows:,} rows into {table_name}")

    finally:
        conn.close()

    print("Instacart ingestion completed.")


if __name__ == "__main__":
    main()
