from pathlib import Path
import os
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = [
    "departments",
    "aisles",
    "products",
    "orders",
    "order_products_prior",
    "order_products_train",
]


def get_connection():
    load_dotenv(PROJECT_ROOT / ".env")

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "retail_recommender"),
        user=os.getenv("POSTGRES_USER", "retail_user"),
        password=os.getenv("POSTGRES_PASSWORD", "retail_password"),
    )


def fetch_one(conn, query: str):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchone()[0]


def main() -> None:
    conn = get_connection()

    try:
        print("Raw table row counts")
        print("-" * 40)

        for table in EXPECTED_TABLES:
            count = fetch_one(conn, f"SELECT COUNT(*) FROM {table};")
            print(f"{table}: {count:,}")

            if count == 0:
                raise ValueError(f"Table {table} is empty.")

        print("\nOrders by eval_set")
        print("-" * 40)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT eval_set, COUNT(*)
                FROM orders
                GROUP BY eval_set
                ORDER BY eval_set;
            """)
            for eval_set, count in cur.fetchall():
                print(f"{eval_set}: {count:,}")

        orphan_products = fetch_one(conn, """
            SELECT COUNT(*)
            FROM products p
            LEFT JOIN aisles a ON p.aisle_id = a.aisle_id
            LEFT JOIN departments d ON p.department_id = d.department_id
            WHERE a.aisle_id IS NULL
               OR d.department_id IS NULL;
        """)

        orphan_prior_items = fetch_one(conn, """
            SELECT COUNT(*)
            FROM order_products_prior op
            LEFT JOIN orders o ON op.order_id = o.order_id
            WHERE o.order_id IS NULL;
        """)

        orphan_train_items = fetch_one(conn, """
            SELECT COUNT(*)
            FROM order_products_train op
            LEFT JOIN orders o ON op.order_id = o.order_id
            WHERE o.order_id IS NULL;
        """)

        print("\nReferential checks")
        print("-" * 40)
        print(f"Products without valid aisle/department: {orphan_products:,}")
        print(f"Prior order items without order: {orphan_prior_items:,}")
        print(f"Train order items without order: {orphan_train_items:,}")

        if orphan_products > 0:
            raise ValueError("There are products without valid aisle/department.")

        if orphan_prior_items > 0:
            raise ValueError("There are prior order items without matching order.")

        if orphan_train_items > 0:
            raise ValueError("There are train order items without matching order.")

    finally:
        conn.close()

    print("\nRaw Instacart validation completed successfully.")


if __name__ == "__main__":
    main()
