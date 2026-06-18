import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine


DATABASE_CONNECT_TIMEOUT_SECONDS = 5


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": DATABASE_CONNECT_TIMEOUT_SECONDS},
    )
