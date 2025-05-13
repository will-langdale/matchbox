from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import TypeEngine


def dataset_parquet_to_local_warehouse(
    parquetpath: Path, schema: str, table: str, dtype: dict[str, TypeEngine]
) -> None:
    """Load dataset parquet file to local warehouse instance.

    Args:
        parquetpath: Path to parquet file.
        schema: Name of schema in local database.
        table: Name of new table in local database.
        dtype: Dictionary of dataset column names and sqlalchemy types.
    """

    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = 7654

    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )
    with engine.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        connection.commit()

    dataset = pd.read_parquet(parquetpath, dtype_backend="pyarrow")
    dataset.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="replace",
        dtype=dtype,
        chunksize=100000,
        index=False,
    )
    engine.dispose()
