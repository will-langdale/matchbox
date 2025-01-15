import sqlite3
from pathlib import Path

import pandas as pd

from matchbox.common.sources import Source, SourceWarehouse, SQLiteWarehouse
from matchbox.server import MatchboxDBAdapter, inject_backend


@inject_backend
def create_source(
    backend: MatchboxDBAdapter,
    alias: str,
    db_table: str,
    warehouse: SourceWarehouse,
    range_left: int,
    range_right: int,
):
    # We need at least a column other than the id, with unique values in each
    df = pd.DataFrame(
        {"id": range(range_left, range_right), "val": range(range_left, range_right)}
    )

    # SQLite doesn't have proper "schemas"
    df.to_sql(
        db_table,
        warehouse.engine,
        if_exists="replace",
        index=False,
    )

    source = Source(
        alias=alias,
        db_pk="id",
        db_table=db_table,
        database=warehouse,
    )

    backend.index(source)


@inject_backend
def main(backend: MatchboxDBAdapter) -> None:
    source_len = 10_000

    backend.clear(certain=True)

    DB_NAME = "dummy_warehouse.db"
    DB_PATH = Path.cwd() / "data" / DB_NAME
    # Create DB file if not exists
    sqlite3.connect(DB_PATH)

    warehouse = SQLiteWarehouse(alias="sqllite", database=str(DB_PATH))

    create_source("alias1", "companies_house", warehouse, 0, source_len)
    create_source("alias2", "hmrc_exporters", warehouse, source_len, source_len * 2)


if __name__ == "__main__":
    main()
