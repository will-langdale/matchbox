import sqlite3
from pathlib import Path

from matchbox.server import MatchboxDBAdapter, inject_backend


@inject_backend
def main(backend: MatchboxDBAdapter) -> None:
    DB_NAME = "dummy_warehouse.db"
    DB_PATH = Path.cwd() / "data" / DB_NAME
    sqlite3.connect(DB_PATH)

    # warehouse = SQLiteWarehouse(alias="sqllite", database=DB_NAME)

    # ds1 = Source(
    #     alias="alias1",
    #     db_pk="id",
    #     db_schema="companieshouse",
    #     db_table="companies",
    #     columns=[Source],
    # )
    # ds2 = Source(...)
    # backend.index(ds1)
    # backend.index(ds2)


if __name__ == "__main__":
    main()
