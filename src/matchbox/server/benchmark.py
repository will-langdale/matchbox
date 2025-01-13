import sqlite3

from matchbox.common.db import SourceWarehouse
from matchbox.server import MatchboxDBAdapter, inject_backend
from matchbox.server.base import Source


@inject_backend
def main(backend: MatchboxDBAdapter) -> None:
    con = sqlite3.connect("dummy_warehouse.db")

    warehouse = SourceWarehouse(alias="sqllite", db_type="sqlite")

    ds1 = Source(
        alias="alias1",
        db_pk="id",
        db_schema="companieshouse",
        db_table="companies",
        columns=[Source],
    )
    ds2 = Source(...)
    backend.index(ds1)
    backend.index(ds2)

    """
    database = "pg_warehouse"
    """


if __name__ == "__main__":
    main()
