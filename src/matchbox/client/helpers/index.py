"""Functions to index data sources to the Matchbox server."""

from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.client.helpers.selector import SourceReader
from matchbox.client.warehouse import _engine_fallback


def index(
    full_name: str,
    db_pk: str,
    engine: Engine,
    columns: list[str] | None = None,
    batch_size: int | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        full_name: the full name of the source
        db_pk: the primary key of the source
        engine: the engine to connect to a data warehouse
        columns: the columns to index
            If not set, all available columns other than `db_pk` will be indexed
        batch_size: the size of each batch when fetching data from the warehouse,
            which helps reduce the load on the database. Default is None.

    Examples:
        ```python
        index("mb.test_orig", "id", engine=engine)
        ```
        ```python
        index("mb.test_cl2", "id", engine=engine, columns=["name", "age"])
        ```
        ```python
        index("mb.test_orig", "id", engine=engine, batch_size=10_000)
        ```
    """
    engine = _engine_fallback(engine)
    reader = SourceReader(
        engine=engine, full_name=full_name, db_pk=db_pk, fields=columns
    )

    _handler.index(reader=reader, batch_size=batch_size)
