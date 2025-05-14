"""Functions to index data sources to the Matchbox server."""

from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.common.dtos import SourceResolutionName
from matchbox.common.sources import SourceAddress, SourceConfig, SourceField


def _process_fields(
    fields: list[str] | list[dict[str, dict[str, str]]] | None,
) -> tuple[SourceField]:
    if fields is None:
        return []

    if isinstance(fields[0], str):
        return (SourceField(name=field) for field in fields)

    return (SourceField(name=field["name"], type=field["type"]) for field in fields)


def index(
    full_name: str,
    key_field: str,
    engine: Engine,
    name: SourceResolutionName | None = None,
    fields: list[str] | list[dict[str, dict[str, str]]] | None = None,
    batch_size: int | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        full_name: the full name of the source
        key_field: the unique identifier of the entity the source config describes
        engine: the engine to connect to a data warehouse
        name: a custom resolution name
            If missing, will use the default name for a `SourceConfig`
        fields: the fields to index
        batch_size: the size of each batch when fetching data from the warehouse,
            which helps reduce the load on the database. Default is None.

    Examples:
        ```python
        index("mb.test_orig", "id", engine=engine)
        ```
        ```python
        index("mb.test_cl2", "id", engine=engine, fields=["name", "age"])
        ```
        ```python
        index(
            "mb.test_cl2",
            "id",
            engine=engine,
            fields=[
                {"name": "name", "type": "TEXT"},
                {"name": "age", "type": "BIGINT"},
            ],
        )
        ```
        ```python
        index("mb.test_orig", "id", engine=engine, batch_size=10_000)
        ```
    """
    index_fields = _process_fields(fields)

    address = SourceAddress.compose(engine=engine, full_name=full_name)
    if name:
        source_config = SourceConfig(
            address=address,
            name=name,
            index_fields=index_fields,
            key_field=key_field,
        )
    else:
        source_config = SourceConfig(
            address=address,
            index_fields=index_fields,
            key_field=key_field,
        )

    source_config.set_engine(engine)
    if not fields:
        source_config = source_config.default_fields()

    _handler.index(source_config=source_config, batch_size=batch_size)
