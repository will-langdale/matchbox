from sqlalchemy import Engine

from matchbox.common.sources import Source, SourceAddress, SourceColumn
from matchbox.server import inject_backend


def process_columns(
    columns: list[str] | list[dict[str, dict[str, str]]] | None,
) -> list[SourceColumn]:
    if columns is None:
        return []

    if isinstance(columns[0], str):
        return [SourceColumn(name=column) for column in columns]

    return [
        SourceColumn(name=column["name"], alias=column["type"], type=column["type"])
        for column in columns
    ]


@inject_backend
def index(
    backend,
    full_name: str,
    engine: Engine,
    columns: list[str] | list[dict[str, dict[str, str]]] | None,
) -> None:
    """Indexes data in Matchbox.

    Examples:
        ```python
        index("mb.test_orig", engine=engine)
        ```
        ```python
        index("mb.test_cl2", engine=engine, columns=["name", "age"])
        ```
        ```python
        index(
            "mb.test_cl2",
            engine=engine,
            columns={
                "name": {"name": "name", "alias": "person_name", "type": "string"},
                "age": {"name": "age", "alias": "person_age", "type": "int"},
            }
        )
        ```
    """
    source = Source(
        address=SourceAddress(full_name=full_name, engine=engine),
        columns=process_columns(columns),
    ).set_engine(engine)

    backend.index(source, source.hash_data())
