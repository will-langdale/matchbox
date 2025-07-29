"""Functions to index data sources to the Matchbox server."""

from matchbox.client import _handler
from matchbox.common.sources import Location, SourceConfig


def index(
    source_config: SourceConfig,
    batch_size: int | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        source_config: A SourceConfig with client set
        batch_size: the size of each batch when fetching data from the warehouse,
            which helps reduce the load on the database. Default is None.
    """
    if not source_config.location.client:
        raise ValueError("Source client not set")

    data_hashes = source_config.hash_data(batch_size=batch_size)
    _handler.index(source_config=source_config, data_hashes=data_hashes)


def get_source(
    name: str,
    location: Location | None = None,
    extract_transform: str | None = None,
    key_field: str | None = None,
    index_fields: list[str] | None = None,
) -> SourceConfig:
    """Get a SourceConfig for an existing source.

    Args:
        name: The name of the source resolution.
        location: If provided, will validate the returned SourceConfig
            against the location.
        extract_transform: If provided, will validate the returned SourceConfig
            against the extract/transform logic.
        key_field: If provided, will validate the returned SourceConfig
            against the key field.
        index_fields: If provided, will validate the returned SourceConfig
            against the index fields.

    Returns:
        A SourceConfig object.
    """
    config = _handler.get_source_config(name=name)

    validations = [
        (location, config.location, "location"),
        (extract_transform, config.extract_transform, "extract/transform"),
        (key_field, config.key_field, "key field"),
    ]

    for provided, actual, field_name in validations:
        if provided is not None and actual != provided:
            raise ValueError(
                f"Source {name} does not match the provided {field_name}: {provided}"
            )

    if index_fields is not None and set(config.index_fields) != set(index_fields):
        raise ValueError(
            f"Source {name} does not match the provided index fields: {index_fields}"
        )

    return config
