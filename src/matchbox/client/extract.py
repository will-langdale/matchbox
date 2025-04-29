"""Functions to extract data out of the Matchbox server."""

from pyarrow import Table as ArrowTable
from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.common.exceptions import MatchboxSourceNotFoundError
from matchbox.common.sources import Source, SourceAddress


def _combined_colname(source: Source, col_name: str):
    return source.address.full_name.replace(".", "_") + "_" + col_name


def primary_keys_map(
    resolution_name: str,
    engine: Engine | None = None,
) -> ArrowTable:
    """Return matchbox IDs to source IDs mapping, optionally filtering by an engine."""
    # Get all sources in scope of the resolution
    sources = _handler.get_resolution_sources(resolution_name=resolution_name)

    if engine:
        # Filter only sources compatible with engine
        warehouse_hash_b64 = SourceAddress.compose(
            full_name="", engine=engine
        ).warehouse_hash_b64

        sources = [
            s for s in sources if s.address.warehouse_hash_b64 == warehouse_hash_b64
        ]

    if not sources:
        raise MatchboxSourceNotFoundError("No compatible source was found")

    source_mb_ids: list[ArrowTable] = []
    db_pks: dict[str, str] = {}

    for s in sources:
        # Get Matchbox IDs from backend
        source_mb_ids.append(
            _handler.query(
                source_address=s.address,
                resolution_name=resolution_name,
            )
        )

        db_pks[s.address.full_name] = s.db_pk

    # Join Matchbox IDs to form mapping table
    mapping = source_mb_ids[0]
    mapping = mapping.rename_columns(
        {
            "source_pk": _combined_colname(
                sources[0], db_pks[sources[0].address.full_name]
            )
        }
    )
    if len(sources) > 1:
        for s, mb_ids in zip(sources[1:], source_mb_ids[1:], strict=True):
            mapping = mapping.join(
                right_table=mb_ids, keys="id", join_type="full outer"
            )
            mapping = mapping.rename_columns(
                {"source_pk": _combined_colname(s, db_pks[s.address.full_name])}
            )

    return mapping
