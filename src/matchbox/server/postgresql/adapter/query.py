"""Query PostgreSQL mixin for Matchbox server."""

from typing import TYPE_CHECKING, Any

from matchbox.common.dtos import Match, ResolverStepPath, SourceStepPath
from matchbox.server.postgresql.utils.query import match, query

if TYPE_CHECKING:
    from pyarrow import Table as ArrowTable
else:
    ArrowTable = Any


class MatchboxPostgresQueryMixin:
    """Query mixin for the PostgreSQL adapter for Matchbox."""

    def query(  # noqa: D102
        self,
        source: SourceStepPath,
        resolves_from: ResolverStepPath | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> ArrowTable:
        return query(
            source=source,
            resolves_from=resolves_from,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )

    def match(  # noqa: D102
        self,
        key: str,
        source: SourceStepPath,
        targets: list[SourceStepPath],
        resolves_from: ResolverStepPath,
    ) -> list[Match]:
        return match(
            key=key,
            source=source,
            targets=targets,
            resolves_from=resolves_from,
        )
