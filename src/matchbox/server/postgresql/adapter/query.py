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
        resolver: ResolverStepPath | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> ArrowTable:
        return query(
            source=source,
            resolver=resolver,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )

    def match(  # noqa: D102
        self,
        key: str,
        source: SourceStepPath,
        targets: list[SourceStepPath],
        resolver: ResolverStepPath,
    ) -> list[Match]:
        return match(
            key=key,
            source=source,
            targets=targets,
            resolver=resolver,
        )
