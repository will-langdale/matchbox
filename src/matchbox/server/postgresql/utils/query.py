"""Utilities for querying and matching in the PostgreSQL backend."""

from typing import Literal

import pyarrow as pa
from sqlalchemy import and_, func, join, literal_column, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Select, Subquery

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Match,
    ResolutionType,
    ResolverResolutionPath,
    SourceResolutionPath,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotQueriable,
    MatchboxResolutionTypeError,
)
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ClusterSourceKey,
    Contains,
    ResolutionClusters,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql


def require_complete_resolver(
    session: Session,
    path: ResolverResolutionPath,
) -> Resolutions:
    """Resolve and validate a resolver path for query-time operations."""
    resolver_resolution = Resolutions.from_path(path=path, session=session)
    if resolver_resolution.type != ResolutionType.RESOLVER:
        raise MatchboxResolutionTypeError(
            resolution_name=str(path),
            resolution_type=resolver_resolution.type,
            expected_resolution_types=[ResolutionType.RESOLVER],
        )
    if resolver_resolution.upload_stage != UploadStage.COMPLETE:
        raise MatchboxResolutionNotQueriable
    return resolver_resolution


def resolver_leaf_to_root_subquery(
    resolution_id: int,
    alias: str = "resolver_assignments",
) -> Subquery:
    """Build leaf->root assignment subquery for a resolver."""
    return (
        select(
            Contains.leaf.label("leaf_id"),
            Contains.root.label("root_id"),
        )
        .select_from(Contains)
        .join(
            ResolutionClusters,
            and_(
                ResolutionClusters.cluster_id == Contains.root,
                ResolutionClusters.resolution_id == resolution_id,
            ),
        )
        .subquery(alias)
    )


def resolver_membership_subquery(
    resolution_id: int,
    alias: str = "resolver_membership",
) -> Subquery:
    """Build root_id/leaf_id membership rows for a resolver."""
    roots_query = select(
        ResolutionClusters.cluster_id.label("root_id"),
        ResolutionClusters.cluster_id.label("leaf_id"),
    ).where(ResolutionClusters.resolution_id == resolution_id)

    leaves_query = (
        select(
            ResolutionClusters.cluster_id.label("root_id"),
            Contains.leaf.label("leaf_id"),
        )
        .select_from(ResolutionClusters)
        .join(Contains, Contains.root == ResolutionClusters.cluster_id)
        .where(ResolutionClusters.resolution_id == resolution_id)
    )

    return roots_query.union(leaves_query).subquery(alias)


def build_unified_query(
    resolution: Resolutions,
    sources: list[SourceConfigs] | None = None,
    level: Literal["leaf", "key"] = "leaf",
    *,
    alias_prefix: str = "resolver_assignments",
    include_source_config_id: bool = False,
) -> Select:
    """Build a query that projects records to root IDs through the hierarchy."""
    lineage = resolution.get_lineage(sources=sources, queryable_only=True)
    resolver_ids: list[int] = []
    source_config_ids: list[int] = []
    for resolution_id, source_config_id in lineage:
        if source_config_id is None:
            resolver_ids.append(resolution_id)
        else:
            source_config_ids.append(source_config_id)

    from_clause = ClusterSourceKey
    projected_roots: list[ColumnElement[int]] = []
    for index, resolver_id in enumerate(resolver_ids):
        assignments = resolver_leaf_to_root_subquery(
            resolution_id=resolver_id,
            alias=f"{alias_prefix}_{index}",
        )
        from_clause = join(
            from_clause,
            assignments,
            assignments.c.leaf_id == ClusterSourceKey.cluster_id,
            isouter=True,
        )
        projected_roots.append(assignments.c.root_id)

    root_projection: ColumnElement[int] = (
        func.coalesce(*projected_roots, ClusterSourceKey.cluster_id)
        if projected_roots
        else ClusterSourceKey.cluster_id
    )

    selection: list[ColumnElement] = [
        root_projection.label("root_id"),
        ClusterSourceKey.cluster_id.label("leaf_id"),
    ]
    if level == "key":
        selection.append(ClusterSourceKey.key)
    if include_source_config_id:
        selection.append(ClusterSourceKey.source_config_id.label("source_config_id"))

    query_stmt = (
        select(*selection)
        .select_from(from_clause)
        .where(ClusterSourceKey.source_config_id.in_(source_config_ids))
    )

    if level == "leaf":
        query_stmt = query_stmt.distinct()

    return query_stmt


def _build_target_cluster_cte(
    key: str,
    source_config_id: int,
    resolution: Resolutions,
) -> Select:
    """Build the target cluster query for a source key."""
    source_projection = build_unified_query(
        resolution=resolution,
        level="key",
        alias_prefix="resolver_assignments_match_target",
        include_source_config_id=True,
    ).subquery("source_projection")

    return (
        select(source_projection.c.root_id.label("cluster_id"))
        .where(
            and_(
                source_projection.c.source_config_id == source_config_id,
                source_projection.c.key == key,
            )
        )
        .limit(1)
    )


def _build_matching_leaves_cte(
    source_and_target_ids: list[int],
    resolution: Resolutions,
    cluster: int,
) -> Select:
    """Build the matching keys query for a resolved cluster."""
    full_projection = build_unified_query(
        resolution=resolution,
        level="key",
        alias_prefix="resolver_assignments_match_all",
        include_source_config_id=True,
    ).subquery("full_projection")

    return (
        select(full_projection.c.source_config_id, full_projection.c.key)
        .where(
            and_(
                full_projection.c.source_config_id.in_(source_and_target_ids),
                full_projection.c.root_id == cluster,
            )
        )
        .distinct()
    )


def query(
    source: SourceResolutionPath,
    point_of_truth: ResolverResolutionPath | None = None,
    return_leaf_id: bool = False,
    limit: int | None = None,
) -> pa.Table:
    """Query Matchbox to retrieve linked data for a source."""
    with MBDB.get_session() as session:
        source_resolution: Resolutions = Resolutions.from_path(
            path=source,
            session=session,
        )
        source_config: SourceConfigs = source_resolution.source_config

        if point_of_truth is None:
            truth_resolution = source_resolution
        else:
            truth_resolution = require_complete_resolver(session, point_of_truth)

        if truth_resolution.upload_stage != UploadStage.COMPLETE:
            raise MatchboxResolutionNotQueriable

        query_stmt = build_unified_query(
            resolution=truth_resolution,
            sources=[source_config],
            level="key",
            alias_prefix="resolver_assignments_query",
        )

    query_stmt = query_stmt.order_by(
        literal_column("root_id"),
        literal_column("leaf_id"),
        ClusterSourceKey.key,
    )

    if limit is not None:
        query_stmt = query_stmt.limit(limit)

    with MBDB.get_adbc_connection() as conn:
        stmt = compile_sql(query_stmt)
        logger.debug(f"Query SQL: \n {stmt}")
        id_results = sql_to_df(
            stmt=stmt,
            connection=conn,
            return_type="arrow",
        ).rename_columns({"root_id": "id"})

    selection = ["id", "key"]
    if return_leaf_id:
        selection.append("leaf_id")

    return id_results.select(selection)


def match(
    key: str,
    source: SourceResolutionPath,
    targets: list[SourceResolutionPath],
    point_of_truth: ResolverResolutionPath,
) -> list[Match]:
    """Match a source key against targets under a resolver point-of-truth."""
    with MBDB.get_session() as session:
        source_config: SourceConfigs = Resolutions.from_path(
            path=source,
            session=session,
        ).source_config
        resolver_resolution = require_complete_resolver(session, point_of_truth)

        target_configs: list[SourceConfigs] = [
            Resolutions.from_path(path=target, session=session).source_config
            for target in targets
        ]

        target_cluster_query = _build_target_cluster_cte(
            key=key,
            source_config_id=source_config.source_config_id,
            resolution=resolver_resolution,
        )
        cluster = session.execute(target_cluster_query).scalar_one_or_none()
        if cluster is None:
            return [
                Match(
                    cluster=None,
                    source=source,
                    source_id=set(),
                    target=target,
                    target_id=set(),
                )
                for target in targets
            ]

        source_and_target_ids = [
            source_config.source_config_id,
            *(tc.source_config_id for tc in target_configs),
        ]

        matched_rows = session.execute(
            _build_matching_leaves_cte(
                source_and_target_ids=source_and_target_ids,
                resolution=resolver_resolution,
                cluster=int(cluster),
            )
        ).all()

        matches_by_source_id: dict[int, set[str]] = {}
        for source_config_id_result, key_in_source in matched_rows:
            matches_by_source_id.setdefault(source_config_id_result, set()).add(
                key_in_source
            )

        source_ids = matches_by_source_id.get(source_config.source_config_id, set())
        result: list[Match] = []
        for target, target_config in zip(targets, target_configs, strict=False):
            result.append(
                Match(
                    cluster=int(cluster),
                    source=source,
                    source_id=source_ids,
                    target=target,
                    target_id=matches_by_source_id.get(
                        target_config.source_config_id,
                        set(),
                    ),
                )
            )

        return result
