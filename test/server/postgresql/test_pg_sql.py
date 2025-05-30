"""Test SQL functions for the Matchbox PostgreSQL backend.

The backend adapter tests provide the key coverage we want for any backend. However,
the PostgreSQL SQL functions are complex, and we've found having lower-level tests is
useful for debugging the complex logic required to query a hierarchical system.

Nevertheless, these tests are completely ephemeral, and if the hierarchical
representation changes, they should be rewritten in whatever form best-aids the
development of the new query functions. Don't be precious.
"""

from typing import Generator

import pytest
from sqlalchemy import and_

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
    SourceFields,
)
from matchbox.server.postgresql.utils.db import compile_sql
from matchbox.server.postgresql.utils.query import (
    _build_model_query,
    _build_source_query,
    _build_unified_query,
    _empty_result,
    _get_resolution_priority,
    _resolve_cluster_hierarchy,
    _resolve_hierarchy_assignments,
    _resolve_thresholds,
    get_clusters_with_leaves,
    get_source_config,
    match,
    query,
)


@pytest.fixture(scope="function")
def populated_postgres_db(
    matchbox_postgres: MatchboxPostgres,
) -> Generator[MatchboxPostgres, None, None]:
    """PostgreSQL database with a rich yet simple test dataset.

    - Source A: 6 keys → 5 clusters (keys 1&2 share cluster 101)
    - Source B: 5 keys → 5 clusters (one key per cluster)
    - Dedupe A: Creates 80% cluster (101+102) and 70% 3-way cluster (102+103+104)
        with pairwise (101+103)
    - Dedupe B: Creates simple 70% cluster (201+202)
    - Linker: Caches truth D1=80, D2=70. Creates:
      * 90% clusters: dedupe outputs (301+401) and raw leaves (105+205)
      * 80% clusters: dedupe+raw mix as pairwise (301+205) and 3-way component
        (301+205+105)

    Tests: threshold filtering, role flags, complex hierarchy, cross-source linking,
    truth inheritance.

    This diagram shows the structure of the test dataset:

    ```mermaid
    graph LR
        %% Resolution hierarchy
        subgraph "Resolutions"
            S1[Source A]
            S2[Source B]
            D1[Dedupe A]
            D2[Dedupe B]
            L1[Linker AB]

            S1 --> D1
            S2 --> D2
            D1 --> L1
            D2 --> L1
        end

        %% Source data
        subgraph "Source A: 6 keys → 5 clusters"
            C101[C101: key1,key2]
            C102[C102: key3]
            C103[C103: key4]
            C104[C104: key5]
            C105[C105: key6]
        end

        subgraph "Source B: 5 keys → 5 clusters"
            C201[C201: key1]
            C202[C202: key2]
            C203[C203: key3]
            C204[C204: key4]
            C205[C205: key5]
        end

        %% Dedupe A results
        subgraph "Dedupe A clusters"
            subgraph "80% threshold: C101+C102"
                C301[C301: prob=80, both]
            end
            subgraph "70% threshold: C102+C103+C104"
                C303[C303: prob=70, pairwise]
                C302[C302: prob=70, component]
            end
        end

        %% Dedupe B results
        subgraph "Dedupe B clusters"
            subgraph "70% threshold: C201+C202"
                C401[C401: prob=70, both]
            end
        end

        %% Linker results
        subgraph "Linker clusters"
            subgraph "90% threshold: C105+C205"
                C503[C503: prob=90, both]
            end
            subgraph "90% threshold: C301+C401"
                C501[C501: prob=90, pairwise]
            end
            subgraph "80% threshold: C301+C205+C105"
                C502[C502: prob=80, pairwise]
                C504[C504: prob=80, component]
            end
        end

        %% Relationships
        C101 -.-> C301
        C102 -.-> C301
        C102 -.-> C302
        C103 -.-> C302
        C104 -.-> C302
        C101 -.-> C303
        C103 -.-> C303

        C201 -.-> C401
        C202 -.-> C401

        C301 -.-> C501
        C401 -.-> C501
        C301 -.-> C502
        C205 -.-> C502
        C105 -.-> C503
        C205 -.-> C503
        C301 -.-> C504
        C205 -.-> C504
        C105 -.-> C504

        style S1 fill:#e3f2fd
        style S2 fill:#e3f2fd
        style D1 fill:#fce4ec
        style D2 fill:#fce4ec
        style L1 fill:#fff8e1
    ```
    """

    with MBDB.get_session() as session:
        # === RESOLUTIONS ===
        resolutions = [
            Resolutions(resolution_id=1, name="source_a", type="source", truth=None),
            Resolutions(resolution_id=2, name="source_b", type="source", truth=None),
            Resolutions(resolution_id=3, name="dedupe_a", type="model", truth=80),
            Resolutions(resolution_id=4, name="dedupe_b", type="model", truth=70),
            Resolutions(resolution_id=5, name="linker_ab", type="model", truth=90),
        ]

        # === RESOLUTION LINEAGE ===
        lineage = [
            # Source A -> Dedupe A
            ResolutionFrom(parent=1, child=3, level=1, truth_cache=None),
            # Source B -> Dedupe B
            ResolutionFrom(parent=2, child=4, level=1, truth_cache=None),
            # Dedupe A -> Linker
            ResolutionFrom(parent=3, child=5, level=1, truth_cache=80),
            # Dedupe B -> Linker
            ResolutionFrom(parent=4, child=5, level=1, truth_cache=70),
            # Source A -> Linker (indirect)
            ResolutionFrom(parent=1, child=5, level=2, truth_cache=None),
            # Source B -> Linker (indirect)
            ResolutionFrom(parent=2, child=5, level=2, truth_cache=None),
        ]

        # === SOURCE CONFIGS ===
        source_configs = [
            SourceConfigs(
                source_config_id=1,
                resolution_id=1,
                location_type="test",
                location_uri="test://source_a",
                extract_transform="identity",
            ),
            SourceConfigs(
                source_config_id=2,
                resolution_id=2,
                location_type="test",
                location_uri="test://source_b",
                extract_transform="identity",
            ),
        ]

        # === SOURCE FIELDS ===
        source_fields = [
            # Source A fields
            SourceFields(
                field_id=1,
                source_config_id=1,
                index=0,
                name="key",
                type="TEXT",
                is_key=True,
            ),
            SourceFields(
                field_id=2,
                source_config_id=1,
                index=1,
                name="value",
                type="TEXT",
                is_key=False,
            ),
            # Source B fields
            SourceFields(
                field_id=3,
                source_config_id=2,
                index=0,
                name="key",
                type="TEXT",
                is_key=True,
            ),
            SourceFields(
                field_id=4,
                source_config_id=2,
                index=1,
                name="value",
                type="TEXT",
                is_key=False,
            ),
        ]

        # === CLUSTERS ===
        clusters = [
            # Source A clusters (100-series)
            Clusters(cluster_id=101, cluster_hash=b"hash_101"),
            Clusters(cluster_id=102, cluster_hash=b"hash_102"),
            Clusters(cluster_id=103, cluster_hash=b"hash_103"),
            Clusters(cluster_id=104, cluster_hash=b"hash_104"),
            Clusters(cluster_id=105, cluster_hash=b"hash_105"),
            # Source B clusters (200-series)
            Clusters(cluster_id=201, cluster_hash=b"hash_201"),
            Clusters(cluster_id=202, cluster_hash=b"hash_202"),
            Clusters(cluster_id=203, cluster_hash=b"hash_203"),
            Clusters(cluster_id=204, cluster_hash=b"hash_204"),
            Clusters(cluster_id=205, cluster_hash=b"hash_205"),
            # Dedupe A clusters (300-series)
            Clusters(cluster_id=301, cluster_hash=b"hash_301"),
            Clusters(cluster_id=302, cluster_hash=b"hash_302"),
            Clusters(cluster_id=303, cluster_hash=b"hash_303"),
            # Dedupe B clusters (400-series)
            Clusters(cluster_id=401, cluster_hash=b"hash_401"),
            # Linker clusters (500-series)
            Clusters(cluster_id=501, cluster_hash=b"hash_501"),
            Clusters(cluster_id=502, cluster_hash=b"hash_502"),
            Clusters(cluster_id=503, cluster_hash=b"hash_503"),
            Clusters(cluster_id=504, cluster_hash=b"hash_504"),
        ]

        # === CLUSTER SOURCE KEYS ===
        cluster_keys = [
            # Source A: 6 keys → 5 clusters (key1,key2 share cluster 101)
            ClusterSourceKey(
                key_id=1, cluster_id=101, source_config_id=1, key="src_a_key1"
            ),
            ClusterSourceKey(
                key_id=2, cluster_id=101, source_config_id=1, key="src_a_key2"
            ),
            ClusterSourceKey(
                key_id=3, cluster_id=102, source_config_id=1, key="src_a_key3"
            ),
            ClusterSourceKey(
                key_id=4, cluster_id=103, source_config_id=1, key="src_a_key4"
            ),
            ClusterSourceKey(
                key_id=5, cluster_id=104, source_config_id=1, key="src_a_key5"
            ),
            ClusterSourceKey(
                key_id=6, cluster_id=105, source_config_id=1, key="src_a_key6"
            ),
            # Source B: 5 keys → 5 clusters (one key per cluster)
            ClusterSourceKey(
                key_id=7, cluster_id=201, source_config_id=2, key="src_b_key1"
            ),
            ClusterSourceKey(
                key_id=8, cluster_id=202, source_config_id=2, key="src_b_key2"
            ),
            ClusterSourceKey(
                key_id=9, cluster_id=203, source_config_id=2, key="src_b_key3"
            ),
            ClusterSourceKey(
                key_id=10, cluster_id=204, source_config_id=2, key="src_b_key4"
            ),
            ClusterSourceKey(
                key_id=11, cluster_id=205, source_config_id=2, key="src_b_key5"
            ),
        ]

        # === CONTAINS RELATIONSHIPS ===
        contains = [
            # Dedupe A: C301 contains C101+C102 (80% cluster)
            Contains(root=301, leaf=101),
            Contains(root=301, leaf=102),
            # Dedupe A: C302 contains C102+C103+C104 (70% component)
            Contains(root=302, leaf=102),
            Contains(root=302, leaf=103),
            Contains(root=302, leaf=104),
            # Dedupe A: C303 contains C101+C103 (70% pairwise)
            Contains(root=303, leaf=101),
            Contains(root=303, leaf=103),
            # Dedupe B: C401 contains C201+C202 (70% cluster)
            Contains(root=401, leaf=201),
            Contains(root=401, leaf=202),
            # Linker: C501 links C301+C401 (90% pairwise)
            Contains(root=501, leaf=101),  # from C301
            Contains(root=501, leaf=102),  # from C301
            Contains(root=501, leaf=201),  # from C401
            Contains(root=501, leaf=202),  # from C401
            # Linker: C502 contains C301+C205 (80% pairwise)
            Contains(root=502, leaf=101),  # from C301
            Contains(root=502, leaf=102),  # from C301
            Contains(root=502, leaf=205),  # direct source leaf
            # Linker: C503 contains C105+C205 (90% both)
            Contains(root=503, leaf=105),
            Contains(root=503, leaf=205),
            # Linker: C504 contains C301+C205+C105 (80% component)
            Contains(root=504, leaf=101),  # from C301
            Contains(root=504, leaf=102),  # from C301
            Contains(root=504, leaf=205),  # direct source leaf
            Contains(root=504, leaf=105),  # direct source leaf
        ]

        # === PROBABILITIES ===
        probabilities = [
            # Dedupe A probabilities
            Probabilities(
                resolution=3, cluster=301, probability=80, role_flag=1
            ),  # both
            Probabilities(
                resolution=3, cluster=302, probability=70, role_flag=2
            ),  # component
            Probabilities(
                resolution=3, cluster=303, probability=70, role_flag=0
            ),  # pairwise
            # Dedupe B probabilities
            Probabilities(
                resolution=4, cluster=401, probability=70, role_flag=1
            ),  # both
            # Linker probabilities
            Probabilities(
                resolution=5, cluster=501, probability=90, role_flag=0
            ),  # pairwise
            Probabilities(
                resolution=5, cluster=502, probability=80, role_flag=0
            ),  # pairwise
            Probabilities(
                resolution=5, cluster=503, probability=90, role_flag=1
            ),  # both
            Probabilities(
                resolution=5, cluster=504, probability=80, role_flag=2
            ),  # component
        ]

        # Insert all objects
        for obj in (
            resolutions,
            lineage,
            source_configs,
            source_fields,
            clusters,
            cluster_keys,
            contains,
            probabilities,
        ):
            if isinstance(obj, list):
                session.add_all(obj)
                session.flush()

        session.commit()

    yield matchbox_postgres


class TestGetSourceConfig:
    """Test source config retrieval."""

    def test_get_existing_source(self, populated_postgres_db: MatchboxPostgres):
        """Should return source config for existing source."""
        with MBDB.get_session() as session:
            source_config = get_source_config("source_a", session)
            assert source_config.source_config_id == 1
            assert source_config.resolution_id == 1

    def test_get_nonexistent_source(self, populated_postgres_db: MatchboxPostgres):
        """Should raise exception for nonexistent source."""
        with MBDB.get_session() as session:
            with pytest.raises(MatchboxSourceNotFoundError):
                get_source_config("nonexistent", session)


class TestResolveThresholds:
    """Test threshold resolution logic."""

    def test_resolve_with_no_threshold(self, populated_postgres_db: MatchboxPostgres):
        """Should use default truth values when no threshold provided."""
        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 5)  # linker_ab
            lineage_truths = {1: None, 3: 80, 4: 70, 5: 90}  # mix of sources and models

            result = _resolve_thresholds(lineage_truths, resolution, threshold=None)

            assert result[1] is None  # source keeps None
            assert result[3] == 80  # model keeps default
            assert result[4] == 70  # model keeps default
            assert result[5] == 90  # target resolution keeps default

    def test_resolve_with_threshold_override(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should override target resolution threshold only."""
        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 5)  # linker_ab
            lineage_truths = {3: 80, 4: 70, 5: 90}

            result = _resolve_thresholds(lineage_truths, resolution, threshold=85)

            assert result[3] == 80  # parent keeps default
            assert result[4] == 70  # parent keeps default
            assert result[5] == 85  # target gets override


class TestGetResolutionPriority:
    """Test resolution priority calculation."""

    def test_direct_parent_priority(self, populated_postgres_db: MatchboxPostgres):
        """Should return level 1 for direct parent."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # dedupe_a is level 1 parent of linker
            priority = _get_resolution_priority(3, linker_res)  # dedupe_a -> linker
            assert priority == 1

    def test_indirect_parent_priority(self, populated_postgres_db: MatchboxPostgres):
        """Should return level 2 for indirect parent."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # source_a is level 2 parent of linker (through dedupe_a)
            priority = _get_resolution_priority(1, linker_res)  # source_a -> linker
            assert priority == 2

    def test_nonexistent_relationship(self, populated_postgres_db: MatchboxPostgres):
        """Should return 0 for non-parent resolution."""
        with MBDB.get_session() as session:
            source_res = session.get(Resolutions, 1)  # source_a

            # dedupe_b has no relationship to source_a
            priority = _get_resolution_priority(4, source_res)  # dedupe_b -> source_a
            assert priority == 0


class TestEmptyResult:
    """Test empty result generation."""

    def test_empty_result_structure(self, populated_postgres_db: MatchboxPostgres):
        """Should return empty result with correct columns."""
        query = _empty_result()

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        assert len(result) == 0
        expected_columns = {
            "root_id",
            "root_hash",
            "leaf_id",
            "leaf_hash",
            "leaf_key",
            "source_config_id",
        }
        assert set(result.columns) == expected_columns


class TestBuildSourceQuery:
    """Test source query building."""

    def test_build_simple_source_query(self, populated_postgres_db: MatchboxPostgres):
        """Should build query for source resolutions."""
        # Simple condition: source_config belongs to resolution 1 (source_a)
        source_conditions = [
            (SourceConfigs.resolution_id == 1, 999, 1)  # condition, priority, res_id
        ]
        source_config_filter = True  # no filtering

        query = _build_source_query(source_conditions, source_config_filter)

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return 6 keys from source_a, root_id == leaf_id
        assert len(result) == 6
        assert all(result["root_id"] == result["leaf_id"])
        expected_keys = {
            "src_a_key1",
            "src_a_key2",
            "src_a_key3",
            "src_a_key4",
            "src_a_key5",
            "src_a_key6",
        }
        assert set(result["leaf_key"]) == expected_keys


class TestBuildModelQuery:
    """Test model query building."""

    def test_build_simple_model_query(self, populated_postgres_db: MatchboxPostgres):
        """Should build query for model resolutions."""
        # Condition: probabilities from resolution 3 (dedupe_a) with prob >= 80
        model_conditions = [
            (
                and_(
                    Probabilities.resolution == 3,
                    Probabilities.role_flag >= 1,
                    Probabilities.probability >= 80,
                ),
                1,
                3,
            )  # condition, priority, res_id
        ]
        source_config_filter = SourceConfigs.resolution_id == 1  # only source_a

        query = _build_model_query(model_conditions, source_config_filter)

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return keys that belong to clusters with prob >= 80
        # Only C301 qualifies (prob=80), which contains C101+C102 (3 keys)
        assert len(result) > 0
        # Check that we have hierarchy - some root_ids should be different from leaf_ids
        hierarchical_rows = result.filter(result["root_id"] != result["leaf_id"])
        assert len(hierarchical_rows) > 0


class TestBuildUnifiedQuery:
    """Test unified query building."""

    def test_build_unified_source_only(self, populated_postgres_db: MatchboxPostgres):
        """Should build unified query for source-only scenario."""
        resolved_thresholds = {1: None}  # source_a only
        source_config_filter = True

        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 1)
            query = _build_unified_query(
                resolved_thresholds, source_config_filter, resolution
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return all 6 keys from source_a
        assert len(result) == 6
        assert all(result["root_id"] == result["leaf_id"])

    def test_build_unified_mixed_scenario(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should build unified query mixing sources and models."""
        resolved_thresholds = {
            1: None,  # source_a
            3: 80,  # dedupe_a with threshold 80
        }
        source_config_filter = True

        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 3)  # dedupe_a context
            query = _build_unified_query(
                resolved_thresholds, source_config_filter, resolution
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return keys, with some potentially mapped to dedupe clusters
        assert len(result) > 0
        assert "root_id" in result.columns
        assert "leaf_key" in result.columns

    def test_build_unified_linker_scenario(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should build unified query for complex linker scenario."""
        resolved_thresholds = {
            1: None,  # source_a
            2: None,  # source_b
            3: 80,  # dedupe_a cached
            4: 70,  # dedupe_b cached
            5: 85,  # linker with threshold 85
        }
        source_config_filter = True

        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 5)  # linker context
            query = _build_unified_query(
                resolved_thresholds, source_config_filter, resolution
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return all 11 keys from both sources
        assert len(result) == 11

        # Should have linker clusters as roots for some keys
        root_ids = set(result["root_id"])

        # At threshold 85, only C503 (prob=90) should qualify from linker
        # But we should also see dedupe clusters from cached thresholds
        assert 503 in root_ids  # This is what we're trying to fix!
        assert 301 in root_ids  # from dedupe_a cached=80
        assert 401 in root_ids  # from dedupe_b cached=70


class TestResolveHierarchyAssignments:
    """Test hierarchy assignment resolution."""

    def test_source_only_resolution(self, populated_postgres_db: MatchboxPostgres):
        """Should return direct cluster assignments for source resolution."""
        with MBDB.get_session() as session:
            source_res = session.get(Resolutions, 1)  # source_a

            query = _resolve_hierarchy_assignments(source_res, None, None)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Source A has 6 keys, root_id should equal leaf_id for sources
            assert len(result) == 6
            assert all(result["root_id"] == result["leaf_id"])
            expected_keys = {
                "src_a_key1",
                "src_a_key2",
                "src_a_key3",
                "src_a_key4",
                "src_a_key5",
                "src_a_key6",
            }
            assert set(result["leaf_key"]) == expected_keys

    def test_linker_with_high_threshold(self, populated_postgres_db: MatchboxPostgres):
        """Should filter linker clusters only, using cached truth for parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold=95 - should exclude all linker clusters (max is 90)
            # But parents should still use cached values: dedupe_a=80, dedupe_b=70
            query = _resolve_hierarchy_assignments(linker_res, None, threshold=95)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 11 keys (6 from source_a + 5 from source_b)
            assert len(result) == 11

            # No linker clusters qualify, so keys should fall back to parent assignments
            # From dedupe_a (cached truth=80): C301 should still be used
            # From dedupe_b (cached truth=70): C401 should still be used
            root_ids = set(result["root_id"])
            assert 301 in root_ids  # dedupe_a's cluster (uses cached truth=80)
            assert 401 in root_ids  # dedupe_b's cluster (uses cached truth=70)

            # Linker clusters should NOT appear as roots
            linker_cluster_ids = {501, 502, 503, 504}
            assert linker_cluster_ids.isdisjoint(root_ids)

    def test_linker_with_medium_threshold(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should include some linker clusters, parents still use cached truth."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold=85 - should include C503 (prob=90) but exclude others
            # Parents still use cached: dedupe_a=80, dedupe_b=70
            query = _resolve_hierarchy_assignments(linker_res, None, threshold=85)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 11 keys
            assert len(result) == 11

            root_ids = set(result["root_id"])

            # From linker (threshold=85): only C503 qualifies (prob=90, flag=1)
            assert 503 in root_ids

            # Other linker clusters should not appear
            excluded_linker_clusters = {501, 502, 504}  # prob < 85 or wrong role_flag
            assert excluded_linker_clusters.isdisjoint(root_ids)

            # Parents should still contribute their cached-truth clusters
            assert 301 in root_ids  # from dedupe_a (cached=80)
            assert 401 in root_ids  # from dedupe_b (cached=70)

    def test_linker_with_low_threshold(self, populated_postgres_db):
        """Should include most linker clusters at low threshold."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold=80 - should include C503 (90) and C504 (80, flag=2)
            # C501 (90, flag=0) and C502 (80, flag=0) excluded by role_flag >= 1 filter
            query = _resolve_hierarchy_assignments(linker_res, None, threshold=80)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 11 keys
            assert len(result) == 11

            root_ids = set(result["root_id"])

            # From linker: C503 (prob=90, flag=1) and C504 (prob=80, flag=2)
            # should qualify
            assert 503 in root_ids
            assert 504 in root_ids

            # C501 and C502 have flag=0, so excluded by role_flag >= 1 filter
            pairwise_only_clusters = {501, 502}
            assert pairwise_only_clusters.isdisjoint(root_ids)

            # C301's keys (101,102) should be captured by C504 (higher priority)
            # So C301 should NOT appear as root
            assert 301 not in root_ids  # C301's keys captured by C504
            # Only C401 should appear for uncaptured keys
            assert 401 in root_ids  # C401 has keys not captured by linker clusters


class TestResolveClusterHierarchy:
    """Test cluster hierarchy resolution for specific source."""

    def test_same_resolution_passthrough(self, populated_postgres_db: MatchboxPostgres):
        """Should return direct mapping when truth resolution equals source."""
        with MBDB.get_session() as session:
            source_config = session.get(SourceConfigs, 1)  # source_a
            source_res = session.get(Resolutions, 1)  # source_a resolution

            query = _resolve_cluster_hierarchy(source_config, source_res, None)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 6 keys with their original cluster IDs
            assert len(result) == 6
            # For same resolution, id should equal original cluster assignments
            expected_pairs = {
                ("src_a_key1", 101),
                ("src_a_key2", 101),
                ("src_a_key3", 102),
                ("src_a_key4", 103),
                ("src_a_key5", 104),
                ("src_a_key6", 105),
            }
            actual_pairs = {
                (row["key"], row["id"]) for row in result.iter_rows(named=True)
            }
            assert actual_pairs == expected_pairs

    def test_hierarchy_through_model(self, populated_postgres_db: MatchboxPostgres):
        """Should resolve hierarchy through model resolution."""
        with MBDB.get_session() as session:
            source_config = session.get(SourceConfigs, 1)  # source_a
            dedupe_res = session.get(Resolutions, 3)  # dedupe_a

            query = _resolve_cluster_hierarchy(source_config, dedupe_res, threshold=80)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            print(f"Result: {len(result)} rows")
            print("Key assignments:")
            for row in result.iter_rows(named=True):
                print(f"  {row['key']} -> cluster {row['id']}")

            # At threshold 80, only C301 qualifies (prob=80), which contains
            # clusters 101+102
            # Keys from clusters 101+102: src_a_key1, src_a_key2, src_a_key3
            # Keys 1,2,3 should map to C301, others keep original clusters
            c301_keys = result.filter(result["id"] == 301)
            original_keys = result.filter(result["id"] != 301)

            assert len(c301_keys) == 3  # Keys that qualify for dedupe
            expected_301_keys = {"src_a_key1", "src_a_key2", "src_a_key3"}
            assert set(c301_keys["key"]) == expected_301_keys

            assert len(original_keys) == 3  # Keys that don't qualify
            expected_original_keys = {"src_a_key4", "src_a_key5", "src_a_key6"}
            assert set(original_keys["key"]) == expected_original_keys


class TestGetClustersWithLeaves:
    """Test cluster-leaf relationship extraction."""

    def test_get_clusters_for_model(self, populated_postgres_db: MatchboxPostgres):
        """Should return cluster hierarchy for model's parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Linker has parents dedupe_a and dedupe_b,
            # should get their cluster assignments
            assert len(result) > 0

            # Should contain cluster info with leaves
            for _, cluster_info in result.items():
                assert "root_hash" in cluster_info
                assert "leaves" in cluster_info
                assert isinstance(cluster_info["leaves"], list)

                # Each leaf should have required fields
                for leaf in cluster_info["leaves"]:
                    assert "leaf_id" in leaf
                    assert "leaf_hash" in leaf

    def test_get_clusters_for_deduper(self, populated_postgres_db: MatchboxPostgres):
        """Should return cluster assignments from deduper's parent (source)."""
        with MBDB.get_session() as session:
            dedupe_res = session.get(Resolutions, 3)  # dedupe_a

            result = get_clusters_with_leaves(dedupe_res)

            # Dedupe_a has parent source_a, so should get source_a's cluster assignments
            # Source assignments are 1:1 (each key maps to its own cluster)
            assert len(result) == 5  # source_a has 5 clusters (101-105)

            # Verify we have the expected source clusters
            expected_clusters = {101, 102, 103, 104, 105}
            actual_clusters = set(result.keys())
            assert actual_clusters == expected_clusters

            # Each source cluster should have exactly one leaf (itself)
            for cluster_id, cluster_info in result.items():
                assert len(cluster_info["leaves"]) == 1
                leaf = cluster_info["leaves"][0]
                assert leaf["leaf_id"] == cluster_id  # Source: leaf_id == root_id
                assert leaf["leaf_hash"] == cluster_info["root_hash"]

    def test_get_clusters_for_linker_specific_clusters(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should return specific cluster assignments from linker's parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Should include clusters from both dedupe_a and dedupe_b parents
            # Based on their default thresholds (cached truth values)

            # From dedupe_a (cached truth=80): should include C301 (prob=80)
            assert 301 in result, (
                f"C301 missing from linker's parent clusters: {result.keys()}"
            )

            # From dedupe_b (cached truth=70): should include C401 (prob=70)
            assert 401 in result, (
                f"C401 missing from linker's parent clusters: {result.keys()}"
            )

            # Verify C301's leaves (should contain clusters 101, 102)
            c301_leaves = result[301]["leaves"]
            c301_leaf_ids = {leaf["leaf_id"] for leaf in c301_leaves}
            assert c301_leaf_ids == {101, 102}, (
                f"C301 leaves incorrect: {c301_leaf_ids}"
            )

            # Verify C401's leaves (should contain clusters 201, 202)
            c401_leaves = result[401]["leaves"]
            c401_leaf_ids = {leaf["leaf_id"] for leaf in c401_leaves}
            assert c401_leaf_ids == {201, 202}, (
                f"C401 leaves incorrect: {c401_leaf_ids}"
            )

    def test_get_clusters_excludes_low_probability(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should exclude clusters that don't meet parent's cached threshold."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Should NOT include C302 (prob=70, flag=2) or C303 (prob=70, flag=0)
            # from dedupe_a because they're below/equal to cached truth=80
            assert 302 not in result, (
                f"C302 should be excluded (prob=70 < cached=80): {result.keys()}"
            )
            assert 303 not in result, (
                f"C303 should be excluded (prob=70 < cached=80): {result.keys()}"
            )

            # Should include C301 (prob=80, meets cached=80)
            assert 301 in result, (
                f"C301 should be included (prob=80 >= cached=80): {result.keys()}"
            )

    def test_get_clusters_for_source_returns_empty(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should return empty dict for source resolution (no parents)."""
        with MBDB.get_session() as session:
            source_res = session.get(Resolutions, 1)  # source_a

            result = get_clusters_with_leaves(source_res)

            # Sources have no parents, so should return empty
            assert result == {}, f"Source should have no parent clusters: {result}"

    def test_get_clusters_includes_source_assignments(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should include source cluster assignments from indirect parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Should include source clusters from both source_a and source_b
            # (indirect parents through dedupe_a and dedupe_b)

            result_clusters = set(result.keys())

            # Some source clusters should be present (those not captured by dedupers)
            # At minimum, uncaptured clusters should appear
            uncaptured_a = {103, 104, 105}  # Not in C301 (which captures 101, 102)
            uncaptured_b = {203, 204, 205}  # Not in C401 (which captures 201, 202)

            for cluster_id in uncaptured_a.union(uncaptured_b):
                if cluster_id in result_clusters:
                    # If present, should have itself as only leaf
                    leaves = result[cluster_id]["leaves"]
                    assert len(leaves) == 1
                    assert leaves[0]["leaf_id"] == cluster_id

    def test_get_clusters_leaf_structure(self, populated_postgres_db: MatchboxPostgres):
        """Should return leaves with correct structure and unique entries."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            for cluster_id, cluster_info in result.items():
                leaves = cluster_info["leaves"]

                # Each leaf should have correct structure
                for leaf in leaves:
                    assert "leaf_id" in leaf
                    assert "leaf_hash" in leaf
                    assert isinstance(leaf["leaf_id"], int)
                    assert isinstance(leaf["leaf_hash"], bytes)

                # Should not have duplicate leaves
                leaf_ids = [leaf["leaf_id"] for leaf in leaves]
                assert len(leaf_ids) == len(set(leaf_ids)), (
                    f"Duplicate leaves in cluster {cluster_id}: {leaf_ids}"
                )

                # Leaves should be valid cluster IDs
                for leaf in leaves:
                    leaf_id = leaf["leaf_id"]
                    assert leaf_id in range(101, 206), (
                        f"Invalid leaf cluster ID: {leaf_id}"
                    )


class TestQueryFunction:
    """Test main query function with various scenarios."""

    def test_query_source_only(self, populated_postgres_db: MatchboxPostgres):
        """Should query source data without resolution."""
        result = query("source_a", resolution=None, threshold=None, limit=None)

        # Should return all keys from source_a with their cluster assignments
        assert result.shape[0] == 6
        assert "id" in result.column_names
        assert "key" in result.column_names

        # For source-only queries, id should be the original cluster ID
        assert set(result["id"].to_pylist()) == {
            101,
            102,
            103,
            104,
            105,
        }  # Two keys in 101
        assert set(result["key"].to_pylist()) == {
            "src_a_key1",
            "src_a_key2",
            "src_a_key3",
            "src_a_key4",
            "src_a_key5",
            "src_a_key6",
        }

    def test_query_source_b_only(self, populated_postgres_db: MatchboxPostgres):
        """Should query source_b which has one key per cluster."""
        result = query("source_b", resolution=None, threshold=None, limit=None)

        # Should return all keys from source_b
        assert result.shape[0] == 5
        assert "id" in result.column_names
        assert "key" in result.column_names

        # Source B has one key per cluster
        assert set(result["id"].to_pylist()) == {201, 202, 203, 204, 205}
        assert set(result["key"].to_pylist()) == {
            "src_b_key1",
            "src_b_key2",
            "src_b_key3",
            "src_b_key4",
            "src_b_key5",
        }

    def test_query_through_deduper(self, populated_postgres_db: MatchboxPostgres):
        """Should query source through its deduper resolution."""
        result = query("source_a", resolution="dedupe_a", threshold=None, limit=None)

        # Should return all 6 keys, but some mapped to dedupe clusters
        assert result.shape[0] == 6
        assert "id" in result.column_names
        assert "key" in result.column_names

        # At dedupe_a's default threshold (80), should see C301 for some keys
        cluster_ids = set(result["id"].to_pylist())
        assert 301 in cluster_ids  # C301 should appear

        # Keys 1,2,3 should map to C301 (dedupe of clusters 101+102)
        keys_in_301 = [row["key"] for row in result.to_pylist() if row["id"] == 301]
        expected_301_keys = {"src_a_key1", "src_a_key2", "src_a_key3"}
        assert set(keys_in_301) == expected_301_keys

    def test_query_through_deduper_with_threshold(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should query source through deduper with threshold override."""
        # Test with threshold=90 (higher than dedupe_a's clusters)
        result = query("source_a", resolution="dedupe_a", threshold=90, limit=None)

        # Should return all 6 keys, but no dedupe clusters qualify
        assert result.shape[0] == 6

        # No dedupe clusters should qualify at threshold=90
        cluster_ids = set(result["id"].to_pylist())
        assert 301 not in cluster_ids  # C301 excluded (prob=80 < 90)

        # Should fall back to original source clusters
        expected_source_clusters = {101, 102, 103, 104, 105}
        assert cluster_ids.issubset(expected_source_clusters)

    def test_query_through_linker(self, populated_postgres_db: MatchboxPostgres):
        """Should query source through complex linker resolution."""
        result = query("source_a", resolution="linker_ab", threshold=None, limit=None)

        # Should return all 6 keys with linker cluster assignments
        assert result.shape[0] == 6

        cluster_ids = set(result["id"].to_pylist())

        # Should see linker clusters at default threshold (90)
        assert 503 in cluster_ids  # C503 should appear
        # C504 has prob=80 < default=90, so may not appear depending on role_flag

        # Verify specific key mappings
        key_cluster_map = {row["key"]: row["id"] for row in result.to_pylist()}

        # src_a_key6 should map to C503 (linker cluster containing C105)
        assert key_cluster_map["src_a_key6"] == 503

    def test_query_both_sources_through_linker(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should query both sources through linker with consistent results."""
        result_a = query("source_a", resolution="linker_ab", threshold=80, limit=None)
        result_b = query("source_b", resolution="linker_ab", threshold=80, limit=None)

        # Both should return their respective key counts
        assert result_a.shape[0] == 6  # source_a keys
        assert result_b.shape[0] == 5  # source_b keys

        # Should see overlapping cluster assignments
        clusters_a = set(result_a["id"].to_pylist())
        clusters_b = set(result_b["id"].to_pylist())

        # Both should include linker clusters that link across sources
        linker_clusters = {503, 504}  # Both qualify at threshold=80
        assert linker_clusters.intersection(clusters_a)  # Some linker clusters in A
        assert linker_clusters.intersection(clusters_b)  # Some linker clusters in B

        # Specific cross-source linking: C503 contains C105+C205
        a_key6_cluster = [
            row["id"] for row in result_a.to_pylist() if row["key"] == "src_a_key6"
        ][0]
        b_key5_cluster = [
            row["id"] for row in result_b.to_pylist() if row["key"] == "src_b_key5"
        ][0]
        assert (
            a_key6_cluster == b_key5_cluster == 503
        )  # Both map to same linker cluster

    def test_query_with_limit(self, populated_postgres_db: MatchboxPostgres):
        """Should respect limit parameter."""
        result = query("source_a", resolution=None, threshold=None, limit=3)

        # Should return only 3 rows
        assert result.shape[0] == 3
        assert "id" in result.column_names
        assert "key" in result.column_names

    def test_query_multiple_keys_per_cluster_scenario(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should handle case where multiple keys belong to same cluster."""
        # This tests the scenario causing your test failure
        result = query("source_a", resolution="dedupe_a", threshold=80, limit=None)

        # source_a has 6 keys but some share clusters:
        # - keys 1,2 both in cluster 101 → both map to C301
        # - key 3 in cluster 102 → also maps to C301
        # - keys 4,5,6 in separate clusters → keep original assignments

        assert result.shape[0] == 6  # Still 6 keys total

        # Count unique clusters vs total keys
        unique_clusters = len(set(result["id"].to_pylist()))
        total_keys = result.shape[0]

        # Should have fewer unique clusters than total keys (due to C301 grouping)
        assert unique_clusters < total_keys

        # Verify the specific multiple-keys-to-one-cluster mapping
        key_cluster_map = {row["key"]: row["id"] for row in result.to_pylist()}

        # Keys 1,2,3 should all map to the same dedupe cluster C301
        dedupe_cluster = key_cluster_map["src_a_key1"]
        assert key_cluster_map["src_a_key2"] == dedupe_cluster
        assert key_cluster_map["src_a_key3"] == dedupe_cluster
        assert dedupe_cluster == 301

    def test_query_nonexistent_source(self, populated_postgres_db: MatchboxPostgres):
        """Should raise exception for nonexistent source."""
        with pytest.raises(MatchboxSourceNotFoundError):
            query("nonexistent", resolution=None)

    def test_query_nonexistent_resolution(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should raise exception for nonexistent resolution."""
        with pytest.raises(MatchboxResolutionNotFoundError):
            query("source_a", resolution="nonexistent")


class TestMatchFunction:
    """Test matching function."""

    def test_match_within_same_cluster(self, populated_postgres_db: MatchboxPostgres):
        """Should find matches within the same cluster."""
        matches = match(
            key="src_a_key1",
            source="source_a",
            targets=["source_a", "source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 2  # One for each target

        # Should find the key in source_a target
        source_a_match = next(m for m in matches if m.target == "source_a")
        assert "src_a_key1" in source_a_match.source_id
        assert "src_a_key1" in source_a_match.target_id  # Should match itself

    def test_match_no_cross_source_matches(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should return empty matches when no cross-source links exist."""
        # src_a_key6 is in C105, not linked to source_b at dedupe level
        matches = match(
            key="src_a_key6",
            source="source_a",
            targets=["source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert len(source_b_match.target_id) == 0  # No matches in source_b

    def test_match_nonexistent_key(self, populated_postgres_db: MatchboxPostgres):
        """Should handle nonexistent key gracefully."""
        matches = match(
            key="nonexistent_key",
            source="source_a",
            targets=["source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 1
        assert len(matches[0].target_id) == 0  # No matches for nonexistent key


def test_sources_none_vs_none(populated_postgres_db: MatchboxPostgres):
    with MBDB.get_session() as session:
        resolution = session.get(Resolutions, 5)

        # Test 1: sources=None (keyword)
        query1 = _resolve_hierarchy_assignments(resolution, sources=None, threshold=80)
        # Test 2: None as positional
        query2 = _resolve_hierarchy_assignments(resolution, None, threshold=80)

        # Are they the same?
        print("Queries identical:", str(query1) == str(query2))
