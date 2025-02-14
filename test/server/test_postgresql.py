from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sqlalchemy import BIGINT, TEXT, Column, UniqueConstraint, text
from sqlalchemy.orm import Session

from matchbox.common.sources import Source
from matchbox.server import MatchboxDBAdapter
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.benchmark.generate_tables import (
    generate_all_tables,
    generate_result_tables,
)
from matchbox.server.postgresql.benchmark.query import (
    compile_match_sql,
    compile_query_sql,
)
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin
from matchbox.server.postgresql.utils.db import large_ingest
from matchbox.server.postgresql.utils.insert import HashIDMap

from ..fixtures.db import SetupDatabaseCallable


def test_hash_id_map():
    """Test HashIDMap core functionality including basic operations."""
    # Initialize with some existing mappings
    lookup = pa.Table.from_arrays(
        [
            pa.array([1, 2], type=pa.uint64()),
            pa.array([b"hash1", b"hash2"], type=pa.binary()),
        ],
        names=["id", "hash"],
    )
    hash_map = HashIDMap(start=100, lookup=lookup)

    # Test getting existing hashes
    ids = pa.array([2, 1], type=pa.uint64())
    hashes = hash_map.get_hashes(ids)
    assert hashes.to_pylist() == [b"hash2", b"hash1"]

    # Test getting mix of existing and new hashes
    input_hashes = pa.array([b"hash1", b"new_hash", b"hash2"], type=pa.binary())
    returned_ids = hash_map.get_ids(input_hashes)

    # Verify results
    id_list = returned_ids.to_pylist()
    assert id_list[0] == 1  # Existing hash1
    assert id_list[2] == 2  # Existing hash2
    assert id_list[1] == 100  # New hash got next available ID

    # Verify lookup table was updated correctly
    assert hash_map.lookup.shape == (3, 3)
    assert hash_map.next_int == 101

    # Test error handling for missing IDs
    with pytest.raises(ValueError) as exc_info:
        hash_map.get_hashes(pa.array([999], type=pa.uint64()))
    assert "not found in lookup table" in str(exc_info.value)


@pytest.mark.parametrize(
    ("parameters"),
    [
        # Test case 1: CDMS/CRN linker, CRN dataset
        {
            "point_of_truth": "deterministic_naive_test.cdms_naive_test.crn",
            "source_index": 0,  # CRN
            "unique_ids": 1_000,
            "unique_pks": 3_000,
        },
        # Test case 2: CDMS/CRN linker, CDMS dataset
        {
            "point_of_truth": "deterministic_naive_test.cdms_naive_test.crn",
            "source_index": 2,  # CDMS
            "unique_ids": 1_000,
            "unique_pks": 2_000,
        },
        # Test case 3: CRN/DUNS linker, CRN dataset
        {
            "point_of_truth": "deterministic_naive_test.crn_naive_test.duns",
            "source_index": 0,  # CRN
            "unique_ids": 1_000,
            "unique_pks": 3_000,
        },
        # Test case 4: CRN/DUNS linker, DUNS dataset
        {
            "point_of_truth": "deterministic_naive_test.crn_naive_test.duns",
            "source_index": 1,  # DUNS
            "unique_ids": 500,
            "unique_pks": 500,
        },
    ],
    ids=["cdms-crn_crn", "cdms-crn_cdms", "crn-duns_crn", "crn-duns_duns"],
)
def test_benchmark_query_generation(
    setup_database: SetupDatabaseCallable,
    matchbox_postgres: MatchboxPostgres,
    warehouse_data: list[Source],
    parameters: dict[str, Any],
):
    setup_database(matchbox_postgres, warehouse_data, "link")

    engine = MBDB.get_engine()
    point_of_truth = parameters["point_of_truth"]
    idx = parameters["source_index"]

    sql_query = compile_query_sql(
        point_of_truth=point_of_truth,
        source_address=warehouse_data[idx].address,
    )

    assert isinstance(sql_query, str)

    with engine.connect() as conn:
        res = conn.execute(text(sql_query)).all()

    df = pd.DataFrame(res, columns=["id", "pk"])

    assert df.id.nunique() == parameters["unique_ids"]
    assert df.pk.nunique() == parameters["unique_pks"]


def test_benchmark_match_query_generation(
    setup_database: SetupDatabaseCallable,
    matchbox_postgres: MatchboxPostgres,
    warehouse_data: list[Source],
    revolution_inc: dict[str, list[str]],
):
    setup_database(matchbox_postgres, warehouse_data, "link")

    engine = MBDB.get_engine()
    source_pks = revolution_inc["duns"]
    target_pks = revolution_inc["crn"]

    sql_match = compile_match_sql(
        source_pk=source_pks[0],
        source_name=warehouse_data[1].address.full_name,  # DUNS
        point_of_truth="deterministic_naive_test.crn_naive_test.duns",
    )

    assert isinstance(sql_match, str)

    with engine.connect() as conn:
        res = conn.execute(text(sql_match)).all()

    df = pd.DataFrame(res, columns=["cluster", "dataset", "source_pk"]).dropna()

    assert df.cluster.nunique() == 1
    assert df.dataset.nunique() == 2
    assert set(df.source_pk) == set(source_pks + target_pks)


@pytest.mark.parametrize(
    ("left_ids", "right_ids", "next_id", "n_components", "n_probs"),
    (
        [range(10_000), None, 10_000, 8000, 2000],
        [range(8000), range(8000, 16_000), 16_000, 6000, 10_000],
    ),
    ids=["dedupe", "link"],
)
def test_benchmark_result_tables(left_ids, right_ids, next_id, n_components, n_probs):
    resolution_id = None

    (top_clusters, _, _, _, _) = generate_result_tables(
        left_ids, right_ids, resolution_id, next_id, n_components, n_probs
    )

    assert len(top_clusters) == n_components


@pytest.mark.parametrize(
    ("cluster_start_id", "dataset_start_id", "expected_datasets"),
    [
        (0, 1, {1, 2, None}),  # Original test case
        (1000, 1, {1, 2, None}),  # Different cluster start
        (0, 3, {3, 4, None}),  # Different dataset start
        (86475, 3, {3, 4, None}),  # Both different
    ],
)
def test_benchmark_generate_tables_parameterized(
    matchbox_postgres: MatchboxDBAdapter,
    cluster_start_id: int,
    dataset_start_id: int,
    expected_datasets: set,
):
    schema = MBDB.MatchboxBase.metadata.schema
    matchbox_postgres.clear(certain=True)

    def array_encode(array: Iterable[str]):
        if not array:
            return None
        escaped_l = [f'"{s}"' for s in array]
        list_rep = ", ".join(escaped_l)
        return "{" + list_rep + "}"

    with MBDB.get_engine().connect() as con:
        results = generate_all_tables(
            source_len=20,
            dedupe_components=5,
            dedupe_len=25,
            link_components=5,
            link_len=25,
            cluster_start_id=cluster_start_id,
            dataset_start_id=dataset_start_id,
        )

        # Test number of tables
        assert len(results) == len(MBDB.MatchboxBase.metadata.tables)

        # Test dataset IDs
        assert (
            set(pc.unique(results["clusters"]["dataset"]).to_pylist())
            == expected_datasets
        )

        # Test cluster IDs start correctly
        min_cluster_id = min(results["clusters"]["cluster_id"].to_pylist())
        assert min_cluster_id == cluster_start_id

        # Test resolution IDs in sources
        source_resolution_ids = set(results["sources"]["resolution_id"].to_pylist())
        assert source_resolution_ids == {dataset_start_id, dataset_start_id + 1}

        # Test resolution IDs in resolutions
        resolution_ids = set(results["resolutions"]["resolution_id"].to_pylist())
        expected_resolution_ids = set(
            range(
                dataset_start_id,
                dataset_start_id + 5,  # We expect 5 resolutions
            )
        )
        assert resolution_ids == expected_resolution_ids

        # Write to database
        for table_name, table_arrow in results.items():
            df = table_arrow.to_pandas()
            # Pandas' `to_sql` dislikes arrays
            array_cols = ["source_pk", "column_types", "column_aliases", "column_names"]
            active_array_cols = set(df.columns.tolist()).intersection(array_cols)
            for col in active_array_cols:
                df[col] = df[col].apply(array_encode)
            # Pandas' `to_sql` dislikes large unsigned ints
            for c in df.columns:
                if df[c].dtype == "uint64":
                    df[c] = df[c].astype("int64")
            df.to_sql(
                name=table_name, con=con, schema=schema, index=False, if_exists="append"
            )

        # Verify relationships in resolution_from table match dataset_start_id
        resolution_from = results["resolution_from"]
        parent_ids = set(resolution_from["parent"].to_pylist())
        child_ids = set(resolution_from["child"].to_pylist())
        all_ids = parent_ids.union(child_ids)
        assert min(all_ids) >= dataset_start_id
        assert max(all_ids) < dataset_start_id + 5  # We expect 5 resolutions

        # Verify probabilities reference correct resolution IDs
        prob_resolution_ids = set(results["probabilities"]["resolution"].to_pylist())
        expected_model_ids = {
            dataset_start_id + 2,  # dedupe1
            dataset_start_id + 3,  # dedupe2
            dataset_start_id + 4,  # link
        }
        assert prob_resolution_ids == expected_model_ids


def test_large_ingest(matchbox_postgres: MatchboxPostgres):
    with Session(MBDB.get_engine()) as session:
        # Dummy table to ingest to
        class DummyTable(CountMixin, MBDB.MatchboxBase):
            __tablename__ = "dummytable"
            foo = Column(BIGINT, primary_key=True)
            bar = Column(TEXT, nullable=False)

            __table_args__ = (UniqueConstraint("bar", name="dummy_unique_bar"),)

        MBDB.MatchboxBase.metadata.create_all(
            MBDB.get_engine(), tables=[DummyTable.__table__]
        )

        row1 = DummyTable(foo=1, bar="First dummy row")
        session.add(row1)
        session.commit()

    # Dummy data to ingest
    schema = pa.schema([("foo", pa.int64()), ("bar", pa.string())])
    data = pa.Table.from_pylist(
        [
            {"foo": 10, "bar": "abc"},
            {"foo": 11, "bar": "def"},
        ],
        schema=schema,
    )

    large_ingest(data, DummyTable)
    # TODO: check constraints still apply; check indices apply
    # TODO: what happens if we ingest data that doesn't satisfy constraints?
    # TODO: allow custom subset selection before table copy
    # TODO: replace batch_ingest with large_ingest

    assert DummyTable.count() == 3
