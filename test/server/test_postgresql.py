from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pytest
from sqlalchemy import text

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


def test_benchmark_generate_tables(matchbox_postgres: MatchboxDBAdapter):
    schema = MBDB.MatchboxBase.metadata.schema
    matchbox_postgres.clear(certain=True)

    def array_encode(array: Iterable[str]):
        if not array:
            return None
        escaped_l = [f'"{s}"' for s in array]
        list_rep = ", ".join(escaped_l)
        return "{" + list_rep + "}"

    with MBDB.get_engine().connect() as con:
        results = generate_all_tables(20, 5, 25, 5, 25)

        assert len(results) == len(MBDB.MatchboxBase.metadata.tables)

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
