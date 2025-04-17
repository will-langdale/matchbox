import itertools

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sqlalchemy import BIGINT, TEXT, Column, Engine, MetaData, UniqueConstraint, text
from sqlalchemy.orm import declarative_base

from matchbox.common.exceptions import MatchboxDatabaseWriteError
from matchbox.common.factories.entities import SourceEntity
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

from ...fixtures.db import setup_scenario


def test_hash_id_map():
    """Test HashIDMap core functionality including basic operations."""
    # Initialize with some existing mappings
    lookup = pa.Table.from_arrays(
        [
            pa.array([1, 2], type=pa.uint64()),
            pa.array([b"hash1", b"hash2"], type=pa.large_binary()),
        ],
        names=["id", "hash"],
    )
    hash_map = HashIDMap(start=100, lookup=lookup)

    # Test getting existing hashes
    ids = pa.array([2, 1], type=pa.uint64())
    hashes = hash_map.get_hashes(ids)
    assert hashes.to_pylist() == [b"hash2", b"hash1"]

    # Test getting mix of existing and new hashes
    input_hashes = pa.array([b"hash1", b"new_hash", b"hash2"], type=pa.large_binary())
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
    ("point_of_truth", "source"),
    [
        # Test case 1: CDMS/CRN linker, CRN dataset
        pytest.param(
            "probabilistic_naive_test.crn_naive_test.cdms", "crn", id="cdms-crn_crn"
        ),
        # Test case 2: CDMS/CRN linker, CDMS dataset
        pytest.param(
            "probabilistic_naive_test.crn_naive_test.cdms", "cdms", id="cdms-crn_cdms"
        ),
        # Test case 3: CRN/DUNS linker, CRN dataset
        pytest.param(
            "deterministic_naive_test.crn_naive_test.duns", "crn", id="crn-duns_crn"
        ),
        # Test case 4: CRN/DUNS linker, DUNS dataset
        pytest.param(
            "deterministic_naive_test.crn_naive_test.duns", "duns", id="crn-duns_duns"
        ),
    ],
)
@pytest.mark.docker
def test_benchmark_query_generation(
    matchbox_postgres: MatchboxPostgres,
    postgres_warehouse: Engine,
    point_of_truth: str,
    source: str,
):
    with setup_scenario(matchbox_postgres, "link", warehouse=postgres_warehouse) as dag:
        engine = MBDB.get_engine()

        sources_dict = dag.get_sources_for_model(point_of_truth)
        assert len(sources_dict) == 1
        linked = dag.linked[next(iter(sources_dict))]

        true_entities = linked.true_entity_subset(source)
        true_pks = set(
            itertools.chain.from_iterable(
                s for e in true_entities for s in e.source_pks.values()
            )
        )

        sql_query = compile_query_sql(
            point_of_truth=point_of_truth,
            source_address=dag.sources[source].source.address,
        )

        assert isinstance(sql_query, str)

        with engine.connect() as conn:
            res = conn.execute(text(sql_query)).all()

        df = pd.DataFrame(res, columns=["id", "pk"])

        assert df.id.nunique() == len(true_entities)
        assert set(df.pk) == true_pks


@pytest.mark.docker
def test_benchmark_match_query_generation(
    matchbox_postgres: MatchboxPostgres,
    postgres_warehouse: Engine,
):
    with setup_scenario(matchbox_postgres, "link", warehouse=postgres_warehouse) as dag:
        engine = MBDB.get_engine()

        linker_name = "deterministic_naive_test.crn_naive_test.duns"
        duns_testkit = dag.sources.get("duns")

        sources_dict = dag.get_sources_for_model(linker_name)
        assert len(sources_dict) == 1
        linked = dag.linked[next(iter(sources_dict))]

        # A random one:many entity
        source_entity: SourceEntity = linked.find_entities(
            min_appearances={"crn": 2, "duns": 1},
            max_appearances={"duns": 1},
        )[0]

        sql_match = compile_match_sql(
            source_pk=next(iter(source_entity.source_pks["duns"])),
            source_address=duns_testkit.source.address,
            point_of_truth="deterministic_naive_test.crn_naive_test.duns",
        )

        assert isinstance(sql_match, str)

        with engine.connect() as conn:
            res = conn.execute(text(sql_match)).all()

        df = pd.DataFrame(res, columns=["cluster", "dataset", "source_pk"]).dropna()

        assert df.cluster.nunique() == 1
        assert df.dataset.nunique() == 2
        assert (
            set(df.source_pk)
            == source_entity.source_pks["duns"] | source_entity.source_pks["crn"]
        )


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
        (0, 1, {1, 2}),  # Original test case
        (1000, 1, {1, 2}),  # Different cluster start
        (0, 3, {3, 4}),  # Different dataset start
        (86475, 3, {3, 4}),  # Both different
    ],
    ids=[
        "original",
        "different_cluster_start",
        "different_dataset_start",
        "both_different",
    ],
)
@pytest.mark.docker
def test_benchmark_generate_tables_parameterised(
    matchbox_postgres: MatchboxDBAdapter,
    cluster_start_id: int,
    dataset_start_id: int,
    expected_datasets: set,
):
    schema = MBDB.MatchboxBase.metadata.schema
    matchbox_postgres.clear(certain=True)

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

    # Test dataset IDs in cluster_source_pks table
    assert (
        set(pc.unique(results["cluster_source_pks"]["source_id"]).to_pylist())
        == expected_datasets
    )

    # Test dataset IDs in source_columns table
    assert (
        set(pc.unique(results["source_columns"]["source_id"]).to_pylist())
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

    with MBDB.get_adbc_connection() as conn, conn.cursor() as cur:
        # Write to database
        for table_name, table_arrow in results.items():
            cur.adbc_ingest(
                table_name, table_arrow, db_schema_name=schema, mode="append"
            )
        conn.commit()

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


@pytest.mark.docker
def test_large_ingest_simple(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
):
    """Test append-only mode of large ingest."""
    engine = MBDB.get_engine()
    metadata = MetaData(schema=MBDB.MatchboxBase.metadata.schema)

    # Initialise DummyTable to which we'll ingest
    class DummyTable(CountMixin, declarative_base(metadata=metadata)):
        __tablename__ = "dummytable"
        pk = Column(BIGINT, primary_key=True)
        foo = Column(TEXT, nullable=False)

    metadata.create_all(engine, tables=[DummyTable.__table__])
    metadata.reflect(engine)
    original_tables = len(metadata.tables)

    # Ingest data with manual PK and chunking
    large_ingest(
        data=pa.Table.from_pylist(
            [{"pk": 0, "foo": "val1"}],
        ),
        table_class=DummyTable,
        max_chunksize=100,
    )

    # Ingest data without PK and no chunking
    large_ingest(data=pa.Table.from_pylist([{"foo": "val2"}]), table_class=DummyTable)

    # Both rows were fine
    assert DummyTable.count() == 2
    with MBDB.get_session() as session:
        second_id = (
            session.query(DummyTable.pk).filter(DummyTable.foo == "val2").scalar()
        )
    assert 0 < second_id

    # By default, upserting not allowed
    with pytest.raises(MatchboxDatabaseWriteError):
        large_ingest(
            data=pa.Table.from_pylist([{"pk": 0, "foo": "val3"}]),
            table_class=DummyTable,
        )

    # Failed ingestion has no effect
    assert DummyTable.count() == 2
    assert len(metadata.tables) == original_tables

    # Columns not available in the target table are rejected
    with pytest.raises(ValueError, match="does not have columns"):
        large_ingest(
            data=pa.Table.from_pylist([{"pk": 10, "bar": "val3"}]),
            table_class=DummyTable,
        )


@pytest.mark.docker
def test_large_ingest_upsert_custom_update(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
):
    """Test large ingest with upsertion and custom columns to update."""
    engine = MBDB.get_engine()
    metadata = MetaData(schema=MBDB.MatchboxBase.metadata.schema)

    # Initialise DummyTable to which we'll ingest
    class DummyTable(CountMixin, declarative_base(metadata=metadata)):
        __tablename__ = "dummytable"
        pk = Column(BIGINT, primary_key=True)
        foo = Column(TEXT, nullable=False)
        bar = Column(TEXT, nullable=False)

        __table_args__ = (UniqueConstraint("foo", name="unique_foo"),)

    metadata.create_all(engine, tables=[DummyTable.__table__])
    metadata.reflect(engine)
    original_tables = len(metadata.tables)

    # Initialise with one original row
    with MBDB.get_session() as session:
        row1 = DummyTable(pk=1, foo="original foo", bar="original bar")
        session.add(row1)
        session.commit()

    # Some choices of parameters are not allowed
    with pytest.raises(ValueError, match="Cannot update a custom upsert key"):
        large_ingest(
            data=pa.Table.from_pylist([{"pk": 1, "foo": "new foo", "bar": "new bar"}]),
            table_class=DummyTable,
            update_columns=["foo"],
            upsert_keys=["foo"],
        )

    with pytest.raises(ValueError, match="different custom upsert key"):
        large_ingest(
            data=pa.Table.from_pylist([{"pk": 1, "foo": "new foo", "bar": "new bar"}]),
            table_class=DummyTable,
            update_columns=["pk"],
        )

    # Ingest updated data
    large_ingest(
        data=pa.Table.from_pylist([{"pk": 1, "foo": "new foo", "bar": "new bar"}]),
        table_class=DummyTable,
        update_columns=["foo"],
    )

    # Number of rows unchanged
    assert DummyTable.count() == 1

    # Only foo has changed
    with MBDB.get_session() as session:
        new_foo = session.query(DummyTable.foo).filter(DummyTable.pk == 1).scalar()
        new_bar = session.query(DummyTable.bar).filter(DummyTable.pk == 1).scalar()
    assert "new" in new_foo
    assert "original" in new_bar

    # No lingering temp tables
    metadata.clear()  # clear all Table objects from this MetaData, doesn't touch DB
    metadata.reflect(engine)
    assert len(metadata.tables) == original_tables

    # Cannot update column when constraints violated
    with pytest.raises(MatchboxDatabaseWriteError):
        large_ingest(
            pa.Table.from_pylist([{"pk": 2, "foo": "new foo", "bar": "new bar"}]),
            DummyTable,
            update_columns=["foo"],
        )

    # Failed ingestion has no effect
    assert DummyTable.count() == 1
    metadata.clear()
    metadata.reflect(engine)
    assert len(metadata.tables) == original_tables

    # Constraints are not violated when upserting
    large_ingest(
        pa.Table.from_pylist([{"pk": 1, "foo": "new foo", "bar": "new bar"}]),
        DummyTable,
        update_columns=["foo"],
    )

    # Nothing changed still
    assert DummyTable.count() == 1


@pytest.mark.docker
def test_large_ingest_upsert_custom_key(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
):
    """Test large ingest with upsertion on custom keys."""
    engine = MBDB.get_engine()
    metadata = MetaData(schema=MBDB.MatchboxBase.metadata.schema)

    # Initialise DummyTable to which we'll ingest
    class DummyTable(CountMixin, declarative_base(metadata=metadata)):
        __tablename__ = "dummytable"
        pk = Column(BIGINT, primary_key=True)
        other_key = Column(TEXT, nullable=False)
        foo = Column(TEXT, nullable=False)

        __table_args__ = (UniqueConstraint("other_key", name="unique_other_key"),)

    metadata.create_all(engine, tables=[DummyTable.__table__])

    # Initialise with one original row
    with MBDB.get_session() as session:
        metadata.reflect(engine)
        original_tables = len(metadata.tables)

        row1 = DummyTable(pk=1, other_key="a", foo="original foo")
        session.add(row1)
        session.commit()

    # Ingest updated data
    large_ingest(
        data=pa.Table.from_pylist([{"pk": 2, "other_key": "a", "foo": "new foo"}]),
        table_class=DummyTable,
        upsert_keys=["other_key"],
    )

    # Number of rows unchanged
    assert DummyTable.count() == 1

    with MBDB.get_session() as session:
        new_foo = (
            session.query(DummyTable.foo).filter(DummyTable.other_key == "a").scalar()
        )
        new_pk = (
            session.query(DummyTable.pk).filter(DummyTable.other_key == "a").scalar()
        )

    # We can update standard columns and primary keys
    assert "new" in new_foo
    assert new_pk == 2

    # No lingering temp tables
    metadata.clear()  # clear all Table objects from this MetaData, doesn't touch DB
    metadata.reflect(engine)
    assert len(metadata.tables) == original_tables
