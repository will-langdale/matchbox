import pytest
from dotenv import find_dotenv, load_dotenv
from matchbox.common.exceptions import MatchboxDBDataError
from matchbox.common.hash import HASH_FUNC
from matchbox.helpers.selector import query, selector, selectors
from matchbox.server.models import Source
from matchbox.server.postgresql import MatchboxPostgres
from pandas import DataFrame

from ..fixtures.db import (
    AddDedupeModelsAndDataCallable,
    AddIndexedDataCallable,
    AddLinkModelsAndDataCallable,
)
from ..fixtures.models import (
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def test_index(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    crn_companies: DataFrame,
    duns_companies: DataFrame,
    cdms_companies: DataFrame,
):
    """Test that indexing data works."""
    assert matchbox_postgres.data.count() == 0

    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    def count_deduplicates(df: DataFrame) -> int:
        return df.drop(columns=["id"]).drop_duplicates().shape[0]

    unique = sum(
        count_deduplicates(df) for df in [crn_companies, duns_companies, cdms_companies]
    )

    assert matchbox_postgres.data.count() == unique


def test_query_single_table(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    """Test querying data from the database."""
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    # Test
    crn = warehouse_data[0]

    select_crn = selector(
        table=str(crn),
        fields=["id", "crn"],
        engine=crn.database.engine,
    )

    df_crn_sample = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
        limit=10,
    )

    assert isinstance(df_crn_sample, DataFrame)
    assert df_crn_sample.shape[0] == 10

    df_crn_full = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
    )

    assert df_crn_full.shape[0] == 3000
    assert set(df_crn_full.columns) == {
        "data_hash",
        "test_crn_id",
        "test_crn_crn",
    }


def test_query_multi_table(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    """Test querying data from multiple tables from the database."""
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    # Test
    crn = warehouse_data[0]
    duns = warehouse_data[1]

    select_crn = selector(
        table=str(crn),
        fields=["id", "crn"],
        engine=crn.database.engine,
    )
    select_duns = selector(
        table=str(duns),
        fields=["id", "duns"],
        engine=duns.database.engine,
    )
    select_crn_duns = selectors(select_crn, select_duns)

    df_crn_duns_full = query(
        selector=select_crn_duns,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
    )

    assert df_crn_duns_full.shape[0] == 3500
    assert df_crn_duns_full[df_crn_duns_full["test_duns_id"].notnull()].shape[0] == 500
    assert df_crn_duns_full[df_crn_duns_full["test_crn_id"].notnull()].shape[0] == 3000

    assert set(df_crn_duns_full.columns) == {
        "data_hash",
        "test_crn_id",
        "test_crn_crn",
        "test_duns_id",
        "test_duns_duns",
    }


def test_query_with_dedupe_model(
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from a deduplication point of truth."""
    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    # Test
    crn = warehouse_data[0]

    select_crn = selector(
        table=str(crn),
        fields=["company_name", "crn"],
        engine=crn.database.engine,
    )

    df_crn = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model="naive_test.crn",
        return_type="pandas",
    )

    assert isinstance(df_crn, DataFrame)
    assert df_crn.shape[0] == 3000
    assert set(df_crn.columns) == {
        "cluster_hash",
        "data_hash",
        "test_crn_crn",
        "test_crn_company_name",
    }
    assert df_crn.data_hash.nunique() == 3000
    assert df_crn.cluster_hash.nunique() == 1000


def test_query_with_link_model(
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    db_add_link_models_and_data: AddLinkModelsAndDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from a link point of truth."""
    # Setup
    db_add_link_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        link_data=link_data_test_params,
        link_models=[link_model_test_params[0]],  # Deterministic linker,
        request=request,
    )

    # Test
    linker_name = "deterministic_naive_test.crn_naive_test.duns"

    crn_wh = warehouse_data[0]
    select_crn = selector(
        table=str(crn_wh),
        fields=["crn"],
        engine=crn_wh.database.engine,
    )

    duns_wh = warehouse_data[1]
    select_duns = selector(
        table=str(duns_wh),
        fields=["duns"],
        engine=duns_wh.database.engine,
    )

    select_crn_duns = selectors(select_crn, select_duns)

    crn_duns = query(
        selector=select_crn_duns,
        backend=matchbox_postgres,
        model=linker_name,
        return_type="pandas",
    )

    assert isinstance(crn_duns, DataFrame)
    assert crn_duns.shape[0] == 3500
    assert set(crn_duns.columns) == {
        "cluster_hash",
        "data_hash",
        "test_crn_crn",
        "test_duns_duns",
    }
    assert crn_duns.data_hash.nunique() == 3500
    assert crn_duns.cluster_hash.nunique() == 1000


def test_validate_hashes(
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test validating data hashes."""
    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    crn = warehouse_data[0]
    select_crn = selector(
        table=str(crn),
        fields=["company_name", "crn"],
        engine=crn.database.engine,
    )
    df_crn = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model="naive_test.crn",
        return_type="pandas",
    )

    # Test validating data hashes
    matchbox_postgres.validate_hashes(
        hashes=df_crn.data_hash.to_list(), hash_type="data"
    )

    # Test validating cluster hashes
    matchbox_postgres.validate_hashes(
        hashes=df_crn.cluster_hash.drop_duplicates().to_list(), hash_type="cluster"
    )

    # Test validating nonexistant hashes errors
    with pytest.raises(MatchboxDBDataError):
        matchbox_postgres.validate_hashes(
            hashes=[HASH_FUNC(b"nonexistant").digest()], hash_type="data"
        )


def test_get_dataset(matchbox_postgres: MatchboxPostgres):
    # Test getting an existing model
    pass


def test_get_model_subgraph(matchbox_postgres: MatchboxPostgres):
    # Test getting the model subgraph
    pass


def test_get_model(matchbox_postgres: MatchboxPostgres):
    # Test getting an existing model
    pass


def test_delete_leaf_model(matchbox_postgres: MatchboxPostgres):
    """Test deletion of a model with no dependencies."""
    pass


def test_delete_node_model(matchbox_postgres: MatchboxPostgres):
    """Test deletion of a model with downstream dependencies."""
    pass


def test_insert_model(matchbox_postgres: MatchboxPostgres):
    def test_insert_deduper_model():
        pass

    def test_insert_linker_model():
        pass

    def test_insert_duplicate_model():
        pass


# Additional tests for other properties and methods


def test_datasets_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_models_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_models_from_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_data_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_clusters_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_creates_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_merges_property(matchbox_postgres: MatchboxPostgres):
    pass


def test_proposes_property(matchbox_postgres: MatchboxPostgres):
    pass
