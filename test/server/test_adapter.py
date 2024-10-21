import random

import pytest
import rustworkx as rx
from dotenv import find_dotenv, load_dotenv
from matchbox.common.exceptions import (
    MatchboxDataError,
    MatchboxDatasetError,
    MatchboxModelError,
)
from matchbox.common.hash import HASH_FUNC
from matchbox.helpers.selector import query, selector, selectors
from matchbox.server import MatchboxDBAdapter
from matchbox.server.base import MatchboxModelAdapter
from matchbox.server.models import Cluster, Probability, Source
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


backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(params=backends)
def matchbox_backend(request: pytest.FixtureRequest) -> MatchboxDBAdapter:
    """Fixture to provide different backend implementations."""
    return request.param()


@pytest.mark.parametrize("backend", backends)
def test_index(
    backend: MatchboxDBAdapter,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    crn_companies: DataFrame,
    duns_companies: DataFrame,
    cdms_companies: DataFrame,
    request: pytest.FixtureRequest,
):
    """Test that indexing data works."""
    backend = request.getfixturevalue(backend)

    assert backend.data.count() == 0

    db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)

    def count_deduplicates(df: DataFrame) -> int:
        return df.drop(columns=["id"]).drop_duplicates().shape[0]

    unique = sum(
        count_deduplicates(df) for df in [crn_companies, duns_companies, cdms_companies]
    )

    assert backend.data.count() == unique


@pytest.mark.parametrize("backend", backends)
def test_query_single_table(
    backend: MatchboxDBAdapter,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from the database."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)

    # Test
    crn = warehouse_data[0]

    select_crn = selector(
        table=str(crn),
        fields=["id", "crn"],
        engine=crn.database.engine,
    )

    df_crn_sample = query(
        selector=select_crn,
        backend=backend,
        model=None,
        return_type="pandas",
        limit=10,
    )

    assert isinstance(df_crn_sample, DataFrame)
    assert df_crn_sample.shape[0] == 10

    df_crn_full = query(
        selector=select_crn,
        backend=backend,
        model=None,
        return_type="pandas",
    )

    assert df_crn_full.shape[0] == 3000
    assert set(df_crn_full.columns) == {
        "data_hash",
        "test_crn_id",
        "test_crn_crn",
    }


@pytest.mark.parametrize("backend", backends)
def test_query_multi_table(
    backend: MatchboxDBAdapter,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from multiple tables from the database."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)

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
        backend=backend,
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


@pytest.mark.parametrize("backend", backends)
def test_query_with_dedupe_model(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from a deduplication point of truth."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
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
        backend=backend,
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


@pytest.mark.parametrize("backend", backends)
def test_query_with_link_model(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    db_add_link_models_and_data: AddLinkModelsAndDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from a link point of truth."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_link_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        backend=backend,
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
        backend=backend,
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


@pytest.mark.parametrize("backend", backends)
def test_validate_hashes(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test validating data hashes."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
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
        backend=backend,
        model="naive_test.crn",
        return_type="pandas",
    )

    # Test validating data hashes
    backend.validate_hashes(hashes=df_crn.data_hash.to_list(), hash_type="data")

    # Test validating cluster hashes
    backend.validate_hashes(
        hashes=df_crn.cluster_hash.drop_duplicates().to_list(), hash_type="cluster"
    )

    # Test validating nonexistant hashes errors
    with pytest.raises(MatchboxDataError):
        backend.validate_hashes(
            hashes=[HASH_FUNC(b"nonexistant").digest()], hash_type="data"
        )


@pytest.mark.parametrize("backend", backends)
def test_get_dataset(
    backend: MatchboxDBAdapter,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test querying data from the database."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)
    crn = warehouse_data[0]

    # Test get a real dataset
    backend.get_dataset(
        db_schema=crn.db_schema, db_table=crn.db_table, engine=crn.database.engine
    )

    # Test getting a dataset that doesn't exist in Matchbox
    with pytest.raises(MatchboxDatasetError):
        backend.get_dataset(
            db_schema="nonexistant", db_table="nonexistant", engine=crn.database.engine
        )


@pytest.mark.parametrize("backend", backends)
def test_get_model_subgraph(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    db_add_link_models_and_data: AddLinkModelsAndDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test getting model from the model subgraph."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_link_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        link_data=link_data_test_params,
        link_models=[link_model_test_params[0]],  # Deterministic linker,
        request=request,
    )

    # Test
    subgraph = backend.get_model_subgraph()

    assert isinstance(subgraph, rx.PyDiGraph)

    assert subgraph.num_nodes() > 0
    assert subgraph.num_edges() > 0


@pytest.mark.parametrize("backend", backends)
def test_get_model(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test getting a model from the database."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    # Test getting a real model
    model = backend.get_model(model="naive_test.crn")
    assert isinstance(model, MatchboxModelAdapter)

    # Test getting a model that doesn't exist
    with pytest.raises(MatchboxModelError):
        backend.get_model(model="nonexistant")


@pytest.mark.parametrize("backend", backends)
def test_delete_model(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    db_add_link_models_and_data: AddLinkModelsAndDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """
    Tests the deletion of:

    * The model from the model table
    * The creates edges the model made
    * Any models that depended on this model, and their creates edges
    * Any probability values associated with the model
    * All of the above for all parent models. As every model is defined by
        its children, deleting a model means cascading deletion to all ancestors
    """
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_link_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        link_data=link_data_test_params,
        link_models=[link_model_test_params[0]],  # Deterministic linker,
        request=request,
    )

    # Expect it to delete itself, its probabilities,
    # its parents, and their probabilities
    deduper_to_delete = "naive_test.crn"
    total_models = len(dedupe_data_test_params) + len(link_data_test_params)

    model_list_pre_delete = backend.models.count()
    cluster_count_pre_delete = backend.clusters.count()
    cluster_assoc_count_pre_delete = backend.creates.count()
    proposed_merge_probs_pre_delete = backend.proposes.count()
    actual_merges_pre_delete = backend.merges.count()

    assert model_list_pre_delete == total_models
    assert cluster_count_pre_delete > 0
    assert cluster_assoc_count_pre_delete > 0
    assert proposed_merge_probs_pre_delete > 0
    assert actual_merges_pre_delete > 0

    # Perform deletion
    backend.delete_model(deduper_to_delete, certain=True)

    model_list_post_delete = backend.models.count()
    cluster_count_post_delete = backend.clusters.count()
    cluster_assoc_count_post_delete = backend.creates.count()
    proposed_merge_probs_post_delete = backend.proposes.count()
    actual_merges_post_delete = backend.merges.count()

    # Deletes deduper and parent linkers: 3 models gone
    assert model_list_post_delete == model_list_pre_delete - 3

    # Cluster, dedupe and link count unaffected
    assert cluster_count_post_delete == cluster_count_pre_delete
    assert actual_merges_post_delete == actual_merges_pre_delete

    # But count of propose and create edges has dropped
    assert cluster_assoc_count_post_delete < cluster_assoc_count_pre_delete
    assert proposed_merge_probs_post_delete < proposed_merge_probs_pre_delete


@pytest.mark.parametrize("backend", backends)
def test_insert_model(
    backend: MatchboxDBAdapter,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test that models can be inserted."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_indexed_data(backend=backend, warehouse_data=warehouse_data)
    crn = warehouse_data[0]
    duns = warehouse_data[1]

    # Test deduper insertion
    model_count = backend.models.count()

    backend.insert_model("dedupe_1", left=str(crn), description="Test deduper 1")
    backend.insert_model("dedupe_2", left=str(duns), description="Test deduper 1")

    assert backend.models.count() == model_count + 2

    # Test linker insertion
    backend.insert_model(
        "link_1", left="dedupe_1", right="dedupe_2", description="Test linker 1"
    )

    assert backend.models.count() == model_count + 3


@pytest.mark.parametrize("backend", backends)
def test_model_insert_probabilities(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test that model insert probabilities are correct."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    crn = warehouse_data[0]
    select_crn = selector(
        table=str(crn),
        fields=["crn"],
        engine=crn.database.engine,
    )
    df_crn = query(
        selector=select_crn,
        backend=backend,
        model=None,
        return_type="pandas",
    )

    duns = warehouse_data[1]
    select_duns = selector(
        table=str(duns),
        fields=["id", "duns"],
        engine=duns.database.engine,
    )
    df_crn_deduped = query(
        selector=select_crn,
        backend=backend,
        model="naive_test.crn",
        return_type="pandas",
    )
    df_duns_deduped = query(
        selector=select_duns,
        backend=backend,
        model="naive_test.duns",
        return_type="pandas",
    )

    backend.insert_model("dedupe_1", left=str(crn), description="Test deduper 1")
    backend.insert_model("dedupe_2", left=str(duns), description="Test deduper 1")
    backend.insert_model(
        "link_1", left="dedupe_1", right="dedupe_2", description="Test linker 1"
    )

    # Test dedupe probabilities
    dedupe_probabilities = [
        Probability(
            hash=HASH_FUNC(random.randbytes(32)).digest(),
            left=crn_prob_1,
            right=crn_prob_2,
            probability=1.0,
        )
        for crn_prob_1, crn_prob_2 in zip(
            df_crn["data_hash"].to_list()[:10],
            reversed(df_crn["data_hash"].to_list()[:10]),
            strict=True,
        )
    ]
    dedupe_1 = backend.get_model(model="dedupe_1")

    assert dedupe_1.probabilities.count() == 0

    dedupe_1.insert_probabilities(
        probabilities=dedupe_probabilities,
        probability_type="deduplications",
        batch_size=10,
    )

    assert dedupe_1.probabilities.count() == 10

    # Test link probabilities
    link_probabilities = [
        Probability(
            hash=HASH_FUNC(random.randbytes(32)).digest(),
            left=crn_prob,
            right=duns_prob,
            probability=1.0,
        )
        for crn_prob, duns_prob in zip(
            df_crn_deduped["cluster_hash"].to_list()[:10],
            df_duns_deduped["cluster_hash"].to_list()[:10],
            strict=True,
        )
    ]
    link_1 = backend.get_model(model="link_1")

    assert link_1.probabilities.count() == 0

    link_1.insert_probabilities(
        probabilities=link_probabilities,
        probability_type="links",
        batch_size=10,
    )

    assert link_1.probabilities.count() == 10


@pytest.mark.parametrize("backend", backends)
def test_model_insert_clusters(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test that model insert clusters are correct."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    crn = warehouse_data[0]
    select_crn = selector(
        table=str(crn),
        fields=["crn"],
        engine=crn.database.engine,
    )
    df_crn = query(
        selector=select_crn,
        backend=backend,
        model=None,
        return_type="pandas",
    )

    duns = warehouse_data[1]
    select_duns = selector(
        table=str(duns),
        fields=["id", "duns"],
        engine=duns.database.engine,
    )
    df_crn_deduped = query(
        selector=select_crn,
        backend=backend,
        model="naive_test.crn",
        return_type="pandas",
    )
    df_duns_deduped = query(
        selector=select_duns,
        backend=backend,
        model="naive_test.duns",
        return_type="pandas",
    )

    backend.insert_model("dedupe_1", left=str(crn), description="Test deduper 1")
    backend.insert_model("dedupe_2", left=str(duns), description="Test deduper 1")
    backend.insert_model(
        "link_1", left="dedupe_1", right="dedupe_2", description="Test linker 1"
    )

    # Test dedupe clusters
    dedupe_clusters = [
        Cluster(parent=crn_prob_1, child=crn_prob_2)
        for crn_prob_1, crn_prob_2 in zip(
            df_crn["data_hash"].to_list()[:10],
            reversed(df_crn["data_hash"].to_list()[:10]),
            strict=True,
        )
    ]
    dedupe_1 = backend.get_model(model="dedupe_1")

    assert dedupe_1.clusters.count() == 0

    dedupe_1.insert_clusters(
        clusters=dedupe_clusters,
        cluster_type="deduplications",
        batch_size=10,
    )

    assert dedupe_1.clusters.count() == 10

    # Test link clusters
    link_clusters = [
        Cluster(parent=crn_prob, child=duns_prob)
        for crn_prob, duns_prob in zip(
            df_crn_deduped["cluster_hash"].to_list()[:10],
            df_duns_deduped["cluster_hash"].to_list()[:10],
            strict=True,
        )
    ]
    link_1 = backend.get_model(model="link_1")

    assert link_1.clusters.count() == 0

    link_1.insert_clusters(
        clusters=link_clusters,
        cluster_type="links",
        batch_size=10,
    )

    assert link_1.clusters.count() == 10


@pytest.mark.parametrize("backend", backends)
def test_model_properties(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test that model properties obey their protocol restrictions."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    naive_crn = backend.get_model(model="naive_test.crn")

    # Test hash exists
    assert naive_crn.hash

    # Test name exists
    assert naive_crn.name

    # Test probabilities is countable
    assert isinstance(naive_crn.probabilities.count(), int)

    # Test clusters is countable
    assert isinstance(naive_crn.clusters.count(), int)


@pytest.mark.parametrize("backend", backends)
def test_properties(
    backend: MatchboxDBAdapter,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: pytest.FixtureRequest,
):
    """Test that properties obey their protocol restrictions."""
    backend = request.getfixturevalue(backend)

    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=backend,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    # Test dataset is listable and countable
    assert isinstance(backend.datasets.list(), list)
    assert isinstance(backend.datasets.count(), int)

    # Test models is countable
    assert isinstance(backend.models.count(), int)

    # Test data is countable
    assert isinstance(backend.data.count(), int)

    # Test clusters is countable
    assert isinstance(backend.clusters.count(), int)

    # Test creates is countable
    assert isinstance(backend.creates.count(), int)

    # Test merges is countable
    assert isinstance(backend.merges.count(), int)

    # Test proposes is countable
    assert isinstance(backend.proposes.count(), int)


def test_clear(matchbox_postgres: MatchboxPostgres):
    """Test clearing the database."""
    pass
