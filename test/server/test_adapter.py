import random
from typing import Callable

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
from matchbox.server.base import MatchboxModelAdapter
from matchbox.server.models import Cluster, Probability, Source
from pandas import DataFrame

from ..fixtures.models import (
    dedupe_data_test_params,
    link_data_test_params,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str):
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj


@pytest.mark.parametrize("backend", backends)
class TestMatchboxBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        request: pytest.FixtureRequest,
        backend_instance: str,
        setup_database: Callable,
        warehouse_data: list[Source],
    ):
        self.backend = backend_instance
        self.setup_database = lambda level: setup_database(
            self.backend, warehouse_data, level
        )
        self.warehouse_data = warehouse_data

    def test_properties(self):
        """Test that properties obey their protocol restrictions."""
        self.setup_database("index")
        assert isinstance(self.backend.datasets.list(), list)
        assert isinstance(self.backend.datasets.count(), int)
        assert isinstance(self.backend.models.count(), int)
        assert isinstance(self.backend.data.count(), int)
        assert isinstance(self.backend.clusters.count(), int)
        assert isinstance(self.backend.creates.count(), int)
        assert isinstance(self.backend.merges.count(), int)
        assert isinstance(self.backend.proposes.count(), int)

    def test_model_properties(self):
        """Test that model properties obey their protocol restrictions."""
        self.setup_database("dedupe")
        naive_crn = self.backend.get_model(model="naive_test.crn")
        assert naive_crn.hash
        assert naive_crn.name
        assert isinstance(naive_crn.probabilities.count(), int)
        assert isinstance(naive_crn.clusters.count(), int)

    def test_validate_hashes(self):
        """Test validating data hashes."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]
        select_crn = selector(
            table=str(crn),
            fields=["company_name", "crn"],
            engine=crn.database.engine,
        )
        df_crn = query(
            selector=select_crn,
            backend=self.backend,
            model="naive_test.crn",
            return_type="pandas",
        )

        self.backend.validate_hashes(hashes=df_crn.hash.to_list())
        with pytest.raises(MatchboxDataError):
            self.backend.validate_hashes(hashes=[HASH_FUNC(b"nonexistant").digest()])

    def test_get_dataset(self):
        """Test querying data from the database."""
        self.setup_database("index")

        crn = self.warehouse_data[0]

        self.backend.get_dataset(
            db_schema=crn.db_schema, db_table=crn.db_table, engine=crn.database.engine
        )
        with pytest.raises(MatchboxDatasetError):
            self.backend.get_dataset(
                db_schema="nonexistant",
                db_table="nonexistant",
                engine=crn.database.engine,
            )

    def test_get_model_subgraph(self):
        """Test getting model from the model subgraph."""
        self.setup_database("link")

        subgraph = self.backend.get_model_subgraph()

        assert isinstance(subgraph, rx.PyDiGraph)
        assert subgraph.num_nodes() > 0
        assert subgraph.num_edges() > 0

    def test_get_model(self):
        """Test getting a model from the database."""
        self.setup_database("dedupe")

        model = self.backend.get_model(model="naive_test.crn")
        assert isinstance(model, MatchboxModelAdapter)

        with pytest.raises(MatchboxModelError):
            self.backend.get_model(model="nonexistant")

    def test_delete_model(self):
        """
        Tests the deletion of:

        * The model from the model table
        * The creates edges the model made
        * Any models that depended on this model, and their creates edges
        * Any probability values associated with the model
        * All of the above for all parent models. As every model is defined by
            its children, deleting a model means cascading deletion to all ancestors
        """
        self.setup_database("link")

        # Expect it to delete itself, its probabilities,
        # its parents, and their probabilities
        deduper_to_delete = "naive_test.crn"
        total_models = len(dedupe_data_test_params) + len(link_data_test_params)

        model_list_pre_delete = self.backend.models.count()
        cluster_count_pre_delete = self.backend.clusters.count()
        cluster_assoc_count_pre_delete = self.backend.creates.count()
        proposed_merge_probs_pre_delete = self.backend.proposes.count()
        actual_merges_pre_delete = self.backend.merges.count()

        assert model_list_pre_delete == total_models
        assert cluster_count_pre_delete > 0
        assert cluster_assoc_count_pre_delete > 0
        assert proposed_merge_probs_pre_delete > 0
        assert actual_merges_pre_delete > 0

        # Perform deletion
        self.backend.delete_model(deduper_to_delete, certain=True)

        model_list_post_delete = self.backend.models.count()
        cluster_count_post_delete = self.backend.clusters.count()
        cluster_assoc_count_post_delete = self.backend.creates.count()
        proposed_merge_probs_post_delete = self.backend.proposes.count()
        actual_merges_post_delete = self.backend.merges.count()

        # Deletes deduper and parent linkers: 3 models gone
        assert model_list_post_delete == model_list_pre_delete - 3

        # Cluster, dedupe and link count unaffected
        assert cluster_count_post_delete == cluster_count_pre_delete
        assert actual_merges_post_delete == actual_merges_pre_delete

        # But count of propose and create edges has dropped
        assert cluster_assoc_count_post_delete < cluster_assoc_count_pre_delete
        assert proposed_merge_probs_post_delete < proposed_merge_probs_pre_delete

    def test_insert_model(self):
        """Test that models can be inserted."""
        self.setup_database("index")

        crn = self.warehouse_data[0]
        duns = self.warehouse_data[1]

        # Test deduper insertion
        model_count = self.backend.models.count()

        self.backend.insert_model(
            "dedupe_1", left=str(crn), description="Test deduper 1"
        )
        self.backend.insert_model(
            "dedupe_2", left=str(duns), description="Test deduper 1"
        )

        assert self.backend.models.count() == model_count + 2

        # Test linker insertion
        self.backend.insert_model(
            "link_1", left="dedupe_1", right="dedupe_2", description="Test linker 1"
        )

        assert self.backend.models.count() == model_count + 3

    def test_model_insert_probabilities(self):
        """Test that model insert probabilities are correct."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]
        select_crn = selector(
            table=str(crn),
            fields=["crn"],
            engine=crn.database.engine,
        )
        df_crn = query(
            selector=select_crn,
            backend=self.backend,
            model=None,
            return_type="pandas",
        )

        duns = self.warehouse_data[1]
        select_duns = selector(
            table=str(duns),
            fields=["id", "duns"],
            engine=duns.database.engine,
        )
        df_crn_deduped = query(
            selector=select_crn,
            backend=self.backend,
            model="naive_test.crn",
            return_type="pandas",
        )
        df_duns_deduped = query(
            selector=select_duns,
            backend=self.backend,
            model="naive_test.duns",
            return_type="pandas",
        )

        self.backend.insert_model(
            "dedupe_1", left=str(crn), description="Test deduper 1"
        )
        self.backend.insert_model(
            "dedupe_2", left=str(duns), description="Test deduper 1"
        )
        self.backend.insert_model(
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
                df_crn["hash"].to_list()[:10],
                reversed(df_crn["hash"].to_list()[:10]),
                strict=True,
            )
        ]
        dedupe_1 = self.backend.get_model(model="dedupe_1")

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
                df_crn_deduped["hash"].to_list()[:10],
                df_duns_deduped["hash"].to_list()[:10],
                strict=True,
            )
        ]
        link_1 = self.backend.get_model(model="link_1")

        assert link_1.probabilities.count() == 0

        link_1.insert_probabilities(
            probabilities=link_probabilities,
            probability_type="links",
            batch_size=10,
        )

        assert link_1.probabilities.count() == 10

    def test_model_insert_clusters(self):
        """Test that model insert clusters are correct."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]
        select_crn = selector(
            table=str(crn),
            fields=["crn"],
            engine=crn.database.engine,
        )
        df_crn = query(
            selector=select_crn,
            backend=self.backend,
            model=None,
            return_type="pandas",
        )

        duns = self.warehouse_data[1]
        select_duns = selector(
            table=str(duns),
            fields=["id", "duns"],
            engine=duns.database.engine,
        )
        df_crn_deduped = query(
            selector=select_crn,
            backend=self.backend,
            model="naive_test.crn",
            return_type="pandas",
        )
        df_duns_deduped = query(
            selector=select_duns,
            backend=self.backend,
            model="naive_test.duns",
            return_type="pandas",
        )

        self.backend.insert_model(
            "dedupe_1", left=str(crn), description="Test deduper 1"
        )
        self.backend.insert_model(
            "dedupe_2", left=str(duns), description="Test deduper 1"
        )
        self.backend.insert_model(
            "link_1", left="dedupe_1", right="dedupe_2", description="Test linker 1"
        )

        # Test dedupe clusters
        dedupe_clusters = [
            Cluster(parent=crn_prob_1, child=crn_prob_2)
            for crn_prob_1, crn_prob_2 in zip(
                df_crn["hash"].to_list()[:10],
                reversed(df_crn["hash"].to_list()[:10]),
                strict=True,
            )
        ]
        dedupe_1 = self.backend.get_model(model="dedupe_1")

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
                df_crn_deduped["hash"].to_list()[:10],
                df_duns_deduped["hash"].to_list()[:10],
                strict=True,
            )
        ]
        link_1 = self.backend.get_model(model="link_1")

        assert link_1.clusters.count() == 0

        link_1.insert_clusters(
            clusters=link_clusters,
            cluster_type="links",
            batch_size=10,
        )

        assert link_1.clusters.count() == 10

    def test_index(
        self,
        crn_companies: DataFrame,
        duns_companies: DataFrame,
        cdms_companies: DataFrame,
    ):
        """Test that indexing data works."""
        assert self.backend.data.count() == 0

        self.setup_database("index")

        def count_deduplicates(df: DataFrame) -> int:
            return df.drop(columns=["id"]).drop_duplicates().shape[0]

        unique = sum(
            count_deduplicates(df)
            for df in [crn_companies, duns_companies, cdms_companies]
        )

        assert self.backend.data.count() == unique

    def test_query_single_table(self):
        """Test querying data from the database."""
        self.setup_database("index")

        crn = self.warehouse_data[0]
        select_crn = selector(
            table=str(crn),
            fields=["id", "crn"],
            engine=crn.database.engine,
        )
        df_crn_sample = query(
            selector=select_crn,
            backend=self.backend,
            model=None,
            return_type="pandas",
            limit=10,
        )

        assert isinstance(df_crn_sample, DataFrame)
        assert df_crn_sample.shape[0] == 10

        df_crn_full = query(
            selector=select_crn,
            backend=self.backend,
            model=None,
            return_type="pandas",
        )

        assert df_crn_full.shape[0] == 3000
        assert set(df_crn_full.columns) == {
            "hash",
            "test_crn_id",
            "test_crn_crn",
        }

    def test_query_multi_table(self):
        """Test querying data from multiple tables from the database."""
        self.setup_database("index")

        crn = self.warehouse_data[0]
        duns = self.warehouse_data[1]

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
            backend=self.backend,
            model=None,
            return_type="pandas",
        )

        assert df_crn_duns_full.shape[0] == 3500
        assert (
            df_crn_duns_full[df_crn_duns_full["test_duns_id"].notnull()].shape[0] == 500
        )
        assert (
            df_crn_duns_full[df_crn_duns_full["test_crn_id"].notnull()].shape[0] == 3000
        )

        assert set(df_crn_duns_full.columns) == {
            "hash",
            "test_crn_id",
            "test_crn_crn",
            "test_duns_id",
            "test_duns_duns",
        }

    def test_query_with_dedupe_model(self):
        """Test querying data from a deduplication point of truth."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]

        select_crn = selector(
            table=str(crn),
            fields=["company_name", "crn"],
            engine=crn.database.engine,
        )

        df_crn = query(
            selector=select_crn,
            backend=self.backend,
            model="naive_test.crn",
            return_type="pandas",
        )

        assert isinstance(df_crn, DataFrame)
        assert df_crn.shape[0] == 3000
        assert set(df_crn.columns) == {
            "hash",
            "test_crn_crn",
            "test_crn_company_name",
        }
        assert df_crn.hash.nunique() == 1000

    def test_query_with_link_model(self):
        """Test querying data from a link point of truth."""
        self.setup_database("link")

        linker_name = "deterministic_naive_test.crn_naive_test.duns"

        crn_wh = self.warehouse_data[0]
        select_crn = selector(
            table=str(crn_wh),
            fields=["crn"],
            engine=crn_wh.database.engine,
        )

        duns_wh = self.warehouse_data[1]
        select_duns = selector(
            table=str(duns_wh),
            fields=["duns"],
            engine=duns_wh.database.engine,
        )

        select_crn_duns = selectors(select_crn, select_duns)

        crn_duns = query(
            selector=select_crn_duns,
            backend=self.backend,
            model=linker_name,
            return_type="pandas",
        )

        assert isinstance(crn_duns, DataFrame)
        assert crn_duns.shape[0] == 3500
        assert set(crn_duns.columns) == {
            "hash",
            "test_crn_crn",
            "test_duns_duns",
        }
        assert crn_duns.hash.nunique() == 1000

    def test_clear(self):
        """Test clearing the database."""
        self.setup_database("dedupe")

        assert self.backend.datasets.count() > 0
        assert self.backend.data.count() > 0
        assert self.backend.models.count() > 0
        assert self.backend.clusters.count() > 0
        assert self.backend.creates.count() > 0
        assert self.backend.merges.count() > 0
        assert self.backend.proposes.count() > 0

        self.backend.clear(certain=True)

        assert self.backend.datasets.count() == 0
        assert self.backend.data.count() == 0
        assert self.backend.models.count() == 0
        assert self.backend.clusters.count() == 0
        assert self.backend.creates.count() == 0
        assert self.backend.merges.count() == 0
        assert self.backend.proposes.count() == 0
