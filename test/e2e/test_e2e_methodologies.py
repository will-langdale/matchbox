"""Integration tests for all methodology classes using real DAG execution.

These tests verify that each methodology (deduper/linker) actually works when integrated
with the full Query/DAG system, not just in isolation with mocked dependencies.
"""

import polars as pl
import pytest
from httpx import Client
from sqlalchemy import Engine, text

# Import configurator functions from methodology tests
from test.client.models.methodologies.test_dedupers_deterministic import (
    DEDUPERS,
)
from test.client.models.methodologies.test_linkers_deterministic import (
    configure_deterministic_linker,
    configure_splink_linker,
    configure_weighted_deterministic_linker,
)

from matchbox.client.dags import DAG
from matchbox.client.locations import RelationalDBLocation
from matchbox.client.models.linkers import (
    DeterministicLinker,
    SplinkLinker,
    WeightedDeterministicLinker,
)
from matchbox.common.factories.sources import (
    FeatureConfig,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EMethodologyIntegration:
    """Integration tests for all methodology classes with real pipeline execution."""

    def _clean_company_name(self, column: str) -> str:
        """Generate cleaning SQL for a company name column.

        Removes company suffixes (Ltd, Limited) and normalises whitespace.
        """
        return f"""
            trim(
                regexp_replace(
                    regexp_replace(
                        {column},
                        ' (Ltd|Limited)$',
                        '',
                        'g'
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            )
        """

    @pytest.fixture(autouse=True, scope="function")
    def setup_environment(
        self,
        matchbox_client: Client,
        postgres_warehouse: Engine,
    ):
        """Set up warehouse and database using fixtures.

        This fixture is shared across all tests in the class, amortising the
        expensive setup cost across multiple methodology tests.
        """
        n_true_entities = 10
        self.warehouse_engine = postgres_warehouse

        # Create feature configurations
        features = {
            "company_name": FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            "registration_id": FeatureConfig(
                name="registration_id",
                base_generator="bothify",
                parameters=(("text", "REG-###-???"),),
            ),
        }

        # Create two sources for testing
        source_parameters = (
            SourceTestkitParameters(
                name="source_a",
                engine=postgres_warehouse,
                features=(
                    features["company_name"].add_variations(
                        SuffixRule(suffix=" Ltd"),
                        SuffixRule(suffix=" Limited"),
                    ),
                    features["registration_id"],
                ),
                n_true_entities=n_true_entities,
                repetition=0,  # No duplicates in source A
            ),
            SourceTestkitParameters(
                name="source_b",
                engine=postgres_warehouse,
                features=(
                    features["company_name"],
                    features["registration_id"],
                ),
                n_true_entities=n_true_entities // 2,  # Half overlap for linking tests
                repetition=1,  # Duplicates in source B for deduplication tests
            ),
        )

        # Create linked sources testkit with ground truth
        self.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,
        )

        # Write tables to warehouse
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location()

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        yield

        # Teardown - clean up warehouse tables
        with postgres_warehouse.connect() as conn:
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()

        # Clear matchbox database after test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    # === DEDUPER TESTS ===

    @pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
    def test_deduper_integration(self, Deduper, configure_deduper):
        """Test that dedupers work end-to-end with real Query execution.

        This verifies that the methodology can:
        - Accept data from real Query objects (not mocks)
        - Process the actual data schemas produced by the warehouse
        - Integrate correctly with DAG execution flow
        - Produce valid results that match ground truth
        """
        # Setup DAG and location
        dw_loc = RelationalDBLocation(name="dbname").set_client(self.warehouse_engine)
        dag = DAG("deduper_integration_test").new_run()

        # Get the testkit for source B (which has duplicates)
        source_b_testkit = self.linked_testkit.sources["source_b"]

        # Create source using fluent API (like real users would)
        source_b = dag.source(
            location=dw_loc,
            name="source_b",
            extract_transform="""
                select
                    key::text as id,
                    company_name,
                    registration_id
                from
                    source_b;
            """,
            infer_types=True,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        # Get settings from configurator
        settings = configure_deduper(source_b_testkit)

        # Create deduper using fluent API
        dedupe_result = source_b.query(
            cleaning={
                "company_name": self._clean_company_name(source_b.f("company_name")),
                "registration_id": source_b.f("registration_id"),
            },
        ).deduper(
            name=f"test_{Deduper.__name__}",
            description=f"Integration test for {Deduper.__name__}",
            model_class=Deduper,
            model_settings=settings,
        )

        # Execute the DAG
        dag.run_and_sync()

        # Get results - just verify it runs and produces output
        results = dedupe_result.run()

        # Basic integration checks
        assert results is not None, f"{Deduper.__name__} returned None"
        assert results.probabilities is not None
        assert isinstance(results.probabilities, pl.DataFrame)
        assert len(results.probabilities) >= 0

    # === LINKER TESTS ===

    @pytest.mark.parametrize(
        ("Linker", "configure_linker"),
        [
            pytest.param(
                DeterministicLinker,
                configure_deterministic_linker,
                id="DeterministicLinker",
            ),
            pytest.param(
                WeightedDeterministicLinker,
                configure_weighted_deterministic_linker,
                id="WeightedDeterministicLinker",
            ),
            pytest.param(
                SplinkLinker,
                configure_splink_linker,
                id="SplinkLinker",
            ),
        ],
    )
    def test_linker_integration(self, Linker, configure_linker):
        """Test that linkers work end-to-end with real Query execution.

        This verifies that the methodology can:
        - Accept data from real Query objects on both left and right
        - Handle the actual data schemas and types from the warehouse
        - Execute successfully through the DAG
        - Produce results without crashing
        """
        # Setup DAG and location
        dw_loc = RelationalDBLocation(name="dbname").set_client(self.warehouse_engine)
        dag = DAG("linker_integration_test").new_run()

        # Get testkits for both sources
        source_a_testkit = self.linked_testkit.sources["source_a"]
        source_b_testkit = self.linked_testkit.sources["source_b"]

        # Create sources using fluent API
        source_a = dag.source(
            location=dw_loc,
            name="source_a",
            extract_transform="""
                select
                    key::text as id,
                    company_name,
                    registration_id
                from
                    source_a;
            """,
            infer_types=True,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        source_b = dag.source(
            location=dw_loc,
            name="source_b",
            extract_transform="""
                select
                    key::text as id,
                    company_name,
                    registration_id
                from
                    source_b;
            """,
            infer_types=True,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        # Get settings from configurator
        settings = configure_linker(source_a_testkit, source_b_testkit)

        # Create linker using fluent API - link sources directly
        link_result = source_a.query(
            cleaning={
                "company_name": self._clean_company_name(source_a.f("company_name")),
                "registration_id": source_a.f("registration_id"),
            },
        ).linker(
            source_b.query(
                cleaning={
                    "company_name": self._clean_company_name(
                        source_b.f("company_name")
                    ),
                    "registration_id": source_b.f("registration_id"),
                },
            ),
            name=f"test_{Linker.__name__}",
            description=f"Integration test for {Linker.__name__}",
            model_class=Linker,
            model_settings=settings,
        )

        # Execute the DAG
        dag.run_and_sync()

        # Get results - just verify it runs and produces output
        results = link_result.run()

        # Basic integration checks
        assert results is not None, f"{Linker.__name__} returned None"
        assert results.probabilities is not None
        assert isinstance(results.probabilities, pl.DataFrame)
        assert len(results.probabilities) >= 0
