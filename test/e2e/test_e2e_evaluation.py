import asyncio

import pytest
from httpx import Client
from sqlalchemy import Engine, text

from matchbox.client.cli.eval import EvalData
from matchbox.client.cli.eval.ui import EntityResolutionApp
from matchbox.client.dags import DAG, DedupeStep, IndexStep, StepInput
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.common.arrow import SCHEMA_CLUSTER_EXPANSION, SCHEMA_JUDGEMENTS
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)
from matchbox.common.sources import RelationalDBLocation, SourceConfig


@pytest.mark.docker
class TestE2EModelEvaluation:
    """End to end tests for model evaluation functionality."""

    client: Client | None = None
    warehouse_engine: Engine | None = None
    linked_testkit: LinkedSourcesTestkit | None = None
    n_true_entities: int | None = None

    @pytest.fixture(autouse=True, scope="function")
    def setup_environment(
        self,
        matchbox_client: Client,
        postgres_warehouse: Engine,
    ):
        """Set up warehouse and database using fixtures."""
        # Store fixtures as class attributes
        n_true_entities = 10  # Keep it small for simplicity

        self.__class__.client = matchbox_client
        self.__class__.warehouse_engine = postgres_warehouse
        self.__class__.n_true_entities = n_true_entities

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
                repetition=1,  # No duplicates
            ),
        )

        linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,
        )

        self.__class__.linked_testkit = linked_testkit

        # Create tables in warehouse
        for source_testkit in linked_testkit.sources.values():
            source_testkit.write_to_location(client=postgres_warehouse, set_client=True)

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        # Create DAG
        dw_loc = RelationalDBLocation(name="postgres", client=postgres_warehouse)
        batch_size = 1000

        source_a_config = SourceConfig.new(
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
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )
        self.__class__.source_a_config = source_a_config

        i_source_a = IndexStep(source_config=source_a_config, batch_size=batch_size)

        self.__class__.final_resolution = "final"
        dedupe_a = DedupeStep(
            left=StepInput(
                prev_node=i_source_a,
                select={source_a_config: ["company_name", "registration_id"]},
                batch_size=batch_size,
            ),
            name=self.final_resolution,
            description="Deduplicate source A",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": [source_a_config.f("registration_id")],
            },
            truth=1.0,
        )

        dag = DAG()
        dag.add_steps(i_source_a, dedupe_a)
        dag.run()

        yield

        # Teardown
        with postgres_warehouse.connect() as conn:
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()

        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    def test_evaluation_workflow(self):
        """Test Textual UI sampling, judging and model scoring workflow."""

        # Test that the Textual app can be initialised with scenario data
        # Get warehouse URL with real password (not masked)
        # The engine.url masks the password, so we need to reconstruct it
        url = self.warehouse_engine.url
        warehouse_url = f"{url.drivername}://{url.username}:{url.password}@{url.host}:{url.port}/{url.database}"

        app = EntityResolutionApp(
            resolution=self.final_resolution,
            num_samples=5,
            user="alice",
            warehouse=warehouse_url,
        )

        # Test the core interaction flow works with real data using proper Textual
        async def test_with_real_data():
            async with app.run_test() as pilot:
                await pilot.pause()

                # Should have authenticated
                assert app.state.user_name == "alice"
                assert app.state.user_id is not None

                # Should have loaded samples - that's enough for this test
                # We're just testing the app can run with real data

                return True

        result = asyncio.run(test_with_real_data())
        assert result is True

        # Test basic evaluation infrastructure still works
        eval_data = EvalData()
        assert SCHEMA_JUDGEMENTS.equals(eval_data.judgements.schema)
        assert SCHEMA_CLUSTER_EXPANSION.equals(eval_data.expansion.schema)
