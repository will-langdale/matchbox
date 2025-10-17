import pytest
from httpx import Client
from sqlalchemy import Engine, text

from matchbox.client import _handler
from matchbox.client.cli.eval import EntityResolutionApp
from matchbox.client.dags import DAG
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.sources import RelationalDBLocation
from matchbox.common.factories.sources import (
    FeatureConfig,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EModelEvaluation:
    """End to end tests for model evaluation functionality."""

    def _clean_company_name(self, column: str) -> str:
        """Generate cleaning SQL for a company name column."""
        return f"""
            trim(
                regexp_replace(
                    {column},
                    ' (Ltd|Limited)$',
                    '',
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
        """Set up warehouse and database using fixtures."""
        # Persist shared setup for use in the test body
        n_true_entities = 10
        final_resolution_1_name = "final_1"
        final_resolution_2_name = "final_2"
        self.warehouse_engine = postgres_warehouse

        # Create a SINGLE source with duplicates
        source_parameters = (
            SourceTestkitParameters(
                name="source_a",
                engine=postgres_warehouse,
                features=(
                    FeatureConfig(
                        name="company_name",
                        base_generator="company",
                    ).add_variations(
                        SuffixRule(suffix=" Ltd"),
                        SuffixRule(suffix=" Limited"),
                    ),
                    FeatureConfig(
                        name="registration_id",
                        base_generator="bothify",
                        parameters=(("text", "REG-###-???"),),
                    ),
                ),
                n_true_entities=n_true_entities,
                repetition=1,  # Duplicate all rows for deduplication
            ),
        )
        self.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters, seed=42
        )
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location()

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        # === SHARED CONFIGURATION ===
        dw_loc = RelationalDBLocation(name="postgres", client=postgres_warehouse)

        # === DAG 1: Created by User 1 (Strict Deduplication) ===
        dag1 = DAG("companies1").new_run()

        source_a_dag1 = dag1.source(
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

        source_a_dag1.query().deduper(
            name=final_resolution_1_name,
            description="Deduplicate by registration ID",
            model_class=NaiveDeduper,
            model_settings={
                "id": "id",
                "unique_fields": [source_a_dag1.f("registration_id")],
            },
        )

        dag1.run_and_sync()

        # Retain DAG for use in tests
        self.dag1 = dag1

        # === DAG 2: Created by User 2 (Loose Deduplication) ===
        dag2 = DAG("companies2").new_run()

        source_a_dag2 = dag2.source(
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

        source_a_dag2.query(
            cleaning={
                "company_name": self._clean_company_name(
                    source_a_dag2.f("company_name")
                )
            }
        ).deduper(
            name=final_resolution_2_name,
            description="Deduplicate by company name",
            model_class=NaiveDeduper,
            model_settings={"id": "id", "unique_fields": ["company_name"]},
        )

        dag2.run_and_sync()

        self.dag2 = dag2

        yield

        # Teardown
        with postgres_warehouse.connect() as conn:
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    @pytest.mark.asyncio
    async def test_evaluation_workflow(self):
        """Test complete end-to-end evaluation workflow.

        app startup → user painting → submission → model evaluation.
        """
        # Create warehouse location
        warehouse_location = RelationalDBLocation(
            name="test_warehouse", client=self.warehouse_engine
        )

        # Load DAG from server with warehouse location
        dag = DAG(str(self.dag1.name)).load_pending(location=warehouse_location)

        # Pass loaded DAG to app
        app = EntityResolutionApp(
            resolution=dag.final_step.resolution_path,
            num_samples=5,
            user="alice",
            dag=dag,
        )

        # Test the complete user workflow with Textual UI
        async with app.run_test() as pilot:
            await pilot.pause()

            # Authenticate the app
            await app.authenticate()

            # Phase 1: App should be authenticated and have samples
            assert app.user_name == "alice"
            assert app.user_id is not None

            # Let the app fetch samples as a user would experience
            if not app.queue.items:
                await app.load_samples()

            # Should now have samples to work with
            assert len(app.queue.items) > 0, "App should have loaded samples"

            # Phase 2: Simulate user painting clusters (as user would do)
            initial_items = list(app.queue.items)
            painted_count = 0

            for item in initial_items[:2]:  # Paint first 2 items like a user would
                # Paint each display column to different groups
                for i, _ in enumerate(item.display_columns):
                    group = "a" if i % 2 == 0 else "b"  # Alternate assignments
                    item.assignments[i] = group
                painted_count += 1

            # Verify we have painted items
            painted_items = [
                item
                for item in app.queue.items
                if len(item.assignments) == len(item.display_columns)
            ]
            assert (
                len(painted_items) >= 1
            ), "Should have painted items ready for submission"

            # Phase 3: Submit judgements to backend
            initial_judgements, _ = _handler.download_eval_data()
            initial_count = len(initial_judgements)

            # Submit painted items one at a time using the app's method
            for item in painted_items:
                if len(item.assignments) == len(item.display_columns):
                    await app.action_submit()

            # Verify judgements were submitted
            final_judgements, _ = _handler.download_eval_data()
            final_count = len(final_judgements)

            assert (
                final_count > initial_count
            ), "Should have more judgements after submission"

        # Phase 4: Test evaluation infrastructure with submitted judgements
        final_judgements, expansion = _handler.download_eval_data()
        assert len(final_judgements) > 0, "Should have judgements to evaluate with"
        assert len(expansion) > 0, "Should have cluster expansion data"

        # Phase 5: Compare the two deduper models
        comparison = _handler.compare_models(
            [
                dag.final_step.resolution_path,
                self.dag2.final_step.resolution_path,
            ]
        )
        expected_keys = {
            str(dag.final_step.resolution_path),
            str(self.dag2.final_step.resolution_path),
        }
        assert expected_keys.issubset(
            comparison.keys()
        ), "Comparison should include both models"
        for key in expected_keys:
            assert len(comparison[key]) == 2, "Each model should have precision/recall"
