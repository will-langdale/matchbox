import logging

import polars as pl
import pytest
from httpx import Client
from sqlalchemy import Engine, text

from matchbox import clean, index, make_model, query, select
from matchbox.client import _handler
from matchbox.client.helpers import delete_resolution
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.entities import query_to_cluster_entities
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EAnalyticalUser:
    """End to end tests for analytical user functionality."""

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
        # Store fixtures as class attributes for use in tests with self.*
        n_true_entities = 100

        self.__class__.client = matchbox_client
        self.__class__.warehouse_engine = postgres_warehouse
        self.__class__.n_true_entities = n_true_entities

        # Create feature configurations
        features = {
            "company_name": FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            "crn": FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
            "duns": FeatureConfig(
                name="duns",
                base_generator="numerify",
                parameters=(("text", "########"),),
            ),
            "cdms": FeatureConfig(
                name="cdms",
                base_generator="numerify",
                parameters=(("text", "ORG-########"),),
            ),
        }

        # Create source configurations that match our test fixtures
        source_parameters = (
            SourceTestkitParameters(
                name="crn",
                engine=postgres_warehouse,
                features=(
                    features["company_name"].add_variations(
                        SuffixRule(suffix=" Limited"),
                        SuffixRule(suffix=" UK"),
                        SuffixRule(suffix=" Company"),
                    ),
                    features["crn"],
                ),
                n_true_entities=n_true_entities,
                repetition=0,  # No duplicates within the variations
            ),
            SourceTestkitParameters(
                name="duns",
                engine=postgres_warehouse,
                features=(
                    features["company_name"],
                    features["duns"],
                ),
                n_true_entities=n_true_entities // 2,  # Half the companies
                repetition=0,
            ),
            SourceTestkitParameters(
                name="cdms",
                engine=postgres_warehouse,
                features=(
                    features["crn"],
                    features["cdms"],
                ),
                n_true_entities=n_true_entities,
                repetition=1,  # Duplicate all rows
            ),
        )

        # Create linked sources testkit with our configurations
        self.__class__.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,  # For reproducibility
        )

        # Setup code - Create tables in warehouse
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location(client=postgres_warehouse, set_client=True)

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})

        assert response.status_code == 200, "Failed to clear matchbox database"

        yield

        # Teardown code

        # Clean up database tables
        with postgres_warehouse.connect() as conn:
            # Drop all tables created by the test
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))

            conn.commit()

        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    def _get_cleaning_dict(
        self, source_prefix: str, columns: list[str]
    ) -> dict[str, str] | None:
        """Get cleaning dictionary for a source based on its columns.

        Mirrors the perturbations made to the company_name field in the
        linked_sources_factory testkit.
        """
        company_name_col = f"{source_prefix}company_name"

        if company_name_col in columns:
            return {
                company_name_col: rf"""
                    trim(
                        regexp_replace(
                            {company_name_col}, 
                            '\b(Limited|UK|Company)\b', 
                            '', 
                            'gi'
                        )
                    )
                """
            }

        return None

    def test_e2e_deduplication_and_linking_pipeline(self):
        """Runs an end to end test of the entire entity resolution pipeline.

        1. Create synthetic test data with linked_sources_factory
        2. Index the data in PostgreSQL
        3. Deduplicate each source, verify results
        4. Link pairs of deduplicated sources, verify results
        5. Link the linked pairs to create a full graph, verify results
        6. Verify the final linked data matches the original entity structure
        """
        # === SETUP PHASE ===
        # Get all sources and true entities for later verification
        all_true_entities = list(self.linked_testkit.true_entities)

        # Basic validation of our test data
        assert len(self.linked_testkit.sources) == 3, "Expected 3 sources"
        assert len(all_true_entities) == self.n_true_entities

        # Index all sources in the PostgreSQL database
        for source_testkit in self.linked_testkit.sources.values():
            source = source_testkit.source_config
            index(source_config=source)
            logging.debug(f"Indexed source: {source.name}")

        # === DEDUPLICATION PHASE ===
        deduper_names = {}

        for source_testkit in self.linked_testkit.sources.values():
            # Get source config
            source_config = source_testkit.source_config

            # Query data from the source
            # keys included then dropped to create ClusterEntity objects for later diff
            source_select = select(
                {
                    source_config.name: ["key"]
                    + [field.name for field in source_config.index_fields]
                },
                client=self.warehouse_engine,
            )
            raw_df = query(source_select, return_type="polars")
            clusters = query_to_cluster_entities(
                query=raw_df,
                keys={source_config.name: source_config.qualified_key},
            )
            df = raw_df.drop(source_config.qualified_key)

            # Apply cleaning
            cleaning_dict = self._get_cleaning_dict(source_config.prefix, df.columns)
            cleaned = clean(df, cleaning_dict)

            # Get feature names with prefix for deduplication
            feature_names = [
                f"{source_config.prefix}{feature.name}"
                for feature in source_testkit.features
            ]

            # Create and run a deduper model
            deduper_name = f"deduper_{source_config.name}"
            deduper = make_model(
                name=deduper_name,
                description=f"Deduplication of {source_config.name}",
                model_class=NaiveDeduper,
                model_settings={
                    "id": "id",
                    "unique_fields": feature_names,
                },
                left_data=cleaned,
                left_resolution=source_testkit.source_config.name,
            )

            # Run the deduper and store results
            results = deduper.run()

            # Verify deduplication results match expected true entities
            identical, report = self.linked_testkit.diff_results(
                probabilities=results.probabilities,
                left_clusters=clusters,
                right_clusters=None,
                sources=[source_config.name],
                threshold=0,
            )

            assert identical, f"Deduplication of {source_config.name} failed: {report}"

            results.to_matchbox()
            deduper.truth = 1.0

            logging.debug(f"Successfully deduplicated {source_config.name}")

            # Store the deduper resolution name for later use
            deduper_names[source_config.name] = deduper_name

        # === PAIRWISE LINKING PHASE ===
        linker_names = {}

        # Define linking pairs based on common fields
        linking_pairs = [
            (
                self.linked_testkit.sources["crn"],
                self.linked_testkit.sources["duns"],
                "company_name",
            ),  # CRN-DUNS via company_name
            (
                self.linked_testkit.sources["crn"],
                self.linked_testkit.sources["cdms"],
                "crn",
            ),  # CRN-CDMS via crn
        ]

        for left_testkit, right_testkit, common_field in linking_pairs:
            # Get sources
            left_source = left_testkit.source_config
            right_source = right_testkit.source_config

            # Query deduplicated data
            # keys included then dropped to create ClusterEntity objects for later diff
            left_raw_df = query(
                select(
                    {left_source.name: ["key", common_field]},
                    client=self.warehouse_engine,
                ),
                resolution=deduper_names[left_source.name],
                return_type="polars",
            )
            left_clusters = query_to_cluster_entities(
                query=left_raw_df,
                keys={left_source.name: left_source.qualified_key},
            )
            left_df = left_raw_df.drop(left_source.qualified_key)

            right_raw_df = query(
                select(
                    {right_source.name: ["key", common_field]},
                    client=self.warehouse_engine,
                ),
                resolution=deduper_names[right_source.name],
                return_type="polars",
            )
            right_clusters = query_to_cluster_entities(
                query=right_raw_df,
                keys={right_source.name: right_source.qualified_key},
            )
            right_df = right_raw_df.drop(right_source.qualified_key)

            # Apply cleaning
            left_cleaning_dict = self._get_cleaning_dict(
                left_source.prefix, left_df.columns
            )
            left_cleaned = clean(left_df, left_cleaning_dict)

            right_cleaning_dict = self._get_cleaning_dict(
                right_source.prefix, right_df.columns
            )
            right_cleaned = clean(right_df, right_cleaning_dict)

            # Build comparison clause
            comparison_clause = (
                f"l.{left_source.prefix}{common_field} "
                f"= r.{right_source.prefix}{common_field}",
            )

            # Create and run linker model
            linker_name = f"linker_{left_source.name}_{right_source.name}"
            linker = make_model(
                name=linker_name,
                description=f"Linking {left_testkit.name} and {right_testkit.name}",
                model_class=DeterministicLinker,
                model_settings={
                    "left_id": "id",
                    "right_id": "id",
                    "comparisons": comparison_clause,
                },
                left_data=left_cleaned,
                left_resolution=deduper_names[left_source.name],
                right_data=right_cleaned,
                right_resolution=deduper_names[right_source.name],
            )

            # Run the linker and store results
            results = linker.run()

            # Verify linking results
            identical, report = self.linked_testkit.diff_results(
                probabilities=results.probabilities,
                left_clusters=left_clusters,
                right_clusters=right_clusters,
                sources=[left_source.name, right_source.name],
                threshold=0,
            )

            report_counts = {
                k: len(v) if isinstance(v, list) else v for k, v in report.items()
            }

            assert identical, (
                f"Linking of {left_testkit.name} to {right_testkit.name} failed: "
                f"{report_counts}"
            )

            results.to_matchbox()
            linker.truth = 1.0

            logging.debug(
                f"Successfully linked {left_testkit.name} and {right_testkit.name}"
            )

            # Store the linker resolution name for later use
            linker_names[(left_source.name, right_source.name)] = linker_name

        # === FINAL LINKING PHASE ===
        # Now link the first linked pair (crn-duns) with the third source (cdms)
        crn_source = self.linked_testkit.sources["crn"].source_config
        duns_source = self.linked_testkit.sources["duns"].source_config
        cdms_source = self.linked_testkit.sources["cdms"].source_config
        first_pair = (crn_source.name, duns_source.name)

        # Query data from the first linked pair and the third source
        # keys included then dropped to create ClusterEntity objects for later diff
        left_raw_df = query(
            select({crn_source.name: ["key", "crn"]}, client=self.warehouse_engine),
            select({duns_source.name: ["key"]}, client=self.warehouse_engine),
            resolution=linker_names[first_pair],
            return_type="polars",
        )
        left_clusters = query_to_cluster_entities(
            query=left_raw_df,
            keys={
                crn_source.name: crn_source.qualified_key,
                duns_source.name: duns_source.qualified_key,
            },
        )
        left_df = left_raw_df.drop(crn_source.qualified_key, duns_source.qualified_key)

        right_raw_df = query(
            select({cdms_source.name: ["key", "crn"]}, client=self.warehouse_engine),
            resolution=deduper_names[cdms_source.name],
            return_type="polars",
        )
        right_clusters = query_to_cluster_entities(
            query=right_raw_df, keys={cdms_source.name: cdms_source.qualified_key}
        )
        right_df = right_raw_df.drop(cdms_source.qualified_key)

        # Apply cleaning
        left_cleaning_dict = self._get_cleaning_dict(crn_source.prefix, left_df.columns)
        left_cleaned = clean(left_df, left_cleaning_dict)

        right_cleaning_dict = self._get_cleaning_dict(
            cdms_source.prefix, right_df.columns
        )
        right_cleaned = clean(right_df, right_cleaning_dict)

        # Create and run final linker with the common "crn" field
        final_linker_name = "__DEFAULT__"
        final_linker = make_model(
            name=final_linker_name,
            description="Final linking of all sources",
            model_class=DeterministicLinker,
            model_settings={
                "left_id": "id",
                "right_id": "id",
                "comparisons": [
                    f"l.{crn_source.prefix}crn = r.{cdms_source.prefix}crn"
                ],
            },
            left_data=left_cleaned,
            left_resolution=linker_names[first_pair],
            right_data=right_cleaned,
            right_resolution=deduper_names[cdms_source.name],
        )

        # Run the final linker and store results
        results = final_linker.run()

        # Verify final linking results
        identical, report = self.linked_testkit.diff_results(
            probabilities=results.probabilities,
            left_clusters=left_clusters,
            right_clusters=right_clusters,
            sources=[crn_source.name, duns_source.name, cdms_source.name],
            threshold=0,
        )

        report_counts = {
            k: len(v) if isinstance(v, list) else v for k, v in report.items()
        }

        assert identical, f"Final linking failed: {report_counts}"

        results.to_matchbox()
        final_linker.truth = 1.0

        logging.debug("Successfully linked all sources")

        # === FINAL VERIFICATION PHASE ===
        # Query the final linked data with specific fields
        crn_source = self.linked_testkit.sources["crn"].source_config
        duns_source = self.linked_testkit.sources["duns"].source_config
        cdms_source = self.linked_testkit.sources["cdms"].source_config

        # Get necessary field from each source
        final_df = query(
            select(
                {
                    crn_source.name: ["key", "company_name", "crn"],
                    duns_source.name: ["key", "company_name", "duns"],
                    cdms_source.name: ["key", "crn", "cdms"],
                },
                client=self.warehouse_engine,
            ),
            resolution=final_linker_name,
            return_type="polars",
        )

        final_clusters = query_to_cluster_entities(
            query=final_df,
            keys={
                crn_source.name: crn_source.qualified_key,
                duns_source.name: duns_source.qualified_key,
                cdms_source.name: cdms_source.qualified_key,
            },
        )

        # Verify the final data structure - number of unique entities
        assert (
            final_df["id"].n_unique()
            == len(self.linked_testkit.true_entities)
            == self.n_true_entities
        ), (
            f"Expected {len(self.linked_testkit.true_entities)} unique entities, "
            f"got {final_df['id'].n_unique()}"
        )

        # Verify the final cluster membership -- the golden check

        true_entities = {
            entity.to_cluster_entity(
                crn_source.name, duns_source.name, cdms_source.name
            )
            for entity in self.linked_testkit.true_entities
            if entity.to_cluster_entity(
                crn_source.name, duns_source.name, cdms_source.name
            )
            is not None
        }

        assert true_entities == set(final_clusters), "Final clusters do not match"

        # Delete some resolutions as if my experimental model wasn't good enough

        final_linker_name = "__DEFAULT__"
        crn_source_name = self.linked_testkit.sources["crn"].source_config.name

        counts = _handler.count_backend_items()
        source_config_count = counts["entities"]["sources"]
        model_count = counts["entities"]["models"]

        # Delete the final linker resolution
        delete_resolution(name=final_linker_name, certain=True)

        counts = _handler.count_backend_items()
        assert counts["entities"]["sources"] == source_config_count
        assert counts["entities"]["models"] == model_count - 1, (
            "Expected one less model after deleting the final linker"
        )

        # Delete a source resolution
        delete_resolution(name=crn_source_name, certain=True)

        counts = _handler.count_backend_items()
        assert counts["entities"]["sources"] == source_config_count - 1, (
            "Expected one less source after deleting crn source"
        )
        assert counts["entities"]["models"] == model_count - 4, (
            "Expected all CRN descendant models to be deleted"
        )

        logging.debug("E2E test completed successfully!")

    def test_query_with_probabilities_e2e(self):
        """Test that get_probabilities works end-to-end in analytical workflow."""
        # === SETUP PHASE ===
        # Index sources
        for source_testkit in self.linked_testkit.sources.values():
            source = source_testkit.source_config
            index(source_config=source)

        # Create a simple deduper for one source to get some probabilistic results
        crn_source = self.linked_testkit.sources["crn"]
        source_config = crn_source.source_config

        # Get and clean data
        source_select = select(
            {
                source_config.name: ["key"]
                + [field.name for field in source_config.index_fields]
            },
            client=self.warehouse_engine,
        )
        raw_df = query(source_select, return_type="polars")
        df = raw_df.drop(source_config.qualified_key)
        cleaning_dict = self._get_cleaning_dict(source_config.prefix, df.columns)
        cleaned = clean(df, cleaning_dict)

        # Create and run deduper
        deduper_name = "test_deduper"
        feature_names = [
            f"{source_config.prefix}{feature.name}" for feature in crn_source.features
        ]

        deduper = make_model(
            name=deduper_name,
            description="Test deduper for probability validation",
            model_class=NaiveDeduper,
            model_settings={
                "id": "id",
                "unique_fields": feature_names,
            },
            left_data=cleaned,
            left_resolution=source_config.name,
        )

        results = deduper.run()
        results.to_matchbox()
        deduper.truth = 1.0

        # === PROBABILITY TESTING ===

        # Test 1: Query source-only (should have all NULL probabilities)
        source_only_result = query(
            select(
                {source_config.name: ["company_name"]}, client=self.warehouse_engine
            ),
            get_probabilities=True,
            return_type="polars",
        )

        # Verify schema and NULL probabilities for source-only
        assert "probability" in source_only_result.columns, (
            "Probability column missing from source-only query"
        )
        assert source_only_result["probability"].dtype == pl.UInt8, (
            "Probability column should be UInt8"
        )
        assert source_only_result["probability"].null_count() == len(
            source_only_result
        ), "All source-only probabilities should be NULL"

        # Test 2: Query with deduper resolution (should have some non-NULL probs)
        deduped_result = query(
            select(
                {source_config.name: ["company_name"]}, client=self.warehouse_engine
            ),
            resolution=deduper_name,
            get_probabilities=True,
            return_type="polars",
        )

        # Verify schema and probability values for model results
        assert "probability" in deduped_result.columns, (
            "Probability column missing from deduped query"
        )
        assert deduped_result["probability"].dtype == pl.UInt8, (
            "Probability column should be UInt8"
        )

        # Should have some non-NULL probabilities (from merged clusters)
        non_null_probs = deduped_result.filter(pl.col("probability").is_not_null())[
            "probability"
        ]
        if len(non_null_probs) > 0:
            # Verify probability range (0-100)
            assert non_null_probs.min() >= 0, "Probabilities should be >= 0"
            assert non_null_probs.max() <= 100, "Probabilities should be <= 100"

        # Test 3: Query with both leaf_id and get_probabilities
        combined_result = query(
            select(
                {source_config.name: ["company_name"]}, client=self.warehouse_engine
            ),
            resolution=deduper_name,
            return_leaf_id=True,
            get_probabilities=True,
            return_type="polars",
        )

        # Verify both columns present
        assert "leaf_id" in combined_result.columns, "leaf_id column missing"
        assert "probability" in combined_result.columns, "Probability column missing"
        assert combined_result["probability"].dtype == pl.UInt8, (
            "Probability column should be UInt8"
        )

        logging.debug("E2E probability test completed successfully!")
