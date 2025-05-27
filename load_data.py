"""Load data for pentest."""

import logging
from typing import TYPE_CHECKING, Any, Generator

from httpx import Client
from sqlalchemy import Engine, create_engine

from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings, settings
from matchbox.client.clean import steps
from matchbox.client.clean.utils import cleaning_function
from matchbox.client.dags import DAG, DedupeStep, IndexStep, LinkStep, StepInput
from matchbox.client.helpers import cleaner, cleaners
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.entities import SuffixRule
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    linked_sources_factory,
)
from matchbox.common.sources import RelationalDBLocation, SourceConfig

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def matchbox_client_settings() -> ClientSettings:
    """Client settings for the Matchbox API running in Docker."""
    return settings


def matchbox_client() -> Client:
    """Client for the Matchbox API running in Docker."""
    return create_client(settings=matchbox_client_settings())


def postgres_warehouse() -> Generator[Engine, None, None]:
    """Creates an engine for the test warehouse database."""
    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = 7654

    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )
    return engine


class PipelineBuilder:
    """End to end tests for DAG pipeline functionality."""

    client: Client | None = None
    warehouse_engine: Engine | None = None
    linked_testkit: LinkedSourcesTestkit | None = None
    n_true_entities: int | None = None

    def setup_environment(
        self,
    ):
        """Set up warehouse and database using fixtures."""
        # Store fixtures as class attributes
        n_true_entities = 10  # Keep it small for simplicity

        self.client = matchbox_client()
        self.warehouse_engine = postgres_warehouse()
        self.n_true_entities = n_true_entities

        # Create simple feature configurations - just two sources
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

        # Create two simple sources that can be linked
        source_parameters = (
            SourceTestkitParameters(
                name="source_a",
                engine=postgres_warehouse(),
                features=(
                    features["company_name"].add_variations(
                        SuffixRule(suffix=" Ltd"),
                        SuffixRule(suffix=" Limited"),
                    ),
                    features["registration_id"],
                ),
                n_true_entities=n_true_entities,
                repetition=0,  # No duplicates
            ),
            SourceTestkitParameters(
                name="source_b",
                engine=postgres_warehouse(),
                features=(
                    features["company_name"],
                    features["registration_id"],
                ),
                n_true_entities=n_true_entities // 2,  # Half overlap
                repetition=1,  # Duplicate all rows for deduplication testing
            ),
        )

        # Create linked sources testkit
        self.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,
        )

        # Setup - Create tables in warehouse
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location(
                credentials=postgres_warehouse(), set_credentials=True
            )

    def run_dag(self):
        """Test DAG API with simple two-source pipeline.

        Rerun to test overwriting.
        """
        # === SETUP PHASE ===
        dw_loc = RelationalDBLocation.from_engine(self.warehouse_engine)
        batch_size = 1000

        # Create source configs
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

        source_b_config = SourceConfig.new(
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
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        # Create simple cleaners
        clean_company_name = cleaning_function(
            steps.tokenise,
            lambda tokens: [t for t in tokens if t not in ["Ltd", "Limited"]],
            steps.list_join_to_string,
            steps.trim,
        )

        source_a_cleaners = cleaners(
            cleaner(
                clean_company_name,
                {"column": "source_a_company_name"},
            ),
        )

        source_b_cleaners = cleaners(
            cleaner(
                clean_company_name,
                {"column": "source_b_company_name"},
            ),
        )

        # === DAG DEFINITION ===
        # Index steps
        i_source_a = IndexStep(source_config=source_a_config, batch_size=batch_size)
        i_source_b = IndexStep(source_config=source_b_config, batch_size=batch_size)

        # Dedupe steps
        dedupe_a = DedupeStep(
            left=StepInput(
                prev_node=i_source_a,
                select={source_a_config: ["company_name", "registration_id"]},
                cleaners=source_a_cleaners,
                batch_size=batch_size,
            ),
            name="dedupe_source_a",
            description="Deduplicate source A",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": ["source_a_registration_id"],
            },
            truth=1.0,
        )

        dedupe_b = DedupeStep(
            left=StepInput(
                prev_node=i_source_b,
                select={source_b_config: ["company_name", "registration_id"]},
                cleaners=source_b_cleaners,
                batch_size=batch_size,
            ),
            name="dedupe_source_b",
            description="Deduplicate source B",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": ["source_b_registration_id"],
            },
            truth=1.0,
        )

        # Link step
        link_ab = LinkStep(
            left=StepInput(
                prev_node=dedupe_a,
                select={source_a_config: ["company_name", "registration_id"]},
                cleaners=source_a_cleaners,
                batch_size=batch_size,
            ),
            right=StepInput(
                prev_node=dedupe_b,
                select={source_b_config: ["company_name", "registration_id"]},
                cleaners=source_b_cleaners,
                batch_size=batch_size,
            ),
            name="__DEFAULT__",
            description="Link sources A and B on registration_id",
            model_class=DeterministicLinker,
            settings={
                "left_id": "id",
                "right_id": "id",
                "comparisons": (
                    "l.source_a_registration_id = r.source_b_registration_id"
                ),
            },
            truth=1.0,
        )

        # Create and run DAG
        dag = DAG()
        dag.add_steps(i_source_a, i_source_b)
        dag.add_steps(dedupe_a, dedupe_b)
        dag.add_steps(link_ab)

        logging.info("Running DAG")
        dag.run()
        logging.info("Finished running DAG")


if __name__ == "__main__":
    pb = PipelineBuilder()
    pb.setup_environment()
    pb.run_dag()
