#!/usr/bin/env python3
"""Script to run Textual eval app with scenario data."""

import sys
from contextlib import contextmanager

from test.fixtures.db import SCENARIO_REGISTRY, setup_scenario

from matchbox.client.cli.eval.ui import EntityResolutionApp
from matchbox.common.graph import DEFAULT_RESOLUTION


@contextmanager
def scenario_app(scenario_name: str):
    """Context manager that sets up scenario, runs app, and cleans up."""
    if scenario_name not in SCENARIO_REGISTRY:
        available = ", ".join(SCENARIO_REGISTRY.keys())
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {available}")
        sys.exit(1)

    # Import required fixtures
    import tempfile
    from pathlib import Path

    from sqlalchemy import create_engine

    from matchbox.server.postgresql import MatchboxPostgres

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        warehouse_engine = create_engine(f"sqlite:///{tmp_path}")

        # Set up postgres backend (requires Docker)
        from test.fixtures.db import DevelopmentSettings

        dev_settings = DevelopmentSettings()

        from matchbox.server.base import MatchboxDatastoreSettings

        datastore_settings = MatchboxDatastoreSettings(
            host="localhost",
            port=dev_settings.datastore_port,
            access_key_id="access_key_id",
            secret_access_key="secret_access_key",
            default_region="eu-west-2",
            cache_bucket_name="cache",
        )

        from matchbox.server.postgresql import MatchboxPostgresSettings

        postgres_settings = MatchboxPostgresSettings(
            batch_size=250_000,
            postgres={
                "host": "localhost",
                "port": dev_settings.postgres_backend_port,
                "user": "matchbox_user",
                "password": "matchbox_password",
                "database": "matchbox",
                "db_schema": "mb",
                "alembic_config": "src/matchbox/server/postgresql/alembic.ini",
            },
            datastore=datastore_settings,
        )

        backend = MatchboxPostgres(settings=postgres_settings)
        backend.clear(certain=True)

        print(f"Setting up {scenario_name} scenario...")

        with setup_scenario(
            backend=backend,
            scenario_type=scenario_name,
            warehouse=warehouse_engine,
            n_entities=10,
            seed=42,
        ) as dag:
            print("Scenario ready! Starting Textual eval app...")

            # Create app with scenario resolution
            if scenario_name in ["bare", "index"]:
                # These scenarios don't have models, use source resolution
                resolution = (
                    list(dag.sources.keys())[0] if dag.sources else DEFAULT_RESOLUTION
                )
            else:
                # Use first model resolution
                resolution = (
                    list(dag.models.keys())[0] if dag.models else DEFAULT_RESOLUTION
                )

            app = EntityResolutionApp(resolution=resolution, num_samples=20)

            yield app

    finally:
        # Clean up
        try:
            backend.clear(certain=True)
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    """Run the scenario-based eval app."""
    if len(sys.argv) != 2:
        available = ", ".join(SCENARIO_REGISTRY.keys())
        print(f"Usage: {sys.argv[0]} <scenario>")
        print(f"Available scenarios: {available}")
        sys.exit(1)

    scenario_name = sys.argv[1]

    try:
        with scenario_app(scenario_name) as app:
            app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
