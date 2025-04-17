"""Tests migrations for the Matchbox PostgreSQL backend."""

import pytest
from alembic.command import downgrade, upgrade
from alembic.script import ScriptDirectory

from matchbox.common.logging import logger
from matchbox.server.postgresql import MatchboxPostgresSettings


def test_migrations_stairway(matchbox_postgres_settings: MatchboxPostgresSettings):
    """Tests that all migrations can be applied and then rolled back in sequence.

    This test runs migrations in their natural order and shows which one fails.
    """
    # Set up revisions in reverse order
    alembic_config = matchbox_postgres_settings.postgres.get_alembic_config()
    revisions_dir = ScriptDirectory.from_config(alembic_config)
    revisions = list(revisions_dir.walk_revisions("base", "heads"))
    revisions.reverse()

    # Reset database to base state first
    downgrade(alembic_config, "base")

    for i, revision in enumerate(revisions):
        revision_id = revision.revision
        down_revision = revision.down_revision or "-1"

        logger.debug(f"Testing migration {i + 1}/{len(revisions)}: {revision_id}")

        try:
            # Apply the migration
            upgrade(alembic_config, revision_id)

            # Roll it back
            downgrade(alembic_config, down_revision)

            # Re-apply it to ensure it works in both directions
            upgrade(alembic_config, revision_id)

            print(f"✓ Migration {revision_id} passed")
        except Exception as e:
            pytest.fail(
                f"Migration {i + 1}/{len(revisions)} '{revision_id}' failed: {str(e)}"
            )
