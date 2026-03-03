"""Split model edges/resolver configs and drop truth cache.

Revision ID: e06333c2daaf
Revises: ecb39d1cc5c2
Create Date: 2026-02-19 11:18:35.699303

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e06333c2daaf"
down_revision: str | None = "ecb39d1cc5c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(
        "resolution_type_constraints",
        "resolutions",
        schema="mb",
        type_="check",
    )
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source', 'resolver')",
        schema="mb",
    )

    # Keep existing model edge rows by renaming the results table
    op.rename_table("results", "model_edges", schema="mb")
    op.drop_index(op.f("ix_results_resolution"), table_name="model_edges", schema="mb")
    op.create_index(
        "ix_model_edges_resolution",
        "model_edges",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )

    # Rename probabilities to an association table and drop probability payload.
    op.rename_table("probabilities", "resolution_clusters", schema="mb")
    op.drop_index(
        op.f("ix_probabilities_resolution"),
        table_name="resolution_clusters",
        schema="mb",
    )
    op.create_index(
        "ix_resolution_clusters_resolution",
        "resolution_clusters",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )
    op.drop_constraint(
        op.f("valid_probability"),
        "resolution_clusters",
        schema="mb",
        type_="check",
    )
    op.drop_column("resolution_clusters", "probability", schema="mb")

    # Create resolver_configs directly with final column names and types.
    op.create_table(
        "resolver_configs",
        sa.Column(
            "resolver_config_id",
            sa.BIGINT(),
            sa.Identity(always=False, start=1),
            nullable=False,
        ),
        sa.Column("resolution_id", sa.BIGINT(), nullable=False),
        sa.Column("resolver_class", sa.TEXT(), nullable=False),
        sa.Column(
            "resolver_settings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["resolution_id"], ["mb.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("resolver_config_id"),
        sa.UniqueConstraint("resolution_id", name="resolver_configs_resolution_key"),
        schema="mb",
    )

    # For each existing model resolution, create a paired resolver resolution
    # Models keep their model_edges but lose their resolution_clusters, which move
    # to the new resolver
    # The resolver builds its Components settings from the threshold stored in
    # resolutions.truth (read here before the column is dropped)
    #
    # upload_stage is supplied explicitly because the READY default is ORM-level only
    # and will not fire on a raw SQL INSERT
    #
    # Fingerprint is set to 32 zero bytes -- a sentinel that will never match a real
    # upload hash, so all clients will be forced to re-upload and replace it. 32 bytes
    # matches the SHA-256 output of matchbox.common.hash.HASH_FUNC
    op.execute(
        """
        INSERT INTO mb.resolutions (
            name, 
            type, 
            description, 
            fingerprint, 
            run_id, 
            upload_stage
        )
        SELECT
            r.name || '_resolver',
            'resolver',
            'Resolver for ' || r.name,
            decode(repeat('00', 32), 'hex'),
            r.run_id,
            'READY'
        FROM mb.resolutions r
        WHERE r.type = 'model'
        """
    )

    op.execute(
        """
        INSERT INTO mb.resolver_configs (
            resolution_id, 
            resolver_class, 
            resolver_settings
        )
        SELECT
            resolver.resolution_id,
            'Components',
            jsonb_build_object(
                'thresholds', jsonb_build_object(model.name, model.truth)
            )
        FROM mb.resolutions model
        JOIN mb.resolutions resolver
            ON resolver.name = model.name || '_resolver'
            AND resolver.run_id = model.run_id
        WHERE model.type = 'model'
        """
    )

    # Move cluster associations from the model to its paired resolver
    op.execute(
        """
        UPDATE mb.resolution_clusters rc
        SET resolution_id = resolver.resolution_id
        FROM mb.resolutions model
        JOIN mb.resolutions resolver
            ON resolver.name = model.name || '_resolver'
            AND resolver.run_id = model.run_id
        WHERE rc.resolution_id = model.resolution_id
            AND model.type = 'model'
        """
    )

    # Add the resolver into the closure table
    # First the direct model -> resolver edge, then all transitive ancestors so
    # the closure table remains complete
    op.execute(
        """
        INSERT INTO mb.resolution_from (parent, child, level)
        SELECT
            model.resolution_id,
            resolver.resolution_id,
            1
        FROM mb.resolutions model
        JOIN mb.resolutions resolver
            ON resolver.name = model.name || '_resolver'
            AND resolver.run_id = model.run_id
        WHERE model.type = 'model'
        """
    )
    op.execute(
        """
        INSERT INTO mb.resolution_from (parent, child, level)
        SELECT
            rf.parent,
            resolver.resolution_id,
            rf.level + 1
        FROM mb.resolution_from rf
        JOIN mb.resolutions model ON model.resolution_id = rf.child
        JOIN mb.resolutions resolver
            ON resolver.name = model.name || '_resolver'
            AND resolver.run_id = model.run_id
        WHERE model.type = 'model'
        """
    )

    op.drop_column("resolutions", "truth", schema="mb")
    op.drop_column("resolution_from", "truth_cache", schema="mb")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "resolution_from",
        sa.Column("truth_cache", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema="mb",
    )

    op.drop_constraint(
        "resolution_type_constraints",
        "resolutions",
        schema="mb",
        type_="check",
    )
    # Downgrading to a schema without resolver type, so remove resolver rows first.
    # Cascades clear resolver lineage, resolver_configs, and resolution_clusters
    # The paired model resolutions are left intact but will have no cluster
    # associations
    # This split is not reversible
    op.execute("DELETE FROM mb.resolutions WHERE type = 'resolver'")
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source')",
        schema="mb",
    )

    op.add_column(
        "resolutions",
        sa.Column("truth", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema="mb",
    )

    # resolver_configs is dropped entirely. No need to alter its columns
    op.drop_table("resolver_configs", schema="mb")

    op.drop_index(
        "ix_resolution_clusters_resolution",
        table_name="resolution_clusters",
        schema="mb",
    )
    op.add_column(
        "resolution_clusters",
        sa.Column("probability", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema="mb",
    )
    op.execute(
        "UPDATE mb.resolution_clusters SET probability = 100 WHERE probability IS NULL"
    )
    op.alter_column(
        "resolution_clusters",
        "probability",
        existing_type=sa.SMALLINT(),
        nullable=False,
        schema="mb",
    )
    op.create_check_constraint(
        op.f("valid_probability"),
        "resolution_clusters",
        "probability >= 0 AND probability <= 100",
        schema="mb",
    )
    op.rename_table("resolution_clusters", "probabilities", schema="mb")
    op.create_index(
        op.f("ix_probabilities_resolution"),
        "probabilities",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )

    op.drop_index("ix_model_edges_resolution", table_name="model_edges", schema="mb")
    op.rename_table("model_edges", "results", schema="mb")
    op.create_index(
        op.f("ix_results_resolution"),
        "results",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )
