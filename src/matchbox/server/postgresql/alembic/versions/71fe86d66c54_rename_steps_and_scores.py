"""Rename resolutions to steps and probabilities to scores.

Revision ID: 71fe86d66c54
Revises: 6f4d6d768f26
Create Date: 2026-03-17 16:05:59.953899

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "71fe86d66c54"
down_revision: str | None = "6f4d6d768f26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "mb"


def _rename_constraint(table_name: str, old_name: str, new_name: str) -> None:
    """Rename a table constraint."""
    op.execute(
        sa.text(
            f"ALTER TABLE {SCHEMA}.{table_name} "
            f"RENAME CONSTRAINT {old_name} TO {new_name}"
        )
    )


def _rename_index(old_name: str, new_name: str) -> None:
    """Rename an index in the Matchbox schema."""
    op.execute(sa.text(f"ALTER INDEX {SCHEMA}.{old_name} RENAME TO {new_name}"))


def _rename_sequence(old_name: str, new_name: str) -> None:
    """Rename a sequence in the Matchbox schema."""
    op.execute(sa.text(f"ALTER SEQUENCE {SCHEMA}.{old_name} RENAME TO {new_name}"))


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table("resolutions", "steps", schema=SCHEMA)
    op.rename_table("resolution_from", "step_from", schema=SCHEMA)
    op.rename_table("resolution_clusters", "resolver_clusters", schema=SCHEMA)

    op.alter_column(
        "steps",
        "resolution_id",
        new_column_name="step_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "source_configs",
        "resolution_id",
        new_column_name="step_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "model_configs",
        "resolution_id",
        new_column_name="step_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "resolver_configs",
        "resolution_id",
        new_column_name="step_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "model_edges",
        "resolution_id",
        new_column_name="step_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "resolver_clusters",
        "resolution_id",
        new_column_name="step_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "model_edges",
        "probability",
        new_column_name="score",
        schema=SCHEMA,
    )

    _rename_sequence("resolutions_resolution_id_seq", "steps_step_id_seq")

    _rename_constraint("steps", "resolutions_pkey", "steps_pkey")
    _rename_constraint("steps", "resolutions_name_key", "steps_name_key")
    _rename_constraint("steps", "resolutions_run_fkey", "steps_run_fkey")
    _rename_constraint("steps", "resolution_type_constraints", "step_type_constraints")

    _rename_constraint("step_from", "resolution_from_pkey", "step_from_pkey")
    _rename_constraint(
        "step_from",
        "resolution_from_parent_fkey",
        "step_from_parent_fkey",
    )
    _rename_constraint(
        "step_from",
        "resolution_from_child_fkey",
        "step_from_child_fkey",
    )

    _rename_constraint(
        "resolver_clusters",
        "probabilities_pkey",
        "resolver_clusters_pkey",
    )
    _rename_constraint(
        "resolver_clusters",
        "probabilities_resolution_id_fkey",
        "resolver_clusters_step_id_fkey",
    )
    _rename_constraint(
        "resolver_clusters",
        "probabilities_cluster_id_fkey",
        "resolver_clusters_cluster_id_fkey",
    )
    _rename_index(
        "ix_resolution_clusters_resolution",
        "ix_resolver_clusters_step",
    )

    _rename_constraint(
        "model_configs",
        "model_configs_resolution_id_fkey",
        "model_configs_step_id_fkey",
    )

    _rename_constraint(
        "resolver_configs",
        "resolver_configs_resolution_id_fkey",
        "resolver_configs_step_id_fkey",
    )
    _rename_constraint(
        "resolver_configs",
        "resolver_configs_resolution_key",
        "resolver_configs_step_key",
    )

    _rename_constraint(
        "source_configs",
        "sources_resolution_id_fkey",
        "source_configs_step_id_fkey",
    )

    _rename_constraint("model_edges", "results_pkey", "model_edges_pkey")
    _rename_constraint(
        "model_edges",
        "results_resolution_id_fkey",
        "model_edges_step_id_fkey",
    )
    _rename_constraint(
        "model_edges",
        "results_left_id_fkey",
        "model_edges_left_id_fkey",
    )
    _rename_constraint(
        "model_edges",
        "results_right_id_fkey",
        "model_edges_right_id_fkey",
    )
    _rename_constraint(
        "model_edges",
        "results_resolution_id_left_id_right_id_key",
        "model_edges_step_id_left_id_right_id_key",
    )
    _rename_constraint("model_edges", "valid_probability", "valid_score")
    _rename_index("ix_model_edges_resolution", "ix_model_edges_step")


def downgrade() -> None:
    """Downgrade schema."""
    _rename_index("ix_model_edges_step", "ix_model_edges_resolution")
    _rename_constraint("model_edges", "valid_score", "valid_probability")
    _rename_constraint(
        "model_edges",
        "model_edges_step_id_left_id_right_id_key",
        "results_resolution_id_left_id_right_id_key",
    )
    _rename_constraint(
        "model_edges",
        "model_edges_right_id_fkey",
        "results_right_id_fkey",
    )
    _rename_constraint(
        "model_edges",
        "model_edges_left_id_fkey",
        "results_left_id_fkey",
    )
    _rename_constraint(
        "model_edges",
        "model_edges_step_id_fkey",
        "results_resolution_id_fkey",
    )
    _rename_constraint("model_edges", "model_edges_pkey", "results_pkey")

    _rename_constraint(
        "source_configs",
        "source_configs_step_id_fkey",
        "sources_resolution_id_fkey",
    )

    _rename_constraint(
        "resolver_configs",
        "resolver_configs_step_key",
        "resolver_configs_resolution_key",
    )
    _rename_constraint(
        "resolver_configs",
        "resolver_configs_step_id_fkey",
        "resolver_configs_resolution_id_fkey",
    )

    _rename_constraint(
        "model_configs",
        "model_configs_step_id_fkey",
        "model_configs_resolution_id_fkey",
    )

    _rename_index(
        "ix_resolver_clusters_step",
        "ix_resolution_clusters_resolution",
    )
    _rename_constraint(
        "resolver_clusters",
        "resolver_clusters_cluster_id_fkey",
        "probabilities_cluster_id_fkey",
    )
    _rename_constraint(
        "resolver_clusters",
        "resolver_clusters_step_id_fkey",
        "probabilities_resolution_id_fkey",
    )
    _rename_constraint(
        "resolver_clusters",
        "resolver_clusters_pkey",
        "probabilities_pkey",
    )

    _rename_constraint(
        "step_from",
        "step_from_child_fkey",
        "resolution_from_child_fkey",
    )
    _rename_constraint(
        "step_from",
        "step_from_parent_fkey",
        "resolution_from_parent_fkey",
    )
    _rename_constraint("step_from", "step_from_pkey", "resolution_from_pkey")

    _rename_constraint("steps", "step_type_constraints", "resolution_type_constraints")
    _rename_constraint("steps", "steps_run_fkey", "resolutions_run_fkey")
    _rename_constraint("steps", "steps_name_key", "resolutions_name_key")
    _rename_constraint("steps", "steps_pkey", "resolutions_pkey")

    _rename_sequence("steps_step_id_seq", "resolutions_resolution_id_seq")

    op.alter_column(
        "model_edges",
        "score",
        new_column_name="probability",
        schema=SCHEMA,
    )
    op.alter_column(
        "resolver_clusters",
        "step_id",
        new_column_name="resolution_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "model_edges",
        "step_id",
        new_column_name="resolution_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "resolver_configs",
        "step_id",
        new_column_name="resolution_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "model_configs",
        "step_id",
        new_column_name="resolution_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "source_configs",
        "step_id",
        new_column_name="resolution_id",
        schema=SCHEMA,
    )
    op.alter_column(
        "steps",
        "step_id",
        new_column_name="resolution_id",
        schema=SCHEMA,
    )

    op.rename_table("resolver_clusters", "resolution_clusters", schema=SCHEMA)
    op.rename_table("step_from", "resolution_from", schema=SCHEMA)
    op.rename_table("steps", "resolutions", schema=SCHEMA)
