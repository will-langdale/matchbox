"""Renaming sources to source config.

Revision ID: 95c0b5c23446
Revises: beba75a24962
Create Date: 2025-05-12 12:24:29.340571

"""

from typing import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "95c0b5c23446"
down_revision: str | None = "beba75a24962"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # First, drop the foreign key constraints that reference the sources table
    op.drop_constraint(
        "cluster_source_pks_source_id_fkey",
        "cluster_source_pks",
        schema="mb",
        type_="foreignkey",
    )
    op.drop_constraint(
        "source_columns_source_id_fkey",
        "source_columns",
        schema="mb",
        type_="foreignkey",
    )

    # Rename the sources table to source_configs
    op.rename_table("sources", "source_configs", schema="mb")

    # Recreate the foreign key constraints with explicit names
    op.create_foreign_key(
        "cluster_source_pks_source_id_fkey",
        "cluster_source_pks",
        "source_configs",
        ["source_id"],
        ["source_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "source_columns_source_id_fkey",
        "source_columns",
        "source_configs",
        ["source_id"],
        ["source_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop the foreign key constraints that reference the source_configs table
    op.drop_constraint(
        "source_columns_source_id_fkey",
        "source_columns",
        schema="mb",
        type_="foreignkey",
    )
    op.drop_constraint(
        "cluster_source_pks_source_id_fkey",
        "cluster_source_pks",
        schema="mb",
        type_="foreignkey",
    )

    # Rename the source_configs table back to sources
    op.rename_table("source_configs", "sources", schema="mb")

    # Recreate the original foreign key constraints with the same names
    op.create_foreign_key(
        "source_columns_source_id_fkey",
        "source_columns",
        "sources",
        ["source_id"],
        ["source_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "cluster_source_pks_source_id_fkey",
        "cluster_source_pks",
        "sources",
        ["source_id"],
        ["source_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
