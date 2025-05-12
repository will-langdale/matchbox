"""Renaming sources to source config and source_id to source_config_id.

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

    # Rename source_id column to source_config_id in all tables
    op.alter_column(
        "source_configs", "source_id", new_column_name="source_config_id", schema="mb"
    )
    op.alter_column(
        "cluster_source_pks",
        "source_id",
        new_column_name="source_config_id",
        schema="mb",
    )
    op.alter_column(
        "source_columns", "source_id", new_column_name="source_config_id", schema="mb"
    )

    # Update indexes to use the new column name
    op.drop_index(
        "ix_source_columns_source_id", table_name="source_columns", schema="mb"
    )
    op.create_index(
        "ix_source_columns_source_config_id",
        "source_columns",
        ["source_config_id"],
        unique=False,
        schema="mb",
    )

    # Update unique constraints to use new column name
    op.drop_constraint(
        "unique_pk_source", "cluster_source_pks", schema="mb", type_="unique"
    )
    op.create_unique_constraint(
        "unique_pk_source",
        "cluster_source_pks",
        ["pk_id", "source_config_id"],
        schema="mb",
    )

    op.drop_constraint(
        "unique_column_index", "source_columns", schema="mb", type_="unique"
    )
    op.create_unique_constraint(
        "unique_column_index",
        "source_columns",
        ["source_config_id", "column_index"],
        schema="mb",
    )

    # Recreate the foreign key constraints with the new column and table names
    op.create_foreign_key(
        "cluster_source_pks_source_id_fkey",
        "cluster_source_pks",
        "source_configs",
        ["source_config_id"],
        ["source_config_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "source_columns_source_id_fkey",
        "source_columns",
        "source_configs",
        ["source_config_id"],
        ["source_config_id"],
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

    # Drop constraints that use the new column names before renaming columns
    op.drop_constraint(
        "unique_column_index", "source_columns", schema="mb", type_="unique"
    )
    op.drop_constraint(
        "unique_pk_source", "cluster_source_pks", schema="mb", type_="unique"
    )
    op.drop_index(
        "ix_source_columns_source_config_id", table_name="source_columns", schema="mb"
    )

    # Rename source_config_id column back to source_id in all tables
    op.alter_column(
        "source_columns", "source_config_id", new_column_name="source_id", schema="mb"
    )
    op.alter_column(
        "cluster_source_pks",
        "source_config_id",
        new_column_name="source_id",
        schema="mb",
    )
    op.alter_column(
        "source_configs", "source_config_id", new_column_name="source_id", schema="mb"
    )

    # Now recreate constraints using the old column names
    op.create_unique_constraint(
        "unique_column_index",
        "source_columns",
        ["source_id", "column_index"],
        schema="mb",
    )
    op.create_unique_constraint(
        "unique_pk_source", "cluster_source_pks", ["pk_id", "source_id"], schema="mb"
    )
    op.create_index(
        "ix_source_columns_source_id",
        "source_columns",
        ["source_id"],
        unique=False,
        schema="mb",
    )

    # Rename the source_configs table back to sources
    op.rename_table("source_configs", "sources", schema="mb")

    # Recreate the original foreign key constraints with the original names
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
