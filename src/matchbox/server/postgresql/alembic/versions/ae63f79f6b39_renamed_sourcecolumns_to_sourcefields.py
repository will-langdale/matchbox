"""Renamed SourceColumns to SourceFields.

Revision ID: ae63f79f6b39
Revises: e4122bdf9b0d
Create Date: 2025-05-14 12:08:49.377998

"""

from typing import Sequence

from alembic import op

revision: str = "ae63f79f6b39"
down_revision: str | None = "e4122bdf9b0d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename SourceColumns table and columns to SourceFields."""
    # Drop the old index first
    op.drop_index(
        "ix_source_columns_source_config_id", table_name="source_columns", schema="mb"
    )

    # Drop the foreign key constraint before renaming to control the name
    # PostgreSQL's automatic control was causing issues
    op.drop_constraint(
        "source_columns_source_id_fkey",
        table_name="source_columns",
        type_="foreignkey",
        schema="mb",
    )

    # Rename the table
    op.rename_table(
        old_table_name="source_columns", new_table_name="source_fields", schema="mb"
    )

    # Rename columns to match new naming convention
    op.alter_column(
        table_name="source_fields",
        column_name="column_id",
        new_column_name="field_id",
        schema="mb",
    )

    op.alter_column(
        table_name="source_fields",
        column_name="column_index",
        new_column_name="index",
        schema="mb",
    )

    op.alter_column(
        table_name="source_fields",
        column_name="column_name",
        new_column_name="name",
        schema="mb",
    )

    op.alter_column(
        table_name="source_fields",
        column_name="column_type",
        new_column_name="type",
        schema="mb",
    )

    # Update the unique constraint name
    op.drop_constraint(
        "unique_column_index", table_name="source_fields", type_="unique", schema="mb"
    )

    op.create_unique_constraint(
        "unique_index",
        table_name="source_fields",
        columns=["source_config_id", "index"],
        schema="mb",
    )

    # Recreate the foreign key with the new table/column but keep a name that will
    # revert to the original when we downgrade
    op.create_foreign_key(
        "source_fields_source_config_id_fkey",
        source_table="source_fields",
        referent_table="source_configs",
        local_cols=["source_config_id"],
        remote_cols=["source_config_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )

    # Create the index with the same name (it will point to the new table)
    op.create_index(
        "ix_source_columns_source_config_id",
        "source_fields",
        ["source_config_id"],
        unique=False,
        schema="mb",
    )


def downgrade() -> None:
    """Revert SourceFields table and columns back to SourceColumns."""
    # Drop the index
    op.drop_index(
        "ix_source_columns_source_config_id", table_name="source_fields", schema="mb"
    )

    # Drop the foreign key constraint
    op.drop_constraint(
        "source_fields_source_config_id_fkey",
        table_name="source_fields",
        type_="foreignkey",
        schema="mb",
    )

    # Drop unique constraint while we rename columns
    op.drop_constraint(
        "unique_index", table_name="source_fields", type_="unique", schema="mb"
    )

    # Rename columns back to original names
    op.alter_column(
        table_name="source_fields",
        column_name="field_id",
        new_column_name="column_id",
        schema="mb",
    )

    op.alter_column(
        table_name="source_fields",
        column_name="index",
        new_column_name="column_index",
        schema="mb",
    )

    op.alter_column(
        table_name="source_fields",
        column_name="name",
        new_column_name="column_name",
        schema="mb",
    )

    op.alter_column(
        table_name="source_fields",
        column_name="type",
        new_column_name="column_type",
        schema="mb",
    )

    # Rename the table back
    op.rename_table(
        old_table_name="source_fields", new_table_name="source_columns", schema="mb"
    )

    # Create the original index
    op.create_index(
        "ix_source_columns_source_config_id",
        "source_columns",
        ["source_config_id"],
        unique=False,
        schema="mb",
    )

    # Create the unique constraint with the original name
    op.create_unique_constraint(
        "unique_column_index",
        table_name="source_columns",
        columns=["source_config_id", "column_index"],
        schema="mb",
    )

    # Recreate the foreign key constraint
    op.create_foreign_key(
        "source_columns_source_id_fkey",
        source_table="source_columns",
        referent_table="source_configs",
        local_cols=["source_config_id"],
        remote_cols=["source_config_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
