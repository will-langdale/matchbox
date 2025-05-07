"""Add PKSpace table.

Revision ID: beba75a24962
Revises: f3c9279437f4
Create Date: 2025-05-01 16:53:16.071565

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "beba75a24962"
down_revision: str | None = "f3c9279437f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "pk_space",
        sa.Column("id", sa.BIGINT(), nullable=False),
        sa.Column("next_cluster_id", sa.BIGINT(), nullable=False),
        sa.Column("next_cluster_source_pk_id", sa.BIGINT(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema="mb",
    )

    op.alter_column(
        "sources",
        "resolution_id",
        existing_type=sa.BIGINT(),
        nullable=False,
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("pk_space", schema="mb")
    op.alter_column(
        "sources",
        "resolution_id",
        existing_type=sa.BIGINT(),
        nullable=True,
        schema="mb",
    )
