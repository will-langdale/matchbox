"""Make all ID columns non-nullable.

Revision ID: 8156dfc0f7d8
Revises: beba75a24962
Create Date: 2025-05-04 11:53:07.972534

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8156dfc0f7d8"
down_revision: str | None = "beba75a24962"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "pk_space",
        "next_cluster_id",
        existing_type=sa.BIGINT(),
        nullable=False,
        schema="mb",
    )
    op.alter_column(
        "pk_space",
        "next_cluster_source_pk_id",
        existing_type=sa.BIGINT(),
        nullable=False,
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
    op.alter_column(
        "sources",
        "resolution_id",
        existing_type=sa.BIGINT(),
        nullable=True,
        schema="mb",
    )
    op.alter_column(
        "pk_space",
        "next_cluster_source_pk_id",
        existing_type=sa.BIGINT(),
        nullable=True,
        schema="mb",
    )
    op.alter_column(
        "pk_space",
        "next_cluster_id",
        existing_type=sa.BIGINT(),
        nullable=True,
        schema="mb",
    )
