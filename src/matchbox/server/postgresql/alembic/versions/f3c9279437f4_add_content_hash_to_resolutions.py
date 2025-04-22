"""Add content hash to Resolutions.

Revision ID: f3c9279437f4
Revises: b694eb292dea
Create Date: 2025-04-22 08:36:37.054592

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f3c9279437f4"
down_revision: str | None = "b694eb292dea"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "resolutions",
        sa.Column("content_hash", postgresql.BYTEA(), nullable=True),
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("resolutions", "content_hash", schema="mb")
