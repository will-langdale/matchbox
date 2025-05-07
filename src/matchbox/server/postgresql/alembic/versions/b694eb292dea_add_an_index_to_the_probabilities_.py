"""Add an index to the probabilities.resolution column.

Revision ID: b694eb292dea
Revises: 1907c34cfa1f
Create Date: 2025-04-17 12:28:17.150346

"""

from typing import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b694eb292dea"
down_revision: str | None = "1907c34cfa1f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "ix_probabilities_resolution",
        "probabilities",
        ["resolution"],
        unique=False,
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_probabilities_resolution", table_name="probabilities", schema="mb"
    )
