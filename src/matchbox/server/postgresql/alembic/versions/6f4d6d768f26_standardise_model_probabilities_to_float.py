"""Standardise model probabilities to float.

Revision ID: 6f4d6d768f26
Revises: e06333c2daaf
Create Date: 2026-03-17 09:48:39.456864

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6f4d6d768f26"
down_revision: str | None = "e06333c2daaf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(
        op.f("valid_probability"),
        "model_edges",
        schema="mb",
        type_="check",
    )
    op.alter_column(
        "model_edges",
        "probability",
        existing_type=sa.SMALLINT(),
        type_=sa.REAL(),
        existing_nullable=False,
        postgresql_using="probability::real / 100.0",
        schema="mb",
    )
    op.create_check_constraint(
        op.f("valid_probability"),
        "model_edges",
        "probability >= 0.0 AND probability <= 1.0",
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        op.f("valid_probability"),
        "model_edges",
        schema="mb",
        type_="check",
    )
    op.alter_column(
        "model_edges",
        "probability",
        existing_type=sa.REAL(),
        type_=sa.SMALLINT(),
        existing_nullable=False,
        postgresql_using="ROUND(probability * 100.0)::smallint",
        schema="mb",
    )
    op.create_check_constraint(
        op.f("valid_probability"),
        "model_edges",
        "probability BETWEEN 0 AND 100",
        schema="mb",
    )
