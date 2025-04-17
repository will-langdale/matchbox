"""Create schema without tables.

Revision ID: 40a8e5ed48f2
Revises: This is the first migration
"""

from typing import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "40a8e5ed48f2"
# down_revision=None indicates that this is the first migration file
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA mb")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP SCHEMA IF EXISTS mb CASCADE")
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
    op.execute('DROP EXTENSION IF EXISTS "pgcrypto"')
