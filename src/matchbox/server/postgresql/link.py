from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import BOOLEAN, NUMERIC, VARCHAR, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Mapped, mapped_column, relationship

from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin, SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from matchbox.server.postgresql.models import Models


class Links(SHA1Mixin, CountMixin, MBDB.MatchboxBase):
    __tablename__ = "mb__links"
    __table_args__ = (UniqueConstraint("left", "right"),)

    left: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("mb__clusters.sha1"))
    right: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("mb__clusters.sha1"))

    validation: Mapped[List["LinkValidation"]] = relationship()
    proposers: Mapped[List["LinkProbabilities"]] = relationship(back_populates="links")


class LinkProbabilities(CountMixin, MBDB.MatchboxBase):
    """
    The associationn object betweenn Models and Links
    """

    __tablename__ = "mb__link_probabilities"

    link: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__links.sha1"), primary_key=True
    )
    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    model: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[float] = mapped_column(NUMERIC(6, 5))

    links: Mapped["Links"] = relationship(
        back_populates="proposers", cascade="save-update, merge"
    )
    proposed_by: Mapped["Models"] = relationship(back_populates="proposes_links")


class LinkContains(MBDB.MatchboxBase):
    __tablename__ = "mb__link_contains"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    parent: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__clusters.sha1"), primary_key=True
    )
    child: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__clusters.sha1"), primary_key=True
    )


class LinkValidation(UUIDMixin, MBDB.MatchboxBase):
    __tablename__ = "mb__link_validation"
    __table_args__ = (UniqueConstraint("link", "user"),)

    link: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("mb__links.sha1"))
    user: Mapped[str] = mapped_column(VARCHAR(100))
    valid: Mapped[bool] = mapped_column(BOOLEAN)
