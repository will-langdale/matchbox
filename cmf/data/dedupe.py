from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import BOOLEAN, NUMERIC, VARCHAR, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from cmf.data.models import Models


class Dedupes(SHA1Mixin, CMFBase):
    __tablename__ = "cmf__ddupes"
    __table_args__ = (UniqueConstraint("left", "right"),)

    left: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("cmf__source_data.sha1"))
    right: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("cmf__source_data.sha1"))

    validation: Mapped[List["DDupeValidation"]] = relationship()
    proposers: Mapped[List["DDupeProbabilities"]] = relationship(
        back_populates="dedupes"
    )


class DDupeProbabilities(CMFBase):
    """
    The associationn object betweenn Models and Dedupes
    """

    __tablename__ = "cmf__ddupe_probabilities"

    ddupe: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("cmf__ddupes.sha1"), primary_key=True
    )
    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    model: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[float] = mapped_column(NUMERIC(6, 5))

    dedupes: Mapped["Dedupes"] = relationship(
        back_populates="proposers", cascade="save-update, merge"
    )
    proposed_by: Mapped["Models"] = relationship(back_populates="proposes_dedupes")


class DDupeContains(CMFBase):
    __tablename__ = "cmf__ddupe_contains"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    parent: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("cmf__clusters.sha1"), primary_key=True
    )
    child: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("cmf__source_data.sha1"), primary_key=True
    )


class DDupeValidation(UUIDMixin, CMFBase):
    __tablename__ = "cmf__ddupe_validation"
    __table_args__ = (UniqueConstraint("ddupe", "user"),)

    ddupe: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("cmf__ddupes.sha1"))
    user: Mapped[str] = mapped_column(VARCHAR(100))
    valid: Mapped[bool] = mapped_column(BOOLEAN)
