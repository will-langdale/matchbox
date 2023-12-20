from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from cmf.data.models import Models


class Dedupes(SHA1Mixin, CMFBase):
    __tablename__ = "ddupes"
    __table_args__ = (UniqueConstraint("left", "right"),)

    left: Mapped[bytes] = mapped_column(ForeignKey("source_data.sha1"))
    right: Mapped[bytes] = mapped_column(ForeignKey("source_data.sha1"))

    validation: Mapped[List["DDupeValidation"]] = relationship()
    proposers: Mapped["DDupeProbabilities"] = relationship(back_populates="dedupes")


class DDupeProbabilities(CMFBase):
    """
    The associationn object betweenn Models and Dedupes
    """

    __tablename__ = "ddupe_probabilities"

    ddupe: Mapped[bytes] = mapped_column(ForeignKey("ddupes.sha1"), primary_key=True)
    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    model: Mapped[bytes] = mapped_column(
        ForeignKey("models.sha1", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[float]

    dedupes: Mapped["Dedupes"] = relationship(
        back_populates="proposers", cascade="save-update, merge"
    )
    proposed_by: Mapped["Models"] = relationship(back_populates="dedupe_associations")


class DDupeContains(UUIDMixin, CMFBase):
    __tablename__ = "ddupe_contains"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    parent: Mapped[bytes] = mapped_column(ForeignKey("clusters.sha1"))
    child: Mapped[bytes] = mapped_column(ForeignKey("source_data.sha1"))


class DDupeValidation(UUIDMixin, CMFBase):
    __tablename__ = "ddupe_validation"
    __table_args__ = (UniqueConstraint("ddupe", "user"),)

    ddupe: Mapped[bytes] = mapped_column(ForeignKey("ddupes.sha1"))
    user: Mapped[str]
    valid: Mapped[bool]
