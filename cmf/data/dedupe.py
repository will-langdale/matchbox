from typing import List

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin


class Dedupes(SHA1Mixin, CMFBase):
    __tablename__ = "ddupes"
    __table_args__ = (UniqueConstraint("left", "right"),)

    left: Mapped[bytes] = mapped_column(ForeignKey("source_data.sha1"))
    right: Mapped[bytes] = mapped_column(ForeignKey("source_data.sha1"))

    validation: Mapped[List["DDupeValidation"]] = relationship()


class DDupeProbabilities(CMFBase):
    """
    The associationn object betweenn Models and Dedupes
    """

    __tablename__ = "ddupe_probabilities"

    ddupe: Mapped[bytes] = mapped_column(ForeignKey("ddupes.sha1"), primary_key=True)
    model: Mapped[bytes] = mapped_column(
        ForeignKey("models.sha1", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[float]

    comparison: Mapped["Dedupes"] = relationship(
        backref="proposers", cascade="save-update, merge"
    )


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
