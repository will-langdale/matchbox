from typing import List

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin


class Links(SHA1Mixin, CMFBase):
    __tablename__ = "links"
    __table_args__ = (UniqueConstraint("left", "right"),)

    left: Mapped[bytes] = mapped_column(ForeignKey("clusters.sha1"))
    right: Mapped[bytes] = mapped_column(ForeignKey("clusters.sha1"))

    validation: Mapped[List["LinkValidation"]] = relationship()


class LinkProbabilities(CMFBase):
    """
    The associationn object betweenn Models and Links
    """

    __tablename__ = "link_probabilities"

    link: Mapped[bytes] = mapped_column(ForeignKey("links.sha1"), primary_key=True)
    model: Mapped[bytes] = mapped_column(
        ForeignKey("models.sha1", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[float]

    comparison: Mapped["Links"] = relationship(
        backref="proposers", cascade="save-update, merge"
    )


class LinkContains(UUIDMixin, CMFBase):
    __tablename__ = "link_contains"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    parent: Mapped[bytes] = mapped_column(ForeignKey("clusters.sha1"))
    child: Mapped[bytes] = mapped_column(ForeignKey("clusters.sha1"))


class LinkValidation(UUIDMixin, CMFBase):
    __tablename__ = "link_validation"
    __table_args__ = (UniqueConstraint("link", "user"),)

    link: Mapped[bytes] = mapped_column(ForeignKey("links.sha1"))
    user: Mapped[str]
    valid: Mapped[bool]
