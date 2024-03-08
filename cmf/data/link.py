from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from cmf.data.models import Models


class Links(SHA1Mixin, CMFBase):
    __tablename__ = "cmf__links"
    __table_args__ = (UniqueConstraint("left", "right"),)

    left: Mapped[bytes] = mapped_column(ForeignKey("cmf__clusters.sha1"))
    right: Mapped[bytes] = mapped_column(ForeignKey("cmf__clusters.sha1"))

    validation: Mapped[List["LinkValidation"]] = relationship()
    proposers: Mapped[List["LinkProbabilities"]] = relationship(back_populates="links")


class LinkProbabilities(CMFBase):
    """
    The associationn object betweenn Models and Links
    """

    __tablename__ = "cmf__link_probabilities"

    link: Mapped[bytes] = mapped_column(ForeignKey("cmf__links.sha1"), primary_key=True)
    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    model: Mapped[bytes] = mapped_column(
        ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[float]

    links: Mapped["Links"] = relationship(
        back_populates="proposers", cascade="save-update, merge"
    )
    proposed_by: Mapped["Models"] = relationship(back_populates="proposes_links")


class LinkContains(UUIDMixin, CMFBase):
    __tablename__ = "cmf__link_contains"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    parent: Mapped[bytes] = mapped_column(ForeignKey("cmf__clusters.sha1"))
    child: Mapped[bytes] = mapped_column(ForeignKey("cmf__clusters.sha1"))


class LinkValidation(UUIDMixin, CMFBase):
    __tablename__ = "cmf__link_validation"
    __table_args__ = (UniqueConstraint("link", "user"),)

    link: Mapped[bytes] = mapped_column(ForeignKey("cmf__links.sha1"))
    user: Mapped[str]
    valid: Mapped[bool]
