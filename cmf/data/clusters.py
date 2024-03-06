from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import Column, ForeignKey, Table, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from cmf.data import Models


# ORM Many to Many pattern -- models/clusters association table
# https://docs.sqlalchemy.org/en/20/orm/
# basic_relationships.html#many-to-many

clusters_association = Table(
    "cmf__models_create_clusters",
    CMFBase.metadata,
    Column(
        "parent", ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
    ),
    Column("child", ForeignKey("cmf__clusters.sha1"), primary_key=True),
)


class Clusters(SHA1Mixin, CMFBase):
    __tablename__ = "cmf__clusters"

    created_by: Mapped[List["Models"]] = relationship(
        secondary=clusters_association, back_populates="creates"
    )
    clusters_validation: Mapped[List["ClusterValidation"]] = relationship()


class ClusterValidation(UUIDMixin, CMFBase):
    __tablename__ = "cmf__cluster_validation"
    __table_args__ = (UniqueConstraint("cluster", "user"),)

    cluster: Mapped[bytes] = mapped_column(ForeignKey("cmf__clusters.sha1"))
    user: Mapped[str]
    valid: Mapped[bool]
