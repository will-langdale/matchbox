from __future__ import annotations

from cmf.data.db import CMFBase
from cmf.data.mixin import UUIDMixin, SHA1Mixin

from sqlalchemy import ForeignKey, UniqueConstraint, Column, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from cmf.data import Models


# ORM Many to Many pattern -- models/clusters association table
# https://docs.sqlalchemy.org/en/20/orm/
# basic_relationships.html#many-to-many

clusters_association = Table(
    "models_create_clusters",
    CMFBase.metadata,
    Column("parent", ForeignKey("models.sha1"), primary_key=True),
    Column("child", ForeignKey("clusters.sha1"), primary_key=True),
)


class Clusters(SHA1Mixin, CMFBase):
    __tablename__ = "clusters"

    created_by: Mapped[List["Models"]] = relationship(
        secondary=clusters_association, back_populates="creates"
    )
    clusters_validation: Mapped[List["ClusterValidation"]] = relationship()


class ClusterValidation(UUIDMixin, CMFBase):
    __tablename__ = "cluster_validation"
    __table_args__ = (UniqueConstraint("cluster", "user"),)

    cluster: Mapped[bytes] = mapped_column(ForeignKey("clusters.sha1"))
    user: Mapped[str]
    valid: Mapped[bool]
