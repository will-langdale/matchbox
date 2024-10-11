from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import BOOLEAN, VARCHAR, Column, ForeignKey, Table, UniqueConstraint
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Mapped, mapped_column, relationship

from matchbox.server.postgresql.db import MatchboxBase
from matchbox.server.postgresql.mixin import CountMixin, SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from matchbox.server.postgresql import Models


# ORM Many to Many pattern -- models/clusters association table
# https://docs.sqlalchemy.org/en/20/orm/
# basic_relationships.html#many-to-many

clusters_association = Table(
    "mb__models_create_clusters",
    MatchboxBase.metadata,
    Column(
        "parent",
        BYTEA,
        ForeignKey("mb__models.sha1", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("child", BYTEA, ForeignKey("mb__clusters.sha1"), primary_key=True),
)


class Clusters(SHA1Mixin, CountMixin, MatchboxBase):
    __tablename__ = "mb__clusters"

    created_by: Mapped[List["Models"]] = relationship(
        secondary=clusters_association, back_populates="creates"
    )
    clusters_validation: Mapped[List["ClusterValidation"]] = relationship()


class ClusterValidation(UUIDMixin, MatchboxBase):
    __tablename__ = "mb__cluster_validation"
    __table_args__ = (UniqueConstraint("cluster", "user"),)

    cluster: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("mb__clusters.sha1"))
    user: Mapped[str] = mapped_column(VARCHAR(100))
    valid: Mapped[bool] = mapped_column(BOOLEAN)
