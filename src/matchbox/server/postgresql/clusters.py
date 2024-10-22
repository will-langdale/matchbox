from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BOOLEAN, VARCHAR, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Mapped, mapped_column, relationship

from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin, SHA1Mixin, UUIDMixin

if TYPE_CHECKING:
    from matchbox.server.postgresql.models import Models


# ORM Many to Many pattern -- models/clusters association table
# https://docs.sqlalchemy.org/en/20/orm/
# basic_relationships.html#many-to-many

# clusters_association = Table(
#     "mb__models_create_clusters",
#     MBDB.MatchboxBase.metadata,
#     Column(
#         "parent",
#         BYTEA,
#         ForeignKey("mb__models.sha1", ondelete="CASCADE"),
#         primary_key=True,
#     ),
#     Column("child", BYTEA, ForeignKey("mb__clusters.sha1"), primary_key=True),
# )


# ORM Many to Many pattern -- models/clusters association object
# https://docs.sqlalchemy.org/en/20/orm/
# basic_relationships.html#association-object
class Creates(CountMixin, MBDB.MatchboxBase):
    __tablename__ = "mb__creates"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    parent: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    child: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__clusters.sha1"), primary_key=True
    )

    model: Mapped["Models"] = relationship(back_populates="creates")
    cluster: Mapped["Clusters"] = relationship(back_populates="created_by")


class Clusters(SHA1Mixin, CountMixin, MBDB.MatchboxBase):
    __tablename__ = "mb__clusters"

    created_by: Mapped[list["Creates"]] = relationship(back_populates="cluster")
    clusters_validation: Mapped[list["ClusterValidation"]] = relationship()


class ClusterValidation(UUIDMixin, MBDB.MatchboxBase):
    __tablename__ = "mb__cluster_validation"
    __table_args__ = (UniqueConstraint("cluster", "user"),)

    cluster: Mapped[bytes] = mapped_column(BYTEA, ForeignKey("mb__clusters.sha1"))
    user: Mapped[str] = mapped_column(VARCHAR(100))
    valid: Mapped[bool] = mapped_column(BOOLEAN)
