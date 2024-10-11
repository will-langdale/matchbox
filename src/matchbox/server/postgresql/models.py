from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional
from uuid import UUID as uuUUID

from sqlalchemy import UUID, VARCHAR, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from matchbox.server.base import Cluster, MatchboxModelAdapter, Probability
from matchbox.server.postgresql.clusters import clusters_association
from matchbox.server.postgresql.db import MatchboxBase
from matchbox.server.postgresql.dedupe import DDupeProbabilities
from matchbox.server.postgresql.link import LinkProbabilities
from matchbox.server.postgresql.mixin import SHA1Mixin
from matchbox.server.postgresql.utils.insert import (
    insert_deduplication_probabilities,
    insert_link_probabilities,
)

if TYPE_CHECKING:
    from matchbox.server.postgresql import Clusters


class CombinedProbabilities:
    def __init__(self, dedupes, links):
        self._dedupes = dedupes
        self._links = links

    def count(self):
        return self._dedupes.count() + self._links.count()


class Models(SHA1Mixin, MatchboxModelAdapter, MatchboxBase):
    __tablename__ = "mb__models"
    __table_args__ = (UniqueConstraint("name"),)

    name: Mapped[str] = mapped_column(VARCHAR(100))
    description: Mapped[str] = mapped_column(VARCHAR(1000))
    deduplicates: Mapped[Optional[uuUUID]] = mapped_column(
        UUID, ForeignKey("mb__source_dataset.uuid")
    )

    # ORM Many to Many pattern
    # https://docs.sqlalchemy.org/en/20/orm/
    # basic_relationships.html#many-to-many
    creates: WriteOnlyMapped[list["Clusters"]] = relationship(
        secondary=clusters_association,
        back_populates="created_by",
        passive_deletes=True,
    )

    # Association object pattern
    # https://docs.sqlalchemy.org/en/20/orm
    # /basic_relationships.html#association-object
    proposes_dedupes: WriteOnlyMapped[list["DDupeProbabilities"]] = relationship(
        back_populates="proposed_by", passive_deletes=True
    )
    proposes_links: WriteOnlyMapped[list["LinkProbabilities"]] = relationship(
        back_populates="proposed_by", passive_deletes=True
    )

    # This approach taken from the SQLAlchemy examples
    # https://github.com/sqlalchemy/sqlalchemy/
    # blob/main/examples/graphs/directed_graph.py
    child_edges: Mapped[list["ModelsFrom"]] = relationship(
        back_populates="child_model",
        primaryjoin="Models.sha1 == ModelsFrom.child",
        cascade="all, delete",
        passive_deletes=True,
    )
    parent_edges: Mapped[list["ModelsFrom"]] = relationship(
        back_populates="parent_model",
        primaryjoin="Models.sha1 == ModelsFrom.parent",
        cascade="all, delete",
        passive_deletes=True,
    )

    @property
    def clusters(self) -> Clusters:
        return self.creates

    @property
    def probabilities(self) -> CombinedProbabilities:
        return CombinedProbabilities(self.proposes_dedupes, self.proposes_links)

    def parent_neighbours(self) -> list["Models"]:
        return [x.parent_model for x in self.child_edges]

    def child_neighbours(self) -> list["Models"]:
        return [x.child_model for x in self.parent_edges]

    def insert_probabilities(
        self,
        probabilities: list[Probability],
        probability_type: Literal["deduplications", "links"],
        batch_size: int,
    ) -> None:
        if probability_type == "deduplications":
            insert_deduplication_probabilities(
                model=self.name,
                engine=self.get_session().get_bind(),
                probabilities=probabilities,
                batch_size=batch_size,
            )
        elif probability_type == "links":
            insert_link_probabilities(
                model=self.name,
                engine=self.get_session().get_bind(),
                probabilities=probabilities,
                batch_size=batch_size,
            )

    def insert_clusters(self, clusters: list[Cluster]) -> None:
        for cluster in clusters:
            self.creates.add(cluster)
        self.session.flush()


# From


class ModelsFrom(MatchboxBase):
    __tablename__ = "mb__models_from"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    parent: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    child: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("mb__models.sha1", ondelete="CASCADE"), primary_key=True
    )

    child_model = relationship(
        Models,
        primaryjoin="ModelsFrom.child == Models.sha1",
        back_populates="child_edges",
    )
    parent_model = relationship(
        Models,
        primaryjoin="ModelsFrom.parent == Models.sha1",
        back_populates="parent_edges",
    )

    def __init__(self, parent, child):
        self.parent_model = parent
        self.child_model = child
