from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional
from uuid import UUID as uuUUID

from sqlalchemy import UUID, VARCHAR, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship
from sqlalchemy.sql.selectable import Select

from matchbox.server.base import Cluster, Probability
from matchbox.server.postgresql.clusters import clusters_association
from matchbox.server.postgresql.db import MatchboxBase
from matchbox.server.postgresql.dedupe import DDupeProbabilities
from matchbox.server.postgresql.link import LinkProbabilities
from matchbox.server.postgresql.mixin import SHA1Mixin

if TYPE_CHECKING:
    from matchbox.server.postgresql import Clusters


class CombinedProbabilities:
    def __init__(self, dedupes, links):
        self._dedupes = dedupes
        self._links = links

    def count(self):
        return self._dedupes.count() + self._links.count()


class Models(SHA1Mixin, MatchboxBase):
    __tablename__ = "cmf__models"
    __table_args__ = (UniqueConstraint("name"),)

    name: Mapped[str] = mapped_column(VARCHAR(100))
    description: Mapped[str] = mapped_column(VARCHAR(1000))
    deduplicates: Mapped[Optional[uuUUID]] = mapped_column(
        UUID, ForeignKey("cmf__source_dataset.uuid")
    )

    # ORM Many to Many pattern
    # https://docs.sqlalchemy.org/en/20/orm/
    # basic_relationships.html#many-to-many
    creates: WriteOnlyMapped[List["Clusters"]] = relationship(
        secondary=clusters_association,
        back_populates="created_by",
        passive_deletes=True,
    )

    # Association object pattern
    # https://docs.sqlalchemy.org/en/20/orm
    # /basic_relationships.html#association-object
    proposes_dedupes: WriteOnlyMapped[List["DDupeProbabilities"]] = relationship(
        back_populates="proposed_by", passive_deletes=True
    )
    proposes_links: WriteOnlyMapped[List["LinkProbabilities"]] = relationship(
        back_populates="proposed_by", passive_deletes=True
    )

    # This approach taken from the SQLAlchemy examples
    # https://github.com/sqlalchemy/sqlalchemy/
    # blob/main/examples/graphs/directed_graph.py
    child_edges: Mapped[List["ModelsFrom"]] = relationship(
        back_populates="child_model",
        primaryjoin="Models.sha1 == ModelsFrom.child",
        cascade="all, delete",
        passive_deletes=True,
    )
    parent_edges: Mapped[List["ModelsFrom"]] = relationship(
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

    def parent_neighbours(self) -> List["Models"]:
        return [x.parent_model for x in self.child_edges]

    def child_neighbours(self) -> List["Models"]:
        return [x.child_model for x in self.parent_edges]

    def _count_mapped(self, attr: WriteOnlyMapped) -> Select:
        return attr.select().with_only_columns(func.count())

    def creates_count(self) -> Select:
        return self._count_mapped(self.creates)

    def dedupes_count(self) -> Select:
        return self._count_mapped(self.proposes_dedupes)

    def links_count(self) -> Select:
        return self._count_mapped(self.proposes_links)

    def insert_probabilities(self, probabilities: Iterable[Probability]) -> None:
        for prob in probabilities:
            if isinstance(prob, DDupeProbabilities):
                self.proposes_dedupes.add(prob)
            elif isinstance(prob, LinkProbabilities):
                self.proposes_links.add(prob)
        self.session.flush()

    def insert_clusters(self, clusters: Iterable[Cluster]) -> None:
        for cluster in clusters:
            self.creates.add(cluster)
        self.session.flush()


# From


class ModelsFrom(MatchboxBase):
    __tablename__ = "cmf__models_from"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    parent: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    child: Mapped[bytes] = mapped_column(
        BYTEA, ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
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
