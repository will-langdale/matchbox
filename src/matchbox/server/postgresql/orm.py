from sqlalchemy import (
    FLOAT,
    INTEGER,
    VARCHAR,
    CheckConstraint,
    Column,
    ForeignKey,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA
from sqlalchemy.orm import Session, relationship

from matchbox.common.graph import ResolutionNodeType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ResolutionFrom(CountMixin, MBDB.MatchboxBase):
    """Resolution lineage closure table with cached truth values."""

    __tablename__ = "resolution_from"

    # Columns
    parent = Column(
        BYTEA, ForeignKey("resolutions.hash", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BYTEA, ForeignKey("resolutions.hash", ondelete="CASCADE"), primary_key=True
    )
    level = Column(INTEGER, nullable=False)
    truth_cache = Column(FLOAT, nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint("parent != child", name="no_self_reference"),
        CheckConstraint("level > 0", name="positive_level"),
    )


class Resolutions(CountMixin, MBDB.MatchboxBase):
    """Table of resolution points: models, datasets and humans.

    Resolutions produce probabilities or own data in the clusters table.
    """

    __tablename__ = "resolutions"

    # Columns
    hash = Column(BYTEA, primary_key=True)
    type = Column(VARCHAR, nullable=False)
    name = Column(VARCHAR, nullable=False, unique=True)
    description = Column(VARCHAR)
    truth = Column(FLOAT)

    # Relationships
    source = relationship("Sources", back_populates="dataset_resolution", uselist=False)
    probabilities = relationship(
        "Probabilities", back_populates="proposed_by", cascade="all, delete-orphan"
    )
    children = relationship(
        "Resolutions",
        secondary=ResolutionFrom.__table__,
        primaryjoin="Resolutions.hash == ResolutionFrom.parent",
        secondaryjoin="Resolutions.hash == ResolutionFrom.child",
        backref="parents",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "type IN ('model', 'dataset', 'human')",
            name="resolution_type_constraints",
        ),
    )

    @property
    def ancestors(self) -> set["Resolutions"]:
        """Returns all ancestors (parents, grandparents, etc.) of this resolution."""
        with Session(MBDB.get_engine()) as session:
            ancestor_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(ResolutionFrom, Resolutions.hash == ResolutionFrom.parent)
                .where(ResolutionFrom.child == self.hash)
            )
            return set(session.execute(ancestor_query).scalars().all())

    @property
    def descendants(self) -> set["Resolutions"]:
        """Returns descendants (children, grandchildren, etc.) of this resolution."""
        with Session(MBDB.get_engine()) as session:
            descendant_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(ResolutionFrom, Resolutions.hash == ResolutionFrom.child)
                .where(ResolutionFrom.parent == self.hash)
            )
            return set(session.execute(descendant_query).scalars().all())

    def get_lineage_to_dataset(
        self, dataset: "Resolutions"
    ) -> tuple[bytes, dict[bytes, float]]:
        """Returns the resolution lineage and cached truth values to a dataset."""
        if dataset.type != ResolutionNodeType.DATASET.value:
            raise ValueError(
                f"Target resolution must be of type 'dataset', got {dataset.type}"
            )

        if self.hash == dataset.hash:
            return {}

        with Session(MBDB.get_engine()) as session:
            path_query = (
                select(
                    ResolutionFrom.parent, ResolutionFrom.truth_cache, Resolutions.type
                )
                .join(Resolutions, Resolutions.hash == ResolutionFrom.parent)
                .where(ResolutionFrom.child == self.hash)
                .order_by(ResolutionFrom.level.desc())
            )

            results = session.execute(path_query).all()

            if not any(parent == dataset.hash for parent, _, _ in results):
                raise ValueError(
                    f"No path between resolution {self.name}, dataset {dataset.name}"
                )

            lineage = {
                parent: truth
                for parent, truth, type in results
                if type != ResolutionNodeType.DATASET.value
            }

            lineage[self.hash] = self.truth

            return lineage


class Sources(CountMixin, MBDB.MatchboxBase):
    """Table of sources of data for Matchbox."""

    __tablename__ = "sources"

    # Columns
    resolution = Column(
        BYTEA, ForeignKey("resolutions.hash", ondelete="CASCADE"), primary_key=True
    )
    schema = Column(VARCHAR, nullable=False)
    table = Column(VARCHAR, nullable=False)
    id = Column(VARCHAR, nullable=False)

    # Relationships
    dataset_resolution = relationship("Resolutions", back_populates="source")
    clusters = relationship("Clusters", back_populates="source")

    @classmethod
    def list(cls) -> list["Sources"]:
        with Session(MBDB.get_engine()) as session:
            return session.query(cls).all()


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    parent = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
    )

    # Constraints
    __table_args__ = (CheckConstraint("parent != child", name="no_self_containment"),)


class Clusters(CountMixin, MBDB.MatchboxBase):
    """Table of indexed data and clusters that match it."""

    __tablename__ = "clusters"

    # Columns
    hash = Column(BYTEA, primary_key=True)
    dataset = Column(BYTEA, ForeignKey("sources.resolution"), nullable=True)
    # Uses array as source data may have identical rows. We can't control this
    # Must be indexed or PostgreSQL incorrectly tries to use nested joins
    # when retrieving small datasets in query() -- extremely slow
    id = Column(ARRAY(VARCHAR(36)), index=True, nullable=True)

    # Relationships
    source = relationship("Sources", back_populates="clusters")
    probabilities = relationship(
        "Probabilities", back_populates="proposes", cascade="all, delete-orphan"
    )
    children = relationship(
        "Clusters",
        secondary=Contains.__table__,
        primaryjoin="Clusters.hash == Contains.parent",
        secondaryjoin="Clusters.hash == Contains.child",
        backref="parents",
    )


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    __tablename__ = "probabilities"

    # Columns
    resolution = Column(
        BYTEA, ForeignKey("resolutions.hash", ondelete="CASCADE"), primary_key=True
    )
    cluster = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
    )
    probability = Column(FLOAT, nullable=False)

    # Relationships
    proposed_by = relationship("Resolutions", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 1", name="valid_probability"),
    )
