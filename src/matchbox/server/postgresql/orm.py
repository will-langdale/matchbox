from sqlalchemy import (
    BIGINT,
    FLOAT,
    INTEGER,
    SMALLINT,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, TEXT
from sqlalchemy.orm import Session, relationship

from matchbox.common.graph import ResolutionNodeType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ResolutionFrom(CountMixin, MBDB.MatchboxBase):
    """Resolution lineage closure table with cached truth values."""

    def __init__(self, suffix=""):
        self.__tablename__ = f"resolution_from{suffix}"

        # Columns
        self.parent = Column(
            BIGINT,
            ForeignKey(f"resolutions{suffix}.resolution_id", ondelete="CASCADE"),
            primary_key=True,
        )
        self.child = Column(
            BIGINT,
            ForeignKey(f"resolutions{suffix}.resolution_id", ondelete="CASCADE"),
            primary_key=True,
        )
        self.level = Column(INTEGER, nullable=False)
        self.truth_cache = Column(FLOAT, nullable=True)

        # Constraints
        self.__table_args__ = (
            CheckConstraint("parent != child", name="no_self_reference"),
            CheckConstraint("level > 0", name="positive_level"),
        )


class Resolutions(CountMixin, MBDB.MatchboxBase):
    """Table of resolution points: models, datasets and humans.

    Resolutions produce probabilities or own data in the clusters table.
    """

    def __init__(self, suffix=""):
        self.__tablename__ = f"resolutions{suffix}"

        # Columns
        self.resolution_id = Column(BIGINT, primary_key=True)
        self.resolution_hash = Column(BYTEA, nullable=False)
        self.type = Column(TEXT, nullable=False)
        self.name = Column(TEXT, nullable=False)
        self.description = Column(TEXT)
        self.truth = Column(FLOAT)

        # Relationships
        self.source = relationship("Sources", back_populates="dataset_resolution", uselist=False)
        self.probabilities = relationship(
            "Probabilities", back_populates="proposed_by", cascade="all, delete-orphan"
        )
        self.children = relationship(
            "Resolutions",
            secondary=ResolutionFrom.__table__,
            primaryjoin="Resolutions.resolution_id == ResolutionFrom.parent",
            secondaryjoin="Resolutions.resolution_id == ResolutionFrom.child",
            backref="parents",
        )

        # Constraints
        self.__table_args__ = (
            CheckConstraint(
                "type IN ('model', 'dataset', 'human')",
                name="resolution_type_constraints",
            ),
            UniqueConstraint("resolution_hash", name="resolutions_hash_key"),
            UniqueConstraint("name", name="resolutions_name_key"),
        )

    @property
    def ancestors(self) -> set["Resolutions"]:
        """Returns all ancestors (parents, grandparents, etc.) of this resolution."""
        with Session(MBDB.get_engine()) as session:
            ancestor_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(
                    ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent
                )
                .where(ResolutionFrom.child == self.resolution_id)
            )
            return set(session.execute(ancestor_query).scalars().all())

    @property
    def descendants(self) -> set["Resolutions"]:
        """Returns descendants (children, grandchildren, etc.) of this resolution."""
        with Session(MBDB.get_engine()) as session:
            descendant_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.child)
                .where(ResolutionFrom.parent == self.resolution_id)
            )
            return set(session.execute(descendant_query).scalars().all())

    def get_lineage(self) -> dict[int, float]:
        """Returns all ancestors and their cached truth values from this model."""
        with Session(MBDB.get_engine()) as session:
            lineage_query = (
                select(ResolutionFrom.parent, ResolutionFrom.truth_cache)
                .where(ResolutionFrom.child == self.resolution_id)
                .order_by(ResolutionFrom.level.desc())
            )

            results = session.execute(lineage_query).all()

            lineage = {parent: truth for parent, truth in results}
            lineage[self.resolution_id] = self.truth

            return lineage

    def get_lineage_to_dataset(
        self, dataset: "Resolutions"
    ) -> tuple[bytes, dict[int, float]]:
        """Returns the resolution lineage and cached truth values to a dataset."""
        if dataset.type != ResolutionNodeType.DATASET.value:
            raise ValueError(
                f"Target resolution must be of type 'dataset', got {dataset.type}"
            )

        if self.resolution_id == dataset.resolution_id:
            return {dataset.resolution_id: None}

        with Session(MBDB.get_engine()) as session:
            path_query = (
                select(ResolutionFrom.parent, ResolutionFrom.truth_cache)
                .join(Resolutions, Resolutions.resolution_id == ResolutionFrom.parent)
                .where(ResolutionFrom.child == self.resolution_id)
                .order_by(ResolutionFrom.level.desc())
            )

            results = session.execute(path_query).all()

            if not any(parent == dataset.resolution_id for parent, _ in results):
                raise ValueError(
                    f"No path between resolution {self.name}, dataset {dataset.name}"
                )

            lineage = {parent: truth for parent, truth in results}
            lineage[self.resolution_id] = self.truth

            return lineage

    @classmethod
    def next_id(cls) -> int:
        """Returns the next available resolution_id."""
        with Session(MBDB.get_engine()) as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.resolution_id), 0))
            ).scalar()
            return result + 1


class Sources(CountMixin, MBDB.MatchboxBase):
    """Table of sources of data for Matchbox."""

    def __init__(self, suffix="", contains_temporary=False):
        self.__tablename__ = "sources"

        # Columns
        self.resolution_id = Column(
            BIGINT,
            ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
            primary_key=True,
        )
        self.alias = Column(TEXT, nullable=False)
        self.full_name = Column(TEXT, nullable=False)
        self.warehouse_hash = Column(BYTEA, nullable=False)
        self.id = Column(TEXT, nullable=False)
        self.column_names = Column(ARRAY(TEXT), nullable=False)
        self.column_aliases = Column(ARRAY(TEXT), nullable=False)
        self.column_types = Column(ARRAY(TEXT), nullable=False)

        # Relationships
        self.dataset_resolution = relationship("Resolutions", back_populates="source")
        self.clusters = relationship("Clusters", back_populates="source")

        # Constraints
        self.__table_args__ = (
            UniqueConstraint("full_name", "warehouse_hash", name="unique_source_address"),
        )

    @classmethod
    def list(cls) -> list["Sources"]:
        with Session(MBDB.get_engine()) as session:
            return session.query(cls).all()


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    def __init__(self, suffix="", clusters_temporary=False):
        self.__tablename__ = f"contains{suffix}"

        # Columns
        if clusters_temporary:
            clusters_name = f"clusters{suffix}"
        else:
            clusters_name = "clusters"

        self.parent = Column(
            BIGINT, ForeignKey(f"{clusters_name}.cluster_id", ondelete="CASCADE"), primary_key=True
        )
        self.child = Column(
            BIGINT, ForeignKey(f"{clusters_name}.cluster_id", ondelete="CASCADE"), primary_key=True
        )

        # Constraints and indices
        self.__table_args__ = (
            CheckConstraint("parent != child", name="no_self_containment"),
            Index(f"ix_contains_parent_child{suffix}", "parent", "child"),
            Index(f"ix_contains_child_parent", "child", "parent"),
        )


class Clusters(CountMixin, MBDB.MatchboxBase):
    """Table of indexed data and clusters that match it."""

    def __init__(self, suffix="", contains_temporary=False):
        self.__tablename__ = f"clusters{suffix}"

        # Columns
        self.cluster_id = Column(BIGINT, primary_key=True)
        self.cluster_hash = Column(BYTEA, nullable=False)

        self.dataset = Column(BIGINT, ForeignKey("sources.resolution_id"), nullable=True)
        # Uses array as source data may have identical rows. We can't control this
        # Must be indexed or PostgreSQL incorrectly tries to use nested joins
        # when retrieving small datasets in query() -- extremely slow
        self.source_pk = Column(ARRAY(TEXT), index=True, nullable=True)

        # Relationships
        self.source = relationship("Sources", back_populates="clusters")
        self.probabilities = relationship(
            "Probabilities", back_populates="proposes", cascade="all, delete-orphan"
        )

        if contains_temporary:
            contains_name = f"{Contains.__table__}{suffix}"
        else:
            contains_name = Contains.__table__
        self.children = relationship(
            "Clusters",
            secondary=contains_name,
            primaryjoin="Clusters.cluster_id == Contains.parent",
            secondaryjoin="Clusters.cluster_id == Contains.child",
            backref="parents",
        )

        # Constraints and indices
        self.__table_args__ = (
            Index(f"ix_clusters_id_gin{suffix}", self.source_pk, postgresql_using="gin"),
            UniqueConstraint("cluster_hash", name=f"clusters_hash_key{suffix}"),
        )

    @classmethod
    def next_id(cls) -> int:
        """Returns the next available cluster_id."""
        with Session(MBDB.get_engine()) as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.cluster_id), 0))
            ).scalar()
            return result + 1


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    def __init__(self, suffix="", resolutions_temporary=False, clusters_temporary=False):
        self.__tablename__ = f"probabilities{suffix}"

        # Columns
        if resolutions_temporary:
            resolutions_name = f"resolutions{suffix}"
        else:
            resolutions_name = "resolutions"
        if clusters_temporary:
            clusters_name = f"clusters{suffix}"
        else:
            clusters_name = "clusters"

        self.resolution = Column(
            BIGINT,
            ForeignKey(f"{resolutions_name}.resolution_id", ondelete="CASCADE"),
            primary_key=True,
        )
        self.cluster = Column(
            BIGINT, ForeignKey(f"{clusters_name}.cluster_id", ondelete="CASCADE"), primary_key=True
        )
        self.probability = Column(SMALLINT, nullable=False)

        # Relationships
        self.proposed_by = relationship("Resolutions", back_populates="probabilities")
        self.proposes = relationship("Clusters", back_populates="probabilities")

        # Constraints
        self.__table_args__ = (
            CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        )
