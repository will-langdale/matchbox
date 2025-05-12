"""ORM classes for the Matchbox PostgreSQL database."""

from typing import Literal

from sqlalchemy import (
    BIGINT,
    INTEGER,
    SMALLINT,
    CheckConstraint,
    Column,
    ForeignKey,
    Identity,
    Index,
    UniqueConstraint,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import BYTEA, TEXT, insert
from sqlalchemy.orm import Session, relationship

from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.sources import SourceAddress
from matchbox.common.sources import SourceColumn as CommonSourceColumn
from matchbox.common.sources import SourceConfig as CommonSourceConfig
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ResolutionFrom(CountMixin, MBDB.MatchboxBase):
    """Resolution lineage closure table with cached truth values."""

    __tablename__ = "resolution_from"

    # Columns
    parent = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    child = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    level = Column(INTEGER, nullable=False)
    truth_cache = Column(SMALLINT, nullable=True)

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
    resolution_id = Column(BIGINT, primary_key=True, autoincrement=True)
    resolution_hash = Column(BYTEA, nullable=False)
    content_hash = Column(BYTEA, nullable=True)
    type = Column(TEXT, nullable=False)
    name = Column(TEXT, nullable=False)
    description = Column(TEXT, nullable=True)
    truth = Column(SMALLINT, nullable=True)

    # Relationships
    source_config = relationship(
        "SourceConfigs", back_populates="source_resolution", uselist=False
    )
    probabilities = relationship(
        "Probabilities",
        back_populates="proposed_by",
        passive_deletes=True,
    )
    children = relationship(
        "Resolutions",
        secondary=ResolutionFrom.__table__,
        primaryjoin="Resolutions.resolution_id == ResolutionFrom.parent",
        secondaryjoin="Resolutions.resolution_id == ResolutionFrom.child",
        backref="parents",
    )

    # Constraints
    __table_args__ = (
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
        with MBDB.get_session() as session:
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
        with MBDB.get_session() as session:
            descendant_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.child)
                .where(ResolutionFrom.parent == self.resolution_id)
            )
            return set(session.execute(descendant_query).scalars().all())

    def get_lineage(self) -> dict[int, float]:
        """Returns all ancestors and their cached truth values from this model."""
        with MBDB.get_session() as session:
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

        with MBDB.get_session() as session:
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
    def from_name(
        cls,
        resolution_name: str,
        res_type: Literal["model", "dataset", "human"] | None = None,
        session: Session | None = None,
    ) -> "Resolutions":
        """Resolves a model name to a Resolution object.

        Args:
            resolution_name: The name of the model to resolve.
            res_type: A resolution type to use as filter.
            session: A session to get the resolution for updates.

        Raises:
            MatchboxResolutionNotFoundError: If the model doesn't exist.
        """
        query = select(cls).where(cls.name == resolution_name)
        if res_type:
            query = query.where(cls.type == res_type)

        if session:
            resolution = session.execute(query).scalar()
        else:
            with MBDB.get_session() as session:
                resolution = session.execute(query).scalar()

        if resolution:
            return resolution

        res_type = res_type or "any"
        raise MatchboxResolutionNotFoundError(
            message=f"No resolution {resolution_name} of {res_type}."
        )


class PKSpace(MBDB.MatchboxBase):
    """Table used to reserve ranges of primary keys."""

    __tablename__ = "pk_space"

    id = Column(BIGINT, primary_key=True)
    next_cluster_id = Column(BIGINT, nullable=False)
    next_cluster_source_pk_id = Column(BIGINT, nullable=False)

    @classmethod
    def initialise(cls) -> None:
        """Create PKSpace tracking row if not exists."""
        with MBDB.get_session() as session:
            init_statement = (
                insert(cls)
                .values(id=1, next_cluster_id=1, next_cluster_source_pk_id=1)
                .on_conflict_do_nothing()
            )

            session.execute(init_statement)
            session.commit()

    @classmethod
    def reserve_block(
        cls, table: Literal["clusters", "cluster_source_pks"], block_size: int
    ) -> int:
        """Atomically get next available ID for table, and increment it."""
        if block_size < 1:
            raise ValueError("Block size must be at least 1.")

        match table:
            case "clusters":
                next_id_col = "next_cluster_id"
            case "cluster_source_pks":
                next_id_col = "next_cluster_source_pk_id"

        with MBDB.get_session() as session:
            try:
                next_id = session.execute(
                    update(cls)
                    .values(**{next_id_col: getattr(cls, next_id_col) + block_size})
                    .returning(getattr(cls, next_id_col) - block_size)
                ).scalar_one()
                session.commit()
            except:
                session.rollback()
                raise

            return next_id


class SourceColumns(CountMixin, MBDB.MatchboxBase):
    """Table for storing column details for SourceConfigs."""

    __tablename__ = "source_columns"

    # Columns
    column_id = Column(BIGINT, primary_key=True)
    source_config_id = Column(
        BIGINT,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    )
    column_index = Column(INTEGER, nullable=False)
    column_name = Column(TEXT, nullable=False)
    column_type = Column(TEXT, nullable=False)

    # Relationships
    source_config = relationship("SourceConfigs", back_populates="columns")

    # Constraints and indices
    __table_args__ = (
        UniqueConstraint(
            "source_config_id", "column_index", name="unique_column_index"
        ),
        Index("ix_source_columns_source_config_id", "source_config_id"),
    )


class ClusterSourcePK(CountMixin, MBDB.MatchboxBase):
    """Table for storing source primary keys for clusters."""

    __tablename__ = "cluster_source_pks"

    # Columns
    pk_id = Column(BIGINT, primary_key=True)
    cluster_id = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    source_config_id = Column(
        BIGINT,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_pk = Column(TEXT, nullable=False)

    # Relationships
    cluster = relationship("Clusters", back_populates="source_pks")
    source_config = relationship("SourceConfigs", back_populates="cluster_source_pks")

    # Constraints and indices
    __table_args__ = (
        Index("ix_cluster_source_pks_cluster_id", "cluster_id"),
        Index("ix_cluster_source_pks_source_pk", "source_pk"),
        UniqueConstraint("pk_id", "source_config_id", name="unique_pk_source"),
    )


class SourceConfigs(CountMixin, MBDB.MatchboxBase):
    """Table of source_configs of data for Matchbox."""

    __tablename__ = "source_configs"

    # Columns
    source_config_id = Column(BIGINT, Identity(start=1), primary_key=True)
    resolution_id = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        nullable=False,
    )
    resolution_name = Column(TEXT, nullable=False)
    full_name = Column(TEXT, nullable=False)
    warehouse_hash = Column(BYTEA, nullable=False)
    db_pk = Column(TEXT, nullable=False)

    # Relationships
    source_resolution = relationship("Resolutions", back_populates="source_config")
    columns = relationship(
        "SourceColumns",
        back_populates="source_config",
        passive_deletes=True,
    )
    cluster_source_pks = relationship(
        "ClusterSourcePK",
        back_populates="source_config",
        passive_deletes=True,
    )
    clusters = relationship(
        "Clusters",
        secondary=ClusterSourcePK.__table__,
        primaryjoin=(
            "SourceConfigs.source_config_id == ClusterSourcePK.source_config_id"
        ),
        secondaryjoin="ClusterSourcePK.cluster_id == Clusters.cluster_id",
        viewonly=True,
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("full_name", "warehouse_hash", name="unique_source_address"),
    )

    @classmethod
    def list_all(cls) -> list["SourceConfigs"]:
        """Returns all source_configs in the database."""
        with MBDB.get_session() as session:
            return session.query(cls).all()

    def to_dto(self) -> list[CommonSourceConfig]:
        """Convert ORM source to a matchbox.common SourceConfig object."""
        with MBDB.get_session() as session:
            columns: list[SourceColumns] = (
                session.query(SourceColumns)
                .filter(SourceColumns.source_config_id == self.source_config_id)
                .order_by(SourceColumns.column_index)
                .all()
            )

        return CommonSourceConfig(
            resolution_name=self.resolution_name,
            address=SourceAddress(
                full_name=self.full_name, warehouse_hash=self.warehouse_hash
            ),
            db_pk=self.db_pk,
            columns=[
                CommonSourceColumn(
                    name=column.column_name,
                    type=column.column_type,
                )
                for column in columns
            ],
        )


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    parent = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )

    # Constraints and indices
    __table_args__ = (
        CheckConstraint("parent != child", name="no_self_containment"),
        Index("ix_contains_parent_child", "parent", "child"),
        Index("ix_contains_child_parent", "child", "parent"),
    )


class Clusters(CountMixin, MBDB.MatchboxBase):
    """Table of indexed data and clusters that match it."""

    __tablename__ = "clusters"

    # Columns
    cluster_id = Column(BIGINT, primary_key=True)
    cluster_hash = Column(BYTEA, nullable=False)

    # Relationships
    source_pks = relationship(
        "ClusterSourcePK",
        back_populates="cluster",
        passive_deletes=True,
    )
    probabilities = relationship(
        "Probabilities",
        back_populates="proposes",
        passive_deletes=True,
    )
    children = relationship(
        "Clusters",
        secondary=Contains.__table__,
        primaryjoin="Clusters.cluster_id == Contains.parent",
        secondaryjoin="Clusters.cluster_id == Contains.child",
        backref="parents",
    )
    # Add relationship to SourceConfigs through ClusterSourcePK
    source_configs = relationship(
        "SourceConfigs",
        secondary=ClusterSourcePK.__table__,
        primaryjoin="Clusters.cluster_id == ClusterSourcePK.cluster_id",
        secondaryjoin=(
            "ClusterSourcePK.source_config_id == SourceConfigs.source_config_id"
        ),
        viewonly=True,
    )

    # Constraints and indices
    __table_args__ = (UniqueConstraint("cluster_hash", name="clusters_hash_key"),)


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    __tablename__ = "probabilities"

    # Columns
    resolution = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    probability = Column(SMALLINT, nullable=False)

    # Relationships
    proposed_by = relationship("Resolutions", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        Index("ix_probabilities_resolution", "resolution"),
    )
