from enum import Enum

from sqlalchemy import (
    FLOAT,
    VARCHAR,
    CheckConstraint,
    Column,
    ForeignKey,
    func,
    select,
    union,
)
from sqlalchemy import (
    text as sqltext,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, JSONB
from sqlalchemy.orm import Session, column_property, relationship

from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ModelType(Enum):
    MODEL = "model"
    DATASET = "dataset"
    HUMAN = "human"


class ModelsFrom(CountMixin, MBDB.MatchboxBase):
    """Model lineage table."""

    __tablename__ = "models_from"

    # Columns
    parent = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )

    # Constraints
    __table_args__ = (CheckConstraint("parent != child", name="no_self_reference"),)


class Models(CountMixin, MBDB.MatchboxBase):
    """Table of models and model-like objects: models, datasets and humans.

    By model-like objects, we mean objects that produce probabilities or own
    data in the clusters table.
    """

    __tablename__ = "models"

    # Columns
    hash = Column(BYTEA, primary_key=True)
    type = Column(VARCHAR, nullable=False)
    name = Column(VARCHAR, nullable=False)
    description = Column(VARCHAR)
    truth = Column(FLOAT)
    ancestors_cache = Column(JSONB, nullable=False, server_default="{}")

    # Relationships
    source = relationship("Sources", back_populates="dataset_model", uselist=False)
    probabilities = relationship(
        "Probabilities", back_populates="proposed_by", cascade="all, delete-orphan"
    )
    children = relationship(
        "Models",
        secondary=ModelsFrom.__table__,
        primaryjoin="Models.hash == ModelsFrom.parent",
        secondaryjoin="Models.hash == ModelsFrom.child",
        backref="parents",
    )

    # Ancestry
    # Recursive CTE for descendants
    descendants = column_property(
        select(func.array_agg(sqltext("descendant")))
        .select_from(
            select(sqltext("child as descendant"))
            .select_from(
                union(
                    select(ModelsFrom.child, ModelsFrom.child).where(
                        ModelsFrom.parent == hash
                    ),
                    select(ModelsFrom.child, ModelsFrom.parent).where(
                        ModelsFrom.parent.in_(
                            select(ModelsFrom.child)
                            .where(ModelsFrom.parent == hash)
                            .scalar_subquery()
                        )
                    ),
                ).cte(recursive=True)
            )
            .subquery()
        )
        .scalar_subquery(),
        deferred=True,
    )

    # Recursive CTE for ancestors
    ancestors = column_property(
        select(func.array_agg(sqltext("ancestor")))
        .select_from(
            select(sqltext("parent as ancestor"))
            .select_from(
                union(
                    select(ModelsFrom.parent, ModelsFrom.parent).where(
                        ModelsFrom.child == hash
                    ),
                    select(ModelsFrom.parent, ModelsFrom.child).where(
                        ModelsFrom.child.in_(
                            select(ModelsFrom.parent)
                            .where(ModelsFrom.child == hash)
                            .scalar_subquery()
                        )
                    ),
                ).cte(recursive=True)
            )
            .subquery()
        )
        .scalar_subquery(),
        deferred=True,
    )

    @property
    def all_descendants(self) -> list["Models"]:
        """Get all descendants as Model objects"""
        with MBDB.get_session() as session:
            if self.descendants is None:
                return []
            return session.query(Models).filter(Models.hash.in_(self.descendants)).all()

    @property
    def all_ancestors(self) -> list["Models"]:
        """Get all ancestors as Model objects"""
        with MBDB.get_session() as session:
            if self.ancestors is None:
                return []
            return session.query(Models).filter(Models.hash.in_(self.ancestors)).all()

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "type IN ('model', 'dataset', 'human')",
            name="model_type_constraints",
        ),
        CheckConstraint(
            "NOT (ancestors_cache ?| ARRAY[hash::text])", name="no_self_ancestor"
        ),
    )


class Sources(CountMixin, MBDB.MatchboxBase):
    """Table of sources of data for Matchbox."""

    __tablename__ = "sources"

    # Columns
    model = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    schema = Column(VARCHAR, nullable=False)
    table = Column(VARCHAR, nullable=False)
    id = Column(VARCHAR, nullable=False)

    # Relationships
    dataset_model = relationship("Models", back_populates="source")
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
    dataset = Column(BYTEA, ForeignKey("sources.model"), nullable=False)
    # Uses array as source data may have identical rows. We can't control this
    # Must be indexed or PostgreSQL incorrectly tries to use nested joins
    # when retrieving small datasets in query() -- extremely slow
    id = Column(ARRAY(VARCHAR(36)), index=True)

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
    """Table of probabilities that a cluster merge is correct, according to a model."""

    __tablename__ = "probabilities"

    # Columns
    model = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    cluster = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
    )
    probability = Column(FLOAT, nullable=False)

    # Relationships
    proposed_by = relationship("Models", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 1", name="valid_probability"),
    )
