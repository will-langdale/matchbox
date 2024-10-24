from enum import Enum

from sqlalchemy import (
    FLOAT,
    VARCHAR,
    CheckConstraint,
    Column,
    ForeignKey,
    func,
    select,
    text,
    union,
)
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, JSONB
from sqlalchemy.orm import column_property, relationship

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
    type = Column(SQLAEnum(ModelType), nullable=False)
    name = Column(VARCHAR, nullable=False)
    description = Column(VARCHAR)
    truth = Column(FLOAT)
    ancestors = Column(JSONB, nullable=False, server_default="{}")

    # Relationships
    source = relationship("Sources", back_populates="model", uselist=False)
    probabilities = relationship(
        "Probabilities", back_populates="model", cascade="all, delete-orphan"
    )
    children = relationship(
        "Models",
        secondary="models_from",
        primaryjoin="Models.hash == ModelsFrom.parent",
        secondaryjoin="Models.hash == ModelsFrom.child",
        backref="parents",
    )

    # Ancestry
    # Recursive CTE for descendants
    descendants = column_property(
        select([func.array_agg(text("descendant"))])
        .select_from(
            select([text("child as descendant")]).select_from(
                union(
                    select([ModelsFrom.child, ModelsFrom.child]).where(
                        ModelsFrom.parent == hash
                    ),
                    select([ModelsFrom.child, ModelsFrom.parent]).where(
                        ModelsFrom.parent.in_(
                            select([ModelsFrom.child]).where(ModelsFrom.parent == hash)
                        )
                    ),
                ).cte(recursive=True)
            )
        )
        .scalar_subquery(),
        deferred=True,
    )

    # Recursive CTE for ancestors
    ancestors = column_property(
        select([func.array_agg(text("ancestor"))])
        .select_from(
            select([text("parent as ancestor")]).select_from(
                union(
                    select([ModelsFrom.parent, ModelsFrom.parent]).where(
                        ModelsFrom.child == hash
                    ),
                    select([ModelsFrom.parent, ModelsFrom.child]).where(
                        ModelsFrom.child.in_(
                            select([ModelsFrom.parent]).where(ModelsFrom.child == hash)
                        )
                    ),
                ).cte(recursive=True)
            )
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
            if self.ancestors_array is None:
                return []
            return (
                session.query(Models)
                .filter(Models.hash.in_(self.ancestors_array))
                .all()
            )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            """
            CASE
                WHEN type = 'model' THEN 
                    NOT EXISTS(SELECT 1 FROM sources WHERE model = hash) AND
                    EXISTS(SELECT 1 FROM models_from WHERE child = hash)
                WHEN type = 'dataset' THEN 
                    EXISTS(SELECT 1 FROM sources WHERE model = hash)
                WHEN type = 'human' THEN 
                    NOT EXISTS(SELECT 1 FROM sources WHERE model = hash) AND
                    NOT EXISTS(SELECT 1 FROM models_from WHERE child = hash)
            END
            """,
            name="model_type_constraints",
        ),
        CheckConstraint(
            "ancestors::jsonb ?& ARRAY"
            "(SELECT hash::text FROM models WHERE hash = ANY(ancestors::jsonb))",
            name="ancestors_valid_hashes",
        ),
        CheckConstraint(
            "NOT (ancestors ?| ARRAY[hash::text])", name="no_self_ancestor"
        ),
        CheckConstraint(
            "CASE WHEN ancestors <> '{}'::jsonb THEN \
             ALL(SELECT jsonb_object_values(ancestors)::float BETWEEN 0 AND 1) \
             ELSE true END",
            name="ancestor_weights_valid",
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
    model_rel = relationship("Models", back_populates="source")
    clusters = relationship("Clusters", back_populates="source")


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
        "Probabilities", back_populates="cluster", cascade="all, delete-orphan"
    )
    children = relationship(
        "Clusters",
        secondary="contains",
        primaryjoin="Clusters.hash == Contains.parent",
        secondaryjoin="Clusters.hash == Contains.child",
        backref="parents",
    )


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
    model_rel = relationship("Models", back_populates="probabilities")
    cluster_rel = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 1", name="valid_probability"),
    )
