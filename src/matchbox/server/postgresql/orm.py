from enum import Enum

from sqlalchemy import (
    FLOAT,
    VARCHAR,
    CheckConstraint,
    Column,
    ForeignKey,
    literal_column,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, JSONB
from sqlalchemy.orm import Session, relationship

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
    name = Column(VARCHAR, nullable=False, unique=True)
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

    @property
    def ancestors(self) -> set["Models"]:
        """
        Returns all ancestors (parents, grandparents, etc.) of this model.
        Uses recursive CTE to efficiently query the ancestry chain.
        """
        with Session(MBDB.get_engine()) as session:
            # Create recursive CTE to find all ancestors
            base_query = (
                select(
                    ModelsFrom.parent.label("ancestor"),
                    ModelsFrom.child.label("descendant"),
                    literal_column("1").label("depth"),
                )
                .select_from(ModelsFrom)
                .where(ModelsFrom.child == self.hash)
            )

            cte = base_query.cte(recursive=True)

            # Add recursive term
            parent_alias = ModelsFrom.__table__.alias()
            recursive_term = (
                select(parent_alias.c.parent, cte.c.descendant, cte.c.depth + 1)
                .select_from(parent_alias)
                .join(cte, parent_alias.c.child == cte.c.ancestor)
            )

            # Combine base and recursive terms
            cte = cte.union_all(recursive_term)

            # Query all ancestor hashes
            ancestor_query = (
                select(Models)
                .select_from(Models)
                .where(Models.hash.in_(select(cte.c.ancestor).distinct()))
            )

            return set(session.execute(ancestor_query).scalars().all())

    @property
    def descendants(self) -> set["Models"]:
        """
        Returns all descendants (children, grandchildren, etc.) of this model.
        Uses recursive CTE to efficiently query the descendant chain.
        """
        with Session(MBDB.get_engine()) as session:
            # Create recursive CTE to find all descendants
            base_query = (
                select(
                    ModelsFrom.parent.label("ancestor"),
                    ModelsFrom.child.label("descendant"),
                    literal_column("1").label("depth"),
                )
                .select_from(ModelsFrom)
                .where(ModelsFrom.parent == self.hash)
            )

            cte = base_query.cte(recursive=True)

            # Add recursive term
            child_alias = ModelsFrom.__table__.alias()
            recursive_term = (
                select(cte.c.ancestor, child_alias.c.child, cte.c.depth + 1)
                .select_from(child_alias)
                .join(cte, child_alias.c.parent == cte.c.descendant)
            )

            # Combine base and recursive terms
            cte = cte.union_all(recursive_term)

            # Query all descendant hashes
            descendant_query = (
                select(Models)
                .select_from(Models)
                .where(Models.hash.in_(select(cte.c.descendant).distinct()))
            )

            return set(session.execute(descendant_query).scalars().all())

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
    dataset = Column(BYTEA, ForeignKey("sources.model"), nullable=True)
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
