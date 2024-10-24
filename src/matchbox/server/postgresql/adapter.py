from typing import Literal

import pandas as pd
from pydantic import BaseModel
from rustworkx import PyDiGraph
from sqlalchemy import Engine, and_, bindparam, func
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import Session

from matchbox.common.exceptions import (
    MatchboxDataError,
    MatchboxDatasetError,
    MatchboxModelError,
)
from matchbox.server.base import MatchboxDBAdapter, MatchboxModelAdapter
from matchbox.server.models import Probability, Source, SourceWarehouse
from matchbox.server.postgresql.db import MBDB, MatchboxPostgresSettings
from matchbox.server.postgresql.orm import Clusters, Models, Probabilities, Sources
from matchbox.server.postgresql.utils.db import get_model_subgraph
from matchbox.server.postgresql.utils.insert import (
    insert_dataset,
    insert_model,
    insert_probabilities,
)
from matchbox.server.postgresql.utils.query import query


class FilteredClusters(BaseModel):
    """Wrapper class for filtered cluster queries"""

    has_dataset: bool | None = None

    def count(self) -> int:
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Clusters)
            if self.has_dataset is not None:
                if self.has_dataset:
                    query = query.filter(Clusters.dataset.isnot(None))
                else:
                    query = query.filter(Clusters.dataset.is_(None))
            return query.scalar()


class FilteredProbabilities(BaseModel):
    """Wrapper class for filtered probability queries"""

    over_truth: bool = False
    mb_model: Models | None = None

    def count(self) -> int:
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Probabilities)

            if self.mb_model is not None:
                query = query.filter(Probabilities.model == self.mb_model.hash)

            if self.over_truth:
                query = query.join(Models, Probabilities.model == Models.hash).filter(
                    and_(
                        Models.truth.isnot(None),
                        Probabilities.probability > Models.truth,
                    )
                )
            return query.scalar()


class MatchboxPostgresModel(MatchboxModelAdapter):
    """An adapter for Matchbox models in PostgreSQL."""

    def __init__(self, model: Models):
        self.model: Models = model
        self.clusters = FilteredProbabilities(over_truth=True, mb_model=model)
        self.probabilities = FilteredProbabilities(mb_model=model)

    @property
    def hash(self) -> bytes:
        return self.model.hash

    @property
    def name(self) -> str:
        return self.model.name

    def insert_probabilities(
        self, probabilities: list[Probability], batch_size: int
    ) -> None:
        """Inserts probabilities for this model."""
        insert_probabilities(
            model=self.name,
            engine=MBDB.get_engine(),
            probabilities=probabilities,
            batch_size=batch_size,
        )

    def set_truth_threshold(self, probability: float) -> None:
        """Sets the truth threshold for this model, creating clusters."""
        with Session(MBDB.get_engine()) as session:
            self.model.truth = probability
            session.add(self.model)
            session.commit()

    @property
    def ancestors_cache(self) -> dict[str, float]:
        """
        Gets the cached ancestor thresholds, converting hashes to model names.
        Returns a dict mapping model names to their truth thresholds.

        This is required because each point of truth needs to be stable, so we choose
        when to update it, caching the ancestor's values in the model itself.
        """
        with Session(MBDB.get_engine()) as session:
            stored_ancestors = self.model.ancestors_cache or {}
            ancestor_models = (
                session.query(Models.hash, Models.name)
                .filter(
                    Models.hash.in_([bytes.fromhex(h) for h in stored_ancestors.keys()])
                )
                .all()
            )
            hash_to_name = {m.hash.hex(): m.name for m in ancestor_models}

            return {
                hash_to_name[hash_str]: float(threshold)
                for hash_str, threshold in stored_ancestors.items()
            }

    @ancestors_cache.setter
    def ancestors_cache(self, new_values: dict[str, float]) -> None:
        """
        Updates the cached ancestor thresholds.
        Takes a dict mapping model names to their truth thresholds.
        Only updates the float values, preserving the existing hash structure.
        """
        with Session(MBDB.get_engine()) as session:
            name_to_hash = {
                m.name: m.hash.hex()
                for m in session.query(Models)
                .filter(Models.name.in_(new_values.keys()))
                .all()
            }

            # Update only the values in the existing JSON structure
            current = self.model.ancestors_cache or {}
            for name, threshold in new_values.items():
                if hash_str := name_to_hash.get(name):
                    current[hash_str] = threshold

            self.model.ancestors_cache = current
            session.add(self.model)
            session.commit()

    @property
    def ancestors(self) -> dict[str, float]:
        """
        Gets the current truth values of all ancestors.
        Returns a dict mapping model names to their current truth thresholds.

        Unlike ancestors_cache which returns cached values, this property returns
        the current truth values of all ancestor models.
        """
        with Session(MBDB.get_engine()) as session:
            if not self.model.ancestors:
                return {}

            ancestor_models = (
                session.query(Models)
                .filter(Models.hash.in_(self.model.ancestors))
                .all()
            )

            return {
                model.name: float(model.truth)
                for model in ancestor_models
                if model.truth is not None
            }

    @classmethod
    def get_model(cls, model_name: str) -> "MatchboxPostgresModel":
        with Session(MBDB.get_engine()) as session:
            if model := session.query(Models).filter_by(name=model_name).first():
                return cls(model)
            else:
                raise MatchboxModelError(model_name=model_name)


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings):
        self.settings = settings
        MBDB.settings = settings
        MBDB.create_database()

        self.datasets = Sources
        self.models = Models
        self.data = FilteredClusters(has_dataset=True)
        self.clusters = FilteredClusters(has_dataset=False)
        self.creates = FilteredProbabilities(over_truth=True)
        self.merges = FilteredProbabilities()
        self.proposes = FilteredProbabilities()

    def query(
        self,
        selector: dict[str, list[str]],
        model: str | None = None,
        threshold: float | dict[str, float] | None = None,
        return_type: Literal["pandas", "sqlalchemy"] | None = None,
        limit: int = None,
    ) -> pd.DataFrame | ChunkedIteratorResult:
        """Queries the database from an optional model of truth.

        Args:
            selector: the tables and fields to query
            return_type: the form to return data in, one of "pandas" or "arrow"
                Defaults to pandas for ease of use
            model (optional): the model to use for filtering results
            threshold (optional): the threshold to use for creating clusters
                If None, uses the models' default threshold
                If a float, uses that threshold for the specified model, and the
                model's cached thresholds for its ancestors
                If a dictionary, expects a shape similar to model.ancestors, keyed
                by model name and valued by the threshold to use for that model. Will
                use these threshold values instead of the cached thresholds
            limit (optional): the number to use in a limit clause. Useful for testing

        Returns:
            Data in the requested return type
        """
        return query(
            selector=selector,
            engine=MBDB.get_engine(),
            return_type=return_type,
            model=model,
            threshold=threshold,
            limit=limit,
        )

    def index(self, dataset: Source) -> None:
        """Indexes a data from your data warehouse within Matchbox.

        Args:
            dataset: The dataset to index.
            engine: The SQLAlchemy engine of your data warehouse.
        """
        insert_dataset(
            dataset=dataset,
            engine=MBDB.get_engine(),
            batch_size=self.settings.batch_size,
        )

    def validate_hashes(self, hashes: list[bytes]) -> None:
        """Validates a list of hashes exist in the database.

        Args:
            hashes: A list of hashes to validate.
            hash_type: The type of hash to validate.

        Raises:
            MatchboxDataError: If some items don't exist in the target table.
        """
        with Session(MBDB.get_engine()) as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.hash.in_(
                        bindparam(
                            "ins_hashs",
                            hashes,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        if len(data_inner_join) != len(hashes):
            raise MatchboxDataError(
                message=("Some items don't exist in Clusters table"),
                table=Clusters.__tablename__,
            )

    def get_dataset(self, db_schema: str, db_table: str, engine: Engine) -> Source:
        """Get a source dataset from the database.

        Args:
            db_schema: The schema of the dataset.
            db_table: The table of the dataset.
            engine: The engine to use to connect to your data warehouse.
        """
        with Session(MBDB.get_engine()) as session:
            dataset = (
                session.query(Sources)
                .filter_by(schema=db_schema, table=db_table)
                .first()
            )
            if dataset:
                return Source(
                    db_schema=dataset.schema,
                    db_table=dataset.table,
                    db_pk=dataset.id,
                    database=SourceWarehouse.from_engine(engine),
                )
            else:
                raise MatchboxDatasetError(db_schema=db_schema, db_table=db_table)

    def get_model_subgraph(self) -> PyDiGraph:
        """Get the full subgraph of a model."""
        return get_model_subgraph(engine=MBDB.get_engine())

    def get_model(self, model: str) -> MatchboxPostgresModel:
        """Get a model from the database.

        Args:
            model: The model to get.
        """
        with Session(MBDB.get_engine()) as session:
            if model := session.query(Models).filter_by(name=model).first():
                return MatchboxPostgresModel(model)
            else:
                raise MatchboxModelError(model_name=model)

    def delete_model(self, model: str, certain: bool = False) -> None:
        """Delete a model from the database.

        Args:
            model: The model to delete.
            certain: Whether to delete the model without confirmation.
        """
        with Session(MBDB.get_engine()) as session:
            if model := session.query(Models).filter(Models.name == model).first():
                if certain:
                    session.delete(model)
                    session.commit()
                else:
                    childen = model.all_descendants()
                    children_names = ", ".join([m.name for m in childen])
                    raise ValueError(
                        f"This operation will delete the models {children_names}, "
                        "as well as all probabilities they have created. \n\n"
                        "It won't delete validation associated with these "
                        "probabilities. \n\n"
                        "If you're sure you want to continue, rerun with certain=True"
                    )
            raise MatchboxModelError(model_name=model)

    def insert_model(
        self, model: str, left: str, description: str, right: str | None = None
    ) -> None:
        """Writes a model to Matchbox.

        Args:
            model: The name of the model
            left: The name of the model on the left side of this model's join, or the
                name of the dataset it deduplicates
            right: The name of the model on the right side of this model's join
                When deduplicating, this is None
            description: A description of the model

        Raises
            MatchboxDataError if, for a linker, the source models weren't found in
                the database
        """
        insert_model(
            model=model,
            left=left,
            right=right,
            description=description,
            engine=MBDB.get_engine(),
        )

    def clear(self, certain: bool = False) -> None:
        """Clears all data from the database.

        Args:
            certain: Whether to clear the database without confirmation.
        """
        if certain:
            MBDB.clear_database()
        else:
            raise ValueError(
                "This operation will drop the entire database. "
                "It's principally used for testing. \n\n"
                "If you're sure you want to continue, rerun with certain=True"
            )
