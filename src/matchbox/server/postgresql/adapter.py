from typing import Literal

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseSettings, Field
from rustworkx import PyDiGraph
from sqlalchemy import (
    Engine,
    bindparam,
)
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import Session

from matchbox.common.exceptions import MatchboxDBDataError
from matchbox.server.base import (
    IndexableDataset,
    MatchboxDBAdapter,
    MatchboxModelAdapter,
    MatchboxSettings,
)
from matchbox.server.postgresql.clusters import Clusters, clusters_association
from matchbox.server.postgresql.data import SourceData, SourceDataset
from matchbox.server.postgresql.db import connect_to_db
from matchbox.server.postgresql.dedupe import DDupeProbabilities, Dedupes
from matchbox.server.postgresql.link import LinkProbabilities, Links
from matchbox.server.postgresql.models import Models, ModelsFrom
from matchbox.server.postgresql.utils.db import get_model_subgraph
from matchbox.server.postgresql.utils.delete import delete_model
from matchbox.server.postgresql.utils.index import index_dataset
from matchbox.server.postgresql.utils.insert import insert_deduper, insert_linker
from matchbox.server.postgresql.utils.selector import query
from matchbox.server.postgresql.utils.sha1 import table_name_to_uuid

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class MergesUnion:
    """A thin wrapper around Dedupes and Links to provide a count method."""

    def count(self) -> int:
        return Dedupes.count() + Links.count()


class ProposesUnion:
    """A thin wrapper around probability classes to provide a count method."""

    def count(self) -> int:
        return DDupeProbabilities.count() + LinkProbabilities.count()


class MatchboxPostgresSettings(BaseSettings):
    """Settings for Matchbox's PostgreSQL backend."""

    host: str = Field(..., env="MB__POSTGRES_HOST")
    port: int = Field(..., env="MB__POSTGRES_PORT")
    user: str = Field(..., env="MB__POSTGRES_USER")
    password: str = Field(..., env="MB__POSTGRES_PASSWORD")
    database: str = Field(..., env="MB__POSTGRES_DATABASE")

    class Config:
        env_prefix = "MB__POSTGRES_"
        allow_population_by_field_name = True


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxSettings):
        self.settings = settings

        self.datasets = SourceDataset
        self.models = Models
        self.models_from = ModelsFrom
        self.data = SourceData
        self.clusters = Clusters
        self.creates = clusters_association
        self.merges = MergesUnion()
        self.proposes = ProposesUnion()

        MatchboxBase, engine = connect_to_db(settings=MatchboxPostgresSettings())

        self.base = MatchboxBase
        self.engine = engine

        self.base.metadata.create_all(self.engine)

    def query(
        self,
        selector: dict[str, list[str]],
        model: str | None = None,
        return_type: Literal["pandas", "sqlalchemy"] | None = None,
        limit: int = None,
    ) -> pd.DataFrame | ChunkedIteratorResult:
        """Queries the database from an optional model of truth.

        Args:
            selector: A dictionary of the validated table name and fields.
            model (optional): The model of truth to query from.
            return_type (optional): The type of return data.
            limit (optional): The number of rows to return.
        """
        return query(
            selector=selector,
            engine=self.engine,
            return_type=return_type,
            model=model,
            limit=limit,
        )

    def index(self, dataset: IndexableDataset, engine: Engine) -> None:
        """Indexes a data from your data warehouse within Matchbox.

        Args:
            dataset: The dataset to index.
            engine: The SQLAlchemy engine of your data warehouse.
        """
        index_dataset(
            dataset=dataset,
            engine=self.engine,
            warehouse_engine=engine,
        )

    def validate_hashes(
        self, hashes: list[bytes], hash_type: Literal["data", "cluster"]
    ) -> None:
        """Validates a list of hashes exist in the database.

        Args:
            hashes: A list of hashes to validate.
            hash_type: The type of hash to validate.

        Raises:
            MatchboxDBDataError: If some items don't exist in the target table.
        """
        if hash_type == "data":
            Source = SourceData
            tgt_col = "data_sha1"
        elif hash_type == "cluster":
            Source = Clusters
            tgt_col = "cluster_sha1"

        with Session(self.engine) as session:
            data_inner_join = (
                session.query(Source)
                .filter(
                    Source.sha1.in_(
                        bindparam(
                            "ins_sha1s",
                            hashes,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        if len(data_inner_join) != len(hashes):
            raise MatchboxDBDataError(
                message=(
                    f"Some items don't exist the target table. "
                    f"Did you use {tgt_col} as your ID when deduplicating?"
                ),
                source=Source,
            )

    def get_model_subgraph(self) -> PyDiGraph:
        """Get the full subgraph of a model."""
        return get_model_subgraph(engine=self.engine)

    def get_model(self, model: str) -> MatchboxModelAdapter:
        """Get a model from the database.

        Args:
            model: The model to get.
        """
        with Session(self.engine) as session:
            return session.query(Models).filter_by(name=model).first()

    def delete_model(self, model: str, certain: bool = False) -> None:
        """Delete a model from the database.

        Args:
            model: The model to delete.
            certain: Whether to delete the model without confirmation.
        """
        delete_model(model=model, certain=certain)

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
            MatchboxDBDataError if, for a linker, the source models weren't found in
                the database
        """
        if right:
            # Linker
            insert_linker(
                model=model,
                left=left,
                right=right,
                description=description,
            )
        else:
            # Deduper
            deduplicates = table_name_to_uuid(schema_table=left, engine=self.engine)
            insert_deduper(
                model=model,
                deduplicates=deduplicates,
                description=description,
            )
