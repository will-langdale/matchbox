from typing import Literal

import pandas as pd
from rustworkx import PyDiGraph
from sqlalchemy import (
    bindparam,
)
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import (
    Session,
)

from matchbox.common.exceptions import MatchboxDBDataError
from matchbox.server.base import (
    Cluster,
    IndexableDataset,
    MatchboxDBAdapter,
    MatchboxModelAdapter,
    Probability,
)
from matchbox.server.postgresql.clusters import Clusters, clusters_association
from matchbox.server.postgresql.data import SourceData, SourceDataset
from matchbox.server.postgresql.db import MBDB, MatchboxPostgresSettings
from matchbox.server.postgresql.dedupe import DDupeProbabilities, Dedupes
from matchbox.server.postgresql.link import LinkProbabilities, Links
from matchbox.server.postgresql.models import Models, ModelsFrom
from matchbox.server.postgresql.utils.db import get_model_subgraph
from matchbox.server.postgresql.utils.delete import delete_model
from matchbox.server.postgresql.utils.hash import table_name_to_uuid
from matchbox.server.postgresql.utils.index import index_dataset
from matchbox.server.postgresql.utils.insert import (
    insert_clusters,
    insert_deduper,
    insert_linker,
    insert_probabilities,
)
from matchbox.server.postgresql.utils.selector import query


class MergesUnion:
    """A thin wrapper around Dedupes and Links to provide a count method."""

    def count(self) -> int:
        return Dedupes.count() + Links.count()


class ProposesUnion:
    """A thin wrapper around probability classes to provide a count method."""

    def count(self) -> int:
        return DDupeProbabilities.count() + LinkProbabilities.count()


class CombinedProbabilities:
    def __init__(self, dedupes, links):
        self._dedupes = dedupes
        self._links = links

    def count(self):
        return self._dedupes.count() + self._links.count()


class MatchboxPostgresModel(MatchboxModelAdapter):
    """An adapter for Matchbox models in PostgreSQL."""

    def __init__(self, model: Models):
        self.model = model

    @property
    def sha1(self) -> bytes:
        return self.model.sha1

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def clusters(self):
        return self.model.creates

    @property
    def probabilities(self) -> CombinedProbabilities:
        return CombinedProbabilities(
            self.model.proposes_dedupes, self.model.proposes_links
        )

    def insert_probabilities(
        self,
        probabilities: list[Probability],
        probability_type: Literal["deduplications", "links"],
        batch_size: int,
    ) -> None:
        insert_probabilities(
            model=self.name,
            engine=MBDB.get_engine(),
            probabilities=probabilities,
            batch_size=batch_size,
            is_deduper=probability_type == "deduplications",
        )

    def insert_clusters(
        self,
        clusters: list[Cluster],
        cluster_type: Literal["deduplications", "links"],
        batch_size: int,
    ) -> None:
        insert_clusters(
            model=self.name,
            engine=MBDB.get_engine(),
            clusters=clusters,
            batch_size=batch_size,
            is_deduper=cluster_type == "deduplications",
        )

    @classmethod
    def get_model(cls, model_name: str) -> "MatchboxPostgresModel":
        with Session(MBDB.get_engine()) as session:
            model = session.query(Models).filter_by(name=model_name).first()
            if model:
                return cls(model)
            return None


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings):
        self.settings = settings
        MBDB.settings = settings
        MBDB.create_database()

        self.datasets = SourceDataset
        self.models = Models
        self.models_from = ModelsFrom
        self.data = SourceData
        self.clusters = Clusters
        self.creates = clusters_association
        self.merges = MergesUnion()
        self.proposes = ProposesUnion()

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
            engine=MBDB.get_engine(),
            return_type=return_type,
            model=model,
            limit=limit,
        )

    def index(self, dataset: IndexableDataset) -> None:
        """Indexes a data from your data warehouse within Matchbox.

        Args:
            dataset: The dataset to index.
            engine: The SQLAlchemy engine of your data warehouse.
        """
        index_dataset(
            dataset=dataset,
            engine=MBDB.get_engine(),
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
            tgt_col = "data_hash"
        elif hash_type == "cluster":
            Source = Clusters
            tgt_col = "cluster_hash"

        with Session(MBDB.get_engine()) as session:
            data_inner_join = (
                session.query(Source)
                .filter(
                    Source.sha1.in_(
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
            raise MatchboxDBDataError(
                message=(
                    f"Some items don't exist the target table. "
                    f"Did you use {tgt_col} as your ID when deduplicating?"
                ),
                table=Source.__tablename__,
            )

    def get_model_subgraph(self) -> PyDiGraph:
        """Get the full subgraph of a model."""
        return get_model_subgraph(engine=MBDB.get_engine())

    def get_model(self, model: str) -> MatchboxPostgresModel:
        """Get a model from the database.

        Args:
            model: The model to get.
        """
        with Session(MBDB.get_engine()) as session:
            model = session.query(Models).filter_by(name=model).first()
            return MatchboxPostgresModel(model)

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
                engine=MBDB.get_engine(),
            )
        else:
            # Deduper
            deduplicates = table_name_to_uuid(
                schema_table=left, engine=MBDB.get_engine()
            )
            insert_deduper(
                model=model,
                deduplicates=deduplicates,
                description=description,
                engine=MBDB.get_engine(),
            )

    def clear(self, certain: bool = False) -> None:
        """Clears all data from the database.

        Args:
            certain: Whether to clear the database without confirmation.
        """
        MBDB.clear_database()
