from __future__ import annotations

from cmf.data import utils as du
from cmf.data.table import Table
from cmf.data.mixin import TableMixin, DataFrameMixin

import uuid
from dotenv import load_dotenv, find_dotenv
import os
import click
import logging
from pydantic import computed_field, field_validator
from typing import List, Optional, TYPE_CHECKING
from pandas import DataFrame
import pandas as pd

if TYPE_CHECKING:
    from cmf.data.db import CMFDB


class Probabilities(TableMixin):
    """
    A class to interact with the company matching framework's probabilities
    table. Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.
    """

    _expected_fields: List[str] = [
        "uuid",
        "model",
        "target",
        "target_id",
        "source",
        "source_id",
        "probability",
    ]

    @computed_field
    def sources(self) -> list:
        """
        Returns a set of the sources currently present in the probabilities table.

        Returns:
            A list of source strings
        """
        sources = du.query(
            "select distinct source from " f"{self.db_table.db_schema_table}"
        )
        return set(sources["source"].tolist())

    @computed_field
    def targets(self) -> list:
        """
        Returns a set of the targets currently present in the probabilities table.

        Returns:
            A list of target strings
        """
        sources = du.query(
            "select distinct target from " f"{self.db_table.db_schema_table}"
        )
        return set(sources["target"].tolist())

    @computed_field
    def models(self) -> list:
        """
        Returns a set of the models currently present in the probabilities table.

        Returns:
            A list of model strings
        """
        models = du.query(
            "select distinct model from " f"{self.db_table.db_schema_table}"
        )
        return set(models["model"].tolist())

    def create(self, overwrite: bool):
        """
        Creates a new probabilities table.

        Arguments:
            overwrite: Whether or not to overwrite an existing probabilities
            table
        """

        if overwrite:
            drop = f"drop table if exists {self.db_table.db_schema_table};"
        elif self.db_table.exists:
            raise ValueError("Table exists and overwrite set to false")
        else:
            drop = ""

        sql = f"""
            {drop}
            create table {self.db_table.db_schema_table} (
                uuid uuid primary key,
                model text not null,
                target text not null,
                target_id text not null,
                source text not null,
                source_id text not null,
                probability float not null
            );
        """

        du.query_nonreturn(sql)

    def add_probabilities(
        self, probabilities: DataFrame, model: str, overwrite: bool = False
    ):
        """
        Takes an output from Linker.predict() and adds it to the probabilities
        table.

        Arguments:
            probabilities: A data frame produced by Linker.predict(). Should
            contain columns cluster, id, source and probability.
            model: A unique string that represents this model
            overwrite: Whether to overwrite existing probabilities inserted by
            this model

        Raises:
            ValueError:
                * If probabilities doesn't contain columns cluster, model, id
                source and probability
                * If probabilities doesn't contain values between 0 and 1

        Returns:
            The dataframe of probabilities that were added to the table.
        """

        in_cols = sorted(probabilities.columns.tolist())
        check_cols = sorted(
            ["target", "target_id", "source", "source_id", "probability"]
        )
        if in_cols != check_cols:
            raise ValueError(
                f"""
                Probabilities not in an appropriate format for the table.
                Expected: {check_cols}
                Got: {in_cols}
            """
            )
        max_prob = max(probabilities.probability)
        min_prob = min(probabilities.probability)
        if max_prob > 1 or min_prob < 0:
            raise ValueError(
                f"""
                Probability column should contain valid probabilities.
                Max: {max_prob}
                Min: {min_prob}
            """
            )

        pd.options.mode.chained_assignment = None
        probabilities["model"] = model
        probabilities["uuid"] = [uuid.uuid4() for _ in range(len(probabilities.index))]
        pd.options.mode.chained_assignment = "warn"

        if model in self.models and overwrite is not True:
            raise ValueError(f"{model} exists in table and overwrite is False")
        elif model in self.models and overwrite is True:
            sql = f"""
                delete from
                    {self.db_table.db_schema_table}
                where
                    model = '{model}'
            """
            du.query_nonreturn(sql)

        du.data_workspace_write(
            df=probabilities,
            schema=self.db_table.db_schema,
            table=self.db_table.db_table,
            if_exists="append",
        )

        return probabilities


class ProbabilityResults(TableMixin, DataFrameMixin):
    dataframe: Optional[DataFrame] = None
    db_table: Optional[Table] = None
    run_name: str
    description: str
    target: str
    source: str

    _expected_fields: List[str] = [
        "target_id",
        "source_id",
        "probability",
    ]

    @field_validator("target", "source")
    @classmethod
    def validate_source(cls, v: str):
        db_table = Table.from_schema_table(full_name=v, validate=True)
        assert db_table.exists
        return v

    def to_df(self) -> DataFrame:
        if self.dataframe is not None:
            res = self.dataframe
            res["target"] = self.target
            res["source"] = self.target
            res = res[["target", "target_id", "source", "source_id", "probability"]]
            return res

    def to_cmf(
        self,
        cmf_conn: CMFDB,
        overwrite: bool = False,
    ) -> None:
        if self.dataframe is not None:
            cmf_conn.probabilities.add_probabilities(
                probabilities=self.to_df(), model=self.run_name, overwrite=overwrite
            )


@click.command()
@click.option(
    "--overwrite",
    is_flag=True,
    help="Required to overwrite an existing table.",
)
def create_probabilities_table(overwrite):
    """
    Entrypoint if running as script
    """
    logger = logging.getLogger(__name__)

    probabilities = Probabilities(
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("PROBABILITIES_TABLE")
        )
    )

    logger.info(
        "Creating probabilities table " f"{probabilities.db_table.db_schema_table}"
    )

    probabilities.create(overwrite=overwrite)

    logger.info("Written probabilities table")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )

    create_probabilities_table()
