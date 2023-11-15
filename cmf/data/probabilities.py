from __future__ import annotations

from cmf.data import utils as du
from cmf.data.table import Table
from cmf.data.mixin import TableMixin, DataFrameMixin

from dotenv import load_dotenv, find_dotenv
import os
import click
import logging
from pydantic import computed_field, field_validator
from typing import List, Optional, TYPE_CHECKING
from pandas import DataFrame

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
                uuid uuid primary key default uuid_generate_v4(),
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
        self, probabilities: ProbabilityResults, overwrite: bool = False
    ) -> None:
        """
        Takes an output from a linker or deduper and adds it to the
        probabilities table.

        Arguments:
            probabilities: A ProbabilityResults produced by a deduper or linker
            overwrite: Whether to overwrite existing probabilities inserted by
            this model
        """

        if not isinstance(probabilities, ProbabilityResults):
            raise ValueError("Probabilities must be of class ProbabilityResults")

        if probabilities.run_name in self.models and overwrite is not True:
            raise ValueError(
                f"{probabilities.run_name} exists in table and overwrite" " is False"
            )
        elif probabilities.run_name in self.models and overwrite is True:
            sql = f"""
                delete from
                    {self.db_table.db_schema_table}
                where
                    model = '{probabilities.run_name}'
            """
            du.query_nonreturn(sql)

        du.data_workspace_write(
            df=probabilities.to_df(),
            schema=self.db_table.db_schema,
            table=self.db_table.db_table,
            if_exists="append",
        )


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
            return (
                self.dataframe.assign(
                    target=self.target, source=self.source, model=self.run_name
                )
            )[["model", "target", "target_id", "source", "source_id", "probability"]]

    def to_cmf(
        self,
        cmf_conn: CMFDB,
        overwrite: bool = False,
    ) -> None:
        if self.dataframe is not None:
            cmf_conn.probabilities.add_probabilities(
                probabilities=self, overwrite=overwrite
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
