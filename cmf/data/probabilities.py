from cmf.data import utils as du
from cmf.data.table import Table
from cmf.data.mixin import TableMixin, DataFrameMixin

import uuid
from dotenv import load_dotenv, find_dotenv
import os
import click
import logging
from pydantic import computed_field, ConfigDict
from typing import List, Optional
from pandas import DataFrame

load_dotenv(find_dotenv())

load_dotenv(find_dotenv())


class Probabilities(TableMixin):
    """
    A class to interact with the company matching framework's probabilities
    table. Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.
    """

    _db_expected_fields: List[str] = [
        "uuid",
        "link_type",
        "model",
        "source",
        "cluster",
        "id",
        "probability",
    ]

    @computed_field
    def sources(self) -> list:
        """
        Returns a set of the sources currently present in the probabilities table.

        Returns:
            A list of source ints, as appear in the DB table
        """
        sources = du.query(
            "select distinct source from " f"{self.db_table.db_schema_table}"
        )
        return set(sources["source"].tolist())

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
                link_type text not null,
                model text not null,
                source int not null,
                cluster uuid not null,
                id text not null,
                probability float not null
            );
        """

        du.query_nonreturn(sql)

    def add_probabilities(self, probabilities, model: str, overwrite: bool = False):
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

        in_cols = set(probabilities.columns.tolist())
        check_cols = {"cluster", "id", "probability", "source"}
        if len(in_cols - check_cols) != 0:
            raise ValueError(
                """
                Linker.predict() has not produced outputs in an appropriate
                format for the probabilities table.
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

        probabilities["uuid"] = [uuid.uuid4() for _ in range(len(probabilities.index))]
        probabilities["link_type"] = "link"
        probabilities["model"] = model

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: Optional[DataFrame] = None
    db_table: Optional[Table] = None
    run_name: str
    description: str

    _db_expected_fields: Optional[List[str]] = [
        "source",
        "cluster",
        "id",
        "probability",
    ]
    _df_expected_fields: Optional[List[str]] = [
        "source",
        "cluster",
        "id",
        "probability",
    ]

    def to_df(self):
        if self.dataframe is not None:
            return self.dataframe

    def to_cmf(
        self,
        probabilities: Probabilities = Probabilities(
            db_table=Table(
                db_schema=os.getenv("SCHEMA"), db_table=os.getenv("PROBABILITIES_TABLE")
            )
        ),
        overwrite: bool = False,
    ):
        if self.dataframe is not None:
            probabilities.add_probabilities(
                probabilities=self.dataframe, model=self.run_name
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
