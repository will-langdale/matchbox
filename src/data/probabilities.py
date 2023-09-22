from src.data import utils as du
from src.data.star import Star

import uuid
from dotenv import load_dotenv, find_dotenv
import os
import click
import logging


class Probabilities(object):
    """
    A class to interact with the company matching framework's probabilities
    table. Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.

    Attributes:
        * schema: the probabilities table's schema name
        * table: the probabilities table's table name
        * schema_table: the probabilities table's full name
        * star: an object of class Star that wraps the star table

    Methods:
        * create(overwrite): Drops all data and recreates the probabilities
        table
        * read(): Returns the probabilities table
        * add_probabilities(lookup): Add new entries to the probabilities
        table
    """

    def __init__(self, schema: str, table: str, star: Star):
        self.schema = schema
        self.table = table
        self.schema_table = f'"{self.schema}"."{self.table}"'
        self.star = star

    def create(self, overwrite: bool):
        """
        Creates a new probabilities table.

        Arguments:
            overwrite: Whether or not to overwrite an existing probabilities
            table
        """

        if overwrite:
            drop = f"drop table if exists {self.schema_table};"
            exist_clause = ""
        else:
            drop = ""
            exist_clause = "if not exists"

        sql = f"""
            {drop}
            create table {exist_clause} {self.schema_table} (
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

    def read(self):
        return du.dataset(self.schema_table)

    def get_sources(self) -> list:
        """
        Returns a list of the sources currently present in the probabilities table.

        Raises:
            KeyError: if table currently contains no sources

        Returns:
            A list of source ints, as appear in the star table
        """
        sources = du.query(f"select distinct source from {self.schema_table}")

        if len(sources.index) == 0:
            raise KeyError("Probabilities table currently contains no sources")

        return sources["source"].tolist()

    def get_models(self) -> list:
        """
        Returns a list of the models currently present in the probabilities table.

        Returns:
            A list of model strings
        """
        models = du.query(f"select distinct model from {self.schema_table}")

        return models["model"].tolist()

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
                * If the model has already

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

        current_models = self.get_models()

        if model in current_models and overwrite is not True:
            raise ValueError(f"{model} exists in table and overwrite is False")
        elif model in current_models and overwrite is True:
            sql = f"""
                delete from
                    {self.schema_table}
                where
                    model = {model}
            """
            du.query_nonreturn(sql)

        du.data_workspace_write(
            df=probabilities, schema=self.schema, table=self.table, if_exists="append"
        )

        return probabilities


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

    star = Star(schema=os.getenv("SCHEMA"), table=os.getenv("STAR_TABLE"))

    probabilities = Probabilities(
        schema=os.getenv("SCHEMA"), table=os.getenv("PROBABILITIES_TABLE"), star=star
    )

    logger.info(f"Creating probabilities table {probabilities.schema_table}")

    probabilities.create(overwrite=overwrite)

    logger.info("Written probabilities table")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )

    create_probabilities_table()
