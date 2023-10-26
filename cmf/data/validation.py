from cmf.data import utils as du

import logging
from dotenv import load_dotenv, find_dotenv
import os
import uuid
import click


class Validation(object):
    """
    A class to interact with the company matching framework's validation
    table. Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.

    Attributes:
        * schema: the validation table's schema name
        * table: the validation table's table name
        * schema_table: the validation table's full name
        * star: an object of class Star that wraps the star table

    Methods:
        * create(overwrite): Drops all data and recreates the validation
        table
        * read(): Returns the probabilities table

    Raises:
        ValueError: if schema or table not specified
    """

    def __init__(self, schema: str, table: str):
        self.schema = schema
        self.table = table
        self.schema_table = f'"{self.schema}"."{self.table}"'

        if None in [self.schema, self.table]:
            raise ValueError(
                f"""
                Schema and table must be specified
                schema: {schema}
                table: {table}
                Have you used the right environment variable?
            """
            )

    def create(self, overwrite: bool = False):
        """
        Creates a new validation table.

        Arguments:
            overwrite: Whether or not to overwrite an existing validation
            table
        """

        exists = du.check_table_exists(self.schema_table)

        if overwrite:
            drop = f"drop table if exists {self.schema_table};"
        elif exists:
            raise ValueError("Table exists and overwrite set to false")
        else:
            drop = ""

        sql = f"""
            {drop}
            create table {self.schema_table} (
                uuid uuid primary key,
                id text not null,
                cluster uuid not null,
                source int not null,
                "user" text not null,
                match bool
            );
        """

        du.query_nonreturn(sql)

    def read(self):
        return du.dataset(self.schema_table)

    def add_validation(self, validation):
        """
        Takes an dataframe of validation statements and adds them to the table.

        Arguments:
            validation: A dataframe containing columns id, cluster, source,
            user and match

        Raises:
            ValueError:
                * If validation doesn't contain columns id, cluster, source,
                user and match

        Returns:
            The dataframe of validation that was added to the table.
        """

        in_cols = set(validation.columns.tolist())
        check_cols = {"id", "cluster", "source", "user", "match"}
        if in_cols != check_cols:
            raise ValueError(
                """
                Data provided does not contain columns id, cluster, source,
                user and match
            """
            )

        validation["uuid"] = [uuid.uuid4() for _ in range(len(validation.index))]

        du.data_workspace_write(
            df=validation, schema=self.schema, table=self.table, if_exists="append"
        )

        return validation


@click.command()
@click.option(
    "--overwrite",
    is_flag=True,
    help="Required to overwrite an existing table.",
)
def create_validation_table(overwrite):
    """
    Entrypoint if running as script
    """
    logger = logging.getLogger(__name__)

    validation = Validation(
        schema=os.getenv("SCHEMA"), table=os.getenv("VALIDATE_TABLE")
    )

    logger.info(f"Creating validation table {validation.schema_table}")

    validation.create(overwrite=overwrite)

    logger.info("Written validation table")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )

    create_validation_table()
