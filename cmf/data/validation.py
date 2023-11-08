from cmf.data import utils as du
from cmf.data.table import Table
from cmf.data.mixin import TableMixin

import logging
from dotenv import load_dotenv, find_dotenv
import os
import uuid
import click
from typing import List


class Validation(TableMixin):
    """
    A class to interact with the company matching framework's validation
    table. Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.
    """

    _db_expected_fields: List[str] = [
        "uuid",
        "id",
        "cluster",
        "source",
        "user",
        "match",
    ]

    def create(self, overwrite: bool = False):
        """
        Creates a new validation table.

        Arguments:
            overwrite: Whether or not to overwrite an existing validation
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
                id text not null,
                cluster uuid not null,
                source int not null,
                "user" text not null,
                match bool
            );
        """

        du.query_nonreturn(sql)

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
            df=validation,
            schema=self.db_table.db_schema,
            table=self.db_table.db_table,
            if_exists="append",
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
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("VALIDATE_TABLE")
        )
    )

    logger.info("Creating validation table " f"{validation.db_table.db_schema_table}")

    validation.create(overwrite=overwrite)

    logger.info("Written validation table")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )

    create_validation_table()
