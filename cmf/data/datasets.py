from cmf.data import utils as du
from cmf.data.table import Table
from cmf.data.mixin import TableMixin
from cmf.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os
from pydantic import BaseModel, computed_field
import hashlib
from typing import List, Dict
import click
import pandas as pd
from sqlalchemy.sql import text as sql_text


class Datasets(TableMixin):
    """
    A table with metadata about each of the datasets currently in the
    framework. Previously the STAR table.
    """

    _expected_fields: List[str] = ["id", "fact", "dim"]

    @computed_field
    def db_datasets(self) -> Dict[str, Table]:
        sql = f"""
            select id, fact, dim
            from {self.db_table.db_schema_table};
        """

        db_datasets = {}

        with du.sql_engine.connect() as connection:
            res = connection.execute(sql_text(sql))
            for row in res:
                fact_schema, fact_table = du.get_schema_table_names(
                    row.fact, validate=True
                )
                dim_schema, dim_table = du.get_schema_table_names(
                    row.dim, validate=True
                )
                fact = Table(db_schema=fact_schema, db_table=fact_table)
                dim = Table(db_schema=dim_schema, db_table=dim_table)
                dataset = Dataset(db_fact=fact, db_dim=dim)

                db_datasets[dataset.db_dim.db_schema_table] = dataset
                db_datasets[dataset.db_fact.db_schema_table] = dataset
                db_datasets[dataset.db_id] = dataset

        return db_datasets

    def create(self, link_pipeline: dict, overwrite: bool):
        """
        Creates a lookup table from a link_pipeline settings object.

        Arguments:
            link_pipeline: The source link_pipeline settings object
            overwrite: Whether to overwrite the target if it exists

        Raises:
            ValueError
                * If the table exists and overwrite is False
                * If the hash function fails to produce unique keys
        """

        if_exists = "replace" if overwrite else "fail"

        if self.db_table.exists and not overwrite:
            raise ValueError(
                """
                Table exists. Set overwrite to True if you want to proceed.
            """
            )

        out = {"id": [], "fact": [], "dim": []}
        for table in link_pipeline:
            fact = link_pipeline[table]["fact"]
            dim = link_pipeline[table]["dim"]
            hash_hex = hashlib.sha256(
                f"{fact}{dim}".encode(encoding="UTF-8", errors="strict")
            ).hexdigest()
            # We need a hash that:
            # * Is an integer
            # * Is stable
            # * Is unique for the amount of dims we'll ever see
            # I therefore manipulate the hex to 0-65535 to fit in a 16-bit unsigned
            # int field
            hash_int = int(hash_hex, 16) % 65536

            out["id"].append(hash_int)
            out["fact"].append(fact)
            out["dim"].append(dim)

        db = pd.DataFrame.from_dict(out)

        if len(db) != len(db.id.unique()):
            raise ValueError(
                """
                Hash function has failed to produce unique keys. Change it.
            """
            )

        du.data_workspace_write(
            df=db,
            schema=self.db_table.db_schema,
            table=self.db_table.db_table,
            if_exists=if_exists,
        )


class Dataset(BaseModel):
    """
    A class to interact with fact and dimension tables in the company
    matching framework.
    """

    db_dim: Table
    db_fact: Table

    @computed_field
    def db_id(self) -> str:
        to_encode = f"""\
            {self.db_dim.db_schema_table}\
            {self.db_fact.db_schema_table}\
        """
        hash_hex = hashlib.sha256(
            to_encode.encode(encoding="UTF-8", errors="strict")
        ).hexdigest()
        # We need a hash that:
        # * Is an integer
        # * Is stable
        # * Is unique for the amount of dims we'll ever see
        # I therefore manipulate the hex to 0-65535 to fit in a 16-bit unsigned
        # int field
        hash_int = int(hash_hex, 16) % 65536
        return hash_int

    def create_dim(self, unique_fields: list, overwrite: bool):
        """
        Takes a fact table and a list of its unique fields, then writes the naive
        deduplicated version to the specified dimension table.

        Reuses the business identifier from the fact table.

        Arguments:
            unique_fields: A list of fields to derive unique rows from
            overwrite: Whether to overwrite the target if it exists

        Raises:
            IOError: Refuses to make the dimension table unless the schema matches
            the framework's, indicating the framework controls the dimension.
            ValueError: If the table exists and overwrite isn't set to True

        Returns:
            Nothing
        """

        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)

        if os.getenv("SCHEMA") != self.db_dim.db_schema:
            raise IOError(
                f"""
                Dimension schema is not {os.getenv("SCHEMA")}.
                This table is not controlled by the framework"
            """
            )

        unique_fields = ", ".join(unique_fields)

        if self.db_dim.exists and not overwrite:
            raise ValueError(
                "Table exists. Set overwrite to True if you want to proceed."
            )

        if overwrite:
            sql = f"drop table if exists {self.db_dim.db_schema_table};"
            du.query_nonreturn(sql)

        sql = f"""
            create table {self.db_dim.db_schema_table} as (
                select distinct on ({unique_fields})
                    id,
                    {unique_fields}
                from
                    {self.db_fact.db_schema_table}
                order by
                    {unique_fields}
            );
        """

        du.query_nonreturn(sql)


@click.group()
def create_tables():
    pass


@create_tables.command()
def datasets():
    """
    Create Datasets metadata table (previously STAR)
    """
    logger.info(
        f'Creating Datasets table "{os.getenv("SCHEMA")}".'
        f'"{os.getenv("DATASETS_TABLE")}"'
    )

    datasets = Datasets(
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("DATASETS_TABLE")
        )
    )

    datasets.create(
        link_pipeline=link_pipeline,
        overwrite=True,
    )

    logger.info("Written Datasets table")


@create_tables.command()
def dimensions():
    """
    Create individual Dataset dimension tables
    """
    logger.info("Creating dimension tables")

    for table in link_pipeline:
        if link_pipeline[table]["fact"] != link_pipeline[table]["dim"]:

            dim_schema, dim_table = du.get_schema_table_names(
                full_name=link_pipeline[table]["dim"], validate=True
            )
            fact_schema, fact_table = du.get_schema_table_names(
                full_name=link_pipeline[table]["fact"], validate=True
            )

            data = Dataset(
                db_dim=Table(db_schema=dim_schema, db_table=dim_table),
                db_fact=Table(db_schema=fact_schema, db_table=fact_table),
            )

            logger.info(f"Creating {data.db_dim.db_schema_table}")

            data.create_dim(
                unique_fields=link_pipeline[table]["key_fields"], overwrite=True
            )

            logger.info(f"Written {data.db_dim.db_schema_table}")

    logger.info("Written all dimension tables")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )
    logger = logging.getLogger(__name__)

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    create_tables()
