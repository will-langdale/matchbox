from cmf.data import utils as du
from cmf.data.models import Table
from cmf.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os
from pydantic import BaseModel, computed_field
import hashlib


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

        if os.getenv("SCHEMA") != self.db_dim.schema:
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )
    logger = logging.getLogger(__name__)
    logger.info("Creating dim tables")

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

    logger.info("Finished")
