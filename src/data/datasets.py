from src.data import utils as du
from src.data.star import Star
from src.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os


class Dataset(object):
    """
    A class to interact with fact and dimension tables in the company
    matching framework.

    Parameters:
        * star_id: the ID of the row in the STAR table
        * star: a star object from which to populate key fields

    Attributes:
        * id: the key of the fact/dim row in the STAR table
        * dim_schema: the schema of the data's dimension table
        * dim_table: the name of the data's dimension table
        * dim_schema_table: the data's dimention table full name
        * fact_schema: the schema of the data's dimension table
        * fact_table: the name of the data's dimension table
        * fact_schema_table: the data's dimention table full name

    Methods:
        * create_dim(unique_fields, overwrite): Drops all data and recreates the
        dimension table using the unique fields specified
        * read_dim(): Returns the dimension table
        * read_fact(): Returns the fact table
        * get_cols(table): Gets the table column names
    """

    def __init__(self, star_id: int, star: object):
        self.star = star
        self.id = star_id

        self.dim_schema_table = star.get(star_id=self.id, response="dim")
        self.dim_schema, self.dim_table = du.get_schema_table_names(
            full_name=self.dim_schema_table, validate=True
        )
        self.dim_table_clean = du.clean_table_name(self.dim_schema_table)

        self.fact_schema_table = star.get(star_id=self.id, response="fact")
        self.fact_schema, self.fact_table = du.get_schema_table_names(
            full_name=self.fact_schema_table, validate=True
        )
        self.fact_table_clean = du.clean_table_name(self.fact_schema_table)

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

        if os.getenv("SCHEMA") != self.dim_schema:
            raise IOError(
                f"""
                Dimension schema is not {os.getenv("SCHEMA")}.
                This table is not controlled by the framework"
            """
            )

        unique_fields = ", ".join(unique_fields)

        if du.check_table_exists(self.dim_schema_table) and not overwrite:
            raise ValueError(
                "Table exists. Set overwrite to True if you want to proceed."
            )

        if overwrite:
            sql = f"drop table if exists {self.dim_schema_table};"
            du.query_nonreturn(sql)

        sql = f"""
            create table {self.dim_schema_table} as (
                select distinct on ({unique_fields})
                    id,
                    {unique_fields}
                from
                    {self.fact_schema_table}
                order by
                    {unique_fields}
            );
        """

        du.query_nonreturn(sql)

    def read_dim(self, dim_select: list = None):
        fields = "*" if dim_select is None else " ,".join(dim_select)
        return du.query(
            f"""
            select
                {fields}
            from
                {self.dim_schema_table};
        """
        )

    def read_fact(self, fact_select: list = None):
        fields = "*" if fact_select is None else " ,".join(fact_select)
        return du.query(
            f"""
            select
                {fields}
            from
                {self.fact_schema_table};
        """
        )

    def get_cols(self, table: str) -> list:
        """
        Returns te columns of either the fact or dimension table.

        Arguments:
            table: one of 'fact' or 'dim'

        Raises:
            ValueError: if table not one of 'fact' or 'dim'

        Returns:
            A list of columns
        """

        if table == "fact":
            out = du.get_table_columns(self.fact_schema_table)
        elif table == "dim":
            out = du.get_table_columns(self.dim_schema_table)
        else:
            raise ValueError("Table much be one of 'fact' or 'dim'")

        return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )
    logger = logging.getLogger(__name__)
    logger.info("Creating dim tables")

    star = Star(schema=os.getenv("SCHEMA"), table=os.getenv("STAR_TABLE"))

    for table in link_pipeline:
        if link_pipeline[table]["fact"] != link_pipeline[table]["dim"]:
            star_id = star.get(
                fact=link_pipeline[table]["fact"],
                dim=link_pipeline[table]["dim"],
                response="id",
            )
            data = Dataset(star_id=star_id, star=star)

            logger.info(f"Creating {data.dim_schema_table}")

            data.create_dim(
                unique_fields=link_pipeline[table]["key_fields"], overwrite=True
            )

            logger.info(f"Written {data.dim_schema_table}")

    logger.info("Finished")
