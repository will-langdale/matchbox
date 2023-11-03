from cmf.data import utils as du
from cmf.data.models import Table
from cmf.data.datasets import Dataset
from cmf.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os
from hashlib import sha256
import pandas as pd
from pydantic import BaseModel, computed_field, field_validator
from sqlalchemy.sql import text as sql_text
from typing import Dict


class DB(BaseModel):
    """
    The entrypoint to the whole Company Matching Framework database.
    """

    db_table: Table

    @field_validator("db_table")
    @classmethod
    def check_db(cls, v: Table) -> Table:
        db_fields = {"id", "fact", "dim"}
        assert set(v.db_fields) == db_fields
        return v

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
        Creates a STAR lookup table from a link_pipeline settings object.

        Arguments:
            link_pipeline: The source link_pipeline settings object
            overwrite: Whether to overwrite the target if it exists

        Raises:
            ValueError
                * If the table exists and overwrite is False
                * If the hash function fails to produce unique keys
        """

        if_exists = "replace" if overwrite else "fail"

        if self.table.exists and not overwrite:
            raise ValueError(
                """
                Table exists. Set overwrite to True if you want to proceed.
            """
            )

        out = {"id": [], "fact": [], "dim": []}
        for table in link_pipeline:
            fact = link_pipeline[table]["fact"]
            dim = link_pipeline[table]["dim"]
            hash_hex = sha256(
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

        star = pd.DataFrame.from_dict(out)

        if len(star) != len(star.id.unique()):
            raise ValueError(
                """
                Hash function has failed to produce unique keys. Change it.
            """
            )

        du.data_workspace_write(
            df=star,
            schema=self.table.db_schema,
            table=self.table.db_table,
            if_exists=if_exists,
        )


if __name__ == "__main__":
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f'Creating STAR table "{os.getenv("SCHEMA")}"."{os.getenv("STAR_TABLE")}"'
    )

    db = DB(
        db_table=Table(db_schema=os.getenv("SCHEMA"), db_table=os.getenv("STAR_TABLE"))
    )

    db.create(
        link_pipeline=link_pipeline,
        overwrite=True,
    )

    logger.info("Written STAR table")
