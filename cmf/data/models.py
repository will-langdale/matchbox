from cmf.data import utils as du

from typing import List, Optional
from pydantic import BaseModel, computed_field, field_validator
from sqlalchemy.sql import text as sql_text
from sqlalchemy.exc import MultipleResultsFound
from pandas import DataFrame


class Table(BaseModel):
    db_schema: str
    db_table: str

    @field_validator("db_schema", "db_table")
    @classmethod
    def unquote(cls, v: str) -> str:
        return v.replace('"', "")

    @computed_field
    def db_schema_table(self) -> str:
        return f"{self.db_schema}.{self.db_table}"

    @computed_field
    def exists(self) -> bool:
        sql = f"""
            select exists (
                select from information_schema.tables
                where table_schema = '{self.db_schema}'
                and table_name = '{self.db_table}'
            );
        """

        with du.sql_engine.connect() as connection:
            res = connection.execute(sql_text(sql))

            try:
                exists = res.scalar()
            except MultipleResultsFound:
                raise ValueError(
                    "Multiple results found. Table or schema name unclear."
                )

        return exists

    @computed_field
    def empty(self) -> bool:
        sql = f"""
            select
                count(*)
            from (
                select
                    1
                from
                    {self.db_schema_table}
                limit 1
            ) as t;
        """
        if self.exists:
            with du.sql_engine.connect() as connection:
                res = connection.execute(sql_text(sql))
                val = res.fetchone()
            return not bool(val[0])
        else:
            return True

    @computed_field
    def db_fields(self) -> Optional[List[str]]:
        if self.exists:
            with du.sql_engine.connect() as connection:
                res = connection.execute(
                    sql_text(f"select * from {self.db_schema_table} limit 0")
                )
            return list(res._metadata.keys)
        else:
            return None

    def read(self, select: list = None, sample: float = None) -> DataFrame:
        """
        Returns the table as pandas dataframe.

        Arguments:
            select: [optional] a list of columns to select. Aliasing
            and casting permitted
            sample:[optional] the percentage sample to return. Used to
            speed up debugging of downstream processes
        """
        fields = "*" if select is None else " ,".join(select)
        if sample is not None:
            sample_clause = f"tablesample system ({sample})"
        else:
            sample_clause = ""

        return du.query(
            f"""
            select
                {fields}
            from
                {self.db_schema_table} {sample_clause};
        """
        )


if __name__ == "__main__":
    # y = Table(db_schema="_user_eaf4fd9a", db_table="cm_test")
    # print(y)
    # x = Table(db_schema="_user_eaf4fd9a", db_table="cm_star")
    # print(x)
    pass
