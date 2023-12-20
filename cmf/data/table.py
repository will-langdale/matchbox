from typing import List, Optional

from pandas import DataFrame
from pydantic import BaseModel, computed_field, field_validator
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.sql import text as sql_text

from cmf.data import utils as du


class Table(BaseModel):
    db_schema: str
    db_table: str

    @field_validator("db_schema", "db_table")
    @classmethod
    def unquote(cls, v: str) -> str:
        return v.replace('"', "")

    @classmethod
    def from_schema_table(cls, full_name: str, validate: bool = True) -> "Table":
        db_schema, db_table = du.get_schema_table_names(
            full_name=full_name, validate=validate
        )
        return cls(db_schema=db_schema, db_table=db_table)

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

    def read(
        self, select: Optional[List] = None, sample: Optional[float] = None
    ) -> DataFrame:
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
