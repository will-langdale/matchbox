from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Engine, bindparam, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from cmf.data import utils as du
from cmf.data.db import ENGINE
from cmf.data.dedupe import Dedupes
from cmf.data.link import Links
from cmf.data.models import Models, ModelsFrom

if TYPE_CHECKING:
    from pandas import DataFrame
    from sqlalchemy import Engine, Table


class Results(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: DataFrame
    run_name: str
    description: str
    left: str
    right: str

    _expected_fields: List[str]

    @model_validator(mode="after")
    def _check_dataframe(self) -> Table:
        """Verifies the table contains the expected fields."""
        table_fields = sorted(self.dataframe.columns)
        expected_fields = sorted(self._expected_fields)

        if table_fields != expected_fields:
            raise ValueError(f"Expected {expected_fields}. \n" f"Found {table_fields}.")

        return self

    def _validate_tables(self, engine: Engine = ENGINE) -> bool:
        """Validates existence of left and right tables in wider database."""
        try:
            db_left_schema, db_left_table = du.get_schema_table_names(
                full_name=self.left, validate=True
            )
            _ = du.string_to_table(
                db_schema=db_left_schema, db_table=db_left_table, engine=engine
            )
            db_right_schema, db_right_table = du.get_schema_table_names(
                full_name=self.right, validate=True
            )
            _ = du.string_to_table(
                db_schema=db_right_schema, db_table=db_right_table, engine=engine
            )
        except NoSuchTableError:
            return False

        return True

    def _validate_sources(self, engine: Engine = ENGINE) -> bool:
        """Validates existence of left and right tables in CMF database."""
        stmt_left = select(Models.name).where(Models.name == self.left)
        stmt_right = select(Models.name).where(Models.name == self.right)

        with Session(engine) as session:
            res_left = session.execute(stmt_left).scalar()
        with Session(engine) as session:
            res_right = session.execute(stmt_right).scalar()

        if res_left is not None and res_right is not None:
            return True
        else:
            return False

    @abstractmethod
    def inspect_with_source(self) -> DataFrame:
        """Enriches the results with the source data."""
        return

    @abstractmethod
    def to_df(self) -> DataFrame:
        """Returns the results as a DataFrame."""
        return

    @abstractmethod
    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database."""
        return

    @abstractmethod
    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database."""
        return

    def to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results to the CMF database."""
        if self.left == self.right:
            # Deduper
            self._deduper_to_cmf(engine=engine)
        else:
            # Linker
            self._linker_to_cmf(engine=engine)


class ProbabilityResults(Results):
    """Probabilistic matches produced by linkers and dedupers."""

    _expected_fields: List[str] = [
        "left_id",
        "right_id",
        "probability",
    ]

    def inspect_with_source(
        self, left_data: DataFrame, left_key: str, right_data: DataFrame, right_key: str
    ) -> DataFrame:
        """Enriches the results with the source data."""
        df = (
            self.dataframe.filter(["left_id", "right_id"])
            .map(bytes)
            .merge(
                left_data.assign(**{left_key: lambda d: d[left_key].apply(bytes)}),
                how="left",
                left_on="left_id",
                right_on=left_key,
            )
            .drop(columns=[left_key])
            .merge(
                right_data.assign(**{right_key: lambda d: d[right_key].apply(bytes)}),
                how="left",
                left_on="right_id",
                right_on=right_key,
            )
            .drop(columns=[right_key])
        )

        return df

    def to_df(self) -> DataFrame:
        """Returns the results as a DataFrame."""
        df = self.dataframe.assign(
            left=self.left, right=self.right, model=self.run_name
        )[["model", "left", "left_id", "right", "right_id", "probability"]]

        return df

    def _prep_to_cmf(self, df) -> Dict[str, Any]:
        """Transforms data to dictionary and calculates SHA-1 hash."""
        probabilities_to_add = (
            df.assign(
                sha1=du.columns_to_value_ordered_sha1(
                    data=self.dataframe, columns=["left_id", "right_id"]
                )
            )
            .rename(columns={"left_id": "left", "right_id": "right"})
            .to_dict("records")
        )

        return probabilities_to_add

    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Adds the model
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe)

        if not self._validate_tables():
            raise ValueError(
                "Tables not found in database. Check your deduplication sources."
            )

        with Session(engine) as session:
            # Add model
            deduper_sha1 = du.list_to_value_ordered_sha1([self.run_name, self.left])
            model = Models(
                sha1=deduper_sha1,
                name=self.run_name,
                description=self.description,
                deduplicates=du.table_name_to_sha1(self.left, engine=engine),
            )

            session.merge(model)
            session.commit()

            # Add probabilities
            model = (
                session.query(Models).filter_by(name=self.run_name).first()
            )  # Required to add association class to session

            ins_dd_stmt = insert(Dedupes)
            ins_dd_stmt = ins_dd_stmt.on_conflict_do_nothing(
                index_elements=[Dedupes.sha1]
            )
            _ = session.execute(
                ins_dd_stmt,
                [
                    {k: v for k, v in dd.items() if k != "probability"}
                    for dd in probabilities_to_add
                ],
            )
            session.commit()

            to_insert = (
                session.query(Dedupes)
                .filter(
                    Dedupes.sha1.in_(
                        bindparam(
                            "ins_sha1s",
                            [dd["sha1"] for dd in probabilities_to_add],
                            expanding=True,
                        )
                    )
                )
                .all()
            )

            model.proposes_dedupes.clear()

            for dd, r in zip(to_insert, probabilities_to_add):
                model.proposes_dedupes[dd] = r["probability"]

            session.commit()

    def _linker_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a linker to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Adds the model
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe)

        if not self._validate_sources(engine):
            raise ValueError("Source not found in database. Check your link sources.")

        with Session(engine) as session:
            # Add model
            left_sha1 = du.model_name_to_sha1(self.left, engine=engine)
            right_sha1 = du.model_name_to_sha1(self.right, engine=engine)
            link_model_sha1 = du.list_to_value_ordered_sha1(
                [self.run_name, left_sha1, right_sha1]
            )

            model = Models(
                sha1=link_model_sha1,
                name=self.run_name,
                description=self.description,
                deduplicates=None,
            )
            models_from_to_insert = [
                {"parent": link_model_sha1, "child": left_sha1},
                {"parent": link_model_sha1, "child": right_sha1},
            ]

            session.merge(model)
            session.commit()

            ins_stmt = insert(ModelsFrom)
            ins_stmt = ins_stmt.on_conflict_do_nothing(
                index_elements=[
                    ModelsFrom.parent,
                    ModelsFrom.child,
                ]
            )
            session.execute(ins_stmt, models_from_to_insert)
            session.commit()

            # Add probabilities
            model = (
                session.query(Models).filter_by(name=self.run_name).first()
            )  # Required to add association class to session

            ins_li_stmt = insert(Links)
            ins_li_stmt = ins_li_stmt.on_conflict_do_nothing(
                index_elements=[Links.sha1]
            )
            _ = session.execute(
                ins_li_stmt,
                [
                    {k: v for k, v in li.items() if k != "probability"}
                    for li in probabilities_to_add
                ],
            )
            session.commit()

            to_insert = (
                session.query(Links)
                .filter(
                    Links.sha1.in_(
                        bindparam(
                            "ins_sha1s",
                            [li["sha1"] for li in probabilities_to_add],
                            expanding=True,
                        )
                    )
                )
                .all()
            )

            model.proposes_links.clear()

            for dd, r in zip(to_insert, probabilities_to_add):
                model.proposes_links[dd] = r["probability"]

            session.commit()


class ClusterResults(Results):
    """Cluster data produced by using to_clusters on ProbabilityResults."""

    pass
