import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Engine, Table, bindparam, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from cmf.data import utils as du
from cmf.data.data import SourceData
from cmf.data.db import ENGINE
from cmf.data.dedupe import Dedupes
from cmf.data.link import Links
from cmf.data.models import Models, ModelsFrom

logic_logger = logging.getLogger("cmf_logic")


class CMFSourceDataError(Exception):
    "Data doesn't exist in the SourceData table."

    pass


class CMFClusterError(Exception):
    "Data doesn't exist in the Cluster table."

    pass


class ResultsBaseDataclass(BaseModel, ABC):
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

    def _model_to_cmf(
        self, deduplicates: bytes = None, engine: Engine = ENGINE
    ) -> None:
        """Writes the model to the CMF."""
        with Session(engine) as session:
            if deduplicates is None:
                # Linker
                # Construct model SHA1 from parent model SHA1s
                left_sha1 = du.model_name_to_sha1(self.left, engine=engine)
                right_sha1 = du.model_name_to_sha1(self.right, engine=engine)
                model_sha1 = du.list_to_value_ordered_sha1(
                    [self.run_name, left_sha1, right_sha1]
                )
            else:
                # Deduper
                model_sha1 = du.list_to_value_ordered_sha1([self.run_name, self.left])

            model = Models(
                sha1=model_sha1,
                name=self.run_name,
                description=self.description,
                deduplicates=deduplicates,
            )

            session.merge(model)
            session.commit()

            if deduplicates is None:
                # Linker
                # Insert reference to parent models
                models_from_to_insert = [
                    {"parent": model_sha1, "child": left_sha1},
                    {"parent": model_sha1, "child": right_sha1},
                ]

                ins_stmt = insert(ModelsFrom)
                ins_stmt = ins_stmt.on_conflict_do_nothing(
                    index_elements=[
                        ModelsFrom.parent,
                        ModelsFrom.child,
                    ]
                )
                session.execute(ins_stmt, models_from_to_insert)
                session.commit()

    def to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results to the CMF database."""
        if self.left == self.right:
            # Deduper
            # Write model
            logic_logger.info("Registering model")
            self._model_to_cmf(
                deduplicates=du.table_name_to_sha1(self.left, engine=engine),
                engine=engine,
            )

            # Write data
            if self.dataframe.shape[0] == 0:
                logic_logger.info("No deduplication data to insert")
            else:
                logic_logger.info("Writing deduplication data")
                self._deduper_to_cmf(engine=engine)
        else:
            # Linker
            # Write model
            logic_logger.info("Registering model")
            self._model_to_cmf(engine=engine)

            # Write data
            if self.dataframe.shape[0] == 0:
                logic_logger.info("No link data to insert")
            else:
                logic_logger.info("Writing link data")
                self._linker_to_cmf(engine=engine)

        logic_logger.info("Complete!")


class ProbabilityResults(ResultsBaseDataclass):
    """Probabilistic matches produced by linkers and dedupers.

    Inherits the following attributes from ResultsBaseDataclass.

    _expected_fields enforces the shape of the dataframe.

    Args:
        dataframe (pd.DataFrame): the DataFrame holding the results
        run_name (str): the name of the run or experiment
        description (str): a description of the model generating the results
        left (str): the source dataset or source model for the left side of
            the comparison
        right (str): the source dataset or source model for the right side of
            the comparison
    """

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
            self.to_df()
            .filter(["left_id", "right_id"])
            .map(str)
            .merge(
                left_data.assign(**{left_key: lambda d: d[left_key].apply(str)}),
                how="left",
                left_on="left_id",
                right_on=left_key,
            )
            .drop(columns=[left_key])
            .merge(
                right_data.assign(**{right_key: lambda d: d[right_key].apply(str)}),
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

    def _prep_to_cmf(self, df: DataFrame, engine: Engine = ENGINE) -> Dict[str, Any]:
        """Transforms data to dictionary and calculates SHA-1 hash."""
        pre_prep_df = df.copy()
        cols = ["left_id", "right_id"]

        # Verify data is in the CMF
        pre_prep_df[cols] = pre_prep_df[cols].map(bytes)

        for col in cols:
            data_unique = pre_prep_df[col].unique().tolist()

            with Session(engine) as session:
                data_inner_join = (
                    session.query(SourceData)
                    .filter(
                        SourceData.sha1.in_(
                            bindparam(
                                "ins_sha1s",
                                data_unique,
                                expanding=True,
                            )
                        )
                    )
                    .all()
                )
                if len(data_inner_join) != len(data_unique):
                    raise CMFSourceDataError(
                        f"Some items in {col} don't exist in SourceData table. "
                        "Did you use data_sha1 as your ID when deduplicating?"
                    )

        # Transform for insert
        pre_prep_df["sha1"] = du.columns_to_value_ordered_sha1(
            data=self.dataframe, columns=cols
        )
        pre_prep_df = pre_prep_df.rename(
            columns={"left_id": "left", "right_id": "right"}
        )

        return pre_prep_df.to_dict("records")

    def _deduper_to_cmf(self, engine: Engine = ENGINE) -> None:
        """Writes the results of a deduper to the CMF database.

        * Turns data into a left, right, sha1, probability dataframe
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe, engine=engine)

        if not self._validate_tables(engine=engine):
            raise ValueError(
                "Tables not found in database. Check your deduplication sources."
            )

        with Session(engine) as session:
            # Add probabilities
            model = (
                session.query(Models).filter_by(name=self.run_name).first()
            )  # Required to add association class to session

            # Insert any new Dedupe nodes, without probabilities
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

            # Add probabilities to the appropriate model.proposes_dedupes
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
        * Upserts data then queries it back as SQLAlchemy objects
        * Attaches these objects to the model
        """
        probabilities_to_add = self._prep_to_cmf(self.dataframe, engine=engine)

        if not self._validate_sources(engine=engine):
            raise ValueError("Source not found in database. Check your link sources.")

        with Session(engine) as session:
            # Add probabilities
            model = (
                session.query(Models).filter_by(name=self.run_name).first()
            )  # Required to add association class to session

            # Insert any new Link nodes, without probabilities
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

            # Add probabilities to the appropriate model.proposes_links
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


class ClusterResults(ResultsBaseDataclass):
    """Cluster data produced by using to_clusters on ProbabilityResults."""

    pass
