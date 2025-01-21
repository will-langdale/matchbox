import logging
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

from matchbox import process, query
from matchbox.client.clean import company_name
from matchbox.client.helpers import cleaner, cleaners, select
from matchbox.client.helpers.selector import Selector
from matchbox.common.sources import Source

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)
TEST_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def test_root_dir() -> Path:
    return TEST_ROOT


@pytest.fixture(scope="session")
def all_companies(test_root_dir: Path) -> DataFrame:
    """
    Raw, correct company data. Uses UUID as ID to replicate Data Workspace.
    1,000 entries.
    """
    df = pd.read_csv(
        Path(test_root_dir, "data", "all_companies.csv"), encoding="utf-8"
    ).reset_index(names="id")
    df["id"] = df["id"].apply(lambda x: UUID(int=x))
    return df


@pytest.fixture(scope="session")
def crn_companies(all_companies: DataFrame) -> DataFrame:
    """
    Company data split into CRN version.

    Company name has Limited, UK and Company added -- our first three stopwords.

    Tests a link/dedupe situation with dirty duplicates.

    3,000 entries, 1,000 unique.
    """
    df_raw = all_companies.filter(["company_name", "crn"])
    df_crn = pd.concat(
        [
            df_raw.assign(company_name=lambda df: df["company_name"] + " Limited"),
            df_raw.assign(company_name=lambda df: df["company_name"] + " UK"),
            df_raw.assign(company_name=lambda df: df["company_name"] + " Company"),
        ]
    )

    df_crn["id"] = range(df_crn.shape[0])
    df_crn = df_crn.filter(["id", "company_name", "crn"])
    df_crn["id"] = df_crn["id"].apply(lambda x: UUID(int=x))
    df_crn = df_crn.convert_dtypes(dtype_backend="pyarrow")

    return df_crn


@pytest.fixture(scope="session")
def duns_companies(all_companies: DataFrame) -> DataFrame:
    """
    Company data split into DUNS version.

    Data is clean.

    Tests a link/dedupe situation with no duplicates.

    500 entries.
    """
    df_duns = (
        all_companies.filter(["company_name", "duns"])
        .sample(n=500, random_state=1618)
        .reset_index(drop=True)
        .convert_dtypes(dtype_backend="pyarrow")
    )
    df_duns["id"] = df_duns.index

    return df_duns


@pytest.fixture(scope="session")
def cdms_companies(all_companies: DataFrame) -> DataFrame:
    """
    Company data split into CDMS version.

    All rows are repeated twice.

    Tests a link/dedupe situation with clean duplicates: edge case in prod,
    but exists in some of the HMRC tables.

    2,000 entries, 1,000 unique.
    """
    df_cdms = pd.DataFrame(
        np.repeat(all_companies.filter(["crn", "cdms"]).values, 2, axis=0)
    )
    df_cdms.columns = ["crn", "cdms"]

    df_cdms.reset_index(names="id", inplace=True)
    df_cdms["id"] = df_cdms["id"].apply(lambda x: UUID(int=x))
    df_cdms = df_cdms.convert_dtypes(dtype_backend="pyarrow")

    return df_cdms


@pytest.fixture(scope="session")
def revolution_inc(
    crn_companies: DataFrame, duns_companies: DataFrame, cdms_companies: DataFrame
) -> dict[str, str]:
    """
    Revolution Inc. as it exists across all three datasets.

    UUIDs are converted to strings to mirror how Matchbox stores them.

    Based on the above fixtures, should return:

    * Three CRNs
    * One DUNS
    * Two CDMS
    """
    crn_ids = crn_companies[
        crn_companies["company_name"].str.contains("Revolution", case=False)
    ]["id"].tolist()

    duns_ids = duns_companies[
        duns_companies["company_name"].str.contains("Revolution", case=False)
    ]["id"].tolist()

    revolution_crn = crn_companies[
        crn_companies["company_name"].str.contains("Revolution", case=False)
    ]["crn"].iloc[0]

    cdms_ids = cdms_companies[cdms_companies["crn"] == revolution_crn]["id"].tolist()

    revolution = {
        "crn": [str(id) for id in crn_ids],
        "duns": [str(id) for id in duns_ids],
        "cdms": [str(id) for id in cdms_ids],
    }

    assert len(revolution.get("crn", [])) == 3
    assert len(revolution.get("duns", [])) == 1
    assert len(revolution.get("cdms", [])) == 2

    return revolution


@pytest.fixture(scope="session")
def winner_inc(
    crn_companies: DataFrame, duns_companies: DataFrame, cdms_companies: DataFrame
) -> dict[str, list[str]]:
    """
    Winner Inc. as it exists across all three datasets.

    UUIDs are converted to strings to mirror how Matchbox stores them.

    Based on the above fixtures, should return:

    * Three CRNs
    * Zero DUNS
    * Two CDMS
    """
    crn_ids = crn_companies[
        crn_companies["company_name"].str.contains("Winner", case=False)
    ]["id"].tolist()

    duns_ids = duns_companies[
        duns_companies["company_name"].str.contains("Winner", case=False)
    ]["id"].tolist()

    winner_crn = crn_companies[
        crn_companies["company_name"].str.contains("Revolution", case=False)
    ]["crn"].iloc[0]

    cdms_ids = cdms_companies[cdms_companies["crn"] == winner_crn]["id"].tolist()

    winner = {
        "crn": [str(id) for id in crn_ids],
        "duns": [str(id) for id in duns_ids],
        "cdms": [str(id) for id in cdms_ids],
    }

    assert len(winner.get("crn", [])) == 3
    assert len(winner.get("duns", [])) == 0
    assert len(winner.get("cdms", [])) == 2

    return winner


@pytest.fixture(scope="function")
def query_clean_crn(
    warehouse_data: list[Source],
) -> tuple[list[Selector], DataFrame]:
    """Fixture for CRN data, and the selector used to get it."""
    # Select
    crn_source = warehouse_data[0]

    select_crn = select(
        {crn_source.address.full_name: ["crn", "company_name"]},
        engine=crn_source.engine,
    )

    crn = query(select_crn, return_type="pandas")

    # Clean
    col_prefix = "test_crn_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_crn = cleaners(cleaner_name)

    crn_cleaned = process(data=crn, pipeline=cleaner_crn)

    return select_crn, crn_cleaned


@pytest.fixture(scope="function")
def query_clean_duns(
    warehouse_data: list[Source],
) -> tuple[list[Selector], DataFrame]:
    """Fixture for DUNS data, and the selector used to get it."""
    # Select
    duns_source = warehouse_data[1]
    select_duns = select(
        {duns_source.address.full_name: ["duns", "company_name"]},
        engine=duns_source.engine,
    )
    duns = query(select_duns, return_type="pandas")

    # Clean
    col_prefix = "test_duns_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_duns = cleaners(cleaner_name)

    duns_cleaned = process(data=duns, pipeline=cleaner_duns)

    return select_duns, duns_cleaned


@pytest.fixture(scope="function")
def query_clean_cdms(
    warehouse_data: list[Source],
) -> tuple[list[Selector], DataFrame]:
    """Fixture for CDMS data, and the selector used to get it."""
    # Select
    cdms_source = warehouse_data[2]
    select_cdms = select(
        {cdms_source.address.full_name: ["crn", "cdms"]},
        engine=cdms_source.engine,
    )
    cdms = query(select_cdms, return_type="pandas")

    # No cleaning needed, see original data
    return select_cdms, cdms


@pytest.fixture(scope="function")
def query_clean_crn_deduped(
    warehouse_data: list[Source],
) -> tuple[list[Selector], DataFrame]:
    """Fixture for cleaned, deduped CRN data, and the selector used to get it."""
    # Select
    crn_source = warehouse_data[0]
    select_crn = select(
        {crn_source.address.full_name: ["crn", "company_name"]},
        engine=crn_source.engine,
    )
    crn = query(select_crn, return_type="pandas")

    # Clean
    col_prefix = "test_crn_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_crn = cleaners(cleaner_name)

    crn_cleaned = process(data=crn, pipeline=cleaner_crn)

    return select_crn, crn_cleaned


@pytest.fixture(scope="function")
def query_clean_duns_deduped(
    warehouse_data: list[Source],
) -> tuple[list[Selector], DataFrame]:
    """Fixture for cleaned, deduped DUNS data, and the selector used to get it."""
    # Select
    duns_source = warehouse_data[1]
    select_duns = select(
        {duns_source.address.full_name: ["duns", "company_name"]},
        engine=duns_source.engine,
    )
    duns = query(select_duns, return_type="pandas")

    # Clean
    col_prefix = "test_duns_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_duns = cleaners(cleaner_name)

    duns_cleaned = process(data=duns, pipeline=cleaner_duns)

    return select_duns, duns_cleaned


@pytest.fixture(scope="function")
def query_clean_cdms_deduped(
    warehouse_data: list[Source],
) -> tuple[list[Selector], DataFrame]:
    """Fixture for cleaned, deduped CDMS data, and the selector used to get it."""
    # Select
    cdms_source = warehouse_data[2]
    select_cdms = select(
        {cdms_source.address.full_name: ["crn", "cdms"]},
        engine=cdms_source.engine,
    )
    cdms = query(select_cdms, return_type="pandas")

    # No cleaning needed, see original data
    return select_cdms, cdms
