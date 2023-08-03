from src.data import utils as du
from src.config import tables, stopwords
from src.features.clean_complex import clean_comp_names

# from splink.duckdb.linker import DuckDBLinker

# import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

companieshouse_raw = du.query(
    f"""
        select
            id::uuid as dim_uuid,
            company_name,
            postcode
        from
            {tables['"companieshouse"."companies"']["dim"]};
    """
)

companieshouse_proc = clean_comp_names(
    companieshouse_raw,
    primary_col="company_name",
    secondary_col=None,
    stopwords=stopwords,
)

hmrcexporters_raw = du.query(
    f"""
        select
            dim_uuid,
            company_name,
            postcode
        from
            {tables['"hmrc"."trade__exporters"']["dim"]};
    """
)

hmrcexporters_proc = clean_comp_names(
    hmrcexporters_raw,
    primary_col="company_name",
    secondary_col=None,
    stopwords=stopwords,
)
