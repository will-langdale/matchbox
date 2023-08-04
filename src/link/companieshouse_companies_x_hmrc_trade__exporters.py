from src.data import utils as du
from src.models import utils as mu
from src.config import tables, stopwords
from src.features.clean_complex import clean_comp_names

from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl

# import os
import logging
import mlflow
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Clean data

companieshouse_raw = du.query(
    f"""
        select
            id as dim_uuid,
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

# Define Linker, including blocking rules and comparisons to use

settings = {
    "link_type": "link_only",
    "unique_id_column_name": "dim_uuid",
    "retain_matching_columns": False,
    "retain_intermediate_calculation_columns": False,
    "blocking_rules_to_generate_predictions": [
        """
            (l.name_unusual_tokens = r.name_unusual_tokens)
            and (
                l.name_unusual_tokens <> ''
                and r.name_unusual_tokens <> ''
            )
        """,
        """
            (l.postcode = r.postcode)
            and (
                l.postcode <> ''
                and r.postcode <> ''
            )
        """,
        """
            (l.name_unusual_tokens_first5 = r.name_unusual_tokens_first5)
            and (
                length(l.name_unusual_tokens_first5) = 5
                and length(r.name_unusual_tokens_first5) = 5
            )
        """,
        """
            (l.name_unusual_tokens_last5 = r.name_unusual_tokens_last5)
            and (
                length(l.name_unusual_tokens_last5) = 5
                and length(r.name_unusual_tokens_last5) = 5
            )
        """,
    ],
    "comparisons": [
        {
            "output_column_name": "Company name",
            "comparison_description": "Jaro-Winkler of the unusual company name tokens",
            "comparison_levels": [
                {
                    cl.jaro_winkler_at_thresholds(
                        "name_unusual_tokens",
                        [0.9, 0.6],
                        term_frequency_adjustments=True,
                    )
                }
            ],
        },
        {
            "output_column_name": "Company postcode",
            "comparison_description": "Splink's postcode comparison built-in",
            "comparison_levels": [{ctl.postcode_comparison("postcode")}],
        },
    ],
}

linker = DuckDBLinker(
    input_table_or_tables=[companieshouse_proc, hmrcexporters_proc],
    settings_dict=settings,
    input_table_aliases=['"companieshouse"."companies"', '"hmrc"."trade__exporters"'],
)

# Train linker using MLflow

logger = logging.getLogger(__name__)

# with mu.mlflow_run(run_name=run_name, description=description, dev_mode=dev):
with mu.mlflow_run():

    logger.info("Estimating model parameters")

    # ...that random records match
    linker.estimate_probability_two_random_records_match(
        "l.name_unusual_tokens = r.name_unusual_tokens",
        recall=0.7,
    )
    mlflow.log_param(key="random_match_recall", value=0.7)

    # ...u
    linker.estimate_u_using_random_sampling(max_pairs=1e7)
    mlflow.log_param(key="u_sampling_max_pairs", value=1e7)

    # ...m
    linker.estimate_m_from_label_column("comp_num_clean")
    m_by_name_and_postcode_area = """
        l.name_unusual_tokens = r.name_unusual_tokens
        and l.postcode_area = r.postcode_area
    """
    linker.estimate_parameters_using_expectation_maximisation(
        m_by_name_and_postcode_area
    )


# Predict and write to lookup


# Compare output with existing company matching opinions
