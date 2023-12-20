import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl

from cmf.config import stopwords
from cmf.features.clean_complex import clean_comp_names
from cmf.link.make_link import LinkDatasets

settings = {
    "link_type": "link_only",
    "unique_id_column_name": "id",
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
    ],
    "comparisons": [
        cl.jaro_winkler_at_thresholds(
            "name_unusual_tokens", [0.9, 0.6], term_frequency_adjustments=True
        ),
        ctl.postcode_comparison("postcode"),
    ],
}

pipeline = {
    "estimate_probability_two_random_records_match": {
        "function": "estimate_probability_two_random_records_match",
        "arguments": {
            "deterministic_matching_rules": """
                l.name_unusual_tokens = r.name_unusual_tokens
            """,
            "recall": 0.7,
        },
    },
    "estimate_u_using_random_sampling": {
        "function": "estimate_u_using_random_sampling",
        "arguments": {"max_pairs": 1e6},
    },
    "estimate_parameters_using_expectation_maximisation": {
        "function": "estimate_parameters_using_expectation_maximisation",
        "arguments": {
            "blocking_rule": """
                l.name_unusual_tokens = r.name_unusual_tokens
            """
        },
    },
}

ch_settings = {
    "name": '"companieshouse"."companies"',
    "select": ["id::text", "company_name", "postcode"],
    "preproc": {
        "clean_comp_names": {
            "function": clean_comp_names,
            "arguments": {
                "primary_col": "company_name",
                "secondary_col": None,
                "stopwords": stopwords,
            },
        }
    },
}

exp_settings = {
    "name": '"hmrc"."trade__exporters"',
    "select": ["id::text", "company_name", "postcode"],
    "preproc": {
        "clean_comp_names": {
            "function": clean_comp_names,
            "arguments": {
                "primary_col": "company_name",
                "secondary_col": None,
                "stopwords": stopwords,
            },
        }
    },
}

if __name__ == "__main__":
    ch_x_exp = LinkDatasets(
        table_l=ch_settings, table_r=exp_settings, settings=settings, pipeline=pipeline
    )

    ch_x_exp.run_mlflow_experiment(
        run_name="Basic linkage",
        description="""
            - Unusual tokens in name
            - Preset postcode distances
            - Eval vs existing service
        """,
        threshold_match_probability=0.7,
    )
