import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

tables = {
    '"companieshouse"."companies"': {
        "fact": '"companieshouse"."companies"',
        "key_fields": [
            "id",
            "company_number",
            "company_name",
            "address_line_1",
            "postcode",
            "company_status",
        ],
        "dim": '"companieshouse"."companies"',
        "match_v1": '"companieshouse"."companies__match_ids"',
    },
    '"dit"."data_hub__companies"': {
        "fact": '"dit"."data_hub__companies"',
        "key_fields": [
            "id",
            "name",
            "company_number",
            "duns_number",
            "cdms_reference_code",
            "address_1",
            "address_postcode",
            "archived",
        ],
        "dim": '"dit"."data_hub__companies"',
        "match_v1": '"dit"."data_hub__companies__match_ids"',
    },
    '"hmrc"."trade__exporters"': {
        "fact": '"hmrc"."trade__exporters"',
        "key_fields": ["company_name", "address", "postcode"],
        "dim": f'"{os.getenv("SCHEMA")}"."hmrc_trade__exporters__dim"',
        "match_v1": '"hmrc"."trade__exporters__match_ids"',
    },
    '"dit"."export_wins__wins_dataset"': {
        "fact": '"dit"."export_wins__wins_dataset"',
        "key_fields": ["cdms_reference", "company_name"],
        "dim": f'"{os.getenv("SCHEMA")}"."export_wins__wins_dataset__dim"',
        "match_v1": '"dit"."export_wins__wins_dataset__match_ids"',
    },
}

pairs = {
    ('"companieshouse"."companies"', '"dit"."data_hub__companies"'): {
        "experiment": "cm_ch_x_dh",
        "eval": f'"{os.getenv("SCHEMA")}"."ch_x_dh__eval"',
    },
    ('"dit"."data_hub__companies"', '"dit"."export_wins__wins_dataset"'): {
        "experiment": "cm_dh_x_ew",
        "eval": f'"{os.getenv("SCHEMA")}"."dh_x_ew__eval"',
    },
    ('"companieshouse"."companies"', '"hmrc"."trade__exporters"'): {
        "experiment": "cm_ch_x_exp",
        "eval": f'"{os.getenv("SCHEMA")}"."ch_x_exp__eval"',
    },
}

stopwords = [
    "limited",
    "uk",
    "company",
    "international",
    "group",
    "of",
    "the",
    "inc",
    "and",
    "plc",
    "corporation",
    "llp",
    "pvt",
    "gmbh",
    "u k",
    "pte",
    "usa",
    "bank",
    "b v",
    "bv",
]


# Keeping for later right now

# import splink.duckdb.comparison_library as cl
# import splink.duckdb.comparison_template_library as ctl

# settings = {
#     "link_type": "link_and_dedupe",
#     "retain_matching_columns": False,
#     "retain_intermediate_calculation_columns": False,
#     "blocking_rules_to_generate_predictions": [
#         """
#             ((l.comp_num_clean = r.comp_num_clean))
#             and (
#                 l.comp_num_clean <> ''
#                 and r.comp_num_clean <> ''
#             )
#         """,
#         """
#             (l.name_unusual_tokens = r.name_unusual_tokens)
#             and (
#                 l.name_unusual_tokens <> ''
#                 and r.name_unusual_tokens <> ''
#             )
#         """,
#         # """
#         #     (l.name_unusual_tokens_first5 = r.name_unusual_tokens_first5)
#         #     and (
#         #         length(l.name_unusual_tokens_first5) = 5
#         #         and length(r.name_unusual_tokens_first5) = 5
#         #     )
#         # """,
#         # """
#         #     (l.name_unusual_tokens_last5 = r.name_unusual_tokens_last5)
#         #     and (
#         #         length(l.name_unusual_tokens_last5) = 5
#         #         and length(r.name_unusual_tokens_last5) = 5
#         #     )
#         # """,
#         """
#             (l.secondary_name_unusual_tokens = r.secondary_name_unusual_tokens)
#             and (
#                 l.secondary_name_unusual_tokens <> ''
#                 and r.secondary_name_unusual_tokens <> ''
#             )
#         """,
#         """
#             (l.secondary_name_unusual_tokens = r.name_unusual_tokens)
#             and (
#                 l.secondary_name_unusual_tokens <> ''
#                 and r.name_unusual_tokens <> ''
#             )
#         """,
#         """
#             (r.secondary_name_unusual_tokens = l.name_unusual_tokens)
#             and (
#                 r.secondary_name_unusual_tokens <> ''
#                 and l.name_unusual_tokens <> ''
#             )
#         """,
#         # My attempt to reduce computation on first/last 5 while retaining info
#         # """
#         #     (l.name_sig_first5 = r.name_sig_first5)
#         #     and (
#         #         length(l.name_sig_first5) = 5
#         #         and length(r.name_sig_first5) = 5
#         #     )
#         # """,
#         # """
#         #     (l.name_sig_last5 = r.name_sig_last5)
#         #     and (
#         #         length(l.name_sig_last5) = 5
#         #         and length(r.name_sig_last5) = 5
#         #     )
#         # """,
#         # TODO: blocking rule on first token name_unusual_tokens?
#     ],
#     # for comp_num_clean: there may be some typos
#     # for name_unusual_tokens:
#     may be some typos but also a lot of 'duplicate ...' gumph
#     #   hence two different similarity levels
#     "comparisons": [
#         cl.jaro_winkler_at_thresholds(
#             "comp_num_clean", [0.75], term_frequency_adjustments=True
#         ),
#         cl.jaro_winkler_at_thresholds(
#             "name_unusual_tokens", [0.9, 0.6], term_frequency_adjustments=True
#         ),
#         # match on the first alphabetic chars in the postcode
#         # TODO: change to geographic similarity measure?
#         # cl.exact_match("postcode_area", 2),
#         ctl.postcode_comparison("postcode")
#         # TODO: try a comparison on main name and secondary name - how to do?
#         # Add first name to secondary name array
#         # Use ct.array_intersect_at_sizes?
#         # TODO: secondary_name_unusual_tokens comparison
#         # cl.array_intersect_at_sizes("alternative_company_names", [1])
#     ],
# }

# datasets = {
#     '"companieshouse"."companies"': {
#         "cols": """
#             id::text as unique_id,
#             company_number,
#             company_name,
#             array_remove(
#                 array[
#                     previous_name_1,
#                     previous_name_2,
#                     previous_name_3,
#                     previous_name_4,
#                     previous_name_5,
#                     previous_name_6
#                 ],
#                 ''
#             ) as secondary_names,
#             postcode
#         """,
#         "where": "",
#     },
#     '"dit"."data_hub__companies"': {
#         "cols": """
#             id::text as unique_id,
#             company_number,
#             name as company_name,
#             string_to_array(btrim(trading_names, '[]'), ', ') as secondary_names,
#             address_postcode as postcode
#         """,
#         "where": "archived is False",
#     },
#     '"hmrc"."trade__exporters"': {
#         "cols": """
#             id::text as unique_id,
#             null as company_number,
#             company_name,
#             null as secondary_names,
#             postcode
#         """,
#         "where": "",
#     },
#     '"dit"."export_wins__wins_dataset"': {
#         "cols": """
#             id::text as unique_id,
#             cdms_reference as company_number,
#             company_name,
#             null as secondary_names,
#             null as postcode
#         """,
#         "where": "",
#     },
# }
