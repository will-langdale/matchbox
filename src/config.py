import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

link_pipeline = {
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
        "n": 1,
        "experiment": "cm_companies-house",
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
        "n": 2,
        "experiment": "cm_data-hub-companies",
    },
    '"hmrc"."trade__exporters"': {
        "fact": '"hmrc"."trade__exporters"',
        "key_fields": ["company_name", "address", "postcode"],
        "dim": f'"{os.getenv("SCHEMA")}"."hmrc_trade__exporters__dim"',
        "n": 3,
        "experiment": "cm_hmrc-trade-exporters",
    },
    '"dit"."export_wins__wins_dataset"': {
        "fact": '"dit"."export_wins__wins_dataset"',
        "key_fields": ["cdms_reference", "company_name"],
        "dim": f'"{os.getenv("SCHEMA")}"."export_wins__wins_dataset__dim"',
        "n": 3,
        "experiment": "cm_export-wins",
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
