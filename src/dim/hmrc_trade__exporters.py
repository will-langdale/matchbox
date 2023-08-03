from src.dim.make_dim import MakeDim
from src.config import tables

import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Quick count
# 3418561 without group
# 254243 with

if __name__ == "__main__":
    hmrc_trade__exporters = MakeDim(
        unique_fields=["company_name", "address", "postcode"],
        fact_table=tables['"hmrc"."trade__exporters"']["fact"],
        dim_table=f'"{os.getenv("SCHEMA")}"."hmrc_trade__exporters__dim"',
    )

    hmrc_trade__exporters.make_dim_table(overwrite=True)
