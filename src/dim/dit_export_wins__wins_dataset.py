from src.dim.make_dim import MakeDim
from src.config import tables

# Quick count
# 54747 without group
# 29306 with

if __name__ == "__main__":
    dit_export_wins__wins_dataset = MakeDim(
        unique_fields=["cdms_reference", "company_name"],
        fact_table=tables['"dit"."export_wins__wins_dataset"']["fact"],
        dim_table=tables['"dit"."export_wins__wins_dataset"']["dim"],
    )

    dit_export_wins__wins_dataset.make_dim_table(overwrite=True)
