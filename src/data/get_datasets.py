from src.data import utils as du
from src.features.clean_complex import clean_raw_data
from src.config import datasets
from src.locations import DATA_SUBDIR

from os import path, makedirs
import logging
import click
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option(
    "--output_dir",
    required=True,
    type=str,
    help="Directory to save the cleaned parquet files",
)
@click.option(
    "--datasets",
    default=datasets,
    type=dict,
    help="Dataset config dict. Defaults to the one in src.config",
)
@click.option(
    "--sample",
    default=None,
    type=int,
    show_default=True,
    help="Sample size for data, useful for speedy testing",
)
def dw_to_clean_locals(output_dir: str, datasets: dict = datasets, sample: int = None):
    outdir = path.join(DATA_SUBDIR["processed"], output_dir)
    if not path.exists(outdir):
        makedirs(outdir)

    for table in datasets.keys():
        table_name_clean = du.clean_table_name(table)
        outfile = path.join(outdir, f"{table_name_clean}.parquet")

        df = du.get_company_data(
            cols=datasets[table]["cols"],
            dataset=table,
            where=datasets[table]["where"],
            sample=sample,
            # Pandas 2 arrow blocked by MLFlow 1.3 requirement
            # dtype_backend="pyarrow"
        )

        df_clean = clean_raw_data(df)

        df_clean.to_parquet(outfile)


def main():
    """
    Entrypoint
    """
    dw_to_clean_locals()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=du.LOG_FMT)

    load_dotenv(find_dotenv())

    main()
