# from src.data import utils as data_utils

# import logging

# from sklearn.model_selection import train_test_split
# from dotenv import find_dotenv, load_dotenv
# import click


# def download_data_workspace_extract(output_name, split_test_set):
#     """
#     Download from data workspace into raw data folder
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Fetching raw data from data workspace")

#     out = data_utils.data_workspace_ds('dataset_name')
#     if not split_test_set:
#         data_utils.persist_df(out, "raw", output_name)
#     else:
#         train_out, test_out = train_test_split(out)
#         data_utils.persist_df(train_out, "raw", output_name + "__train")
#         data_utils.persist_df(test_out, "raw", output_name + "__test")


# @click.command()
# @click.option(
#     "--output_name", required=True, type=str, help="Name of dataset to be saved"
# )
# @click.option(
#     "--split_test_set",
#     is_flag=True,
#     help="Whether to create separate test set",
# )
# def main(output_name, split_test_set):
#     """
#     Entrypoint
#     """
#     download_data_workspace_extract(output_name, split_test_set)


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     load_dotenv(find_dotenv())

#     main()
