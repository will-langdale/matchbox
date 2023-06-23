# from src.data import utils as du
# from src.models import utils as mu

# import logging

# import click


# @click.command()
# @click.option(
#     "--input_name", required=True, type=str, help="Name of input data extract"
# )
# @click.option("--model_version", required=True, type=int, help="Version number")
# @click.option(
#     "--output_name", required=True, type=str, help="Name of output to be saved"
# )
# def predict(input_name, model_version, output_name):
#     logger = logging.getLogger(__name__)

#     logger.info("Loading input data")
#     df = du.load_df("processed", input_name)
#     logger.info("Loading trained model")
#     model = mu.load_model_from_registry(model_version)
#     logger.info("Computing predictions")
#     model.predict(df)


# def main():
#     """
#     Entrypoint
#     """
#     predict()


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     main()
