# from src.data import utils as du
# from src.models import utils as mu

# import logging

# import click
# import sklearn
# from sklearn.ensemble import RandomForestRegressor
# import mlflow


# @click.command()
# @click.option(
#     "--input_name", required=True, type=str, help="Name of input data extract"
# )
# @click.option("--run_name", required=True, type=str, help="Name of model run")
# @click.option(
#     "--seed",
#     default=None,
#     type=int,
#     help="Seed used for non-deterministic components of the model",
# )
# @click.option(
#     "--dev",
#     is_flag=True,
#     help="""Dev runs allow to run this script with a dirty git repo""",
# )
# @click.option(
#     "--description",
#     default=None,
#     type=str,
#     help="Description of the training run",
# )
# def train_model(input_name, run_name, seed, dev, description):
#     logger = logging.getLogger(__name__)

#     logger.info("Loading training data")
#     df = du.load_df("processed", input_name)

#     logger.info("Training model")

#     with mu.mlflow_run(
#         run_name=run_name,
#         description=description,
#         dev_mode=dev,
#     ):
#         params = {"n_estimators": 5, "random_state": seed}
#         mlflow.log_params(params)

#         model = RandomForestRegressor(**params)

#         # ...train model...
#         # ...evaluate model and log metrics and artifacts...

#         prediction_dependencies = {
#             "scikit-learn": sklearn.__version__,
#         }

#         logger.info("Done. Serialising model and logging it to MLFlow.")
#         all_model_steps = (model.predict,)
#         mu.log_python_pipeline(all_model_steps, prediction_dependencies)


# def main():
#     """
#     Entrypoint
#     """
#     train_model()


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     main()
