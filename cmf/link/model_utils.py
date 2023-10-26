import cmf
from cmf.locations import REFERENCES_HOME, PROJECT_DIR, MODELS_HOME

from git import Repo
import mlflow.pyfunc
import cloudpickle
import jinja2
import markdown

from sys import version_info
from os import environ, path
import re
import contextlib
import traceback

DEFAULT_ARTIFACT_PATH = "model"
DEFAULT_EXPERIMENT_NAME = "company_matching"
DEFAULT_MODEL_NAME = "company_matching"


@contextlib.contextmanager
def mlflow_run(
    run_name,
    experiment_name=DEFAULT_EXPERIMENT_NAME,
    description=None,
    dev_mode=False,
    tags=None,
):
    """
    A custom context manager that prepares the logging of a model run
    through MLFlow, sending some standard metadata first.
    If something run in this context raises an error, the MLFlow run
    will be marked as "failed"

    Params:
        run_name: name of run on the tracking server
        experiment_name: a string that will override the default experiment name
            corresponding to the repo name. Consider changing the value of
            DEFAULT_EXPERIMENT_NAME in this file instead of setting
            this parameter manually all the time
        description: description of the run
        dev_mode: if False, will try to record git commit hash of the repo
        tags: Dict[str, Any] of tags to set for the run

    Raises:
        ValueError:
            - when the repo has uncommitted changes and dev_mode is False
            - when the default or overridden experiment name does not exist
    """
    if dev_mode:
        git_hash = None
    else:
        repo = Repo(PROJECT_DIR)
        if repo.is_dirty():
            raise ValueError(
                """Cannot produce git commit hash: repo has uncommitted changes.
            Use dev mode instead"""
            )
        git_hash = repo.head.object.hexsha

    _init_username()

    if mlflow.get_experiment_by_name(experiment_name) is None:
        raise ValueError(f"Experiment {experiment_name} does not exist on MLFlow")

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name, description=description)
    if tags is None:
        tags = {}
    mlflow.set_tags({"git_hash": git_hash, "dev": dev_mode, **tags})

    status = "FINISHED"
    try:
        yield
    except Exception as e:
        status = "FAILED"
        mlflow.log_text(traceback.format_exc(), "error.txt")
        raise e
    finally:
        mlflow.end_run(status=status)


def _init_username():
    """
    Change default username (e.g. theia) to something like name_surname
    for the purpose of signing model runs before they are logged
    """
    dw_creds = environ["DATABASE_DSN__datasets_1"]
    username_pattern = r"user=user_(\w+)_trade_gov_uk_\w+ "
    match = re.findall(username_pattern, dw_creds)[0]

    environ["LOGNAME"] = match


def log_python_pipeline(steps, predict_dependencies, signature=None):
    """
    Send a complete model pipeline to the tracking server

    Parameters:
        steps: an iterable of functions that will be applied in sequence
        predict_dependencies: dictionary where keys are pip packages, and values
            are versions, representing the dependencies of the model at inference
            time
        signature: input/output schemas, of type
            mlflow.models.signature.ModelSignature, for more info:
            https://www.mlflow.org/docs/latest/models.html#model-signature-and-input-example
    """
    # The following line forces cloudpickle to package this whole module
    # In this way, the environment loading the MLFlow model will not need
    # the `src` package as a dependency
    # See https://github.com/cloudpipe/cloudpickle/pull/417
    cloudpickle.register_pickle_by_value(cmf)

    formatted_dependencies = [f"{k}=={v}" for k, v in predict_dependencies.items()]

    class ModelPipeline(mlflow.pyfunc.PythonModel):
        """
        This class is an MLFlow wrapper for an iterable of custom Python
        functions which will be applied in sequence and treated as a single
        model, saved in the MLFlow format. For more info see
        https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models
        """

        def __init__(self, steps):
            self.steps = steps

        def predict(self, context, raw_input):
            output = raw_input
            # Apply each step in sequence
            for s in self.steps:
                output = s(output)

            return output

    PYTHON_VERSION = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python={}".format(PYTHON_VERSION),
            "pip",
            {
                "pip": [
                    "mlflow",
                    "cloudpickle=={}".format(cloudpickle.__version__),
                    *formatted_dependencies,
                ],
            },
        ],
        "name": "mlflow_model_env",
    }
    pipeline = ModelPipeline(steps)

    mlflow.pyfunc.log_model(
        artifact_path=DEFAULT_ARTIFACT_PATH,
        python_model=pipeline,
        conda_env=conda_env,
        signature=signature,
    )


def load_model_from_registry(version, model_name=DEFAULT_MODEL_NAME):
    """
    Fetches a model version from the MLFlow registry

    Parameters:
        version: the model version number to be fetched
        model_name: string to override the DEFAULT_MODEL_NAME corresponding
            to the repo name

    Returns:
        The fetched model, of type mlflow.pyfunc.PythonModel,
            with a predict(...) method
    """
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{version}",
        dst_path=MODELS_HOME,
    )

    return model


def load_model_from_run(run_id, artifact_path=DEFAULT_ARTIFACT_PATH):
    """
    Fetches an MLFlow model artifact from a run on the tracking server

    Parameters:
        run_id: the run ID owning the model artifacts
        artifact_path: the path where the MLFlow model artifacts are stored

    Returns:
        The fetched model, of type mlflow.pyfunc.PythonModel,
            with a predict(...) method
    """
    model = mlflow.pyfunc.load_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        dst_path=MODELS_HOME,
    )

    return model


def render_methodology(template_path=None):
    """
    Reads references/METHODOLOGY.md and converts to HTML

    Parameters:
        template_path: optional Jinja template that will wrap rendered HTML

    Returns:
        string containing HTML of the rendered methodology document
    """
    with open(path.join(REFERENCES_HOME, "METHODOLOGY.md"), "r") as mf:
        methodology = mf.read()

    rendered = markdown.markdown(methodology)

    if template_path is not None:
        with open(template_path) as f:
            template = jinja2.Template(f.read())
            rendered = template.render(methodology=rendered)

    return rendered


def latest_successful_run_id(experiment: str = DEFAULT_EXPERIMENT_NAME):
    """
    Gets the last successful run in an experiment.
    """
    experiment_runs = mlflow.search_runs(
        experiment_ids=mlflow.get_experiment_by_name(experiment).experiment_id
    )

    return experiment_runs[
        (experiment_runs.end_time == max(experiment_runs.end_time))
        & (experiment_runs.status == "FINISHED")
    ].run_id[0]
