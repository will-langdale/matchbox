from src.data import utils as du
from src.models import utils as mu
from src.locations import OUTPUTS_HOME
from src.config import tables, pairs

from splink.duckdb.linker import DuckDBLinker
from splink.comparison import Comparison

import mlflow
import logging
import json
from os import path, makedirs


class ComparisonEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return obj.__name__
        elif isinstance(obj, Comparison):
            return obj.as_dict()
        else:
            return json.JSONEncoder.default(self, obj)


class LinkDatasets(object):
    def __init__(
        self,
        table_l: dict,
        table_r: dict,
        settings: dict,
        pipeline: dict,
    ):
        self.settings = settings
        self.pipeline = pipeline
        self.table_l_settings = table_l
        self.table_r_settings = table_r

        if (table_l["name"], table_r["name"]) in pairs:
            self.pair = pairs[(table_l["name"], table_r["name"])]
        elif (table_r["name"], table_l["name"]) in pairs:
            self.pair = pairs[(table_r["name"], table_l["name"])]

        self.table_l = tables[table_l["name"]]
        self.table_l_alias = du.clean_table_name(table_l["name"])
        self.table_l_select = ", ".join(table_l["select"])

        self.table_r = tables[table_r["name"]]
        self.table_r_alias = du.clean_table_name(table_r["name"])
        self.table_r_select = ", ".join(table_r["select"])

        self.table_l_raw = None
        self.table_r_raw = None

        self.table_l_proc_pipe = table_l["preproc"]
        self.table_r_proc_pipe = table_r["preproc"]

        self.table_l_proc = None
        self.table_r_proc = None

        self.linker = None

        self.predictions = None

    def get_data(self):
        self.table_l_raw = du.query(
            f"""
                select
                    {self.table_l_select}
                from
                    {self.table_l['dim']};
            """
        )
        self.table_r_raw = du.query(
            f"""
                select
                    {self.table_r_select}
                from
                    {self.table_r['dim']};
            """
        )

    def preprocess_data(self):
        curr = self.table_l_raw
        for func in self.table_l_proc_pipe.keys():
            curr = self.table_l_proc_pipe[func]["function"](
                curr, **self.table_l_proc_pipe[func]["arguments"]
            )
        self.table_l_proc = curr

        curr = self.table_r_raw
        for func in self.table_r_proc_pipe.keys():
            curr = self.table_r_proc_pipe[func]["function"](
                curr, **self.table_r_proc_pipe[func]["arguments"]
            )
        self.table_r_proc = curr

    def create_linker(self):
        self.linker = DuckDBLinker(
            input_table_or_tables=[self.table_l_proc, self.table_r_proc],
            settings_dict=self.settings,
            input_table_aliases=[self.table_l_alias, self.table_r_alias],
        )

    def train_linker(self):
        for k in self.pipeline.keys():
            proc_func = getattr(self.linker, k)
            proc_func(**self.pipeline[k])

    def predict(self, **kwargs):
        self.predictions = self.linker.predict(**kwargs)

    def generate_report(self, sample: int, predictions=None) -> dict:
        """
        Generate a dict report that compares a prediction df
        with the evaluation df for the pair. It contains:

            - The difference in match counts
            - The count of matches that agree
            - The count of matches that disagree
            - A sample of agreeing matches
            - A sample of disagreeing matches from the eval
            - A sample of disagreeing matches from the predictions

        Parameters:
            Sample: The sample size of matches to spot check
            Predictions: A dataframe output by the linker. If none,
            will use predictions in self.prefictions

        Returns:
            A dict with the relevant metrics
        """

        if not predictions:
            predictions = self.predictions

        predictions = (
            predictions.as_pandas_dataframe()
            .sort_values(by=["match_probability"], ascending=False)
            .drop_duplicates(subset=["id_l", "id_r"], keep="first")
            .merge(
                self.table_l_raw.add_suffix("_l"),
                how="left",
                left_on=["id_l"],
                right_on=["id_l"],
                suffixes=("", "_remove"),
            )
            .merge(
                self.table_r_raw.add_suffix("_r"),
                how="left",
                left_on=["id_r"],
                right_on=["id_r"],
                suffixes=("", "_remove"),
            )
            .filter(regex="^((?!remove).)*$")
        )

        existing = (
            du.dataset(self.pair["eval"])
            .merge(
                self.table_l_raw.add_suffix("_l"),
                how="left",
                left_on=["id_l"],
                right_on=["id_l"],
                suffixes=("", "_remove"),
            )
            .merge(
                self.table_r_raw.add_suffix("_r"),
                how="left",
                left_on=["id_r"],
                right_on=["id_r"],
                suffixes=("", "_remove"),
            )
            .filter(regex="^((?!remove).)*$")
        )

        agree = predictions.merge(
            existing, how="inner", on=["id_l", "id_r"], suffixes=("_pred", "_exist")
        ).filter(regex="id|pred|exist|match_probability|score")

        cols = agree.columns.tolist()
        cols.insert(0, cols.pop(cols.index("score")))
        cols.insert(0, cols.pop(cols.index("match_probability")))
        cols.insert(0, cols.pop(cols.index("id_r")))
        cols.insert(0, cols.pop(cols.index("id_l")))

        agree = agree.reindex(columns=cols)

        disagree = predictions.merge(
            existing,
            how="outer",
            on=["id_l", "id_r"],
            suffixes=("_pred", "_exist"),
            indicator=True,
        ).filter(regex="id|pred|exist|match_probability|score|_merge")

        cols = disagree.columns.tolist()
        cols.insert(0, cols.pop(cols.index("score")))
        cols.insert(0, cols.pop(cols.index("match_probability")))
        cols.insert(0, cols.pop(cols.index("id_r")))
        cols.insert(0, cols.pop(cols.index("id_l")))

        disagree = disagree.reindex(columns=cols)

        prediction_only = (
            disagree[(disagree._merge == "left_only")]
            .drop("_merge", axis=1)
            .filter(regex="^((?!_exist).)*$")
            .filter(regex="^((?!score).)*$")
            .sort_values(by=["match_probability"], ascending=False)
        )
        existing_only = (
            disagree[(disagree._merge == "right_only")]
            .drop("_merge", axis=1)
            .filter(regex="^((?!_pred).)*$")
            .filter(regex="^((?!match_probability).)*$")
            .sort_values(by=["score"], ascending=False)
        )

        res = {
            "eval_matches": existing.shape[0],
            "pred_matches": predictions.shape[0],
            "both_eval_and_pred": agree.shape[0],
            "eval_only": existing_only.shape[0],
            "pred_only": prediction_only.shape[0],
            "both_eval_and_pred_sample": (
                agree.sample(sample).to_dict(orient="records")
            ),
            "eval_only_sample": (
                existing_only.sample(sample).to_dict(orient="records")
            ),
            "pred_only_sample": (
                prediction_only.sample(sample).to_dict(orient="records")
            ),
        }

        return res

    def run_mlflow_experiment(self, run_name, description):
        """
        Runs the whole pipeline:

        - Data acquisition
        - Data cleaning
        - Linker creation
        - Linker training
        - Evaluation

        Logs the lot to MLflow.
        """
        logging.basicConfig(
            level=logging.INFO,
            format=du.LOG_FMT,
        )
        logger = logging.getLogger(__name__)
        logger.info("Running pipeline as MLflow experiment")

        with mu.mlflow_run(
            run_name=run_name,
            experiment_name=self.pair["experiment"],
            description=description,
            dev_mode=True,
        ):
            # Data processing

            logger.info("Acquiring raw data")
            self.get_data()

            logger.info("Preprocessing data")
            self.preprocess_data()

            mlflow.log_param(
                key="blocking_rules",
                value=len(self.settings["blocking_rules_to_generate_predictions"]),
            )
            mlflow.log_param(key="comparisons", value=len(self.settings["comparisons"]))
            mlflow.log_param(key="preprocessing_l", value=len(self.table_l_proc_pipe))
            mlflow.log_param(key="preprocessing_r", value=len(self.table_r_proc_pipe))

            # Linker

            logger.info("Creating linker and running training pipeline")
            self.create_linker()
            self.train_linker()

            outdir = path.join(OUTPUTS_HOME, self.pair["experiment"])
            if not path.exists(outdir):
                makedirs(outdir)

            model_file_path = path.join(outdir, f"{self.pair['experiment']}.json")
            self.linker.save_model_to_json(out_path=model_file_path, overwrite=True)
            mlflow.log_artifact(model_file_path, "model")

            pipeline_path = path.join(outdir, "pipeline.json")
            pipeline_json = json.dumps(self.pipeline, indent=4, cls=ComparisonEncoder)
            with open(pipeline_path, "w") as f:
                f.write(pipeline_json)
            mlflow.log_artifact(pipeline_path, "config")

            preproc_l_path = path.join(outdir, f"{self.table_l_alias}_settings.json")
            preproc_l_json = json.dumps(
                self.table_l_settings, indent=4, cls=ComparisonEncoder
            )
            with open(preproc_l_path, "w") as f:
                f.write(preproc_l_json)
            mlflow.log_artifact(preproc_l_path, "config")

            preproc_r_path = path.join(outdir, f"{self.table_r_alias}_settings.json")
            preproc_r_json = json.dumps(
                self.table_r_settings, indent=4, cls=ComparisonEncoder
            )
            with open(preproc_r_path, "w") as f:
                f.write(preproc_r_json)
            mlflow.log_artifact(preproc_r_path, "config")

            # Outcome

            logger.info("Generating predictions")
            self.predict(threshold_match_probability=0.7)

            logger.info("Creating report")
            report = self.generate_report(sample=10)

            mlflow.log_param(key="threshold_match_probability", value=0.7)

            mlflow.log_metric("eval_matches", report["eval_matches"])
            mlflow.log_metric("pred_matches", report["pred_matches"])
            mlflow.log_metric("both_eval_and_pred", report["both_eval_and_pred"])
            mlflow.log_metric("eval_only", report["eval_only"])
            mlflow.log_metric("pred_only", report["pred_only"])

        logger.info("Done!")
