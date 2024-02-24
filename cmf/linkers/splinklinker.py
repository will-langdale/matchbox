import inspect
from typing import Any, Dict, List, Type

from pandas import DataFrame
from pydantic import BaseModel, Field, model_validator
from splink.duckdb.linker import DuckDBLinker
from splink.linker import Linker as SplinkLibLinkerClass

from cmf.linkers.make_linker import Linker, LinkerSettings


class SplinkLinkerFunction(BaseModel):
    """A method of x to train the linker."""

    function: str
    arguments: Dict[str, Any]

    @model_validator(mode="after")
    def validate_function_and_arguments(self) -> "SplinkLinkerFunction":
        if not hasattr(SplinkLibLinkerClass, self.function):
            raise ValueError(
                f"Function {self.function} not found as method of Splink Linker class"
            )

        linker_func = getattr(SplinkLibLinkerClass, self.function)
        linker_func_param_set = set(inspect.signature(linker_func).parameters.keys())
        func_param_set = set(self.arguments.keys())

        if not func_param_set <= linker_func_param_set:
            raise ValueError(
                f"Function {self.function} given incorrect arguments: "
                f"{func_param_set.difference(linker_func_param_set)}. "
                "Consider referring back to the Splink documentation: "
                "https://moj-analytical-services.github.io/splink/linker.html"
            )

        return self


class SplinkSettings(LinkerSettings):
    """
    A data class to enforce the Splink linker's settings dictionary shape.
    """

    linker_class: Type[SplinkLibLinkerClass] = Field(
        default=DuckDBLinker,
        description="""
            A Splink Linker class. Defaults to DuckDBLinker, and has only been tested
            with this class.
        """,
        validate_default=True,
    )
    linker_training: List[SplinkLinkerFunction] = Field(
        description="""
            An ordered list of dictionaries keyed to functions, with values of the
            function's argument dictionary, to be run against the Linker.

            Example:
            
                >>> linker_training=[
                ...     {
                ...         "function": "estimate_probability_two_random_records_match",
                ...         "arguments": {
                ...             "deterministic_matching_rules": \"""
                ...                 l.company_name = r.company_name
                ...             \""",
                ...             "recall": 0.7,
                ...         },
                ...     },
                ...     {
                ...         "function": "estimate_u_using_random_sampling",
                ...         "arguments": {"max_pairs": 1e6},
                ...     }
                ... ]
            
        """
    )
    linker_settings: Dict = Field(
        description="""
            A valid settings dictionary for a Splink linker.

            See Splink's documentation for a full description of available settings.
            https://moj-analytical-services.github.io/splink/settings_dict_guide.html

            The following settings are enforced by the Company Matching Framework:

            * link_type is set to "link_only"
            * unique_id_column_name is set to the value of left_id and right_id, which
                must match

            Example:

                >>> from splink.duckdb.blocking_rule_library import block_on
                ... import splink.duckdb.comparison_library as cl
                ... import splink.duckdb.comparison_template_library as ctl
                ... 
                ... splink_settings={
                ...     "retain_matching_columns": False,
                ...     "retain_intermediate_calculation_columns": False,
                ...     "blocking_rules_to_generate_predictions": [
                ...         \"""
                ...             (l.company_name = r.company_name)
                ...             and (
                ...                 l.name_unusual_tokens <> ''
                ...                 and r.name_unusual_tokens <> ''
                ...             )
                ...         \""",
                ...         \"""
                ...             (l.postcode = r.postcode)
                ...             and (
                ...                 l.postcode <> ''
                ...                 and r.postcode <> ''
                ...             )
                ...         \""",
                ...     ],
                ...     "comparisons": [
                ...         cl.jaro_winkler_at_thresholds(
                ...             "company_name", 
                ...             [0.9, 0.6], 
                ...             term_frequency_adjustments=True
                ...         ),
                ...         ctl.postcode_comparison("postcode"),
                ...     ],
                ... }
            
        """
    )
    threshold: float = Field(
        description="""
            The probability above which matches will be kept. 
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
        """,
        ge=0,
        le=1,
    )

    @model_validator(mode="after")
    def check_ids_match(self) -> "SplinkSettings":
        l_id = self.left_id
        r_id = self.right_id
        if l_id is not None and r_id is not None and l_id != r_id:
            raise ValueError(
                "Left and right ID do not match. "
                "left_id and right_id must match in a Splink linker."
            )
        return self

    @model_validator(mode="after")
    def add_enforced_settings(self) -> "SplinkSettings":
        enforced_settings = {
            "link_type": "link_only",
            "unique_id_column_name": self.left_id,
        }
        for k, v in enforced_settings.items():
            self.linker_settings[k] = v
        return self


class SplinkLinker(Linker):
    settings: SplinkSettings

    _linker: SplinkLibLinkerClass = None

    @classmethod
    def from_settings(
        cls,
        left_id: str,
        right_id: str,
        linker_class: SplinkLibLinkerClass,
        linker_training: List[Dict[str, Any]],
        linker_settings: Dict[str, Any],
        threshold: float,
    ) -> "SplinkLinker":
        settings = SplinkSettings(
            left_id=left_id,
            right_id=right_id,
            linker_class=linker_class,
            linker_training=[SplinkLinkerFunction(**func) for func in linker_training],
            linker_settings=linker_settings,
            threshold=threshold,
        )
        return cls(settings=settings)

    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        if (set(left.columns) != set(right.columns)) or not left.dtypes.equals(
            right.dtypes
        ):
            raise ValueError(
                "SplinkLinker requires input datasets to be conformant, meaning they "
                "share the same column names and data formats."
            )

        self._linker = self.settings.linker_class(
            input_table_or_tables=[left, right],
            input_table_aliases=["l", "r"],
            settings_dict=self.settings.linker_settings,
        )
        for func in self.linker_training.keys():
            proc_func = getattr(
                self._linker, self.settings.linker_training[func]["function"]
            )
            proc_func(**self.settings.linker_training[func]["arguments"])

    def link(self, left: DataFrame = None, right: DataFrame = None) -> DataFrame:
        if left is not None or right is not None:
            raise ValueError(
                "Left and right data is declared in .prepare() for SplinkLinker"
            )

        res = self._linker.predict(threshold_match_probability=self.settings.threshold)

        return (
            res.as_pandas_dataframe()
            .rename(
                {
                    f"{self.settings.left_id}_l": "left_id",
                    f"{self.settings.right_id}_r": "right_id",
                    "match_probability": "probability",
                }
            )
            .filter(["left_id", "right_id", "probability"])
        )


if __name__ == "__main__":
    import splink.duckdb.comparison_library as cl
    import splink.duckdb.comparison_template_library as ctl
    from splink.duckdb.blocking_rule_library import block_on

    linker_training = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": """
                    l.company_name = r.company_name
                """,
                "recall": 0.7,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    linker_settings = {
        "retain_matching_columns": False,
        "retain_intermediate_calculation_columns": False,
        "blocking_rules_to_generate_predictions": [
            block_on("company_name"),
            block_on("postcode"),
        ],
        "comparisons": [
            cl.jaro_winkler_at_thresholds(
                "company_name", [0.9, 0.6], term_frequency_adjustments=True
            ),
            ctl.postcode_comparison("postcode"),
        ],
    }

    splink_linker = SplinkLinker.from_settings(
        left_id="cluster_sha1",
        right_id="cluster_sha1",
        linker_class=DuckDBLinker,
        linker_training=linker_training,
        linker_settings=linker_settings,
        threshold=0.8,
    )

    print(splink_linker.settings.linker_training)
    print(splink_linker.settings.linker_settings)
