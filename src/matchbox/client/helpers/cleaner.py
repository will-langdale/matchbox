"""Functions to pre-process data sources."""

from typing import Any, Callable, Dict

from pandas import DataFrame


def cleaner(function: Callable, arguments: Dict) -> Dict[str, Dict[str, Any]]:
    """Define a function to clean a dataset.

    Args:
        function: the callable implementing the cleaning behaviour
        arguments: a dictionary of keyword arguments to pass to the cleaning function

    Returns:
        A representation of the cleaner ready to be passed to the `cleaners()` function
    """
    return {function.__name__: {"function": function, "arguments": arguments}}


def cleaners(*cleaner: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Combine multiple cleaners in a single object to pass to `process()`.

    Args:
        cleaner: Output of the `cleaner()` function

    Returns:
        A representation of multiple cleaners to be passed to the `process()` function

    Examples:
        ```python
        clean_pipeline = cleaners(
            cleaner(
                normalise_company_number,
                {"column": "company_number"},
            ),
            cleaner(
                normalise_postcode,
                {"column": "postcode"},
            ),
        )
        ```
    """
    return {k: v for d in cleaner for k, v in d.items()}


def process(data: DataFrame, pipeline: Dict[str, Dict[str, Any]]) -> DataFrame:
    """Apply cleaners to input dataframe.

    Args:
        data: The dataframe to process
        pipeline: Output of the `cleaners()` function

    Returns:
        The processed dataset
    """
    curr = data
    for func in pipeline.keys():
        curr = pipeline[func]["function"](curr, **pipeline[func]["arguments"])
    return curr
