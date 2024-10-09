from typing import Any, Callable, Dict

from pandas import DataFrame


def cleaner(function: Callable, arguments: Dict) -> Dict[str, Dict[str, Any]]:
    return {function.__name__: {"function": function, "arguments": arguments}}


def cleaners(*cleaner: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: v for d in cleaner for k, v in d.items()}


def process(data: DataFrame, pipeline: Dict[str, Dict[str, Any]]) -> DataFrame:
    curr = data
    for func in pipeline.keys():
        curr = pipeline[func]["function"](curr, **pipeline[func]["arguments"])
    return curr
