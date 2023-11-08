from typing import Dict, Callable, Any


def cleaner(function: Callable, arguments: Dict) -> Dict[str, Dict[str, Any]]:
    return {function.__name__: {"function": function, "arguments": arguments}}


def cleaners(*cleaner: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: v for d in cleaner for k, v in d.items()}


if __name__ == "__main__":
    pass
