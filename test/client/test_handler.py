from functools import wraps
from pathlib import Path
from typing import Callable

from matchbox.client._handler import get_resolution_graph
from vcr import use_cassette


def vcr_cassette(func: Callable) -> Callable:
    test_dir = Path(__file__).resolve().parents[1]
    cassette_name = f"{func.__name__}.yaml"

    @wraps(func)
    @use_cassette(test_dir / "fixtures" / "vcr_cassettes" / cassette_name)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@vcr_cassette
def test_get_resolution_graph():
    assert get_resolution_graph()
