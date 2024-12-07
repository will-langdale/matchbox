from functools import wraps

from matchbox.client._handler import get_resolution_graph
from vcr import use_cassette


def vcr_cassette(func):
    @wraps(func)
    @use_cassette(f"test/fixtures/vcr_cassettes/{func.__name__}.yaml")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@vcr_cassette
def test_get_resolution_graph():
    assert get_resolution_graph()
