import pandas as pd
from pathlib import Path
import duckdb
import pytest

from src import locations as loc
from src.features.clean_basic import clean_punctuation

# from src.features.clean_complex import duckdb_cleaning_factory


def load_test_data(path):
    dirty = pd.read_csv(Path(path, "dirty.csv"))
    clean = pd.read_csv(Path(path, "clean.csv"))

    return dirty, clean


cleaning_tests = [("clean_punctuation", clean_punctuation)]


@pytest.mark.parametrize("test", cleaning_tests)
def test_basic_functions(test):
    """
    Tests whether the basic cleaning functions do what they're supposed
    to. More complex functions should follow from here.
    """
    test_name = test[0]
    cleaning_function = test[1]

    dirty, clean = load_test_data(Path(loc.PROJECT_DIR, "test", "features", test_name))

    cleaned = duckdb.sql(
        f"""
        select
            {cleaning_function("col")} as col
        from
            dirty
    """
    ).df()

    assert cleaned.equals(clean)


def test_factory():
    """
    Tests whether the cleaning factory is accurately combining basic
    functions.
    """
    pass


def test_nest_unnest():
    """
    Tests whether the nest_unnest function is working.
    """
    pass
