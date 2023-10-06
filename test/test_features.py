import pandas as pd
from pathlib import Path
import duckdb
import pytest
from functools import partial
import ast

from src import locations as loc
from src.features.clean_basic import (
    clean_punctuation,
    array_except,
    list_join_to_string,
    tokenise,
)

# from src.features.utils import duckdb_cleaning_factory, unnest_renest


"""
----------------------------
-- Feature cleaning tests --
----------------------------

To avoid bug-prone unit tests for complex cleaning functions like
src.features.clean_complex.clean_comp_names, we instead test the constituent
parts, and the methods that build those parts into something complex.

To test clean_comp_names, therefore, we test the leaf functions in the stack:

* clean_comp_names
    * clean_company_name
        * tokenise
        * expand_abbreviations
        * clean_punctuation
    * array_except
    * list_join_to_string

And the methods that assemble them:

* duckdb_cleaning_factory
* unnest_renest

See features/ directory for more information on specific tests.

"""


def load_test_data(path):
    dirty = pd.read_csv(Path(path, "dirty.csv"), converters={"list": ast.literal_eval})
    clean = pd.read_csv(Path(path, "clean.csv"), converters={"list": ast.literal_eval})
    dirty.columns = ["col"]
    clean.columns = ["col"]

    return dirty, clean


array_except_partial = partial(array_except, terms_to_remove=["ltd", "plc"])

cleaning_tests = [
    ("clean_punctuation", clean_punctuation),
    ("array_except", array_except_partial),
    ("list_join_to_string", list_join_to_string),
    ("tokenise", tokenise),
]


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
