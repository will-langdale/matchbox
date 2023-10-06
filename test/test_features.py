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
    expand_abbreviations,
)

from src.features.utils import duckdb_cleaning_factory, unnest_renest


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


def passthrough(input_column):
    """
    A passthrough cleaning function that does nothing. Helps test more complex
    building functions.
    """
    return f"{input_column}"


array_except_partial = partial(array_except, terms_to_remove=["ltd", "plc"])

cleaning_tests = [
    ("clean_punctuation", clean_punctuation),
    ("array_except", array_except_partial),
    ("list_join_to_string", list_join_to_string),
    ("tokenise", tokenise),
    ("expand_abbreviations", expand_abbreviations),
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


nest_unnest_tests = [
    ("expand_abbreviations", expand_abbreviations),
    ("pass", passthrough),
]


@pytest.mark.parametrize("test", nest_unnest_tests)
def test_nest_unnest(test):
    """
    Tests whether the nest_unnest function is working.
    """
    test_name = test[0]
    cleaning_function = duckdb_cleaning_factory(test[1])

    dirty, clean = load_test_data(
        Path(loc.PROJECT_DIR, "test", "features", "unnest_renest", test_name)
    )

    cleaning_function_arrayed = unnest_renest(cleaning_function)

    cleaned = cleaning_function_arrayed(dirty, column="col")

    assert cleaned.equals(clean)
