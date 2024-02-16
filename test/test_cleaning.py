import ast
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from cmf import locations as loc
from cmf.clean import drop
from cmf.clean.steps import (
    clean_punctuation,
    expand_abbreviations,
    list_join_to_string,
    remove_stopwords,
    tokenise,
)
from cmf.clean.utils import alias, cleaning_function, unnest_renest

"""
----------------------------
-- Feature cleaning tests --
----------------------------

To avoid bug-prone unit tests for complex cleaning functions like
cmf.clean.company_name, we instead test the constituent
parts, and the methods that build those parts into something complex.

To test company_name, therefore, we test the leaf functions in the stack:

* clean_comp_names
    * clean_punctuation
    * expand_abbreviations
    * tokenise
    * array_except
    * list_join_to_string

And the methods that assemble them:

* cleaning_function
* unnest_renest

See cleaning/ directory for more information on specific tests.

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


remove_stopwords_partial = partial(remove_stopwords, stopwords=["ltd", "plc"])
expand_abbreviations_partial = partial(
    expand_abbreviations, replacements={"co": "company", "ltd": "limited"}
)

cleaning_tests = [
    ("clean_punctuation", clean_punctuation),
    ("remove_stopwords", remove_stopwords_partial),
    ("list_join_to_string", list_join_to_string),
    ("tokenise", tokenise),
    ("expand_abbreviations", expand_abbreviations_partial),
]


@pytest.mark.parametrize("test", cleaning_tests)
def test_basic_functions(test):
    """
    Tests whether the basic cleaning functions do what they're supposed
    to. More complex functions should follow from here.
    """
    test_name = test[0]
    test_cleaning_function = test[1]

    dirty, clean = load_test_data(Path(loc.PROJECT_DIR, "test", "cleaning", test_name))

    cleaned = duckdb.sql(
        f"""
        select
            {test_cleaning_function("col")} as col
        from
            dirty
    """
    ).df()

    assert cleaned.equals(clean)


function_tests = [
    ("tokenise", [tokenise]),
    ("pass", [passthrough]),
    (
        "clean_comp_names",
        [
            clean_punctuation,
            expand_abbreviations_partial,
            tokenise,
            remove_stopwords_partial,
            list_join_to_string,
        ],
    ),
]


@pytest.mark.parametrize("test", function_tests)
def test_function(test):
    """
    Tests whether the cleaning function is accurately combining basic
    functions.
    """
    test_name = test[0]
    test_cleaning_function = cleaning_function(*test[1])

    dirty, clean = load_test_data(
        Path(loc.PROJECT_DIR, "test", "cleaning", "cleaning_function", test_name)
    )

    cleaned = test_cleaning_function(dirty, column="col")

    # Handle Arrow returning arrays but read_csv ingesting lists
    if isinstance(cleaned["col"][0], np.ndarray):
        cleaned["col"] = cleaned["col"].apply(list)

    assert cleaned.equals(clean)


nest_unnest_tests = [
    ("expand_abbreviations", expand_abbreviations_partial),
    ("pass", passthrough),
]


@pytest.mark.parametrize("test", nest_unnest_tests)
def test_nest_unnest(test):
    """
    Tests whether the nest_unnest function is working.
    """
    test_name = test[0]
    test_cleaning_function = cleaning_function(test[1])

    dirty, clean = load_test_data(
        Path(loc.PROJECT_DIR, "test", "cleaning", "unnest_renest", test_name)
    )

    test_cleaning_function_arrayed = unnest_renest(test_cleaning_function)

    cleaned = test_cleaning_function_arrayed(dirty, column="col")

    # Handle Arrow returning arrays but read_csv ingesting lists
    if isinstance(cleaned["col"][0], np.ndarray):
        cleaned["col"] = cleaned["col"].apply(list)

    cleaned = cleaned.sort_values(by="col").reset_index(drop=True)
    clean = clean.sort_values(by="col").reset_index(drop=True)

    assert cleaned.equals(clean)


def test_alias():
    """
    Tests whether the alias function is working.
    """
    test_cleaning_function = cleaning_function(passthrough)

    dirty, clean = load_test_data(Path(loc.PROJECT_DIR, "test", "cleaning", "alias"))

    alias_function = alias(test_cleaning_function, "foo")

    cleaned = alias_function(dirty, column="col")

    assert "foo" in cleaned.columns


def test_drop():
    """
    Tests whether the drop function is working.
    """
    dirty, clean = load_test_data(Path(loc.PROJECT_DIR, "test", "cleaning", "alias"))

    cleaned = drop(dirty, column="col")

    assert len(cleaned.columns) == 0
