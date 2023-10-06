# duckdb_cleaning_factory

We test the function in two environments:

* Wrapping a simple cleaning function, tokenise
* Wrapping a more complex stack of cleaning functions as clean_company_names, specifically
    * clean_punctuation
    * expand_abbreviations
    * tokenise
    * array_except
    * list_join_to_string
* Wrapping no function
