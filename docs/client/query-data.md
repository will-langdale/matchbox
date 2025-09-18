# Data retrieval

## Key mapping

Given a key and a source, retrieves all keys that share its cluster in both the source and target. Useful for making ad-hoc queries about specific items of data.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    from_key, to_key = DAG("companies").map_key(
        from_source="datahub_companies",
        to_source="companies_house",
        key="8534735",
    )

    print(from_key)
    print(to_key)
    ```

=== "Output"
    ```
    ["8534735", "8534736"],
    ["EXP123", "EXP124"]
    ```