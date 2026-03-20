# Look up matches

Use lookups when you want to inspect the entity view produced by a resolver.

## Single keys

Given a key in one source, `lookup_key()` returns the keys that resolve with it in other sources through the DAG's default resolver.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    matches = DAG("companies").load_default().lookup_key(
        from_source="datahub_companies",
        to_sources=["companies_house"],
        key="8534735",
    )

    print(matches["datahub_companies"])
    print(matches["companies_house"])
    ```

=== "Output"
    ```
    ["8534735", "8534736"]
    ["EXP123", "EXP124"]
    ```

## Whole lookups

Use `get_matches()` to download resolved entities for a resolver and turn them into a lookup table.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    dag = DAG("companies").load_default()

    lookup = dag.get_matches().as_lookup()
    strict_lookup = dag.get_matches(resolver="companies_resolver").as_lookup()
    ```

`dag.get_matches()` returns [`ResolverMatches`][matchbox.client.results.ResolverMatches], which also supports `as_dump()`, `view_cluster()`, and `merge()` for [evaluation and inspection workflows](evaluation.md).
