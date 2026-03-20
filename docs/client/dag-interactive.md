# Run DAGs interactively

Interactive work is useful when you are developing a DAG in a notebook or shell and want to inspect each layer before you sync or publish anything.

Consider a small local DAG that deduplicates `source_a`, resolves it, links that resolved view to `source_b`, and then applies a final resolver. `dag.draw()` is often the quickest way to orient yourself before you start changing anything.

```text
Collection: companies
└── Run: ⛓️‍💥 Disconnected

💎 final_resolver [6]
└── ⚙️ link_ab [5]
    ├── 💎 resolve_a [4]
    │   └── ⚙️ dedupe_a [3]
    │       └── 📄 source_a [1]
    └── 📄 source_b [2]
```

The rest of this guide follows that same flow: inspect the graph, work on sources and queries, check model output, then check resolver output.

## Running steps manually

Sources, models, and resolvers can be run and synced one by one.

Different step types have different requirements to run individually:

* Sources have no dependencies
* Models need their depdendencies to be run and synced
* Resolvers need their depdendencies to be run

=== "Example"
    ```python
    source_a.run()
    source_a.sync()

    dedupe_a.run()
    dedupe_a.sync()

    resolve_a.run()
    resolve_a.sync()

    source_b.run()
    source_b.sync()

    link_ab.run()
    link_ab.sync()

    final_resolver.run()
    final_resolver.sync()
    ```

Use `dag.draw(mode="list")` when you want the execution order that `run_and_sync()` follows.

=== "Example"
    ```python
    print(dag.draw(mode="list"))
    ```

    ```text
    Collection: companies
    └── Run: ⛓️‍💥 Disconnected

    1. 📄 source_a
    2. 📄 source_b
    3. ⚙️ dedupe_a
    4. 💎 resolve_a
    5. ⚙️ link_ab
    6. 💎 final_resolver
    ```

You can also run part of the DAG by step name.

=== "Example"
    ```python
    dag.run_and_sync(start="source_a", finish="resolve_a")
    ```

This is useful when you are working on one branch of the DAG and do not want to re-run everything above it.

## Iterating on sources

It's useful to inspect source internals when you are shaping extract-transform logic or checking what will be indexed. While `source.run()` will return a full output, `source.sample()` can be used to examine a smaller subset.

=== "Example"
    ```python
    source_a = dag.source(...)
    source_a.sample()

    source_a.extract_transform = new_sql
    source_a.sample()
    ```

The default return type is Polars. Other return types are available.

=== "Example"
    ```python
    source_a.sample(return_type="pandas")
    ```

Re-running a source refreshes its local cache. If you change a source definition, re-run and sync it so downstream queries and models read the updated data.

## Iterating on queries

Once a source has run, inspect the query output that the next model layer will see.

=== "Example"
    ```python
    source_a_query = source_a.query(
        cleaning={
            "name": f"lower({source_a.f('name')})",
        }
    )

    source_a_query.data()      # cleaned columns that the model will receive
    source_a_query.data_raw()  # raw qualified columns before cleaning
    ```

For speed, you can reuse cached raw data while adjusting cleaning logic.

=== "Example"
    ```python
    raw_query_data = source_a_query.data_raw()

    source_a_query.cleaning = new_cleaning
    source_a_query.data(raw_data=raw_query_data)
    ```

When the cleaned table looks right, re-run the model with that query output.

## Iterating on models

Running a model returns score edges. After a model has run, those scores are also available on the `results` attribute.

=== "Example"
    ```python
    scores = dedupe_a.run()
    scores = dedupe_a.results
    ```

You can also reuse query data while adjusting model settings.

=== "Example"
    ```python
    dedupe_data = dedupe_a.left_query.data()

    dedupe_a.model_settings.unique_fields = ["name", "postcode"]
    dedupe_a.run(left_data=dedupe_data)
    ```

Linkers accept pre-fetched left and right query dataframes.

=== "Example"
    ```python
    left_data = link_ab.left_query.data()
    right_data = link_ab.right_query.data()

    link_ab.run(left_data=left_data, right_data=right_data)
    ```

After you are happy with the scores, re-run the dependent resolver layer.

## Iterating on resolvers

Resolvers consume model outputs and return cluster assignments. Those upstream model results must exist locally in the current session, not only on the server.

=== "Example"
    ```python
    assignments = final_resolver.run()
    assignments = final_resolver.results
    ```

After a local resolver run, `results_eval` gives the leaf mapping used for Matchbox evaluation, as long as the upstream models were run with the default `low_memory=False`. See [Evaluate resolver output](evaluation.md) for the full evaluation workflow.

=== "Example"
    ```python
    final_resolver.results_eval
    ```

If you re-run an upstream model, re-run every dependent resolver before you sync or publish the DAG.

## Replacing a step wholesale

You can replace a step in the DAG by creating a new step with the same name.

=== "Example"
    ```python
    source_a = dag.source(name="source_a", location=location1, ...)
    source_a = dag.source(name="source_a", location=location2, ...)
    ```
