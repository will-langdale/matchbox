# Pre-0.10 to 0.10

Matchbox 0.10 introduces resolver steps and renames several public concepts. This guide is intentionally short and aimed at users updating existing code.

## What changed

The main shift is that models score pairs, and resolvers create clusters.

- Sources still index records.
- Models still deduplicate and link, but their output is a score table.
- Resolvers define the entity view that gets queried and evaluated.

The main naming changes are:

- `resolution` becomes `step`.
- `probability` becomes `score`.
- Model-final workflows become resolver-final workflows.

## Updating pipeline code

The common migration is to insert an explicit resolver after the models you already have.

=== "Before"
    ```python
    linker = left.query().linker(
        right.query(),
        ...,
    )
    ```

=== "After"
    ```python
    from matchbox.client.resolvers import Components, ComponentsSettings

    linker = left.query().linker(
        right.query(),
        ...,
    )

    resolver = linker.resolver(
        name="companies_resolver",
        resolver_class=Components,
        resolver_settings=ComponentsSettings(
            thresholds={linker.name: 0.8}
        ),
    )
    ```

## Updating lookups and querying

Replace `dag.resolve()` style code with `dag.get_matches()`.

=== "Before"
    ```python
    lookup = dag.resolve().as_lookup()
    ```

=== "After"
    ```python
    lookup = dag.get_matches().as_lookup()
    ```

Published runs require a single final resolver, so `dag.set_default()` only succeeds once every step is reachable from one resolver.

## Updating evaluation

Evaluation uses resolver output rather than model results.

=== "Before"
    ```python
    precision, recall = eval_data.precision_recall(results, threshold=0.5)
    ```

=== "After"
    ```python
    from matchbox.client.eval import EvalData

    eval_data = EvalData(tag="companies__15_02_2025")
    precision, recall = eval_data.precision_recall(resolver.results_eval)
    ```

To compare score cut-offs or clustering policies, define more than one resolver and evaluate each resolver separately.

Local sample files also come from resolver matches:

```python
dag.get_matches().as_dump().write_parquet("samples.pq")
```

## Updating direct API usage

If you call the server API directly, expect step-based paths and resolver-based evaluation sampling.

- `/collections/.../resolutions/...` becomes `/collections/.../steps/...`.
- Evaluation samples are requested for a resolver step.
- Backend terminology and DTOs use `step`, `model_edges`, `resolver_clusters`, and `score`.

## Using a migrated DAG to help the rewrite

If your DAG already exists on the server, download it and use the stored structure as a guide while rewriting local code.

```python
dag = DAG("companies").load_default()
print(dag.draw())

pending_dag = DAG("companies").load_pending()
print(pending_dag.draw())
```

Those server-side DAGs have already been migrated, so they are useful for checking step names, resolver names, and the shape of the final pipeline.
