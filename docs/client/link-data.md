# Building Matchbox DAGs

Entity resolution usually needs more than one matching step. Matchbox uses a directed acyclic graph (DAG) to describe that workflow and keep each step explicit.

In Matchbox, a complete pipeline uses three kinds of node:

- [`Source`][matchbox.client.sources.Source] steps index data from warehouses or files.
- [`Model`][matchbox.client.models.Model] steps score pairs of records that may refer to the same entity.
- [`Resolver`][matchbox.client.resolvers.Resolver] steps turn one or more model outputs into entity clusters.

This guide walks through the full flow: define sources, build models, add resolvers, run the DAG, and publish a default run.

## Understanding Matchbox DAGs

Sources provide raw records. Models emit scored edges. Resolvers consume model outputs and materialise the clusters that Matchbox queries. A DAG can alternate between scoring and clustering layers several times.

```mermaid
graph TD
    SA["Source A"] --> MA["Deduper A"]
    MA --> RA["Resolver A"]
    RA --> LAB["Linker AB"]
    SB["Source B"] --> LAB
    LAB --> RF["Final resolver"]
```

All nodes are lazy until you run them.

```python
query_data = companies_house.query().data()
model_scores = dedupe_companies_house.run()
resolver_clusters = companies_resolver.run()
```

## Setting up your environment

When building a pipeline, it helps to configure logging and create the warehouse client you will attach to your sources.

=== "Example"
    ```python
    import logging

    from sqlalchemy import create_engine

    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matchbox").setLevel(logging.INFO)

    engine = create_engine("postgresql://user:password@host:port/database")
    ```

## 1. Defining a DAG

Create a [`DAG`][matchbox.client.dags.DAG] and call
[`new_run()`][matchbox.client.dags.DAG.new_run] to start an editable run.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    dag = DAG(name="companies").new_run()
    ```

The DAG owns every source, model, and resolver you define afterwards. A run is
the stored snapshot of that DAG inside the collection. The
[`new_run()`][matchbox.client.dags.DAG.new_run] method creates the run that you
populate and sync.

## 2. Defining sources

Add each dataset with [`source()`][matchbox.client.dags.DAG.source]. See
[`Source`][matchbox.client.sources.Source] for the full argument list.

=== "Example"
    ```python
    from matchbox.client import RelationalDBLocation

    warehouse = RelationalDBLocation(name="warehouse").set_client(engine)

    companies_house = dag.source(
        name="companies_house",
        location=warehouse,
        extract_transform="""
            select
                id,
                company_number::text as company_number,
                company_name,
                upper(postcode) as postcode
            from companieshouse.companies
        """,
        infer_types=True,
        index_fields=["company_name", "company_number", "postcode"],
        key_field="id",
    )

    exporters = dag.source(
        name="hmrc_exporters",
        location=warehouse,
        extract_transform="""
            select
                id,
                company_name,
                upper(postcode) as postcode
            from hmrc.trade__exporters
        """,
        infer_types=True,
        index_fields=["company_name", "postcode"],
        key_field="id",
    )
    ```

Each source needs:

- A `location`, such as [`RelationalDBLocation`][matchbox.client.locations.RelationalDBLocation].
- An `extract_transform` statement that produces the source data.
- A list of `index_fields` that Matchbox is allowed to match on.
- A `key_field` that uniquely identifies each record in the source output.

## 3. Preparing query data

To prepare model inputs, call [`query()`][matchbox.client.sources.Source.query]
on a source or [`query()`][matchbox.client.resolvers.Resolver.query] on a
resolver. See [`Query`][matchbox.client.queries.Query] for the full argument
list.

### Cleaning query data

The `cleaning` dictionary controls which columns flow into the model.

- The dictionary key becomes the output column name.
- The dictionary value is a DuckDB SQL expression.
- Only the cleaned columns you declare are passed through, plus `id`, `leaf_id`, and key columns.

!!! tip "Include every field the model needs"
    If a field is missing from `cleaning`, it does not reach the model. Include pass-through aliases for every field referenced by `model_settings`.

### Field qualification

Without cleaning, Matchbox qualifies source fields with the source name.

```python
companies_house.query().data()
# Columns: id, companies_house_key, companies_house_company_name, ...
```

With cleaning, the result columns use the aliases you define.

```python
companies_house.query(
    cleaning={
        "name": f"lower({companies_house.f('company_name')})",
        "number": companies_house.f("company_number"),
    }
).data()
# Columns: id, companies_house_key, name, number
```

Use `source.f()` inside cleaning expressions when you need a qualified reference.

```python
companies_house.f("company_name")
```

## 4. Creating models

To create a model, call [`deduper()`][matchbox.client.queries.Query.deduper] or
[`linker()`][matchbox.client.queries.Query.linker] on a query. See
[`Model`][matchbox.client.models.Model] for the full argument list.

### Choosing model methodologies

Matchbox includes deterministic, weighted, and learned linkers, as well as dedupers.

- [`NaiveDeduper`][matchbox.client.models.dedupers.NaiveDeduper] groups records by identical cleaned fields.
- [`DeterministicLinker`][matchbox.client.models.linkers.DeterministicLinker] links records with explicit DuckDB comparison rules.
- [`WeightedDeterministicLinker`][matchbox.client.models.linkers.WeightedDeterministicLinker] combines several deterministic checks into a weighted score.
- [`SplinkLinker`][matchbox.client.models.linkers.SplinkLinker] emits learned scores using [Splink](https://moj-analytical-services.github.io/splink/index.html).

See the [models API](../api/client/models.md) for full settings.

### Creating source-level dedupers

To create a deduper, call [`query()`][matchbox.client.sources.Source.query] on a
source and then [`deduper()`][matchbox.client.queries.Query.deduper].

=== "Example"
    ```python
    from matchbox.client.models.dedupers.naive import NaiveDeduper

    dedupe_companies_house = companies_house.query(
        cleaning={
            "company_name": f"lower({companies_house.f('company_name')})",
            "company_number": companies_house.f("company_number"),
        }
    ).deduper(
        name="dedupe_companies_house",
        description="Deduplicate Companies House companies",
        model_class=NaiveDeduper,
        model_settings={
            "unique_fields": ["company_name", "company_number"],
        },
    )

    dedupe_exporters = exporters.query(
        cleaning={
            "company_name": f"lower({exporters.f('company_name')})",
            "postcode": exporters.f("postcode"),
        }
    ).deduper(
        name="dedupe_exporters",
        description="Deduplicate exporter records",
        model_class=NaiveDeduper,
        model_settings={
            "unique_fields": ["company_name", "postcode"],
        },
    )
    ```

## 5. Creating resolvers and a linker

To create a resolver, call
[`resolver()`][matchbox.client.models.Model.resolver] on a model and, if
needed, pass extra model inputs that should contribute to the same clustering
policy. See [`Resolver`][matchbox.client.resolvers.Resolver] for the full
argument list.

Resolvers can sit between model layers. Call
[`query()`][matchbox.client.resolvers.Resolver.query] on a resolver when you
want the next model layer to work from a resolved entity view.

=== "Example"
    ```python
    from matchbox.client.models.linkers import DeterministicLinker
    from matchbox.client.resolvers import Components, ComponentsSettings

    resolve_companies_house = dedupe_companies_house.resolver(
        name="resolve_companies_house",
        description="Resolve Companies House duplicates",
        resolver_class=Components,
        resolver_settings=ComponentsSettings(
            thresholds={dedupe_companies_house.name: 1.0}
        ),
    )

    resolve_exporters = dedupe_exporters.resolver(
        name="resolve_exporters",
        description="Resolve exporter duplicates",
        resolver_class=Components,
        resolver_settings=ComponentsSettings(
            thresholds={dedupe_exporters.name: 1.0}
        ),
    )

    link_resolved = resolve_companies_house.query().linker(
        resolve_exporters.query(),
        name="link_resolved_companies",
        description="Link resolved company views",
        model_class=DeterministicLinker,
        model_settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": [
                "l.company_name = r.company_name and l.postcode = r.postcode"
            ],
        },
    )

    companies_resolver = link_resolved.resolver(
        dedupe_companies_house,
        dedupe_exporters,
        name="companies_resolver",
        description="Resolve company entities across both sources",
        resolver_class=Components,
        resolver_settings=ComponentsSettings(
            thresholds={
                dedupe_companies_house.name: 1.0,
                dedupe_exporters.name: 1.0,
                link_resolved.name: 0.8,
            }
        ),
    )
    ```

This pattern separates scoring from clustering.

- Models focus on generating candidate matches and scores.
- Resolvers decide which model edges are strong enough to merge and how several model outputs work together.
- A DAG can include several resolvers over the same model graph, each representing a different clustering policy.

## 6. Running and publishing the DAG

Run every step in execution order with
[`run_and_sync()`][matchbox.client.dags.DAG.run_and_sync].

=== "Example"
    ```python
    dag.run_and_sync()
    ```

Publish the run with [`set_default()`][matchbox.client.dags.DAG.set_default]
when other users and services should query it.

=== "Example"
    ```python
    dag.set_default()
    ```

`set_default()` marks the run as the published version that `load_default()` retrieves. It requires all steps in the DAG to be reachable from a single final resolver.

### Visualising execution

Use `dag.draw()` to inspect the pipeline, or `dag.draw(mode="list")` to see the
execution order.

=== "Example"
    ```python
    print(dag.draw(mode="list"))
    ```

    ```text
    Collection: companies
    └── Run: <run_id>

    1. 📄 companies_house
    2. 📄 hmrc_exporters
    3. ⚙️ dedupe_companies_house
    4. ⚙️ dedupe_exporters
    5. 💎 resolve_companies_house
    6. 💎 resolve_exporters
    7. ⚙️ link_resolved_companies
    8. 💎 companies_resolver
    ```

## 7. Querying resolved entities

Use [`get_matches()`][matchbox.client.dags.DAG.get_matches] to fetch
[`ResolverMatches`][matchbox.client.results.ResolverMatches] for the default
resolver, or for a named resolver if you pass one explicitly.

=== "Example"
    ```python
    resolved = dag.get_matches()
    lookup = resolved.as_lookup()
    cluster_dump = resolved.as_dump()

    strict_resolved = dag.get_matches(resolver="companies_resolver")
    ```

You can also call [`query()`][matchbox.client.resolvers.Resolver.query] on a
resolver and then [`data()`][matchbox.client.queries.Query.data] if you want
the tabular entity view behind that resolver.

=== "Example"
    ```python
    entity_query = companies_resolver.query()
    entity_query.data()
    ```

## 8. Re-running a published DAG

You can load a stored DAG from the server with
[`load_default()`][matchbox.client.dags.DAG.load_default] or
[`load_pending()`][matchbox.client.dags.DAG.load_pending], attach a warehouse
client with [`set_client()`][matchbox.client.dags.DAG.set_client], and then
call [`new_run()`][matchbox.client.dags.DAG.new_run] to reuse the same
structure for another run.

=== "Example"
    ```python
    dag = DAG(name="companies").load_default().set_client(engine).new_run()
    dag.run_and_sync()
    dag.set_default()
    ```

## Further reading

- [Explore DAGs](explore-dags.md)
- [Look up matches](look-up.md)
- [Evaluate resolver output](evaluation.md)
- [Client API reference](../api/client/index.md)
