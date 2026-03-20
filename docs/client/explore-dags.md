# Explore DAGs

A Matchbox collection can store several runs of a DAG. Each run is a server-side snapshot of the sources, models, and resolvers that define one entity view for that collection.

This guide shows how to list collections, download a stored run, and inspect the pipeline it contains.

## Listing collections

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    DAG.list_all()
    ```

You can also do lightweight exploration from the [CLI](../server/cli.md).

=== "CLI"
    ```shell
    mbx collections
    ```

Run `mbx collections --help` to inspect the available collection commands and options.

## Downloading a DAG

Load the default or pending run for a collection.

`load_default()` reconstructs the published run for that collection. This is the run other users and services query by default.

=== "Default run"
    ```python
    from matchbox.client.dags import DAG

    dag = DAG(name="companies").load_default()
    ```

`load_pending()` reconstructs the latest non-default run, which is usually the draft currently being worked on before publication.

=== "Pending run"
    ```python
    from matchbox.client.dags import DAG

    dag = DAG(name="companies").load_pending()
    ```

The downloaded DAG includes the serialisable definitions of every source, model, and resolver in that run. 

## Inspecting the pipeline

Use `draw()` to inspect the dependency graph.

=== "Tree view"
    ```python
    print(dag.draw())
    ```

=== "Execution order"
    ```python
    print(dag.draw(mode="list"))
    print(dag.sequence)
    ```

The default resolver is the single final resolver in a complete published DAG. It's used in functions like `DAG.get_matches()` if no resolver is supplied. To make a DAG run the default, a `final_resolver` must be present.

## Inspecting individual steps

You can retrieve sources, models, and resolvers by name.

=== "Example"
    ```python
    source = dag.get_source("companies_house")
    model = dag.get_model("dedupe_companies_house")
    resolver = dag.get_resolver("companies_resolver")
    ```

Once you have the DAG locally, you can attach warehouse clients, query resolved entities, or start a new run from the same structure.
