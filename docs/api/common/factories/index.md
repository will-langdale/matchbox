# Overview

::: matchbox.common.factories
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        extra:
            show_root_docstring: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!_logger$"  # Excludes logger variables
            - "!_path$"    # Excludes path variables
            - "!model_config" # Excludes Pydantic configuration

## Using the system

The factory system aims to provide `*Testkit` objects that facilitate three groups of testing scenarios:

- Realistic mock `Source`, `Model`, and `Resolver` objects to test client-side connectivity functions
- Realistic mock data to test server-side adapter functions
- Realistic mock pipelines with controlled completeness to test client-side methodologies

Four broad factory functions are provided:

- [`source_factory()`][matchbox.common.factories.sources.source_factory] generates [`SourceTestkit`][matchbox.common.factories.sources.SourceTestkit] objects, which contain dummy `Source`s and associated data
- [`linked_sources_factory()`][matchbox.common.factories.sources.linked_sources_factory] generates [`LinkedSourcesTestkit`][matchbox.common.factories.sources.LinkedSourcesTestkit] objects, which contain a collection of interconnected `SourceTestkit` objects, and the true entities this data describes
- [`model_factory()`][matchbox.common.factories.models.model_factory] generates [`ModelTestkit`][matchbox.common.factories.models.ModelTestkit] objects, which mock score tables that can connect both `SourceTestkit` and `ResolverTestkit` objects in ways that fail and succeed predictably
- [`resolver_factory()`][matchbox.common.factories.resolvers.resolver_factory] generates [`ResolverTestkit`][matchbox.common.factories.resolvers.ResolverTestkit] objects, which turn one or more model outputs into cluster assignments

Underneath, these factories and objects use a system of [`SourceEntity`][matchbox.common.factories.entities.SourceEntity] and [`ClusterEntity`][matchbox.common.factories.entities.ClusterEntity] objects to share data. The source entities are the ground truth, and the cluster entities are the intermediate or final groupings as data moves through a pipeline.

All factory functions are configured to provide a sensible, useful default.

The system is designed to be as hashable as possible to enable caching. Often you'll need to provide tuples where you might normally provide lists.

There are some common patterns you might consider using when editing or extending tests.

## Client-side connectivity

We can use the factories to test inserting or retrieving isolated `Source`, `Model`, or `Resolver` objects.

Perhaps you're testing DAG reconstruction and want to add a realistic `Source` DTO to a DAG:

```python
from matchbox.client.dags import DAG

source_testkit = source_factory()

dag = DAG("companies")
dag.add_step(name=source_testkit.name, step=source_testkit.source.to_dto())
```

`source_factory()` can be configured with a powerful range of [`FeatureConfig`][matchbox.common.factories.entities.FeatureConfig] objects, including a [variety of rules][matchbox.common.factories.entities.VariationRule] which distort and duplicate the data in predictable ways. These use [Faker](https://faker.readthedocs.io/) to generate data.

```python
source_factory(
    n_true_entities=1_000,
    features=(
        FeatureConfig(
            name="name",
            base_generator="first_name_female",
            drop_base=False,
            variations=(PrefixRule(prefix="Ms "),),
        ),
        FeatureConfig(
            name="title",
            base_generator="job",
            drop_base=True,
            variations=(
                SuffixRule(suffix=" MBE"),
                ReplaceRule(old="Manager", new="Leader"),
            ),
        ),
    ),
    repetition=3,
)
```

By default, each `SourceTestkit`, `ModelTestkit`, or `ResolverTestkit` creates a new [`DAG`][matchbox.client.dags.DAG]. If membership to the right DAG is important, you can either set it manually:

```python
dag = DAG("companies")
source_testkit = source_factory(dag=dag)
```

Or you can unpack your objects into `DAG` methods:

```python
source_testkit = source_factory()
dag = DAG("companies")
dag.source(**source_testkit.into_dag())
```

[`LinkedSourcesTestkit`][matchbox.common.factories.sources.LinkedSourcesTestkit] objects attach all linked sources to the same new DAG.

## Server-side adapters

The factories can generate data suitable for `MatchboxDBAdapter.create_step()`, `MatchboxDBAdapter.insert_source_data()`, `MatchboxDBAdapter.insert_model_data()`, and `MatchboxDBAdapter.insert_resolver_data()`. Between these functions, we can set up any backend in any configuration we need to test the other adapter methods.

Adding a source step and its indexed data:

```python
source_testkit = source_factory().fake_run()
backend.create_step(step=source_testkit.source.to_dto(), path=source_testkit.source.path)
backend.insert_source_data(
    path=source_testkit.source.path,
    data_hashes=source_testkit.data_hashes,
)
```

Adding a model step and its score data:

```python
model_testkit = model_factory().fake_run()
backend.create_step(step=model_testkit.model.to_dto(), path=model_testkit.model.path)
backend.insert_model_data(
    path=model_testkit.model.path,
    results=model_testkit.scores.to_arrow(),
)
```

Adding a resolver step and its cluster assignments:

```python
model_testkit = model_factory().fake_run()
resolver_testkit = resolver_factory(inputs=(model_testkit,)).fake_run()

backend.create_step(
    step=resolver_testkit.resolver.to_dto(),
    path=resolver_testkit.resolver.path,
)
backend.insert_resolver_data(
    path=resolver_testkit.resolver.path,
    data=resolver_testkit.assignments.to_arrow(),
)
```

`linked_sources_factory()` and the step factories work together to create broader systems of data that connect, or do not connect, in controlled ways.

## Methodologies

Configure the true state of your data with `linked_sources_factory()`. Its default is a set of three tables of ten unique company entities.

- CRN (company name, CRN ID) contains all entities with three unique variations of the company's name
- CDMS (CRN ID, DH ID) contains all entities repeated twice
- DH (company name, DH ID) contains half the entities

`linked_sources_factory()` can be configured using tuples of [`SourceTestkitParameters`][matchbox.common.factories.sources.SourceTestkitParameters] objects. Using these you can create complex sets of interweaving sources for methodologies to be tested against.

The `model_factory()` is designed so you can chain together known processes and then plug in your real scoring methodology. [`LinkedSourcesTestkit.diff_model_edges()`][matchbox.common.factories.sources.LinkedSourcesTestkit.diff_model_edges] makes a score table comparable with the true source entities and gives a detailed diff to help you debug. Pass the `.entities` from the upstream `SourceTestkit` or `ResolverTestkit` objects that fed the methodology you are checking.

```python
linked_testkit = linked_sources_factory()
left = linked_testkit.sources["crn"]
right = linked_testkit.sources["cdms"]

link_model = model_factory(
    left_testkit=left,
    right_testkit=right,
    true_entities=tuple(linked_testkit.true_entities),
).fake_run()

identical, report = linked_testkit.diff_model_edges(
    scores=link_model.scores,
    left_clusters=left.entities,
    right_clusters=right.entities,
    sources=["crn", "cdms"],
    threshold=0,
)

assert identical, report
```

The `resolver_factory()` serves the same role for clustering methodologies. It lets you compare resolver assignments against the true entity grouping while keeping the upstream model scores fixed.

## Testing with scenarios

For more complex integration tests, the factory provides a scenario system. This allows you to stand up a fully-populated backend with a single context manager, [`setup_scenario()`][matchbox.common.factories.scenarios.setup_scenario]. This is particularly useful for testing database adapters and end-to-end methodologies.

The main usage pattern is to call `setup_scenario()` with a backend adapter, a warehouse engine, and a named scenario. The context manager yields a `TestkitDAG` containing all the sources, models, and resolvers created for the scenario, giving you access to the ground truth.

```python
from matchbox.common.factories.scenarios import setup_scenario

def test_my_adapter_function(my_backend_adapter, engine):
    with setup_scenario(my_backend_adapter, "link", warehouse=engine) as dag:
        assert dag.sources
        assert dag.models
        assert dag.resolvers
```

The scenario system is cached, so subsequent runs of the same scenario are significantly faster.

See [the scenarios API][matchbox.common.factories.scenarios] for a full list of available scenarios.

### Creating new scenarios

You can create your own scenarios by writing a builder function and registering it with the [`@register_scenario`][matchbox.common.factories.scenarios.register_scenario] decorator. This allows you to build reusable, complex data setups for your tests.
