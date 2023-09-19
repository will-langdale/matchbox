# 🔗 Company matching framework

A match orchestration framework to allow the comparison, validation, and orchestration of the best match methods for the company matching job.

## Coverage

* [Companies House](https://data.trade.gov.uk/datasets/a777d199-53a4-4d0a-bbbb-1559a86f8c4c#companies-house-company-data)
* [Data Hub companies](https://data.trade.gov.uk/datasets/32918f3e-a727-42e6-8359-9efc61c93aa4#data-hub-companies-master)
* [Export Wins](https://data.trade.gov.uk/datasets/0738396f-d1fd-46f1-a53f-5d8641d032af#export-wins-master-datasets)
* [HMRC UK exporters](https://data.trade.gov.uk/datasets/76fb2db3-ab32-4af8-ae87-d41d36b31265#uk-exporters)

## Quickstart

Clone the repo, then run:

```bash
. setup.sh
```

Create a `.env` with your development schema to write tables into. Copy the sample with `cp .env.sample .env` then fill it in.

* `SCHEMA` is where any tables the service creates will be written by default
* `STAR_TABLE` is where fact and dimension tables will be recorded and checked
* `PROBABILITIES_TABLE` is where match probabilities will be recorded and checked
* `CLUSTERS_TABLE` is where company entities will be recorded and checked
* `VALIDATE_TABLE` is where user validation outputs will be recorded and checked

To set up the database in your specificed schema run:

```bash
make setup
```

## Design

### How does matching happen?

![The methodology of match orchestration](/docs/hybridadditive_links.png "The 🔌hybrid additive methodology")

The matching methodology is often shorthanded as 🔌hybrid additive. This is because there are a core set of tables that are matched additively, one after the other, each using information from all the tables before it. After this, tables can be plugged into this additive core in any order, even in parallel.

{::options parse_block_html="true" /}
<div class="panel panel-info">
**Facts and dimensions**
{: .panel-heading}
<div class="panel-body">

A dimension table contains one row only for each company entity. An example is Companies House. Contrast with a fact table, like HMRC Exporters, where the same company may appear hundreds of times.

</div>
</div>
{::options parse_block_html="false" /}

To implement this, there are three foundational ideas:

1. Every dataset we're matching must be a dimension table
2. A company entity (a "cluster") can only ever have a maximum of one item from each dimension table
3. The left side of a join is always constructed from the clusters table

Everything else flows from this. Matching is done in a pipeline where each step is `n`. For each step in a matching pipeline:

1. Construct the **left table** from clusters, potentially using data from several dimension tables
2. Construct the **right table** from a dimension
3. Match using whatever methodology you like, as long as the result matches the structure of the probabilities table
4. Resolve the probabilities into company entity clusters with a max of one item from the right table, the dimension
5. Repeat for every step in the pipeline

### What does the framework's database look like?

![The entity relationship diagram of the framework](/docs/erdiagram.png "The entity relationship diagram")

The architecture is loosely based on the star schema.

Every dataset we're matching must be a dimension table. If the dataset in Data Workspace isn't already a dimension table, we create its dimension table through naïve deduping (without cleaning) on the fields that demarcate an entity.

The star table is a lookup of these fact and dimension table names, and it's this `id` that's used in the various `source` fields.

The probabilities table contains the raw outputs of a link job.

The validation table contains verified matches made by users.

The clusters table contains the probabilities and validation tables resolved into company entities.

### What does the code structure look like?

![The class diagram of the framework](/docs/classdiagram.png "The class diagram")

Broadly, the repo contains two kinds of classes:

* Data classes are wrappers for tables in the database. They contains read/write functions that safeguard the shape of data moving in and out. These are:
    * Star -- a singleton class that wraps the star table
    * Probabilities -- a singleton class that wraps the probabilities table
    * Validation -- a singleton class that wraps the validation table
    * Clusters -- a singleton class that wraps the clusters table
    * Datasets -- a class with one instance per fact and dimension table combination, providing access to both
* Linker classes define the methodology for a particular link type, such as deterministic or probabilistic links. The parent contains standard functions all Linker subclasses will need
    * Linker subclasses must implement a `prepare()` and `link()` method. In the final prototype we aim to supply:
        * SplinkLinker, for probabilistic linking
        * DeterministicLinker, for straight clean and join links
     
A cold run of the pipeline will:

1. Set up the database
    1. Create all tables
    2. Make missing dimension tables
    3. Add first table in link job to the clusters table
2. Iterate over the pipeline in `config.py`
    1. Use scipts in `/pipeline` to instantiate the configured Linker subclasses
    2. Get cluster and dimension data
    3. Link it and output to probabilities
    4. Resolve probabilities to entities in clusters
    5. Repeat
  
While not yet implemented, we intend to supply a dashboard that will read from probabilities and clusters to allow users to hand-verify matches. The output will be written to the validation table and used:

* To compare models, by getting users to verify matches where models disagree
* To improve the service overall, by getting users to verify weak or disputed matches

## Release metrics

🛠 Coming soon!

## Usage

🛠 Coming soon!

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
