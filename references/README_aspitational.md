# 🔗 Company matching framework

A match orchestration framework to allow the comparison, validation, and orchestration of the best match methods for the company matching job.

⚠️ The below is aspirational, **unimplemented code** to help refine where we want the API to get to. By writing instructions for the end product, we'll flush out the problems with it before they occur.

# v2: Functional API

A quick overview of where we're aiming:

```python
import matchbox

from matchbox import clean
from matchbox.client.helpers import (
    selector, 
    selectors, 
    cleaner, 
    cleaners, 
    comparison,
    comparisons
)

from matchbox.dedupers import Naive
from matchbox.linkers import CMS

# Select and query the data

my_data_selector = selector(
    table="data.data_hub_statistics",
    fields=[
        "company_name",
        "data_hub_id"
    ]
)

ch_selector = selector(
    table="companieshouse.companies",
    fields=[
        "company_name"
    ]
)
dh_selector = selector(
    table="dit.data_hub__companies",
    fields=[
        "data_hub_id"
    ]
)

ch_dh_selector = selectors(
    ch_selector,
    dh_selector
)

cluster_raw = cmf.query(
    select=ch_dh_selector
)
data_raw = cmf.query(
    select=my_data_selector
)

# Clean and process the data

cleaner_dh_id = cleaner(
    function=clean.data_hub_id,
    column="data_hub_id"
)
cleaner_company_name = cleaner(
    function=clean.company_name,
    column="company_name"
)

my_cleaners = cleaners(
    cleaner_dh_id,
    cleaner_company_name
)

cluster_clean = cmf.process(
    data=cluster_raw,
    pipeline=my_cleaners
)
data_clean = cmf.process(
    data=data_raw,
    pipeline=my_cleaners
)

# Prepare and run the deduper

company_name_comparison = comparison(
    output_column="company_name",
    l_column="company_name",
    r_column="company_name",
    sql_condition="l_column = r_column"
)
dh_id_comparison = comparison(
    output_column="data_hub_id",
    l_column="data_hub_id",
    r_column="data_hub_id",
    sql_condition="l_column = r_column"
)

all_comparisons = comparisons(
    ch_selector,
    dh_selector
)

my_deduper = cmf.make_deduper(
    dedupe_run_name="data_hub_stats",
    description="""
        Clean and check company name and ID.
    """,
    deduper=Naive,
    data=data_clean,
    dedupe_settings=all_comparisons
)

dedupe_probabilities = my_deduper()

dedupe_probabilities.to_cmf() # Can now view performance stats, also .to_df() or .to_sql() to unwrap

# Prepare and run the linker

my_linker = cmf.make_linker(
    link_run_name="data_hub_stats",
    description="""
        Using company name and ID with existing cleaning methods
    """,
    linker=CMS, 
    cluster=cluster_clean, 
    data=data_clean,
    dedupe_probabilities=dedupe_probabilities, # or a dedupe_run_name
    dedupe_threshold=0.99,
    link_settings={
        company_name_comparison: 1,
        dh_id_comparison: 3
    }
)

link_probabilities = my_linker()

link_probabilities.to_cmf() # Can now view performance stats, also .to_df() or .to_sql() to unwrap

# Resolve entities

my_clusters = cmf.to_clusters(
    link_probabilities, # or a link_run_name
    link_threshold=0.95
)

my_clusters.to_cmf() # To push to final table, also .to_df() or .to_sql()

```

# v1: Object-oriented API

This version of the aspirational code helped me clarify a lot of my thinking. I leave it here for now so we can steal from it in the final README.

[[_TOC_]]

There are six levels of engagement a user might have with the framework, and each of these is made up of a patchwork of ten tasks.

* Reading data from the company entity clusters
    * Reading
* Reading data from the company entity clusters, then joining new data themselves cleaned with the framework's cleaning functions
    * Reading
    * Cleaning
* Joining new data to the service in an evaluatable pipeline for personal/team-based usage
    * Reading
    * Cleaning
    * Deduplication
    * Link creation
    * Evaluation
    * Entity resolution
    * Persisting the process
* Deduplicating a joined dataset
    * Deduplication
* Joining new data to the service in an evaluatable pipeline to add to the deployed service
    * Reading
    * Cleaning
    * Deduplication
    * Link creation
    * Evaluation
    * Adding to the existing service
* Developing new methodologies and the Company Matching Framework itself
    * Adding linker methodologies
    * Adding deduplication methodologies
 
Those tasks broken out are:

* Reading
* Cleaning
* Deduplication
* Link creation
* Evaluation
* Entity resolution
* Persisting the process
* Adding to the existing service
* Adding linker methodologies
* Adding deduplication methodologies

Let's break these down one by one. In our example we have a dataset of Data Hub statistics, `data.data_hub_statistics`, that we will eventually want to bring into the service.

## Installation

```bash
pip install company-matching-framework
```

## Reading

### `selector` and `selectors`

Lots of functions in the Framework will require a dictionary be passed to a `select` argument. We provide `selector` and `selectors` to aid with this creation.

```python
import matchbox

ch_selector = cmf.selector(
    table="companieshouse.companies",
    fields=[
        "company_name"
    ]
)
dh_selector = cmf.selector(
    table="dit.data_hub__companies",
    fields=[
        "data_hub_id"
    ]
)

ch_dh_selector = cmf.selectors(
    ch_selector,
    dh_selector
)

ch_dh_selector
```

This is just a fancy way to format dictionaries. You can write them out manually if you prefer.

```shell
foo@bar:~$
{
    "companieshouse.companies": [
        "company_name"
    ],
    "dit.data_hub__companies": [
        "data_hub_id"
    ]
}
```

### Querying in Python or PostgreSQL

`cmf.query()` is the workhorse of reading data out, or with functions in PostgreSQL directly.

I want the name of a company as it appears in Companies House, and its ID from Data Hub. We can do this in the DBMS, or in Python. 

```sql
select
    ch.company_name,
    dh.data_hub_id
from
    companies([
        'companieshouse.companies as ch',
        'dit.data_hub__companies as dh'
    ]);
```

```python
import matchbox

cmf.query(
    select={
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ]
    }
)
```

Some companies in Companies House might have several linked entries in Data Hub. By default the service returns them all, but the service can be opinionated about which one is preferred.

```sql
select
    ch.company_name,
    dh.data_hub_id
from
    companies(
        tables => {
            'companieshouse.companies as ch',
            'dit.data_hub__companies as dh'
        },
        preferred => true
    )
```

```python
import matchbox

cmf.query(
    select=ch_dh_selector,
    preferred=True
)
```

Consider the HMRC Exporters table. The same company appears hundreds of times. If you pass it to the service, it will assume you want every row, and the duplicated rows will contain lots of `null`s. To aggregate, use the service in a subquery. The unique company entity ID can be returned with `cms.id`, to help you group.

```sql
select
    agg.id,
    max(agg.company_name),
    max(agg.data_hub_id),
    count(agg."month")
from (
    select
        cms.id as id,
        ch.company_name as company_name,
        dh.data_hub_id as data_hub_id,
        exp."month" as "month"
    from
        companies(
            tables => {
                'companieshouse.companies as ch',
                'dit.data_hub__companies as dh',
                'hmrc.trade__exporters as exp'
            }
        )
) agg
group by
    agg.id;
```

```python
import matchbox

exp_selector = cmf.selector(
    table="hmrc.trade__exporters",
    fields=[
        "month"
    ]
)

ch_dh_exp_selector = cmf.selectors(
    ch_selector,
    dh_selector,
    exp_selector
)

df = cmf.query(
    select=exp_selector,
    return_id=True,
    return_type="pandas"
)

df.groupby("id").agg({"company_name": "max", "data_hub_id": "max", "month": "count"})
```

We can also calculate and alias fields. [Cleaning](#cleaning) and [Array strategies](#array-strategies) contain some ways to deal with arrays created this way.

```sql
select
    array_append(ch.secondary_names, ch.company_name) as all_company_names,
    abs(dh.data_hub_id) as data_hub_id
from
    companies([
        'companieshouse.companies as ch',
        'dit.data_hub__companies as dh'
    ]);
```

```python
import matchbox

cmf.query(
    select={
        "companieshouse.companies": [
            "array_append(secondary_names, company_name) as all_company_names"
        ],
        "dit.data_hub__companies": [
            "abs(data_hub_id) as data_hub_id"
        ]
    }
)
```

## Cleaning

This is all handled by collections of functions in `cmf.clean`. Functions are written in SQL to be run in-DBMS or in duckDB. We can either use some pre-made cleaning functions in collections like `cmf.clean`, or roll our own using `cmf.clean.cleaning_function` to amalgamate functions in collections like `cmf.clean.steps` -- or make your own step!

We want to clean the company name in `data.data_hub_statistics` so we can left join it onto some extracted data ourselves.

We offer several cleaning functions for fields found in lots of datasets.

```python
from matchbox import clean

clean.company_name(df, input_column="company_name")
clean.postcode(df, input_column="postcode")
clean.companies_house_id(df, input_column="ch_id")

# Can also be used on remote tables

clean.company_name("data.data_hub_statistics", input_column="company_name", return_type="pandas")
clean.postcode("data.data_hub_statistics", input_column="postcode", return_type="sql", sql_out="_user_eaf4fd9a.data_hub_statistics_cleaned")
```

Our cleaning functions are all amagamations of steps of small, basic cleaning SQL. Step functions all apply to a single column and need to be wrapped in `cleaning_function` which allows them to be used locally or on the DBMS.

```python
from matchbox import clean

nopunc_lower = clean.cleaning_function(
    clean.steps.clean_punctuation,
    clean.steps.lowercase,
)

nopunc_lower(df, input_column="company_name", return_type="pandas")
```

Sometimes you don't need to clean a company name -- you need to clean a list of them. `cleaning_function` can handle it.

```python
from matchbox import clean

nopunc_lower_array = clean.cleaning_function(
    clean.steps.clean_punctuation,
    clean.steps.lowercase,
    array=True
)

nopunc_lower_array(df, input_column="all_company_names", return_type="pandas")
```

### Array strategies

This is an advanced topic for fine-tuning your linker or deduper.

Sometimes we don't know whether we're going to be cleaning a list, or single item, and we want to be explicit and how we deal with this. Consider selecting this company's `company_name` from Data Hub. 

| cmf.id | data_hub_id | company_name      |
| ------ | ----------- | ----------------- |
| 1      | 12345​      | Wild Pines​       |
| 1      | 1 2345​     | Wild Pines Group​ |

Our team have detected a duplicate company, and resolved it to a single entity -- `1`. But which version of its name do we want to link a new dataset to?

To make this decision we need an **array strategy**. The options are:

* None. The default. Will return two rows for the company
* Most common. Takes the most common value, and the first alphabetically in a tie
* First. Takes the first alphabetically
* Array. Rolls up all options into an array of unique items. This retains all information, but will need more complex matching rules in the linker.
    * For Splink, this would mean something like `array_intersect_level()` in the Comparison Template Library
 
```python
from matchbox import clean

nopunc_lower_most_common = clean.cleaning_function(
    clean.steps.clean_punctuation,
    clean.steps.lowercase,
    array_strategy="most_common"
)

nopunc_lower_most_common(df, input_column="all_company_names", return_type="pandas")
```

Note that if you created an array field with `selecter`, the only permitted array strategies are `none` or `array`, which will merge all the arrays into one, and dedupe.

Whether to define an array strategy is down to your data and your linker. For Splink, `none` violates some mathmatical assumptions that will hit performance, but it might be worth it for the sake of simplicity.

## Deduplication

To be appropriate for linking, a dataset must have **one row per company entity**, otherwise known as observational independence. One way to meet the one row per entity requirement is to pass a `deduper` object to the `cmf.linker(deduper=)` argument. Passing this object will also mean chosing a deduping threshold, so it's important we first know how to deduplicate, and evaluate that deduplication. When passed to a linker, a deduper is _only_ applied to the dataset side of the join, never the cluster.

Dedupers are like linkers that link a dataset to itself. We expect them to be run as standalone tasks while they're being built and analysed to choose a threshold to consider a duplicate, then as arguments to linkers along with `dedupe_threshold`.

Every deduper will need:

* \[Optional\] A cleaning pipeline for the data
* Settings for the deduper

### `cleaner` and `cleaners`

We've seen how to make cleaning functions. Let's see how to make a pipeline of them which we'll use in both dedupers and linkers.

To do this we offer `cleaner` and `cleaners`. Similar to `selector(s)`, they are just ways of making dictionaries that linkers can undersand to run a pipeline of data cleaning.

```python
import matchbox
from matchbox import clean

cleaner_dh_id = cmf.cleaner(
    function=clean.data_hub_id,
    column="data_hub_id"
)
cleaner_company_name = cmf.cleaner(
    function=clean.company_name,
    column="company_name"
)

my_cleaners = cleaners(
    cleaner_dh_id,
    cleaner_company_name
)
```

If any of your clears use a cleaning fuction whose array strategy was `array`, the next function in the pipeline will need to have been created with `array=True`.

### `comparison` and `comparisons`

One common task is building comparisons. Just like `selector` and `selectors`, `comparison` and `comparisons` can help us build a comparison object for some linkers and dedupers. Write SQL conditions using `l_column` and `r_column`.

```python
import matchbox

company_name_comparison = cmf.comparison(
    output_column="company_name",
    l_column="company_name",
    r_column="company_name",
    sql_condition="l_column = r_column"
)
dh_id_comparison = cmf.comparison(
    output_column="data_hub_id",
    l_column="data_hub_id",
    r_column="data_hub_id",
    sql_condition="l_column = r_column"
)

all_comparisons = cmf.comparisons(
    ch_selector,
    dh_selector
)

all_comparisons
```

Again, this is just a fancy way to format dictionaries. You can write them out manually if you prefer.

```shell
foo@bar:~$
{
    "comparisons": [
        {
            "output_column_name": "company_name",
            "sql_condition": "l.company_name = r.company_name"
        },
        {
            "output_column_name": "data_hub_id",
            "sql_condition": "l.data_hub_id = r.data_hub_id"
        }
    ]
}
```

If any of your cleaners used an array strategy of `array`, you will need to make comparisons on arrays instead of single items. For now you can do this yourself in `sql_condition`. Helper functions may eventually be implemented.

### Deduplication settings

The simplest deduper is the naïve deduper, which will detect a duplicate when all the fields it's given are identical.

You can use `cmf.report.dedupers(deduper="naive")` to see the docstring of the deduper, which will help you deal with its requirements.

Because deduplication just links a dataset to itself, `dedupe_settings` can use `comparison` and `comparisons` objects just like a linker.

Every deduper needs a `dedupe_run_name` name and an optional `description`. The `dedupe_run_name` is used to record the probabilities a deduper generates, and means you can either overwrite a previous run with your ever-refined methodology, or start a new one.

```python
import matchbox

data_hub_statistics_deduper = cmf.deduper(
    type="naive",
    dedupe_run_name="data_hub_stats",
    description="""
        Clean and check company name and ID.
    """,
    data_select=cmf.selector(
        table="data.data_hub_statistics",
        fields=[
            "data_hub_id",
            "company_name"
        ]
    ),
    data_cleaner=my_cleaners,
    dedupe_settings=all_comparisons
)
```

### Running the deduper

To get the data:

```python
data_hub_statistics_deduper.get_raw_data()
```

To view the raw data:

```python
data_hub_statistics_deduper.get_dataset(stage="raw")
```

To prepare and clean the data:

```python
data_hub_statistics_deduper.prepare()
data_hub_statistics_deduper.get_dataset(stage="processed")
```

And finally to deduplicate:

```python
data_hub_statistics_deduper.dedupe()
```

Just like the linker, this outputs to the Company Matching Framework's probabilities table by default, where you can use our evaluation tools. See the [Evaluation -- dedupers](#dedupers) section for evaluating your deduper and chosing a threshold to pass to the linker.

For `data.data_hub_statistics` and the naïve deduper, the results are either 0 or 1, which makes a choosing a threshold nice and easy!

To see the results directly:

```python
data_hub_statistics_deduper.get_duplicates()
```

## Link creation

We previously discussed the one row per company entity rule in the [Deduplication](#deduplication) section. The linker will check that every row in your `selector` statement is unique, and will stop if it isn't. This behaviour can be overridden if you wish, or permitted to a threshold of tolerance. We understand that datasets that _should_ have one row per company entity sometimes don't for pragmatic reasons, and the entity resolution algorithm knows it.

One way to meet the one row per entity requirement is to pass a `deduper` object to the `cmf.linker(deduper=)` argument. See [Deduplication](#deduplication) for how to create a deduper -- the syntax is extremely similer to linkers.

Before it can run, every linker will need:

* \[Optional\] Cleaning pipelines for the clusters and the data. You'll get better results if you add them!
* Settings for the linker

### Linker settings

Every linker will need slightly different things in its settings. You can use `cmf.report.linkers(linker="cms")` to see the docstring of the linker, which will help you deal with its requirements.

We've made our cleaning pipeline and our comparisons. The `cms` linker actually requires a weight for each of its comparisons. See [Making linkers: helper functions](#making-linkers-helper-functions) for commands that would help you understand the needs of each linker.

Just like the deduper, every linker needs a `link_run_name` name and an optional `description`. The `link_run_name` is a key your outputted probabilities are stored against.

Two important optional arguments here are `dedupe_threshold` and `link_threshold`. These are the values above which we consider a probability to have become truth. If you've used a deduper, the linker cannot run without a `dedupe_threshold` -- see [Evaluation -- dedupers](#dedupers) for how to choose one. `link_threshold` is only really needed to run your final pipeline. See [Entity resolution](#entity-resolution) for more details.

```python
import matchbox

data_hub_statistics_linker = cmf.linker(
    type="cms",
    link_run_name="data_hub_stats",
    description="""
        Using company name and ID with existing cleaning methods
    """,
    cluster_select=ch_dh_selector,
    cluster_cleaner=my_cleaners,
    data_select=cmf.selector(
        table="data.data_hub_statistics",
        fields=[
            "data_hub_id",
            "company_name"
        ]
    ),
    data_cleaner=my_cleaners,
    deduper=data_hub_statistics_deduper,
    dedupe_threshold=0.99,
    link_settings={
        company_name_comparison: 1,
        dh_id_comparison: 3
    }
)
```

If you happen to use the same object for the `cmdf.linker(data_cleaner=)` and `cmdf.deduper(data_cleaner=)` argument, we'll do some stuff in the background to make it compute quicker. You're welcome.

We're now ready to perform our core operations.

### Running the linker

Functions to run the deduper and linker are very similar. The main difference is some extra `.get_*` commands.

We retrieve the data. Note we could also supply `cluster_select` and `data_select` at this stage if we wanted. This allows us to test our method locally.

```python
data_hub_statistics_linker.get_raw_data()
```

We can view the raw data to check we're happy with it:

```python
data_hub_statistics_linker.get_cluster(stage="raw")
data_hub_statistics_linker.get_dataset(stage="raw")
```

Now we prepare the data. We could also supply `cluster_cleaner`, `data_cleaner` and `link_settings` at this stage, as arguments to `.prepare()`, allowing experimentation. To view the cleaned data at this stage we would use the `processed` stage.

```python
data_hub_statistics_linker.prepare()
data_hub_statistics_linker.get_cluster(stage="processed")
data_hub_statistics_linker.get_dataset(stage="processed")
```

Note that if you're using a more complicated linker, like `splink`, your linker may have lots of smaller functions within `.prepare()` you can use to test and refine your pipeline. Check the linker's demo notebook to find out about its intricacies.

Finally it's time to actually make our link. For most linkers `link_settings` could be supplied at this stage -- `splink` is the notable exception.

```python
data_hub_statistics_linker.link()
```

By default this outputs to the Company Matching Framework's probabilities table, enabling the use of our evaluation tools. You can turn this off and return your probabilities to a different table or pandas dataframe if you wish.

Regardless, we can see the result:

```python
data_hub_statistics_linker.get_links()
```

### Helper functions

To help with all the above we might want:

* To check for the one row per entity rule
* To know what kinds of linkers have worked well in the past
* To know how certain fields were cleaned in the past

How do we know that our data is appropriate for linking, or whether we need to do some deduping? `cmf.report` can help.

```python
import matchbox.report
report.data(
    df, # or data.data_hub_statistics
    select=cmf.selector(
        table="data.data_hub_statistics",
        fields=[
            "data_hub_id"
        ]
    )
)
```

```shell
foo@bar:~$
Field 
data.data_hub_statistics
------------------------

data_hub_id: 98.7% unique

Use cmf.dedupe to perform deduplication in your linking process, or add another field to reach 100%
```

How do we know what linkers have worked well in the past? What fields can we join into and how were they cleaned?

Let's start with the fields that exist.

```python
import matchbox.report
report.fields()
```

```shell
foo@bar:~$
Field           Source                         Coverage    Accuracy
-------------------------------------------------------------------
company_name    companieshouse.companies       95%         99%
company_id      companieshouse.companies       76%         97%
data_hub_id     dit.data_hub__companies        21%         88%
address_1       companieshouse.companies       74%         89%
address_2       dit.data_hub__companies        65%         74%
address_3       dit.export_wins__wins_dataset  22%         45%
...             ...                            ...         ...
```

Accuracy has yet to be determined methodologically, but some canidate ideas are:

* In all verified matches made with this field, how many were right vs wrong? 
* In all verified matches made with this field that were right, in how many did this field match exactly?
* In the production service, what is the Brier score for this field, measured against verified matches?
    * When a linker says "this `company_name` gives a x% chance of this being a match", how well configured is that score overall?

What about linkers that have worked well for fields we want to join onto? We can use the `selector` we built earlier.

```python
import matchbox.report
report.linkers(select=ch_dh_selector)
```

```shell
foo@bar:~$
Table                      Field          Link run              Match %ge   AOC
--------------------------------------------------------------------------------
companieshouse.companies   company_name   n1_li_deterministic   35%         0.23
companieshouse.companies   company_name   n1_li_cms             46%         0.67
companieshouse.companies   company_name   n1_li_splink          87%         0.89
dit.data_hub__companies    data_hub_id    n3_li_deterministic   88%         0.9
dit.data_hub__companies    data_hub_id    n3_li_cms             98%         0.98
dit.data_hub__companies    data_hub_id    n3_li_splink          86%         0.97
...                        ...            ...                   ...         ...
```

And how was a specific field cleaned in a specific linker or deduper?

```python
import matchbox.report
report.cleaners(link_run="n3_cms_dun_and_bradstreet", field="data_hub_id")
report.cleaners(dedupe_run="n4_naive_hmrc_importers", field="data_hub_id")
```

```shell
foo@bar:~$
In n3_cms_dun_and_bradstreet, data_hub_id was cleaned with the following functions:

from matchbox import clean

{
    "data_hub_id": {
        "function": clean.steps.remove_punctuation,
        "arguments": {
            "column": "data_hub_id"
        },
        "function": clean.steps.lowercase,
        "arguments": {
            "column": "data_hub_id"
        }
    }
}

Use cmf.clean to help.
```

We can even get guidelines for using a speficic linker. This can be helpful for tricky `linker_settings`.

```python
import matchbox.report
report.linkers(linker="cms")
```

```shell
foo@bar:~$
The Company Matching Service linker requires the following objects:

cluster_select: A selector to get fields from one or more tables. Use cmf.selector(s) to help
data_select: A selector to get fields from the table you wish to join. Use cmf.selector(s) to help
link_settings: A dictionary of fields you wish to match on, and the weight given to each match. Use cmf.report.linkers(linker="cms", arg="link_settings") to help

You may also wish to define:

cluster_cleaner: A cleaner to clean the clusters data. Use cmf.cleaner(s) to help
dataset_cleaner: A cleaner to clean the dataset. Use cmf.cleaner(s) to help
```

Let's look at a more complex one too.

```python
import matchbox.report
report.linkers(linker="splink")
```

```shell
foo@bar:~$
The Splink linker requires the following objects:

cluster_select: A selector to get fields from one or more tables. Use cmf.selector(s) to help
data_select: A selector to get fields from the table you wish to join. Use cmf.selector(s) to help
cluster_cleaner: A cleaner to clean the clusters data. Use cmf.cleaner(s) to help
dataset_cleaner: A cleaner to clean the dataset. Use cmf.cleaner(s) to help
link_settings: The Splink settings dictionary passed to the Splink Linker object
link_pipeline: The Splink Linker methods to estimate linker parameters. Will be run in the order they're supplied

Use Splink documentation and example notebooks to help you create link_settings and link_pipeline.

* https://moj-analytical-services.github.io/splink/demos/tutorials/00_Tutorial_Introduction.html
* https://moj-analytical-services.github.io/splink/settings_dict_guide.html
* https://moj-analytical-services.github.io/splink/linker.html

Across both link_settings and link_pipeline you will need to define:

* Blocking rules
* Comparisons
* Linker training methods

⚠️ IMPORTANT ⚠️ Array strategies

Splink cleaners can only reach their optimal performance with cleaning functions that have array strategies defined.

Where the strategy is to return an array, this will need to be handled in the comparisons and blocking rules in the link_settings. The array_intersect_level function in the Comparison Level Library is a good place to start.

```

We can use `report.dedupers(deduper=)` in the same way.

## Evaluation

### Dedupers

Let's take a look and see how our `data_hub_statistics_deduper` did.

```python
data_hub_statistics_deduper.report()
```

```shell
foo@bar:~$
Deduplication run              Dedupe %ge   Entities   AOC
----------------------------------------------------------
n5_dd_naive_data_hub_stats     13%          125,234    N/A

Use the validation dashboard at https://matching.data.trade.gov.uk/ to improve AOC calculation.

```

What are those `n5_dd_naive_` bits at the front? They're added automatically based on the deduper or linker you used, and to help with some of the internals this use case doesn't need to worry about.

While ee currently have an entity count, which gives us a bit of a heuristic, we have no real idea of how good deduplication was here. AOC ([area under a receiver operating characteristic curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)) is how we judge quality. We want it to be close to 1, but it's not been calculated. Why?

AOC requires the "truth" to calculate -- it's a measure of how good our balance of true positive vs false positives are. It needs a human to hand-label some data to compare a methodology against. Because no one's tried to dedupe `data.data_hub_statistics` before, no one's labelled any data.

To do it yourself:

* Head to https://matching.data.trade.gov.uk/
* Go to the labelling section and select "dedupers"
* Choose the `n5_dd_naive_data_hub_stats` dedupe run
* Choose a sampling method, one of:
    * Random. Will sample matches to verify completely randomly
    * Improvement. Will weight sampling towards low-probability matches. An AOC against this method will be artificially low, but this will help you focus on what needs improving
    * Disagreement. Will weight sampling towards matches where similar runs disagree. This will artificially increase the difference in AOC between several runs, helping make the best matching method more obvious

While the naïve method should be always be right when it makes a guess, in many instances it won't deduplicate all the entities it should do. Dedupers like `splink` can pick up where it leaves off.
 
Once you've started labelling the results, https://matching.data.trade.gov.uk/ can give you not only AOC, but the receiver operating characteristic (ROC) curve that generated it in the evaluation section. For naïve, this is a point, rather than a curve.

The ROC curve shows the change in false positive and true positive links as you change the threshold of what consitutes a match. You need to choose a `dedupe_threshold` that makes this trade-off in a way you're happy with. Alternatively, we suggest a value that maximiese the sum of true positive and false negatives.

### Linkers

Let's take a look and see how our `data_hub_statistics_linker` did.

```python
data_hub_statistics_linker.report()
```

```shell
foo@bar:~$
Link run                   Match %ge   AOC
------------------------------------------
n5_li_cms_data_hub_stats   78%         N/A

Use the validation dashboard at https://matching.data.trade.gov.uk/ to improve AOC calculation.

```

We've got a match percentage of 78%. Not bad -- domain knowledge suggests the true possible match rate is likely much higher, but it's a start. 

As with the deduper, we can label some data to generate AOC:

* Head to https://matching.data.trade.gov.uk/
* Go to the labelling section and select "linkers".
* Choose the `n5_li_cms_data_hub_stats` link run
* Choose a sampling method, one of:
    * Random
    * Improvement
    * Disagreement
 
Again, the ROC curve can be found in the evaluation section of https://matching.data.trade.gov.uk/ -- use it to choose a `link_threshold`. We suggest a value, but it's very much down to you.
 
## Entity resolution

We've made a deduper and linker that connects `data.data_hub_statistics` to the wider group of company entities. We've rigorously validated our output and measured how good it is. But if we want to use the core `cmf.query()` function to pull data from every entity and include data from `data.data_hub_statistics`, we need to turn our probabilities into entity clusters. 

To do this we need to choose a threshold above which we consider a probability to become truth for both deduping and linking. When we're linking we might _keep_ all probabilities above 0.7 for analysis, but when we're making entities, we might only want to keep those where we're really sure, like 0.95 or above. See [Evaluation](#evaluation) for using a ROC curve to decide these thresholds.

We can create your custom entity clusters locally or remotely:

```python
my_clusters = data_hub_statistics_linker.to_clusters(
    return_type="pandas",
    dedupe_threshold=0.99,
    link_threshold=0.95
)

data_hub_statistics_linker.to_clusters(
    return_type="sql", 
    sql_out="_user_eaf4fd9a.my_clusters",
    dedupe_threshold=0.99,
    link_threshold=0.95
)
```

You can then use the same query service as before with an extra argument:

```sql
select
    ch.company_name,
    exp.data_hub_id,
    dhs.total_clicks
from
    companies(
        tables => {
            'companieshouse.companies as ch',
            'hmrc.trade__exporters as exp',
            'data.data_hub_statistics as dhs'
        },
        cluster_table => '_user_eaf4fd9a.my_clusters'
    )
```

```python
import matchbox

cmf.query(
    select={
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ],
        "data.data_hub_statistics": [
            "total_clicks"
        ]
    },
    preferred=True,
    cluster_table=my_clusters # or "_user_eaf4fd9a.my_clusters"
)
```

## Persisting the process

Your deduper and linker is a process -- it gets data, cleans it, links it, and outputs it. We can `save()` a linker in two ways:

* As a script, to run again
* As a pickle, to reload as-is

```python
data_hub_statistics_linker.save(path="myfiles/dh_stats.py", type="script")
data_hub_statistics_linker.save(path="myfiles/dh_stats.pickle", type="file")
```

We can `load()` the linker in the same way.

```python
data_hub_statistics_linker = CMSLinker.load(path="myfiles/dh_stats.py")
data_hub_statistics_linker = CMSLinker.load(path="myfiles/dh_stats.pickle")
```

You can run your script from the command line, letting it know exactly where you want to save your new clusters:

```shell
foo@bar:~$ python myfiles/dh_stats.py --sql_out=_user_eaf4fd9a.my_clusters --overwrite --dedupe-threshold=0.99 --link_threshold=0.95
```

Re-running this script will ensure your clusters stay up to date with changes in both the service, and your data.

## Adding to the existing service

We want everyone to benefit from the hard work you've put into linking up your data. We've tried to make it as easy as possible to add new methodologies, or replace old ones with better versions. Each dataset will only ever have one "canonical" linking method, even though we might experiment with lots of different ones to improve the service.

The Company Matching Framework exists in three repositories:

* Company Matching Framework. The Python library installed with `pip install company-matching-framework`
* Company Matching Framework pipelines. A collection of scripts which uses `cmf` to run the "canonical" pipeline that everyone uses
* Company Matching Framework reporting. The dashboard that allows everyone to help to validate links

To add to the existing service, we need to use the second repo, the pipelines. To add or update the canonical link for a dataset:

* Go to the Company Matching Framework pipeline repo
* Create a new branch. In there:
    * Place your linker script in the `pipelines/` subdirectory
    * Add any new cleaning functions to the `cleaning/` subdirectory
    * Add any unit tests to the `tests/` subdirectory
    * If it replaces an existing script, delete the old one
    * In `config.py` add or update the appropriate section to include a reference to your script and dataset
        * This is where the threshold for both deduping and cluster creation should be entered
        * For must users, the `EXTRAS` section is where a new dataset will go
        * For data engineers, you may be working on the `CORE` section, which deals with the datasets everything else will join onto. Note changing the order of this section can have huge ramifications for the rest of the pipeline
* Create a merge request. In it state:
    * The change you made
    * The uplift in AOC and match percentage
    * Any tests you've run or built to assure quality
    * Anything else we should know
 
The data team will review your request, and once it's accepted, your data will have joined the service and everyone will be able to make use of it.

## Adding linker methodologies

A matching methodology is a way of linking data that can be applied in different ways to lots of different datasets. The linkers included in the Company Matching Framework should be enough to cover most use cases, and do it inside the DBMS. But let's say you want to implement something new, like the [Python Record Linkage Toolkit](https://recordlinkage.readthedocs.io/en/latest/about.html).

The key things to know are:

* Put your new linker in `cmf/link/` in the Company Matching Framework repo (not the pipeline repo)
* You only need to implement a `prepare()` and `link()` function, plus any helpful functions that go into supporting them
* `prepare()` must
    * Accept at least:
        * `cluster_cleaner` and `data_cleaner` arguments, which it can pass to `cmf.utils.cluster_cleaner` and `cmf.utils.data_cleaner` if you want to let them work as they do in other linkers
        * `cluster_raw` and `data_raw` arguments, which you should assume to be a reference to a PostgreSQL or duckDB table
    * Return a tuple of `(cluster_processed, data_processed)`, which should be references to a PostgreSQL or duckDB table
* `link()` must
    * Accept at least:
        * `link_settings`. This can look however you want, but we try to follow Splink's settings dictionary syntax in `cmf.comparison(s)` so users can move between linkers easily
        * `cluster_processed` and `data_processed` arguments, which you should assume to be a reference to a PostgreSQL or duckDB table
    * Call `cmf.utils.probabilities` to write its output to the proabilities table
* You must implement the `LINKER_NAME` and `LINKER_DESCRIPTION` global variables, which provide the name users will pass to `cmf.linker()`, and added to link_run_name names
* Consider putting a "Help" section in your docstrings for the `cmf.report.linkers(linker=, arg=)` function to show the user. If not present, it'll show the whole docstring
* Consider adding a demonstation notebook in `notebooks/` to help users see how to use the intricacies of your method
 
Linkers are composed based on the above, and the following public methods are added for users:

* `get_raw_data()`
* `get_clusters()`
* `get_dataset()`
* `report()`
* `get_links()`
* `to_clusters()`
* `save()`
* `load()`

Other helpful private methods are also present. Find them in `cmf.link.utils`.

## Adding deduplication methodologies

A deduplucation methodology is a way of deduping data that can be applied in different ways to lots of datasets. We provide `naive` and `splink`, which should be a good basis.

If you want to add more the key things to know are:

* Put the deduper in `cmf/dedupe/` in the Company Matching Framework repo (not the pipeline repo)
* You only need to implement a `prepare()` and `dedupe()` function, plus any helpful functions that go into supporting them
* `prepare()` must
    * Accept at least:
        * A `data_cleaner` argument which it can pass to `cmf.utils.data_cleaner` to use our out-of-the-box methodology
        * A `data_raw` argument, which you should assume to be a reference to a PostgreSQL or duckDB table
    * Return `data_processed`, which should be a reference to a PostgreSQL or duckDB table
* `dedupe()` must
    * Accept at least:
        * `dedupe_settings`. This can look however you want, but we try to follow Splink's settings dictionary syntax in `cmf.comparison(s)` so users can move between linkers easily
        * A `data_processed` argument, which you should assume to be a reference to a PostgreSQL or duckDB table
    * Call `cmf.utils.probabilities` to write its output to the proabilities table
* You must implement the `DEDUPER_NAME` and `DEDUPER_DESCRIPTION` global variables, which provide the name users will pass to `cmf.dedupe()`, and added to dedupe_run_name names
* Consider putting a "Help" section in your docstrings for the `cmf.report.dedupers(deduper=, arg=)` function to show the user. If not present, it'll show the whole docstring
* Consider adding a demonstation notebook in `notebooks/` to help users see how to use the intricacies of your method

Dedupers are composed based on the above, and the following public methods are added for users:

* `get_raw_data()`
* `get_dataset()`
* `report()`
* `get_duplicates()`

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
