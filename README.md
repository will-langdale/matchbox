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

## Usage

The below is **unimplemented code** to help refine where we want the API to get to. By writing instructions for the end product, we'll flush out the problems with it before they occur.

### Installation

```bash
pip install company-matching-framework
```

### I want data from linked datasets

I want the name of a company as it appears in Companies House, and its ID from Data Hub. We can do this in the DBMS, or in Python. 

```sql
select
    ch.company_name,
    dh.data_hub_id
from
    company_matching_service([
        'companieshouse.companies as ch',
        'dit.data_hub__companies as dh'
    ])
```

```python
import cmf.utils as cmfu

cmfu.query(
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
    company_matching_service(
        tables => [
            'companieshouse.companies as ch',
            'dit.data_hub__companies as dh'
        ],
        preferred => true
    )
```

```python
import cmf.utils as cmfu

cmfu.query(
    select={
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ]
    },
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
        company_matching_service(
            tables => [
                'companieshouse.companies as ch',
                'dit.data_hub__companies as dh',
                'hmrc.trade__exporters as exp'
            ]
        )
) agg
group by
    agg.id;
```

```python
import cmf.utils as cmfu

df = cmfu.query(
    select={
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ],
        "hmrc.trade__exporters": [
            "month"
        ]
    },
    preferred=True,
    return_id=True,
    return_type="pandas"
)

df.groupby("id").agg({"company_name": "max", "data_hub_id": "max", "month": "count"})
```

### I have a dataset I want to connect to company enties

Your dataset should first be added to Data Workspace, even if it's in a personal schema. The Company Matching Framework works entirely in the DBMS.

Say you want to connect `data.data_hub_statistics` to the existing company clusters and you want to do this really well -- not just a simple left join. It contains a data hub ID and company name, both of which were entered by hand, so they'll need cleaning up.

What fields exist to connect this dataset to?

```python
import cmf.utils as cmfu
cmfu.cluster_report()
```

```console
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

Great, `company_name` and `data_hub_id` seem right for my dataset.

We're going to be doing a lot of selecting from various tables and clusters. The `selecter` function can help us build the dictionaries to do this. For one dataset, `selecter` is all we need. For getting information across several datasets, we can add several `selecter`s together with `selecters`.

```python
import cmf.utils as cmfu

ch_selecter = cmfu.selecter(
    table="companieshouse.companies",
    fields=[
        "company_name"
    ]
)
dh_selecter = cmfu.selecter(
    table="cdit.data_hub__companies",
    fields=[
        "data_hub_id"
    ]
)

ch_dh_selecter = cmfu.selecters(
    ch_selecter,
    dh_selecter
)

ch_dh_selecter
```

This is just a fancy way to format dictionaries. You can write them out manually if you prefer.

```console
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

What linkers already work with these tables and fields? How have other people cleaned this data?

```python
import cmf.utils as cmfu
cmfu.link_report(select=ch_dh_selecter)
```

```console
foo@bar:~$
Table                      Field          Link experiment          Match %ge   AOC
-----------------------------------------------------------------------------------
companieshouse.companies   company_name   n1_deterministic_basic   35%         0.23
companieshouse.companies   company_name   n1_cms_basic             46%         0.67
companieshouse.companies   company_name   n1_splink_basic          87%         0.89
dit.data_hub__companies    data_hub_id    n3_deterministic_basic   88%         0.9
dit.data_hub__companies    data_hub_id    n3_cms_basic             98%         0.98
dit.data_hub__companies    data_hub_id    n3_splink_basic          86%         0.97
...                        ...            ...                      ...         ...
```

It looks like both a Splink and CMS-based linker might be helpful here. Let's fit both, validate some matches, and see which wins.

Let's start with CMS.

```python
from cmf.linkers import CMSLinker
CMSLinker.help()
```

```console
foo@bar:~$
The Company Matching Service linker requires the following objects:

cluster_select: A selecter to get fields from one or more tables. Use cmf.utils.selecter(s) to help
data_select: A selecter to get fields from the table you wish to join. Use cmf.utils.selecter(s) to help
link_settings: A dictionary of fields you wish to match on. Use CMSLinker.help("link_settings") to help

You may also wish to define:

cluster_cleaner: A cleaner to clean the clusters data. Use cmf.utils.cleaner(s) to help
dataset_cleaner: A cleaner to clean the dataset. Use cmf.utils.cleaner(s) to help
```

Let's start with those required fields.

We've already seen how to build selecters, so lets move onto `link_settings`.

I could look up how other people have done this, but let's let the object itself help me configure it.

```python
CMSLinker.help("link_settings")
```

```console
foo@bar:~$
The Company Matching Service linker requires a link settings in the following shape:

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

Use cmf.utils.comparison(s) to help
```

Just like `selecter` and `selecters`, `comparison` and `comparisons` can help us build a comparison object for this linker. Write SQL conditions using `l_column` and `r_column`.

```python
import cmf.utils as cmfu

company_name_comparison = cmfu.comparison(
    output_column="company_name,
    l_column="company_name",
    r_column="company_name",
    sql_condition="l_column = r_column"
)
dh_id_comparison = cmfu.comparison(
    output_column="data_hub_id,
    l_column="data_hub_id",
    r_column="data_hub_id",
    sql_condition="l_column = r_column"
)

all_comparisons = cmfu.comparisons(
    ch_selecter,
    dh_selecter
)

all_comparisons
```

Again, this is just a fancy way to format dictionaries. You can write them out manually if you prefer.

```console
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

Every linker needs a `link_experiment` name and `description`. The `link_experiment` is used to record the probabilities a linker generates, and means you can either overwrite a previous run with your ever-refined methodology, or start a new one.

```python
data_hub_statistics_linker = CMSLinker(
    link_experiment="data_hub_stats",
    description="""
        Using company name and ID with existing cleaning methods
    """,
    cluster_select=ch_dh_selecter,
    data_select=cmfu.selecter(
        table="data.data_hub_statistics",
        fields=[
            "data_hub_id"
        ]
    ),
    link_settings=all_comparisons
)
data_hub_statistics_linker.get_data()
```

I've now collected my raw data. Let's take a look at those functions that were used to clean the data before, in `n3_cms_basic` for `data_hub_id`, and in `n1_splink_basic` for `company_name`.

```python
import cmf.utils as cmfu
cmfu.cleaning_report(experiment="n3_cms_basic", field="data_hub_id")
```

```console
foo@bar:~$
In n3_cms_basic, data_hub_id was cleaned with the following functions:

import cmf.features.clean_basic as cmf_cb

{
    "data_hub_id": {
        "function": cmf_cb.clean_punctuation,
        "arguments": {
            "column": "data_hub_id"
        },
        "function": cmf_cb.lowercase,
        "arguments": {
            "column": "data_hub_id"
        }
    }
}

Use cmf.utils.cleaner(s) to help.
```

And `company_name`?

```python
import cmf.utils as cmfu
cmfu.cleaning_report(experiment="n1_splink_basic", field="company_name")
```

```console
foo@bar:~$
In n3_cms_basic, data_hub_id was cleaned with the following functions:

import cmf.features.clean_complex as cmf_cc

{
    "company_name": {
        "function": cmf_cc.clean_company_name,
        "arguments": {
            "column": "company_name"
        }
    }
}

Use cmf.utils.cleaner and cmf.utils.cleaners to help.
```

As before, we can use `cleaner` and `cleaners` to build these cleaning dictionaries, though there's nothing stopping us from just making them ourselves. This particular pipeline is made easier because we're cleaning both cluster and dataset using the same functions, but there's no reason you'd have to stick to this.

```python
import cmf.features.clean_basic as cmf_cb
import cmf.features.clean_complex as cmf_cc

dh_clean_punctuation = cleaner(
    function=cmf_cb.clean_punctuation,
    column="data_hub_id"
)
dh_clean_lowercase = cleaner(
    function=cmf_cb.lowercase,
    column="data_hub_id"
)
comp_clean_company_name = cleaner(
    function=cmf_cc.clean_company_name,
    column="company_name"
)

my_cleaners = cleaners(
    dh_clean_punctuation,
    dh_clean_lowercase,
    comp_clean_company_name
)
```

Time to clean. Every linker contains:

* A `prepare()` function to clean the data, which needs at least: 
    * `cluster_cleaner`
    * `data_cleaner`
* A `link()` function to do the linking, which needs at least:
    * `link_settings`

As we've already supplied `link_settings`, we can `link()` right away. 

```python
data_hub_statistics_linker.prepare(
    cluster_cleaner=my_cleaners,
    data_cleaner=my_cleaners
)

data_hub_statistics_linker.link()
```

Let's take a look and see how that did.

```python
data_hub_statistics_linker.report()
```

```console
foo@bar:~$
Link experiment         Match %ge   AOC
---------------------------------------
n5_cms_data_hub_stats   78%         N/A

Use the validation dashboard at https://matching.data.trade.gov.uk/ to improve AOC calculation.

```

What are those `n5_cms_` bits at the front? They're added automatically based on the linker you used, and to help with some of the internals this use case doesn't need to worry about.

We've got a match percentage of 78%. Not bad -- domain knowledge suggests the true possible match rate is likely much higher, but it's a start. But AOC ([area under a receiver operating characteristic curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)) is how we judge quality. We want it to be close to 1, but it's not been calculated. Why?

AOC requires the "truth" to calculate -- it's a measure of how good our balance of true positive vs false positives are. It needs a human to hand-label some data to compare a methodology against. Because no one's tried to connect `data.data_hub_statistics` before, no one's labelled any data.

To do it yourself:

* Head to https://matching.data.trade.gov.uk/
* Go to the labelling section
* Choose the `n5_cms_data_hub_stats` experiment
* Choose a sampling method, one of:
    * Random. Will sample matches to verify completely randomly
    * Improvement. Will weight sampling towards low-probability matches. An AOC against this method will be artificially low, but this will help you focus on what needs improving
    * Disagreement. Will weight sampling towards matches where similar experiments disagree. This will artificially increase the difference in AOC between several experiments, helping make the best matching method more obvious

So you've configured and tested a linker. What now?

#### Test another linker

We identified SplinkLinker as a potential rival to CMSLinker for this dataset. Both linkers share some arguments to their `prepare()` methods, but SplinkLinker has some important differences:

* `prepare()` requires a configuration for training the linker
* `link()`'s `link_settings` is more complex, and needs more things to be configured

You can read about the linker's requirements in its `SplinkLinker.help()` method, or by copying existing linkers in the pipelines repo.

#### Iterate on our linker

We might:

* Make more sophisticated data cleaning methods
* Add new fields and update the settings
* Make more sophisticated SQL matching conditions

Play around and see how the AOC changes. Remember, a new `link_experiment` means a new set of probabilities are output, and you can compare between them.

#### Read data from company entities

You did all this so you could join data from all company entities to your data. To properly add your probabilities to the service see "Add your dataset to the service", but for now you can build your own local version of the clusters table:

```python
my_clusters = data_hub_statistics_linker.to_clusters(return_type="pandas")

data_hub_statistics_linker.to_clusters(return_type="sql", sql_out="_user_eaf4fd9a.my_clusters")
```

You can then use the same query service as before with an extra argument:

```sql
select
    ch.company_name,
    exp.data_hub_id,
    dhs.total_clicks
from
    company_matching_service(
        tables => [
            'companieshouse.companies as ch',
            'hmrc.trade__exporters as exp',
            'data.data_hub_statistics as dhs'
        ],
        cluster_table => '_user_eaf4fd9a.my_clusters'
    )
```

```python
import cmf.utils as cmfu

cmfu.query(
    select={
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ]
    },
    preferred=True,
    cluster_table=my_clusters # or "_user_eaf4fd9a.my_clusters"
)
```

#### Save my linker so I can run it again

Your linker is a process -- it gets data, cleans it, links it, and outputs it. We can `save()` a linker in two ways:

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

```console
foo@bar:~$ python myfiles/dh_stats.py --sql_out=_user_eaf4fd9a.my_clusters --overwrite
```

Re-running this script will ensure your clusters stay up to date with changes in both the service, and your data.

#### Add your dataset to the service

We want everyone to benefit from the hard work you've put into linking up your data. We've tried to make it as easy as possible to add new methodologies, or replace old ones with better versions. Each dataset will only ever have one "canonical" linking method, even though we might experiment with lots of different ones to improve the service.

To add or update the canonical link for a dataset:

* Go to the Company Matching Framework pipeline repo
* Create a new branch. In there:
    * Place your linker script in the `pipelines/` subdirectory
    * Add any new cleaning functions to the `features/` subdirectory
    * If it replaces an existing script, delete the old one
    * Add or update the `config.py` to include a reference to your script and dataset. Unless you're a data engineer, add your dataset to the end of the `n` queue
* Create a merge request. In it state:
    * The change you made
    * The uplift in AOC and match percentage
    * Anything else we should know
 
The data team will review your request, and once it's accepted, your data will have joined the service and everyone will be able to make use of it.

### I have a new matching methodology I want to implement

A matching methodology is a way of linking data that can be applied in different ways to lots of different datasets. The linkers included in the Company Matching Framework should be enough to cover most use cases, and do it inside the DBMS. But let's say you want to implement something new, like the [Python Record Linkage Toolkit](https://recordlinkage.readthedocs.io/en/latest/about.html).

The key things to know are:

* Put your new linker in `cmf/link/` in the Company Matching Framework repo (not the pipeline repo)
* You only need to implement a `prepare()` and `link()` function, plus any helpful functions that go into supporting them
* `prepare()` must
    * Accept at least:
        * `cluster_cleaner` and `data_cleaner` arguments, which it can pass to `cmf.link.utils.cluster_cleaner` and `cmf.link.utils.data_cleaner` if you want to let them work as they do in other linkers
        * `cluster_raw` and `data_raw` arguments, which you should assume to be a reference to a PostgreSQL or duckDB table
    * And return a tuple of `(cluster_processed, data_processed)`, which should be references to a PostgreSQL or duckDB table
* `link()` must
    * Accept at least:
        * `link_settings`. This can look however you want, but we try to follow Splink's settings dictionary syntax in `cmf.utils.comparison(s)` so users can move between linkers easily
        * `cluster_processed` and `data_processed` arguments, which you should assume to be a reference to a PostgreSQL or duckDB table
    * And call `cmf.utils.probabilities` to write its output to the proabilities table
* You must implement the `LINKER_CLASS_NAME` and `LINKER_ABBREVIATION` global variables, which provide the class name users will import, and the abbreviation added to experiment names
* Consider putting a "Help" section in your docstrings for the `help()` method to show the user. If not present, it'll show the whole docstring
 
Linkers are composed based on the above, and the following public methods are added for users:

* `get_data()`
* `report()`
* `to_clusters()`
* `save()`
* `load()`
* `help()`

Other helpful private methods are also present. Find them in `cmf.link.utils`.

### Learning from this README exercise

* Selecting data from the clusters is a top level function, and every time we use it it should have the same syntax
* The non-Splink linkers should use Splink's syntax as much as possible. Their settings dict should be similar, specifying columns to match should be similar, etc
* This repo should do everything in-database as its primary functionality. We hold references to temp tables in a linker, not the actual data
* Like Splink, SQL should be usable in both duckDB and PostgreSQL -- no or little pandas, certainly not for core processing
* There's only so simple all this can get. This has taken a long time to write
* This is so much more complex than I'd realised. Should we scale down the vision? How?

## Release metrics

🛠 Coming soon!

## Usage

🛠 Coming soon!

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
