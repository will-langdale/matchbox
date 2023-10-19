# 🔗 Company matching framework

A match orchestration framework to allow the comparison, validation, and orchestration of the best match methods for the company matching job.

[[_TOC_]]

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

I think there are five levels of engagement a user might have with the framework, and each of these is made up of a patchwork of eight tasks.

* Reading data from the company entity clusters
    * Reading
* Reading data from the company entity clusters, then joining new data themselves cleaned with the framework's cleaning functions
    * Reading
    * Cleaning
* Joining new data to the service in an evaluatable pipeline for personal/team-based usage
    * Reading
    * Cleaning
    * Link creation
    * Evaluation
    * Entity resolution
    * Persisting the process
* Joining new data to the service in an evaluatable pipeline to add to the deployed service
    * Reading
    * Cleaning
    * Link creation
    * Evaluation
    * Adding to the existing service
* Developing new methodologies and the Company Matching Framework itself
    * Adding linker methodologies
 
Those tasks broken out are:

* Reading
* Cleaning
* Link creation
* Evaluation
* Entity resolution
* Persisting the process
* Adding to the existing service
* Adding linker methodologies

Let's break these down one by one. In our example we have a dataset of Data Hub statistics, `data.data_hub_statistics`, that we will eventually want to bring into the service.

### Installation

```bash
pip install company-matching-framework
```

### Reading

This is all handled by `cmf.query()`, or with functions in PostgreSQL directly.

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
import cmf

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

Lots of functions in the Framework will require a dictionary be passed to a `select` argument. We provide `selector` and `selectors` to aid with this creation.

```python
import cmf

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

Let's get back to reading data out. Some companies in Companies House might have several linked entries in Data Hub. By default the service returns them all, but the service can be opinionated about which one is preferred.

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
import cmf

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
import cmf

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
    preferred=True,
    return_id=True,
    return_type="pandas"
)

df.groupby("id").agg({"company_name": "max", "data_hub_id": "max", "month": "count"})
```

### Cleaning

This is all handled by collections of functions in `cmf.clean`. Functions are written in SQL to be run in-DBMS or in duckDB. We can either use some pre-made cleaning functions in collections like `cmf.clean`, or roll our own using `cmf.clean.cleaning_function` to amalgamate functions in collections like `cmf.clean.steps` -- or make your own step!

We want to clean the company name in `data.data_hub_statistics` so we can left join it onto some extracted data ourselves.

We offer several cleaning functions for fields found in lots of datasets.

```python
from cmf import clean

clean.company_name(df, input_column="company_name")
clean.postcode(df, input_column="postcode")
clean.companies_house_id(df, input_column="ch_id")

# Can also be used on remote tables

clean.company_name("data.data_hub_statistics", input_column="company_name", return_type="pandas")
clean.postcode("data.data_hub_statistics", input_column="postcode", return_type="sql", sql_out="_user_eaf4fd9a.data_hub_statistics_cleaned")
```

Our cleaning functions are all amagamations of steps of small, basic cleaning SQL. Step functions all apply to a single column and need to be wrapped in `cleaning_function` which allows them to be used locally or on the DBMS.

```python
from cmf import clean

nopunc_lower = clean.cleaning_function(
    clean.steps.clean_punctuation,
    clean.steps.lowercase,
)

nopunc_lower(df, input_column="company_name", return_type="pandas")
```

Sometimes you don't need to clean a company name -- you need to clean a list of them. `cleaning_function` can handle it.

```python
from cmf import clean

nopunc_lower_array = clean.cleaning_function(
    clean.steps.clean_punctuation,
    clean.steps.lowercase,
    array=True
)

nopunc_lower_array(df, input_column="all_company_names", return_type="pandas")
```

### Link creation

To make a link we're going to need:

* To ensure our data is appropriate for linking
* To create pipelines of cleaning functions
* To create linker settings
* To run a data collection, preparation and link job

And to help with all this we might want:

* To check for the one row per entity rule
* To know what kinds of linkers have worked well in the past
* To know how certain fields were cleaned in the past

#### Making linkers: core functions

To be appropriate for linking, your dataset must have **one row per company entity**, otherwise known as observational independence. The linker will check that every row in your `selector` statement is unique, and will stop if it isn't. This behaviour can be overridden if you wish, or permitted to a threshold of tolerance. We understand that datasets that _should_ have one row per company entity sometimes don't for pragmatic reasons, and the entity resolution algorithm knows it.

One way to meet the one row per entity requirement is to pass a `selector` object to the `cmf.linker(data_deduper=)` argument. The linker will perform what we call naïve deduping -- creating new unique rows based on the fields you specify. The service will handle making sure that when data comes from your table, it still returns _all_ the data in your original table.

Our dataset doesn't have this problem, so we can move on.

Before it can run, every linker will need:

* \[Optional\] Cleaning pipelines for the clusters and the data. You'll get better results if you add them!
* Settings for the linker

We've seen how to make cleaning functions. Let's see how to make a pipeline of them.

To do this we offer `cleaner` and `cleaners`. Similar to `selector(s)`, they are just ways of making dictionaries that linkers can undersand to run a pipeline of data cleaning.

```python
import cmf
from cmf import clean

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

Every linker will need slightly different things in its settings. You can use `cmf.report.linkers(linker="cms")` to see the docstring of the linker, which will help you deal with its requirements.

One common task is building comparisons. Just like `selector` and `selectors`, `comparison` and `comparisons` can help us build a comparison object for some linkers. Write SQL conditions using `l_column` and `r_column`.

```python
import cmf

company_name_comparison = cmf.comparison(
    output_column="company_name,
    l_column="company_name",
    r_column="company_name",
    sql_condition="l_column = r_column"
)
dh_id_comparison = cmf.comparison(
    output_column="data_hub_id,
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

We've made our cleaning pipeline and our comparisons, which is all we need for the `cms` linker. Let's put it all together.

Every linker needs a `link_experiment` name and `description`. The `link_experiment` is used to record the probabilities a linker generates, and means you can either overwrite a previous run with your ever-refined methodology, or start a new one.

```python
import cmf

data_hub_statistics_linker = cmf.linker(
    type="cms",
    link_experiment="data_hub_stats",
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
    link_settings=all_comparisons
)
```

We're now ready to perform our core operations.

We retrieve the data. Note we could also supply `cluster_select` and `data_select` at this stage if we wanted. This allows us to test our method in Jupyter.

```python
data_hub_statistics_linker.get_data()
```

We can view the raw data to check we're happy with it:

```python
data_hub_statistics_linker.get(type="cluster", stage="raw")
data_hub_statistics_linker.get(type="data", stage="raw")
```

Now we prepare the data. We could also supply `cluster_cleaner`, `data_cleaner` and `link_settings` at this stage, as arguments to `.prepare()`, allowing experimentation. To view the cleaned data at this stage we would use the "processed" stage.

```python
data_hub_statistics_linker.prepare()
data_hub_statistics_linker.get(type="cluster", stage="processed")
data_hub_statistics_linker.get(type="data", stage="processed")
```

Note that if you're using a more complicated linker, like `splink`, your linker may have lots of smaller functions within `.prepare()` you can use to test and refine your pipeline. Check the linker's demo notebook to find out about its intricacies.

Finally it's time to actually make our link. For most linkers `link_settings` could be supplied at this stage -- `splink` is the notable exception.

```python
data_hub_statistics_linker.link()
```

By default this outputs to the Company Matching Framework's probabilities table, enabling the use of our evaluation tools. You can turn this off and return your probabilities to a different table or pandas dataframe if you wish.

Regardless, we can see the result:

```python
data_hub_statistics_linker.get(type="link")
```

#### Making linkers: helper functions

How do we know that our data is appropriate for linking, or whether we need to do some naïve deduping? `cmf.report` can help.

```python
import cmf.report
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

```console
foo@bar:~$
Field 
data.data_hub_statistics
------------------------

data_hub_id: 98.7% unique

Use cmf.linker(data_deduper=) to naïve-dedupe in your linking process, or add another field to reach 100%
```

How do we know what linkers have worked well in the past? What fields can we join into and how were they cleaned?

Let's start with the fields that exist.

```python
import cmf.report
report.fields()
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

Accuracy has yet to be determined methodologically, but some canidate ideas are:

* In all verified matches made with this field, how many were right vs wrong? 
* In all verified matches made with this field that were right, in how many did this field match exactly?
* In the production service, what is the Brier score for this field, measured against verified matches?
    * When a linker says "this `company_name` gives a 70% chance of this being a match, and this 90%", how well configured is that score overall?

What about linkers that have worked well for fields we want to join onto? We can use the `selector` we built earlier.

```python
import cmf.report
report.linkers(select=ch_dh_selector)
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

And how was a specific field cleaned in a specific linker?

```python
import cmf.report
report.cleaners(link_experiment="n3_cms_basic", field="data_hub_id")
```

```console
foo@bar:~$
In n3_cms_basic, data_hub_id was cleaned with the following functions:

from cmf import clean

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
import cmf.report
report.linkers(linker="cms")
```

```console
foo@bar:~$
The Company Matching Service linker requires the following objects:

cluster_select: A selector to get fields from one or more tables. Use cmf.selector(s) to help
data_select: A selector to get fields from the table you wish to join. Use cmf.selector(s) to help
link_settings: A dictionary of fields you wish to match on. Use cmf.report.linkers(linker="cms", arg="link_settings") to help

You may also wish to define:

cluster_cleaner: A cleaner to clean the clusters data. Use cmf.cleaner(s) to help
dataset_cleaner: A cleaner to clean the dataset. Use cmf.cleaner(s) to help
```

### Evaluation

Let's take a look and see how our `data_hub_statistics_linker` did.

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

### Entity resolution

We've made a linker that connects `data.data_hub_statistics` to the wider group of company entities. We've rigorously validated our output and measured how good it is. But if we want to use the core `cmf.query()` function to pull data from every entity and include data from `data.data_hub_statistics`, we need to turn our probabilities into entity clusters. 

To do this we need to choose a threshold above which we consider a probability to become truth. When we're linking we might keep all probabilities above 0.7, but when we're making entities, we might only want to keep those where we're really sure, like 0.95 or above. You need to explore your results and decide on this figure for yourself. Some linkers are more sensitive than others.

We can create your custom entity clusters locally or remotely:

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
import cmf

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

### Persisting the process

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

### Adding to the existing service

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
        * For must users, the `EXTRAS` section is where a new dataset will go
        * For data engineers, you may be working on the `CORE` section, which deals with the datasets everything else will join onto. Note changing the order of this section can have huge ramifications for the rest of the pipeline
* Create a merge request. In it state:
    * The change you made
    * The uplift in AOC and match percentage
    * Any tests you've run or built to assure quality
    * Anything else we should know
 
The data team will review your request, and once it's accepted, your data will have joined the service and everyone will be able to make use of it.

### Adding linker methodologies

A matching methodology is a way of linking data that can be applied in different ways to lots of different datasets. The linkers included in the Company Matching Framework should be enough to cover most use cases, and do it inside the DBMS. But let's say you want to implement something new, like the [Python Record Linkage Toolkit](https://recordlinkage.readthedocs.io/en/latest/about.html).

The key things to know are:

* Put your new linker in `cmf/link/` in the Company Matching Framework repo (not the pipeline repo)
* You only need to implement a `prepare()` and `link()` function, plus any helpful functions that go into supporting them
* `prepare()` must
    * Accept at least:
        * `cluster_cleaner` and `data_cleaner` arguments, which it can pass to `cmf.link.utils.cluster_cleaner` and `cmf.link.utils.data_cleaner` if you want to let them work as they do in other linkers
        * `cluster_raw` and `data_raw` arguments, which you should assume to be a reference to a PostgreSQL or duckDB table
    * Return a tuple of `(cluster_processed, data_processed)`, which should be references to a PostgreSQL or duckDB table
* `link()` must
    * Accept at least:
        * `link_settings`. This can look however you want, but we try to follow Splink's settings dictionary syntax in `cmf.comparison(s)` so users can move between linkers easily
        * `cluster_processed` and `data_processed` arguments, which you should assume to be a reference to a PostgreSQL or duckDB table
    * Call `cmf.link.utils.probabilities` to write its output to the proabilities table
* You must implement the `LINKER_NAME` and `LINKER_DESCRIPTION` global variables, which provide the name users will pass to `cmf.linker()`, and added to experiment names
* Consider putting a "Help" section in your docstrings for the `cmf.report.linkers(linker=, arg=)` function to show the user. If not present, it'll show the whole docstring
* Consider adding a demonstation notebook in `notebooks/` to help users see how to use the intricacies of your method
 
Linkers are composed based on the above, and the following public methods are added for users:

* `get_data()`
* `report()`
* `to_clusters()`
* `save()`
* `load()`

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
