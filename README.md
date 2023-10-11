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

I want the name of a company as it appears in Companies House and its ID from Data Hub.

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

Consider the HMRC Exporters table. The same company appears hundreds of times. If you pass it to the service, it will assume you want every row. To aggregate, use the service in a subquery.

```sql
select
    agg.company_name,
    agg.data_hub_id,
    count(agg."month")
from (
    select
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
) agg;

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
    return_type="pandas"
)

df.groupby(["company_name", "data_hub_id"]).count("month")
```

### I have a public dataset I want to connect to existing companies

I want to connect data.data_hub_statistics to the existing company clusters. It contains a data hub ID and company name, both of which were entered by hand, so they'll need cleaning up.

What fields exist to connect my dataset to?

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

What linkers already work with these tables and fields? How have other people cleaned this data?

```python
import cmf.utils as cmfu
cmfu.link_report(
    {
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ]
    }
)
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

clusters: A dictionary of tables and fields you wish to link into
dataset: A dictionary of one table and fields you wish to join in
link_settings: A dictionary of fields you wish to match on

You may also wish to define:

cluster_pipeline: A dictionary of fields and cleaning functions you wish to apply in order to the clusters data
dataset_pipeline: A dictionary of fields and cleaning functions you wish to apply in order to the dataset
```

Let's start with those required fields.

I could look up how other people have done this, but let's let the object itself help me configure it.

```python
CMSLinker.help('clusters')
```

```console
foo@bar:~$
The Company Matching Service linker requires a clusters dictionary in the following shape:

{
    "table_1": [
        "field_1",
        "field_2",
        ...
    ],
    "table_2": [
        "field_1",
        "field_2",
        ...
    ],
}

Use CMSLinker.help('clusters', table="table_1", fields=["field_1, field_2"] to let the linker format this dictionary for you.
```

I'll follow this process for each required field until I'm ready to configure my linker.

```python
data_hub_statistics_linker = CMSLinker(
    clusters={
        "companieshouse.companies": [
            "company_name"
        ],
        "dit.data_hub__companies": [
            "data_hub_id"
        ]
    },
    dataset={
        "data.data_hub_statistics": [
            "comp_name as company_name",
            "dh_id as data_hub_id"
        ]
    },
    link_settings={
        "company_name": {
            "cluster": "company_name",
            "dimension": "company_name"
        },
        "postcode": {
            "cluster": "data_hub_id",
            "dimension": "data_hub_id"
        }
    }
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
```

Let's just use these right out of the box. Every linker contains a `prepare()` function to clean the data, and a `link()` function to do the linking. As we've already supplied `link_settings`, we can `link()` right away. 

```python
import cmf.features.clean_basic as cmf_cb
import cmf.features.clean_complex as cmf_cc

data_hub_statistics_linker.prepare(
    cluster_pipeline={
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
        "company_name": {
            "function": cmf_cc.clean_company_name,
            "arguments": {
                "column": "company_name"
            }
        }
    },
    dataset_pipeline={
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
        "company_name": {
            "function": cmf_cc.clean_company_name,
            "arguments": {
                "column": "company_name"
            }
        }
    }
)

```

### I have a private dataset I want to connect to existing companies

🛠 Coming soon!

### I have a new matching methodology I want to implement

🛠 Coming soon!

## Release metrics

🛠 Coming soon!

## Usage

🛠 Coming soon!

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
