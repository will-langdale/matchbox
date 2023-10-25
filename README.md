# 🔗 Company matching framework

A match orchestration framework to allow the comparison, validation, and orchestration of the best match methods for the company matching job.

We envisage this forming one of three repos in the Company Matching Framework:

* `company-matching-framework`, this repo. A Python library for creating data linkage and deduplication pipelines over a shared relational database
* `company-matching-framework-dash`, or https://matching.data.trade.gov.uk/. A dashboard for verifying links and deduplications, and comparing the performance metrics of different approaches. Uses `company-matching-framework`
* `company-matching-framework-pipeline`. The live pipeline of matching and deduping methods, running in production. Uses `company-matching-framework`

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

See [the aspirational README](references/README_aspitational.md) for how we envisage the finished version of this Python library will be used.

## Release metrics

🛠 Coming soon!

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
