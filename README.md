# 🔥 Matchbox (neé Company Matching Framework)

Record matching is a chore. Matchbox is a march pipeline orchestration tool that aims to:

* Make matching an iterative, collaborative, measurable problem
* Allow organisations to know they have matching records without having to share the data
* Allow matching pipelines to run iteratively

Matchbox doesn't store raw data, instead indexing the data in your warehouse and leaving permissioning at the level of the user, service or pipeline. 

## Use cases

### Data archiects and engineers

* Reconcile entities across disparate datasets
* Rationalise about the quality of different entity matching pipelines and serve up the best
* Run matching pipelines without recomputing them every time
* Lay the foundation for the nouns of a semantic layer

### Data analysts and scientists

* Use your team's best matching methods when retrieving entities, always
* Measurably improve methodologies when they don't work for you
* When you link new datasets, allow others to use your work easily and securely

### Service owners

* Understand the broader business entities in your service, not just what you have
* Enrich other services with data generated in yours without giving away any permissioning powers
* Empower your users to label matched entities and let other services use that information

## Structure

> [!CAUTION]
> Some of the below is aspirational. Matchbox is in alpha and under heavy construction.

The project is loosely formed into a client/server structure.

### Server

The parts of matchbox intended for deployment. Allows different backends as long as they can meet the standards of the adapter and tests.

### Client

The parts of matchbox intended for users and services to call a matchbox server, and to insert matched data in the right structure.

Pipelines using this part of matchbox will:

1. Use `matchbox.query()` to retrive source data from a particular model's perspective
2. Use `matchbox.make_model()` with `matchbox.dedupers` and `matchbox.linkers` to generate probabilities for a new model
3. Upload the probabilites to matchbox with `results.to_matchbox()`
4. Label data, or use existing data, to decide the probability threshold that you're willing to consider "truth" for your new model
5. Use `model.roc_curve()` and other tools to make your decision
6. Use `model.set_truth_threshold()` to codify it

With the truth threshold set to `1.0` by default, deterministic methodologies are ready for others to use from step 3!

## Development

This project is managed by [uv](https://docs.astral.sh/uv/), linted and formated with [ruff](https://docs.astral.sh/ruff/), and tested with [pytest](https://docs.pytest.org/en/stable/).

Task running is done with [just](https://just.systems/man/en/). To see all available commands:

```console
just -l
```
