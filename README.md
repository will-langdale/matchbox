# 🔥 Matchbox DB

Record matching is a chore. Matchbox is a match pipeline orchestration tool that aims to:

* Make matching an iterative, collaborative, measurable problem
* Compose sources, dedupers and linkers and make the results very easy to query
* Allow organisations to know they have matching records without having to share the data
* Allow matching pipelines to run iteratively
* Support batch and real-time matching 

Matchbox doesn't store raw data, instead indexing the data in your warehouse and leaving permissioning at the level of the user, service or pipeline. 

## Installation
To install the matchbox client:
```
pip install "matchbox-db"
```

To install the full package, including the server features:

```
pip install "matchbox-db[server]"
```

## Running the server locally

To run the server locally, run:

```bash
docker compose up --build
```

## Running the server locally with Datadog (monitoring) integration

1. Run:

   ```
   cp ./environments/datadog-agent-private-sample.env ./environments/.datadog-agent-private.env
   ```

2. Populate the newly-created ` ./environments/.datadog-agent-private.env` with a Datadog API key.


3. Run the server using:

   ```bash
   docker compose --profile monitoring up --build
   ```


## Use cases

### Data architects and engineers

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

If the dataset isn't already in matchbox, it'll need to be indexed.

API endpoints with write properties require API Key authentication. The API key should be stored in the client environment or .env as a variable named `MB__CLIENT__API_KEY`.

Pipelines using this part of matchbox will:

1. Use `matchbox.query()` to retrieve source data from the perspective of a particular resolution point
2. Use `matchbox.process()` to clean the data with standardised processes
3. Use `matchbox.make_model()` with `matchbox.dedupers` and `matchbox.linkers` to create a new model
4. Generate probabilistic model outputs using `model.run()`
5. Upload the probabilites to matchbox with `results.to_matchbox()`
6. Label data, or use existing data, to decide the probability threshold that you're willing to consider "truth" for your new model
7. Use `model.roc_curve()` and other tools to make your decision
8. Update `model.truth` to codify it

With the truth threshold set to `1.0` by default, deterministic methodologies are ready for others to use from step five!

## Development

See our full development guide and coding standards in [CONTRIBUTING.md](./docs/contributing.md)
