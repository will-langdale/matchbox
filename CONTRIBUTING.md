# Developer's guide

This document describes how you can get started with developing Matchbox.

## Dependencies

* [Python 3.11+](https://www.python.org)
* [uv](https://docs.astral.sh/uv/)
* [Docker](https://www.docker.com)
* [TruffleHog](https://github.com/trufflesecurity/trufflehog)
* [PostgreSQL](https://www.postgresql.org)
* [pre-commit](https://pre-commit.com)
* [just](https://just.systems/man/en/)
* [Node.js](https://nodejs.org/en)

## Setup

Set up environment variables by creating a `.env` file under project directory. See [/environments/dev_local.env](./environments/dev_local.env) for sensible defaults.

This project is managed by [uv](https://docs.astral.sh/uv/), linted and formated with [ruff](https://docs.astral.sh/ruff/), and tested with [pytest](https://docs.pytest.org/en/stable/). [Docker](https://www.docker.com) is used for local development. Documentation is build using [11ty](https://www.11ty.dev) via [Node.js](https://nodejs.org/en).

To install all dependencies for this project, run:

```cnosole
uv sync --all-extras
```

Secret scanning is done with [TruffleHog](https://github.com/trufflesecurity/trufflehog).

For security, use of [pre-commit](https://pre-commit.com) is expected. Ensure your hooks are installed with `pre-commit install`.

Task running is done with [just](https://just.systems/man/en/). To see all available commands:

```console
just -l
```

## Run tests

A just task is provided to run all tests.

```console
just test
```

If you're running tests with some other method, such as your IDE or pytest directly, you'll need to start the local backends and mock warehouse in Docker.

```console
docker compose up -d --wait
```

## Standards

### Code

When contributing to the main matchbox repository and its associated repos, we try to follow consistent standards. Python code should be:

* Unit tested, and pass new and existing tests
* Documented via docstrings, in the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* Linted and auto-formatted (`just format`)
* Type hinted (within reason)
* Using env files and dotenv for setting environment variables
* Structured as a Python package with `pyprojects.toml`
* Using dependencies managed automatically by uv
* Integrated with justfile when relevant
* Documented, for example, `README.md` files where relevant

### Git

We commit as frequently as possible. We keep our commits as atomic as possible. We never push straight to main, instead we merge feature branches. Before merging to main, branches are peer reviewed.

> [!CAUTION]
> pre-commit **must** be turned on. Any secrets you commit to the repo are your own responsibility.
