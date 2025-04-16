This document describes how you can get started with developing Matchbox.

## Dependencies

* [Python 3.11+](https://www.python.org)
* [uv](https://docs.astral.sh/uv/)
* [Docker](https://www.docker.com)
* [TruffleHog](https://github.com/trufflesecurity/trufflehog)
* [PostgreSQL](https://www.postgresql.org)
* [pre-commit](https://pre-commit.com)
* [just](https://just.systems/man/en/)

## Setup

Set up environment variables by creating a `.env` file under project directory. See [`/environments/dev_local.env`](https://github.com/uktrade/matchbox/blob/main/environments/dev_local.env) for sensible defaults.

This project is managed by [uv](https://docs.astral.sh/uv/), linted and formated with [ruff](https://docs.astral.sh/ruff/), and tested with [pytest](https://docs.pytest.org/en/stable/). [Docker](https://www.docker.com) is used for local development. Documentation is build using [mkdocs](https://www.mkdocs.org).

To install all dependencies for this project, run:

```shell
uv sync --all-extras
```

Secret scanning is done with [TruffleHog](https://github.com/trufflesecurity/trufflehog).

For security, use of [pre-commit](https://pre-commit.com) is expected. Ensure your hooks are installed:

```shell
pre-commit install
```

Task running is done with [just](https://just.systems/man/en/). To see all available commands:

```shell
just -l
```

## Run tests

A just task is provided to run all tests.

```shell
just test
```

If you're running tests with some other method, such as your IDE or pytest directly, you'll need to start the local backends and mock warehouse in Docker.

```shell
docker compose up -d --wait
```

## Database Migrations for PostgreSQL backend

Migrations for the PostgreSQL backend are managed by [alembic](https://alembic.sqlalchemy.org/en/latest/). If you have made an alteration to the database through the ORM code (but not yet applied this), and if you have run `docker compose up` to ensure the database container is running, then you can create verify a migration script would be created (without creating one) with:

```shell
just migration-check
```

Or actually create the new migration script by running:

```shell
just migration-generate "< enter descriptive message >"
```

These commands will auto-detect the difference between the ORM and the database container; you should then check `src/matchbox/server/postgresql/alembic/versions/` for the new migration script - verify that the autogenerate matches your expectations (see the [documentation for known failure modes](https://alembic.sqlalchemy.org/en/latest/autogenerate.html#what-does-autogenerate-detect-and-what-does-it-not-detect)). Then apply with:

```shell
just migration-apply
```

Note in this command that the special name "head" refers to the latest migration script. You can similarly upgrade or downgrade to any specific migration script by referencing its revision ID; "base" is also used to refer to the earliest migration script. Note you should not make alterations to the database through other mechanisms (such as directly running SQL commands) as this will conflict with the completeness of the migration scripts. In such a case it would be best to run `just migration-reset` followed by `just migration-apply` to effectively refresh and restore the current expected "head" state, and then create a migration script for your planned changes.


## Debugging

We have a VSCode default debugging profile called "API debug", which allows to e.g. set breakpoints on the API when running tests. After running this profile, change your `.env` file  as follows:

- Change the `MB__CLIENT__API_ROOT` variable to redirect tests to use the debug port (8080)
- Disable time-outs by commenting out the `MB__CLIENT__TIMEOUT` variable

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

!!! warning
    Pre-commit **must** be turned on. Any secrets you commit to the repo are your own responsibility.

### AI

In order to help reviewers prioritise their time appropriately, we expect any use of AI to be declared in your PR comment.
