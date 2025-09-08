# PostgreSQL migration
mod migrate 'src/matchbox/server/postgresql/justfile'
# Evaluation app
mod eval 'src/matchbox/client/eval/justfile'

# Build and run all containers
build *DOCKER_ARGS:
    uv sync --extra server
    MB_VERSION=$(uv run --frozen python -m setuptools_scm) \
    docker compose --env-file=environments/containers.env up --build {{DOCKER_ARGS}}

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Run a local documentation development server
docs:
    uv run mkdocs serve

# Reformat and lint
format:
    uv run ruff format .
    uv run ruff check . --fix
    uvx uv-sort pyproject.toml

# Scan for secrets
scan:
    bash -c "docker run -v "$(pwd):/repo" -i \
        --rm trufflesecurity/trufflehog:latest git \
        file:///repo  --since-commit HEAD --fail"

# Run Python tests (usage: just test [local|docker])
test ENV="":
    #!/usr/bin/env bash
    if [[ "{{ENV}}" == "local" ]]; then
        uv run pytest -m "not docker"
    elif [[ "{{ENV}}" == "docker" ]]; then
        just build -d
        uv run pytest -m "docker"
    else
        just build -d
        uv run pytest
    fi
