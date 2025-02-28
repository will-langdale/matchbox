# Make datasets table
matchbox:
    uv run python src/matchbox/admin.py --datasets datasets.toml

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Reformat and lint
format:
    uv run ruff format .
    uv run ruff check . --fix

# Scan for secrets
scan:
    bash -c "docker run -v "$(pwd):/repo" -i --rm trufflesecurity/trufflehog:latest git file:///repo"

# Run Python tests (usage: just test [local|docker])
test ENV="":
    #!/usr/bin/env bash
    if [[ "{{ENV}}" == "local" ]]; then
        uv run pytest -m "not docker"
    elif [[ "{{ENV}}" == "docker" ]]; then
        docker compose up --build -d --wait
        uv run pytest -m "docker"
    else
        docker compose up --build -d --wait
        uv run pytest
    fi

# Run a local documentation development server
docs:
    uv run mkdocs serve
