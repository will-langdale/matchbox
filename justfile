export MB__BATCH_SIZE := "10_000"
export MB__BACKEND_TYPE := "postgres"
export MB__DATASETS_CONFIG := "datasets.toml"
export MB__POSTGRES__HOST := "localhost"
export MB__POSTGRES__PORT := "5432"
export MB__POSTGRES__USER := "matchbox_user"
export MB__POSTGRES__PASSWORD := "matchbox_password"
export MB__POSTGRES__DATABASE := "matchbox"
export MB__POSTGRES__DB_SCHEMA := "mb"

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

# Run Python tests
test:
    docker compose up -d --wait
    uv run pytest

# Run development version of API
api:
	uv run fastapi dev src/matchbox/server/api.py