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

# Run Python tests
test:
    docker compose up -d --wait
    uv run pytest

# Run a local documentation development server
docs:
    cd docs && npm start
