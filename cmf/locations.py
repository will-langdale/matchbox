from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_HOME = os.path.join(PROJECT_DIR, "data")
DATA_SUBDIR = {
    "external": os.path.join(DATA_HOME, "external"),
    "raw": os.path.join(DATA_HOME, "raw"),
    "interim": os.path.join(DATA_HOME, "interim"),
    "processed": os.path.join(DATA_HOME, "processed"),
}

MODELS_HOME = os.path.join(PROJECT_DIR, "models")

NOTEBOOKS_HOME = os.path.join(PROJECT_DIR, "notebooks")

REFERENCES_HOME = os.path.join(PROJECT_DIR, "references")

OUTPUTS_HOME = os.path.join(PROJECT_DIR, "outputs")
FIGURES_HOME = os.path.join(OUTPUTS_HOME, "figures")
