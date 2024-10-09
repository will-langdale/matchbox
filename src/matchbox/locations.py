import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

CMF = os.path.join(PROJECT_DIR, "cmf")
TEST = os.path.join(PROJECT_DIR, "test")
