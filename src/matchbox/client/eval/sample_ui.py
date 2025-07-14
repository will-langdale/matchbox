"""Launcher for evaluation UI which writes some mock data."""

import atexit
import subprocess
import sys


def setup_mock_database():
    """Add some mock data to test the evaluation UI."""
    # TODO: complete
    print("Mocking DB")


def cleanup_database():
    """Clean up mock data for the evaluation UI."""
    # TODO: complete
    print("Cleaning up DB")


if __name__ == "__main__":
    setup_mock_database()

    atexit.register(cleanup_database)

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/matchbox/client/eval/ui.py"]
    )
