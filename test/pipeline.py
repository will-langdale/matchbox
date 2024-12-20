import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
from rich.logging import RichHandler

from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
pipeline_logger = logging.getLogger("mb_pipeline")

ROOT = Path(__file__).parent.parent


@contextmanager
def timer(description: str):
    start = time.time()
    yield
    elapsed = time.time() - start

    if elapsed >= 60:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        time_str = f"{minutes} min {seconds:.1f} sec"
    else:
        time_str = f"{elapsed:.2f} seconds"

    pipeline_logger.info(f"{description} in {time_str}")


if __name__ == "__main__":
    with timer("Full pipeline completed"):
        with timer("Read table"):
            table = pq.read_table(Path.cwd() / "data/hierarchical_cc20k.parquet")

        pipeline_logger.info(f"Processing {len(table):,} records")

        with timer("Added components"):
            cc = attach_components_to_probabilities(table)

        with timer("Built hierarchical clusters"):
            out = to_hierarchical_clusters(cc)
