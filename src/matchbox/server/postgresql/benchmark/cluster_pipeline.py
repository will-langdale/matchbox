import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
from rich.logging import RichHandler

from matchbox.common.hash import HASH_FUNC
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


INPUT_NAME = "hierarchical_cc200k"
OUTPUT_PREFIX = "large"

if __name__ == "__main__":
    with timer("Full pipeline completed"):
        with timer("Read table"):
            table = pq.read_table(Path.cwd() / f"data/{INPUT_NAME}.parquet")

        pipeline_logger.info(f"Processing {len(table):,} records")

        with timer("Added components"):
            cc = attach_components_to_probabilities(table)

        with timer("Built hierarchical clusters"):
            hierarchy = to_hierarchical_clusters(cc)

        with timer("Created output tables"):
            fake_resolution_hash = HASH_FUNC(
                "ceci n'est pas un model".encode("utf-8")
            ).digest()

            parents_im, children_im, thresholds = (
                hierarchy.column("parent").to_numpy(),
                hierarchy.column("child").to_numpy(),
                hierarchy.column("probability").to_numpy(),
            )
            import numpy as np
            import pyarrow as pa
            from pyarrow.parquet import write_table

            im_to_pos = dict()
            next_int = max(max(parents_im), 0)
            parents = []
            children = []
            for pim in parents_im:
                if pim >= 0:
                    parents.append(pim)
                elif pim in im_to_pos:
                    parents.append(im_to_pos[pim])
                else:
                    im_to_pos[pim] = next_int
                    parents.append(next_int)
                    next_int += 1

            for cim in children_im:
                if cim >= 0:
                    children.append(cim)
                elif cim in im_to_pos:
                    children.append(im_to_pos[cim])
                else:
                    im_to_pos[cim] = next_int
                    children.append(next_int)
                    next_int += 1

            unique_clusters = np.unique(parents)

            out_clusters = pa.table(
                {
                    "id": pa.array(unique_clusters, type=pa.uint64()),
                    "dataset_id": pa.array(
                        [None] * len(unique_clusters), type=pa.uint64()
                    ),
                    "id_in_dataset": pa.array(
                        [None] * len(unique_clusters), type=pa.string()
                    ),
                }
            )

            out_contains = pa.table(
                {
                    "parent": pa.array(parents, type=pa.uint64()),
                    "child": pa.array(children, type=pa.uint64()),
                }
            )

            out_probabilities = pa.table(
                {
                    "model": pa.array(
                        [fake_resolution_hash] * len(parents), type=pa.binary()
                    ),
                    "cluster": pa.array(parents, type=pa.uint64()),
                    "probability": pa.array(thresholds, type=pa.uint64()),
                }
            )

            write_table(
                out_clusters,
                Path.cwd() / "data" / f"{OUTPUT_PREFIX}_ingest_clusters.parquet",
            )

            write_table(
                out_contains, Path.cwd() / "data" / f"{OUTPUT_PREFIX}_contains.parquet"
            )

            write_table(
                out_probabilities,
                Path.cwd() / "data" / f"{OUTPUT_PREFIX}_ingest_probabilities.parquet",
            )
