import json
from pathlib import Path
from typing import Iterable

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from matchbox.common.factories import generate_dummy_probabilities
from matchbox.common.hash import HASH_FUNC, hash_data, hash_values
from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)


class HashIDMap:
    """An object to help map between IDs and hashes.

    When given a set of IDs, returns their hashes. If any ID doesn't have a hash,
    it will error.

    When given a set of hashes, it will return their IDs. If any don't have IDs, it
    will create one and return it as part of the set.

    Args:
        start: The first integer to use for new IDs
        lookup (optional): A lookup table to use for existing hashes
    """

    def __init__(self, start: int, lookup: pa.Table = None):
        self._next_int = start
        if not lookup:
            self._lookup = pa.Table(
                pa.schema([("id", pa.uint64()), ("hash", pa.large_binary())])
            )
        else:
            self._lookup = lookup

        if self._lookup.column_names != ["id", "hash"]:
            raise ValueError("Lookup table must have columns 'id' and 'hash'")

    def get_hashes(self, ids: pa.UInt64Array) -> pa.LargeBinaryArray:
        """Returns the hashes of the given IDs."""
        indices = pc.index_in(ids, self._lookup["id"])

        if pc.any(pc.is_null(indices)).as_py():
            m_mask = pc.is_null(indices)
            m_ids = pc.filter(ids, m_mask)

            raise ValueError(
                f"The following IDs were not found in lookup table: {m_ids.to_pylist()}"
            )

        return pc.take(self._lookup["hash"], indices)

    def get_ids(self, hashes: pa.LargeBinaryArray) -> pa.UInt64Array:
        """Returns the IDs of the given hashes, assigning new IDs for unknown hashes."""
        indices = pc.index_in(hashes, self._lookup["hash"])
        new_hashes = pc.unique(pc.filter(hashes, pc.is_null(indices)))

        if len(new_hashes) > 0:
            new_ids = pa.array(
                range(self._next_int, self._next_int + len(new_hashes)),
                type=pa.uint64(),
            )

            new_entries = pa.Table.from_arrays(
                [new_ids, new_hashes], names=["id", "hash"]
            )

            self._next_int += len(new_hashes)
            self._lookup = pa.concat_tables([self._lookup, new_entries])

            indices = pc.index_in(hashes, self._lookup["hash"])

        return pc.take(self._lookup["id"], indices)


class IDCreator:
    """
    A generator of incremental integer IDs from positive and negative integers.

    Positive integers will be returned as they are, while a new ID will be generated
    for each negative integer.
    """

    def __init__(self, start: int):
        self.id_map = dict()
        self._next_int = start

    def create(self, temp_ids: list[int]) -> list[int]:
        results = []
        for ti in temp_ids:
            if ti >= 0:
                results.append(ti)
            elif ti in self.id_map:
                results.append(self.id_map[ti])
            else:
                self.id_map[ti] = self._next_int
                results.append(self._next_int)
                self._next_int += 1

        return results

    def reset_mapping(self):
        self.__init__(self._next_int)

        return self


def _hash_list_int(li: list[int]) -> list[bytes]:
    return [HASH_FUNC(str(i).encode("utf-8")).digest() for i in li]


def _unique_clusters(
    all_parents: Iterable[int], all_probabilities: Iterable[int]
) -> tuple[list[int], list[float]]:
    ll = set()
    clusters = []
    probabilities = []
    for parent, prob in zip(all_parents, all_probabilities, strict=True):
        if parent in ll:
            continue
        else:
            ll.add(parent)
            clusters.append(parent)
            probabilities.append(prob / 100)
    return clusters, probabilities


def generate_sources() -> pa.Table:
    """
    Generate sources table.

    Returns:
        PyArrow sources table
    """
    sources_resolution_id = [1, 2]
    sources_alias = ["alias1", "alias2"]
    sources_schema = ["dbt", "dbt"]
    sources_table = ["companies_house", "hmrc_exporters"]
    sources_id = ["company_number", "id"]
    sources_indices = [
        {
            "literal": ["col1", "col2", "col3"],
            "alias": ["col1", "col2", "col3"],
        },
        {
            "literal": ["col1", "col2", "col3"],
            "alias": ["col1", "col2", "col3"],
        },
    ]
    sources_indices = [json.dumps(si) for si in sources_indices]
    return pa.table(
        {
            "resolution_id": pa.array(sources_resolution_id, type=pa.uint64()),
            "alias": pa.array(sources_alias, type=pa.string()),
            "schema": pa.array(sources_schema, type=pa.string()),
            "table": pa.array(sources_table, type=pa.string()),
            "id": pa.array(sources_id, type=pa.string()),
            "indices": pa.array(sources_indices, type=pa.string()),
        }
    )


def generate_resolutions() -> pa.Table:
    """
    Generate resolutions table.

    Returns:
        PyArrow resolutions table
    """
    resolutions_resolution_id = [1, 2, 3, 4, 5]
    resolutions_name = ["source1", "source2", "dedupe1", "dedupe2", "link"]
    resolutions_resolution_hash = [
        HASH_FUNC(rid.encode("utf-8")).digest() for rid in resolutions_name
    ]
    resolutions_type = ["dataset", "dataset", "model", "model", "model"]
    resolutions_float = [None, None, 0.8, 0.8, 0.9]

    return pa.table(
        {
            "resolution_id": pa.array(resolutions_resolution_id, type=pa.uint64()),
            "resolution_hash": pa.array(
                resolutions_resolution_hash, type=pa.large_binary()
            ),
            "type": pa.array(resolutions_type, type=pa.string()),
            "name": pa.array(resolutions_name, type=pa.string()),
            "description": pa.array(resolutions_name, type=pa.string()),
            "truth": pa.array(resolutions_float, type=pa.float64()),
        }
    )


def generate_resolution_from() -> pa.Table:
    """
    Generate resolution_from table.

    Returns:
        PyArrow resolution_from table
    """
    # 1 and 2 are sources; 3 and 4 are dedupers; 5 is a linker
    resolution_parent = [1, 1, 3, 2, 2, 4]
    resolution_child = [3, 5, 5, 4, 5, 5]
    resolution_level = [1, 2, 1, 1, 2, 1]
    resolution_truth_cache = [None, None, 0.7, None, None, 0.7]

    return pa.table(
        {
            "parent": pa.array(resolution_parent, type=pa.uint64()),
            "child": pa.array(resolution_child, type=pa.uint64()),
            "level": pa.array(resolution_level, type=pa.uint32()),
            "truth_cache": pa.array(resolution_truth_cache, type=pa.float64()),
        }
    )


def generate_cluster_source(range_left: int, range_right: int) -> pa.Table:
    """
    Generate cluster table containing rows for source rows.

    Args:
        range_left: first ID to generate
        range_right: last ID to generate, plus one
    Returns:
        PyArrow cluster table
    """

    def create_source_pk(li: list[int]) -> list[list[str]]:
        return [[str(i)] for i in li]

    source = list(range(range_left, range_right))

    return pa.table(
        {
            "cluster_id": pa.array(source, type=pa.uint64()),
            "cluster_hash": pa.array(_hash_list_int(source), type=pa.large_binary()),
            "dataset": pa.array([1] * len(source), type=pa.uint64()),
            "source_pk": pa.array(create_source_pk(source), type=pa.list_(pa.string())),
        }
    )


def generate_result_tables(
    left_ids: Iterable[int],
    right_ids: Iterable[int] | None,
    resolution_id: int,
    id_creator: IDCreator,
    n_components: int,
    n_probs: int,
    prob_min: float = 0.6,
    prob_max: float = 1,
) -> tuple[list[int], pa.Table, pa.Table, pa.Table]:
    """
    Generate probabilities, contains and clusters tables.

    Args:
        left_ids: list of IDs for rows to dedupe, or for left rows to link
        right_ids: list of IDs for right rows to link
        resolution_id: ID of resolution for this dedupe or link model
        id_creator: an IDCreator instance
        n_components: number of implied connected components
        n_probs: total number of probability edges to be generated
        prob_min: minimum value for probabilities to be generated
        prob_max: maximum value for probabilities to be generated

    Returns:
        Tuple with 1 list of top-level clusters and 3 PyArrow tables, for probabilities,
        contains and clusters
    """
    probs = generate_dummy_probabilities(
        left_ids, right_ids, [prob_min, prob_max], n_components, n_probs
    )

    # Create a lookup table for hashes
    all_probs = pa.concat_arrays(
        [probs["left"].combine_chunks(), probs["right"].combine_chunks()]
    )
    lookup = pa.table(
        {
            "id": all_probs,
            "hash": pa.array(
                [hash_data(p) for p in all_probs.to_pylist()], type=pa.large_binary()
            ),
        }
    )

    hm = HashIDMap(start=id_creator._next_int, lookup=lookup)

    # Join hashes, probabilities and components
    probs_with_ccs = attach_components_to_probabilities(
        pa.table(
            {
                "left": hm.get_hashes(probs["left"]),
                "right": hm.get_hashes(probs["right"]),
                "probability": probs["probability"],
            }
        )
    )

    # Calculate hierarchies
    hierarchy = to_hierarchical_clusters(
        probabilities=probs_with_ccs,
        hash_func=hash_values,
        dtype=pa.large_binary,
    )

    # Shape into tables
    parent_ids = hm.get_ids(hierarchy["parent"])
    child_ids = hm.get_ids(hierarchy["child"])
    unique_parent_ids = pc.unique(parent_ids)
    mask = pc.invert(pc.is_in(unique_parent_ids, pc.unique(child_ids)))
    top_clusters = pc.filter(unique_parent_ids, mask)

    probabilities_table = pa.table(
        {
            "resolution": pa.array(
                [resolution_id] * hierarchy.shape[0], type=pa.uint64()
            ),
            "cluster": parent_ids,
            "probability": hierarchy["probability"],
        }
    )

    contains_table = pa.table(
        {
            "parent": parent_ids,
            "child": child_ids,
        }
    )

    clusters_table = pa.table(
        {
            "cluster_id": unique_parent_ids,
            "cluster_hash": hm.get_hashes(unique_parent_ids),
            "dataset": pa.array([None] * len(unique_parent_ids), type=pa.uint64()),
            "source_pk": pa.array(
                [None] * len(unique_parent_ids), type=pa.list_(pa.string())
            ),
        }
    )

    return (top_clusters, probabilities_table, contains_table, clusters_table)


def generate_all_tables(
    source_len: int,
    dedupe_components: int,
    dedupe_len: int,
    link_components: int,
    link_len: int,
) -> dict[str, pa.Table]:
    """
    Make all 6 backend tables. It will create two sources, one deduper for each,
    and one linker from each deduper.

    Args:
        source_len: length of each data source
        dedupe_components: number of connected components implied by each deduper
        dedupe_len: probabilities generated by each deduper
        link_components: number of connected components implied by each linker
        link_len: probabilities generated by each linker
    Returns:
        A dictionary where keys are table names and values are PyArrow tables
    """
    resolutions = generate_resolutions()
    resolution_from = generate_resolution_from()
    sources = generate_sources()

    clusters_source1 = generate_cluster_source(0, source_len)
    clusters_source2 = generate_cluster_source(source_len, source_len * 2)

    id_creator = IDCreator(source_len * 2)
    top_clusters1, probabilities_dedupe1, contains_dedupe1, clusters_dedupe1 = (
        generate_result_tables(
            clusters_source1["cluster_id"].to_pylist(),
            None,
            3,
            id_creator,
            dedupe_components,
            dedupe_len,
        )
    )

    top_clusters2, probabilities_dedupe2, contains_dedupe2, clusters_dedupe2 = (
        generate_result_tables(
            clusters_source2["cluster_id"].to_pylist(),
            None,
            4,
            id_creator.reset_mapping(),
            dedupe_components,
            dedupe_len,
        )
    )

    _, probabilities_link, contains_link, clusters_link = generate_result_tables(
        top_clusters1,
        top_clusters2,
        5,
        id_creator.reset_mapping(),
        link_components,
        link_len,
    )

    probabilities = pa.concat_tables(
        [probabilities_dedupe1, probabilities_dedupe2, probabilities_link]
    )
    contains = pa.concat_tables([contains_dedupe1, contains_dedupe2, contains_link])
    clusters = pa.concat_tables(
        [
            clusters_source1,
            clusters_source2,
            clusters_dedupe1,
            clusters_dedupe2,
            clusters_link,
        ]
    )

    return {
        "resolutions": resolutions,
        "resolution_from": resolution_from,
        "sources": sources,
        "probabilities": probabilities,
        "contains": contains,
        "clusters": clusters,
    }


@click.command()
@click.option("-s", "--settings", type=str, required=True)
@click.option("-o", "--output_dir", type=click.Path(exists=True, path_type=Path))
def main(settings, output_dir):
    PRESETS = {
        "xs": {
            "source_len": 10_000,
            "dedupe_components": 8000,
            "dedupe_len": 2000,
            "link_components": 6000,
            "link_len": 10_000,
        },
        "s": {
            "source_len": 100_000,
            "dedupe_components": 80_000,
            "dedupe_len": 20_000,
            "link_components": 60_000,
            "link_len": 100_000,
        },
        "m": {
            "source_len": 1_000_000,
            "dedupe_components": 800_000,
            "dedupe_len": 200_000,
            "link_components": 600_000,
            "link_len": 1_000_000,
        },
        "l": {
            "source_len": 10_000_000,
            "dedupe_components": 8_000_000,
            "dedupe_len": 2_000_000,
            "link_components": 6_000_000,
            "link_len": 10_000_000,
        },
        "xl": {
            "source_len": 100_000_000,
            "dedupe_components": 80_000_000,
            "dedupe_len": 20_000_000,
            "link_components": 60_000_000,
            "link_len": 100_000_000,
        },
    }

    if not output_dir:
        output_dir = Path.cwd() / "data" / "all_tables"
    if settings not in PRESETS:
        raise ValueError(f"Settings {settings} are invalid")

    config = PRESETS[settings]
    source_len = config["source_len"]
    dedupe_components = config["dedupe_components"]
    dedupe_len = config["dedupe_len"]
    link_len = config["link_len"]
    link_components = config["link_components"]

    all_tables = generate_all_tables(
        source_len=source_len,
        dedupe_components=dedupe_components,
        dedupe_len=dedupe_len,
        link_components=link_components,
        link_len=link_len,
    )

    output_dir /= settings
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in all_tables.items():
        pq.write_table(table, output_dir / f"{name}.parquet")


if __name__ == "__main__":
    main()
