"""PR1 compatibility shim surface to remove in PR2.

This module isolates temporary model-era compatibility logic:
- legacy model->resolver path translation,
- legacy DAG run projection,
- upload-time implicit resolver synthesis from model edge results.
"""

# TODO: remove shim in Resolution PR2
import json

import pyarrow as pa
from pyarrow import Table

from matchbox.common.arrow import SCHEMA_RESOLVER_UPLOAD
from matchbox.common.dtos import (
    ModelResolutionPath,
    Resolution,
    ResolutionPath,
    ResolutionType,
    ResolverConfig,
    ResolverResolutionPath,
    Run,
)
from matchbox.common.hash import hash_arrow_table
from matchbox.common.transform import DisjointSet


def compat_resolver_path(
    resolution: ResolutionPath | None,
) -> ResolverResolutionPath | None:
    """Translate legacy model-oriented paths to resolver paths."""
    if resolution is None:
        return None

    resolver_name = (
        resolution.name
        if str(resolution.name).startswith("resolver_")
        else f"resolver_{resolution.name}"
    )
    return ResolverResolutionPath(
        collection=resolution.collection,
        run=resolution.run,
        name=resolver_name,
    )


def project_run_for_legacy_dag(run: Run) -> Run:
    """Project run payload to legacy DAG shape used during PR1 compatibility."""
    thresholds_by_model: dict[str, int] = {}
    for resolution in run.resolutions.values():
        if resolution.resolution_type != ResolutionType.RESOLVER:
            continue

        resolver_settings = json.loads(resolution.config.resolver_settings)
        raw_thresholds = resolver_settings.get("thresholds", {})
        for model_name, threshold in raw_thresholds.items():
            thresholds_by_model[model_name] = int(threshold)

    projected_resolutions: dict[str, Resolution] = {}
    for resolution_name, resolution in run.resolutions.items():
        if resolution.resolution_type == ResolutionType.RESOLVER:
            continue

        if (
            resolution.resolution_type == ResolutionType.MODEL
            and resolution.truth is None
        ):
            truth = thresholds_by_model.get(str(resolution_name), 100)
            resolution = resolution.model_copy(update={"truth": truth})

        projected_resolutions[resolution_name] = resolution

    return run.model_copy(update={"resolutions": projected_resolutions})


def canonical_resolver_path_for_model(
    model_path: ModelResolutionPath,
) -> ResolverResolutionPath:
    """Return canonical resolver path for a model path."""
    return ResolverResolutionPath(
        collection=model_path.collection,
        run=model_path.run,
        name=f"resolver_{model_path.name}",
    )


def resolver_upload_from_model_results(
    results: Table,
    threshold: int,
) -> Table:
    """Build canonical resolver upload rows from model edge results."""
    node_ids: set[int] = set()
    left_ids = [int(node_id) for node_id in results["left_id"].to_pylist()]
    right_ids = [int(node_id) for node_id in results["right_id"].to_pylist()]
    probabilities = [
        int(probability) for probability in results["probability"].to_pylist()
    ]

    components = DisjointSet[int]()
    node_ids.update(left_ids)
    node_ids.update(right_ids)
    for node_id in node_ids:
        components.add(node_id)

    for left_id, right_id, probability in zip(
        left_ids, right_ids, probabilities, strict=True
    ):
        if probability >= threshold:
            components.union(left_id, right_id)

    rows: list[tuple[int, int]] = []
    for component in components.get_components():
        ordered_nodes = sorted(component)
        if not ordered_nodes:
            continue
        client_cluster_id = ordered_nodes[0]
        rows.extend((client_cluster_id, node_id) for node_id in ordered_nodes)

    if not rows:
        return pa.table(
            {
                "client_cluster_id": pa.array([], type=pa.uint64()),
                "node_id": pa.array([], type=pa.uint64()),
            },
            schema=SCHEMA_RESOLVER_UPLOAD,
        )

    client_cluster_ids, all_node_ids = zip(*rows, strict=True)
    return pa.table(
        {
            "client_cluster_id": pa.array(client_cluster_ids, type=pa.uint64()),
            "node_id": pa.array(all_node_ids, type=pa.uint64()),
        },
        schema=SCHEMA_RESOLVER_UPLOAD,
    )


def canonical_resolver_resolution_for_model(
    model_path: ModelResolutionPath,
    resolver_upload: Table,
    threshold: int,
) -> Resolution:
    """Build canonical Components resolver metadata for a model path."""
    return Resolution(
        description=f"Resolver for {model_path.name}",
        resolution_type=ResolutionType.RESOLVER,
        config=ResolverConfig(
            resolver_class="Components",
            resolver_settings=json.dumps(
                {"thresholds": {str(model_path.name): threshold}}
            ),
            inputs=(model_path.name,),
        ),
        fingerprint=hash_arrow_table(resolver_upload),
    )
