from os import getenv

import httpx

from matchbox.common.graph import ResolutionGraph


def _url(path: str) -> str:
    """
    Return path prefixed by API root, determined from environment
    """
    api_root = getenv("API__ROOT")
    if api_root is None:
        raise RuntimeError("API__ROOT needs to be defined in the environment")

    return api_root + path


def get_resolution_graph() -> ResolutionGraph:
    G = httpx.get(_url("/report/resolutions")).json()
    return ResolutionGraph.model_validate(G)
