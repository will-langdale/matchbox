from os import getenv

import httpx


def _url(path: str) -> str:
    """
    Return path prefixed by API root, determined from environment
    """
    api_root = getenv("API__ROOT")
    if api_root is None:
        raise RuntimeError("API__ROOT needs to be defined in the environment")

    return api_root + path


def _get_resolution_graph() -> str:
    return httpx.get(_url("/report/resolutions")).text
