"""Object to store and retrieve data on the server."""

from matchbox.client.dags import DAG


class Collection:
    """Client to active and pending DAGs on the server."""

    def __init__(self, name: str):
        """Checks existence of collection on server and initialises local object."""
        self.name = name

    def dag(self) -> DAG:
        """Create new pending DAG and returns builder."""
        return DAG(collection_name=self.name)
