from typing import Any, Optional

from matchbox.server.postgresql.db import Base


class MatchboxConnectionError(Exception):
    """Connection to Matchbox's backend database failed."""


class MatchboxDBDataError(Exception):
    """Data doesn't exist in the Matchbox source table."""

    def __init__(
        self,
        message: str = None,
        source: Base = None,
        data: Optional[Any] = None,
    ):
        if message is None:
            message = "Data doesn't exist in Matchbox."
            if source is not None:
                message += f"\nTable: {source.__tablename__}"
            if data is not None:
                message += f"\nData: {str(data)}"

        super().__init__(message)
        self.source = source
        self.data = data


class MatchboxSourceTableError(Exception):
    """Tables not found in wider database, outside of the framework."""

    def __init__(
        self,
        message: str = None,
        table_name: str = None,
    ):
        if message is None:
            message = "Table doesn't exist in wider database."
            if table_name is not None:
                message += f"\nTable name: {table_name}"

        super().__init__(message)
        self.table_name = table_name
