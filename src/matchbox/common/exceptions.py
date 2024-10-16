from typing import Any, Optional


class MatchboxConnectionError(Exception):
    """Connection to Matchbox's backend database failed."""


class MatchboxValidatonError(Exception):
    """Validation of data failed."""


class MatchboxDBDataError(Exception):
    """Data doesn't exist in the Matchbox source table."""

    def __init__(
        self,
        message: str = None,
        table: str = None,
        data: Optional[Any] = None,
    ):
        if message is None:
            message = "Data doesn't exist in Matchbox."
            if table is not None:
                message += f"\nTable: {table}"
            if data is not None:
                message += f"\nData: {str(data)}"

        super().__init__(message)
        self.table = table
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
