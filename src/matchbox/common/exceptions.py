from typing import Any


class MatchboxConnectionError(Exception):
    """Connection to Matchbox's backend database failed."""


class MatchboxSourceColumnError(Exception):
    """Source columns diverge with the warehouse"""


class MatchboxSourceEngineError(Exception):
    """Engine must be available in Source"""


class MatchboxServerFileError(Exception):
    """There was a problem with file transfer."""

    def __init__(self, message: str | None = None, file: str | None = None):
        if message is None:
            message = "There was a problem with file transfer."
            if file is not None:
                message = f"There was a problem with file transfer: {file}."

        super().__init__(message)
        self.file = file


class MatchboxServerResolutionError(Exception):
    """Resolution not found."""

    def __init__(self, message: str | None = None, resolution_name: str | None = None):
        if message is None:
            message = "Resolution not found."
            if resolution_name is not None:
                message = f"Resolution {resolution_name} not found."

        super().__init__(message)
        self.resolution_name = resolution_name


class MatchboxServerSourceError(Exception):
    """Source not found on the server."""

    def __init__(
        self,
        message: str = None,
        address: str | None = None,
    ):
        if message is None:
            message = "Source not found on matchbox."
            if address:
                message = f"Source ({address}) not found."

        super().__init__(message)
        self.address = address


class MatchboxDataError(Exception):
    """Data doesn't exist in the Matchbox source table."""

    def __init__(
        self,
        message: str | None = None,
        table: str | None = None,
        data: Any | None = None,
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
        message: str | None = None,
        table_name: str | None = None,
    ):
        if message is None:
            message = "Table doesn't exist in wider database."
            if table_name is not None:
                message += f"\nTable name: {table_name}"

        super().__init__(message)
        self.table_name = table_name
