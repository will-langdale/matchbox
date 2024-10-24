from typing import Any, Optional


class MatchboxConnectionError(Exception):
    """Connection to Matchbox's backend database failed."""


class MatchboxValidatonError(Exception):
    """Validation of data failed."""


class MatchboxModelError(Exception):
    """Model not found."""

    def __init__(self, message: str = None, model_name: str = None):
        if message is None:
            message = "Model not found."
            if model_name is not None:
                message = f"Model {model_name} not found."

        super().__init__(message)
        self.model_name = model_name


class MatchboxDatasetError(Exception):
    """Dataset not found."""

    def __init__(
        self,
        message: str = None,
        db_schema: str | None = None,
        db_table: str | None = None,
    ):
        if message is None:
            message = "Dataset not found."
            if db_table is not None:
                message = f"Dataset {db_schema or ''}.{db_table} not found."

        super().__init__(message)
        self.db_schema = db_schema
        self.db_table = db_table


class MatchboxDataError(Exception):
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
