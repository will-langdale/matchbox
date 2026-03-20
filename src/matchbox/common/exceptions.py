"""Custom exceptions for Matchbox."""

import http
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pyarrow import Schema

if TYPE_CHECKING:
    from matchbox.common.dtos import (
        BackendResourceType,
        CollectionName,
        PermissionType,
        RunID,
        StepName,
        StepType,
    )
else:
    BackendResourceType = Any
    CollectionName = Any
    PermissionType = Any
    StepName = Any
    StepType = Any
    RunID = Any


# -- Base class for all Matchbox exceptions


class MatchboxException(Exception):
    """An exception has occurred in Matchbox.."""

    def __init__(self, message: str | None = None) -> None:
        """Initialise the exception."""
        super().__init__(message or self.__doc__)

    def to_details(self) -> dict[str, Any] | None:
        """Return exception-specific details for serialisation.

        Override in subclasses that have additional constructor arguments.
        """
        return None


class MatchboxHttpException(MatchboxException):
    """Base class for exceptions that can be transmitted over HTTP.

    Subclasses must define http_status as a class attribute.
    """

    http_status: http.HTTPStatus


# -- Common data objects exceptions --


class MatchboxRuntimeError(MatchboxException, RuntimeError):
    """Runtime error."""

    def __init__(self, message: str) -> None:
        """Initialise the exception."""
        super().__init__(message)


class MatchboxNameError(MatchboxException, ValueError):
    """Name did not pass validation."""

    def __init__(self, message: str) -> None:
        """Initialise the exception."""
        super().__init__(message)


class MatchboxArrowSchemaMismatch(MatchboxException):
    """Arrow schema mismatch."""

    def __init__(self, expected: Schema, actual: Schema) -> None:
        """Initialise the exception."""
        message = f"Schema mismatch. Expected:\n{expected}\nGot:\n{actual}"

        super().__init__(message)


# -- Configuration exceptions --


class MatchboxClientSettingsException(MatchboxException):
    """Incorrect configuration provided to client."""

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Incorrect configuration provided to client."

        super().__init__(message)


# -- Client-side API exceptions --


class MatchboxUnparsedClientRequest(MatchboxException):
    """The API could not parse the content of the client request."""

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "The API could not parse the content of the client request"

        super().__init__(message)


class MatchboxUnhandledServerResponse(MatchboxException):
    """The API sent an unexpected response."""

    def __init__(self, http_status: int, details: str | None = None) -> None:
        """Initialise the exception."""
        message = f"The API sent an unexpected response with status {http_status}"
        if details:
            message += f" and the following message: {details}"

        super().__init__(message)


class MatchboxEmptyServerResponse(MatchboxException):
    """The server returned an empty response when data was expected."""

    def __init__(
        self, message: str | None = None, operation: str | None = None
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "The server returned an empty response when data was expected."
            if operation is not None:
                message = (
                    f"The {operation} operation returned no data from the server "
                    "when data was expected."
                )

        super().__init__(message)
        self.operation = operation


# -- SourceConfig exceptions --


class MatchboxSourceFieldError(MatchboxException):
    """Specified fields diverge with the warehouse."""


class MatchboxSourceClientError(MatchboxException):
    """Location client must be set."""


class MatchboxSourceExtractTransformError(MatchboxException):
    """Invalid ETL logic detected."""

    def __init__(
        self,
        logic: str | None = None,
    ) -> None:
        """Initialise the exception."""
        message = "Invalid ETL logic detected."
        if logic is not None:
            message += f"\n{logic}"

        super().__init__(message)


class MatchboxSourceTableError(MatchboxException):
    """Tables not found in your source data warehouse."""

    def __init__(
        self,
        message: str | None = None,
        table_name: str | None = None,
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Table doesn't exist in your source data warehouse."
            if table_name is not None:
                message += f"\nTable name: {table_name}"

        super().__init__(message)
        self.table_name = table_name


class MatchboxServerFileError(MatchboxHttpException):
    """There was an issue with file upload."""

    http_status = 400

    def __init__(self, message: str | None = None) -> None:
        """Initialise the exception."""
        if message is None:
            message = "There was an issue with file upload."

        super().__init__(message)


# -- Authentication exceptions --


class MatchboxAuthenticationError(MatchboxHttpException):
    """Authentication failed."""

    http_status = 401

    def __init__(self, message: str | None = None) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Authentication failed."

        super().__init__(message)


class MatchboxPermissionDenied(MatchboxHttpException):
    """User lacks required permission for the requested resource."""

    http_status = 403

    def __init__(
        self,
        message: str | None = None,
        permission: PermissionType | None = None,
        resource_type: BackendResourceType | None = None,
        resource_name: str | None = None,
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Permission denied."
            if permission is not None and resource_name is not None:
                message = (
                    f"Permission denied: requires {permission} "
                    f"access on '{resource_name}'."
                )

        super().__init__(message)
        self.permission = permission
        self.resource_type = resource_type
        self.resource_name = resource_name

    def to_details(self) -> dict[str, Any] | None:
        """Return permission, resource_type and resource_name if set."""
        details: dict[str, Any] = {}
        if self.permission is not None:
            details["permission"] = str(self.permission)
        if self.resource_type is not None:
            details["resource_type"] = str(self.resource_type)
        if self.resource_name is not None:
            details["resource_name"] = self.resource_name
        return details if details else None


# -- Resource not found on server exceptions --


class MatchboxUserNotFoundError(MatchboxHttpException):
    """User not found."""

    http_status = 404

    def __init__(
        self, message: str | None = None, user_name: str | None = None
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "User not found."
            if user_name is not None:
                message = f"User {user_name} not found."

        super().__init__(message)
        self.user_name = user_name

    def to_details(self) -> dict[str, Any] | None:
        """Return user_name if set."""
        if self.user_name is not None:
            return {"user_name": self.user_name}
        return None


class MatchboxStepNotFoundError(MatchboxHttpException):
    """Step not found."""

    http_status = 404

    def __init__(
        self, message: str | None = None, name: StepName | None = None
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Step not found."
            if name is not None:
                message = f"Step {name} not found."

        super().__init__(message)
        self.name = name

    def to_details(self) -> dict[str, Any] | None:
        """Return name if set."""
        if self.name is not None:
            return {"name": str(self.name)}
        return None


class MatchboxCollectionNotFoundError(MatchboxHttpException):
    """Collection not found."""

    http_status = 404

    def __init__(
        self, message: str | None = None, name: CollectionName | None = None
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Collection not found."
            if name is not None:
                message = f"Collection {name} not found."

        super().__init__(message)
        self.name = name

    def to_details(self) -> dict[str, Any] | None:
        """Return name if set."""
        if self.name is not None:
            return {"name": str(self.name)}
        return None


class MatchboxRunNotFoundError(MatchboxHttpException):
    """Run not found."""

    http_status = 404

    def __init__(self, message: str | None = None, run_id: RunID | None = None) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Run not found."
            if run_id is not None:
                message = f"Run {run_id} not found."

        super().__init__(message)
        self.run_id = run_id

    def to_details(self) -> dict[str, Any] | None:
        """Return run_id if set."""
        if self.run_id is not None:
            return {"run_id": self.run_id}
        return None


class MatchboxDataNotFound(MatchboxHttpException):
    """Data doesn't exist in the Matchbox source table."""

    http_status = 404

    def __init__(
        self,
        message: str | None = None,
        table: str | None = None,
        data: list[Any] | None = None,
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Data doesn't exist in Matchbox."
            if table is not None:
                message += f"\nTable: {table}"
            if data is not None:
                message += f"\nData: {str(data)}"

        super().__init__(message)
        self.table = table
        self.data = data

    def to_details(self) -> dict[str, Any] | None:
        """Return table and data if set."""
        details: dict[str, Any] = {}
        if self.table is not None:
            details["table"] = self.table
        if self.data is not None:
            details["data"] = self.data
        return details if details else None


# -- Server-side API exceptions --


class MatchboxConnectionError(MatchboxException):
    """Connection to Matchbox's backend database failed."""


class MatchboxPermissionDeniedError(MatchboxHttpException):
    """Raised when a user lacks permission for an action."""

    http_status = 403


class MatchboxDeletionNotConfirmed(MatchboxHttpException):
    """Deletion must be confirmed: if certain, rerun with certain=True."""

    http_status = 409

    def __init__(
        self, message: str | None = None, children: list[str | int] | None = None
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "Deletion must be confirmed: if certain, rerun with certain=True."

        if children is not None:
            children_names = ", ".join(str(child) for child in children)
            message = (
                f"This operation will delete the steps {children_names}, "
                "as well as all scores they have created. \n\n"
                "It won't delete validation associated with these "
                "scores. \n\n"
                "If you're sure you want to continue, rerun with certain=True. "
            )

        super().__init__(message)
        self.children = children

    def to_details(self) -> dict[str, Any] | None:
        """Return children if set."""
        if self.children is not None:
            return {"children": self.children}
        return None


class MatchboxStepAlreadyExists(MatchboxHttpException):
    """Step already exists."""

    http_status = 409


class MatchboxStepUpdateError(MatchboxHttpException):
    """Step metadata cannot be updated."""

    http_status = 422


class MatchboxStepInvalidData(MatchboxHttpException):
    """Step data does not match fingerprint."""

    http_status = 422


class MatchboxStepExistingData(MatchboxHttpException):
    """Data was already set on step."""

    http_status = 409


class MatchboxStepNotQueriable(MatchboxHttpException):
    """The step is not ready to be queried."""

    http_status = 422


class MatchboxStepTypeError(MatchboxHttpException):
    """An invalid operation was attempted using this type of step."""

    http_status = 422

    def __init__(
        self,
        message: str | None = None,
        step_name: StepName | None = None,
        step_type: StepType | None = None,
        expected_step_types: list[StepType] | None = None,
    ) -> None:
        """Initialise the exception."""
        if message is None:
            message = "An invalid operation was attempted using this type of step."
            if step_name is not None and step_type is not None:
                message = (
                    f"Step '{step_name}' is of type {step_type}, "
                    "which does not support this operation."
                )
            if expected_step_types is not None:
                expected = ", ".join(expected_step_types)
                message += f" Expected one of: {expected}."

        super().__init__(message)
        self.step_name = step_name
        self.step_type = step_type
        self.expected_step_types = expected_step_types

    def to_details(self) -> dict[str, Any] | None:
        """Return attributes if set."""
        details: dict[str, Any] = {}
        if self.step_name is not None:
            details["step_name"] = self.step_name
        if self.step_type is not None:
            details["step_type"] = self.step_type
        if self.expected_step_types is not None:
            details["expected_step_types"] = list(self.expected_step_types)
        return details if details else None


class MatchboxCollectionAlreadyExists(MatchboxHttpException):
    """Collection already exists."""

    http_status = 409


class MatchboxRunAlreadyExists(MatchboxHttpException):
    """Run already exists."""

    http_status = 409


class MatchboxRunNotWriteable(MatchboxHttpException):
    """Run is not mutable."""

    http_status = 423


class MatchboxTooManySamplesRequested(MatchboxHttpException):
    """Too many samples have been requested from the server."""

    http_status = 422


class MatchboxGroupNotFoundError(MatchboxHttpException):
    """Raised when a group is not found."""

    http_status = 404


class MatchboxGroupAlreadyExistsError(MatchboxHttpException):
    """Raised when attempting to create a group that already exists."""

    http_status = 409


class MatchboxSystemGroupError(MatchboxHttpException):
    """Raised when attempting to modify or delete a system group."""

    http_status = 422


# -- Adapter DB exceptions --


class MatchboxNoJudgements(MatchboxHttpException):
    """No judgements found in the database when required for operation."""

    http_status = 404


class MatchboxDatabaseWriteError(MatchboxException):
    """Could not be written to the backend DB, likely due to a constraint violation."""


class MatchboxLockError(MatchboxHttpException):
    """Trying to modify locked data."""

    http_status = 423


# -- Exception Registry --


def _get_all_subclasses(cls: type) -> list[type]:
    """Recursively get all subclasses of a class."""
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_all_subclasses(subclass))
    return subclasses


HTTP_EXCEPTION_REGISTRY: dict[str, type[MatchboxHttpException]] = {
    exc.__name__: exc for exc in _get_all_subclasses(MatchboxHttpException)
}

# Auto-generate enum from registry
_enum_members = {name: name for name in HTTP_EXCEPTION_REGISTRY}
_enum_members["MatchboxServerError"] = "MatchboxServerError"  # For generic 500 errors

MatchboxExceptionType = StrEnum("MatchboxExceptionType", _enum_members)
