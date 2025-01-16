import json

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine

from matchbox.common.db import SourceColumn
from matchbox.common.hash import HASH_FUNC, hash_to_base64


class SourceNameAddress(BaseModel):
    full_name: str
    warehouse_hash: bytes

    @classmethod
    def from_engine(cls, engine: Engine, full_name: str) -> "SourceNameAddress":
        """
        Generate a SourceNameAddress from a SQLAlchemy Engine and full source name.
        """
        url = engine.url
        components = {
            "dialect": url.get_dialect().name,
            "database": url.database or "",
            "host": url.host or "",
            "port": url.port or "",
            "schema": getattr(url, "schema", "") or "",
            "service_name": url.query.get("service_name", ""),
        }

        stable_str = json.dumps(components, sort_keys=True).encode()

        return HASH_FUNC(stable_str).digest()


class Selector(BaseModel):
    source_name_address: SourceNameAddress
    fields: list[str]


class SourceBase(BaseModel):
    source_name_address: SourceNameAddress
    db_pk: str
    alias: str
    db_columns: list[SourceColumn]


class SourceColumnName(BaseModel):
    """A column name in the Matchbox database."""

    name: str

    @property
    def hash(self) -> bytes:
        """Generate a unique hash based on the column name."""
        return HASH_FUNC(self.name.encode("utf-8")).digest()

    @property
    def base64(self) -> str:
        """Generate a base64 encoded hash based on the column name."""
        return hash_to_base64(self.hash)


class SourceColumn(BaseModel):
    """A column in a dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    literal: SourceColumnName = Field(
        description="The literal name of the column in the database."
    )
    alias: SourceColumnName = Field(
        default_factory=lambda data: SourceColumnName(name=data["literal"].name),
        description="The alias to use when hashing the dataset in Matchbox.",
    )
    type: str | None = Field(
        default=None, description="The type to cast the column to before hashing data."
    )

    def __eq__(self, other: object) -> bool:
        """Compare SourceColumn with another SourceColumn or bytes object.

        Two SourceColumns are equal if:

        * Their literal names match, or
        * Their alias names match, or
        * The hash of either their literal or alias matches the other object's
        corresponding hash

        A SourceColumn is equal to a bytes object if:

        * The hash of either its literal or alias matches the bytes object

        Args:
            other: Another SourceColumn or a bytes object to compare against

        Returns:
            bool: True if the objects are considered equal, False otherwise
        """
        if isinstance(other, SourceColumn):
            if self.literal == other.literal or self.alias == other.alias:
                return True

            self_hashes = {self.literal.hash, self.alias.hash}
            other_hashes = {other.literal.hash, other.alias.hash}

            return bool(self_hashes & other_hashes)

        if isinstance(other, bytes):
            return other in {self.literal.hash, self.alias.hash}

        return NotImplemented

    @field_validator("literal", "alias", mode="before")
    def string_to_name(cls: "SourceColumn", value: str) -> SourceColumnName:
        if isinstance(value, str):
            return SourceColumnName(name=value)
        else:
            raise ValueError("Column name must be a string.")
