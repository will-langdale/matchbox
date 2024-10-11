from typing import List
from uuid import UUID as uuUUID

from sqlalchemy import UUID, VARCHAR, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from matchbox.server.postgresql.db import MatchboxBase
from matchbox.server.postgresql.mixin import CountMixin, SHA1Mixin, UUIDMixin


class SourceDataset(UUIDMixin, CountMixin, MatchboxBase):
    __tablename__ = "mb__source_dataset"
    __table_args__ = (UniqueConstraint("db_schema", "db_table"),)

    db_schema: Mapped[str] = mapped_column(VARCHAR(100))
    db_id: Mapped[str] = mapped_column(VARCHAR(100))
    db_table: Mapped[str] = mapped_column(VARCHAR(100))

    data: Mapped[List["SourceData"]] = relationship(back_populates="parent_dataset")

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        with cls.get_session() as session:
            return session.query(cls.db_schema, cls.db_table).scalars()


class SourceData(SHA1Mixin, CountMixin, MatchboxBase):
    __tablename__ = "mb__source_data"
    __table_args__ = (UniqueConstraint("sha1", "dataset"),)  # id array can change

    # Uses array as source data may have identical rows. We can't control this
    # Must be indexed or PostgreSQL incorrectly tries to use nested joins
    # when retrieving small datasets in query() -- extremely slow
    id: Mapped[List[str]] = mapped_column(ARRAY(VARCHAR(36)), index=True)
    dataset: Mapped[uuUUID] = mapped_column(UUID, ForeignKey("mb__source_dataset.uuid"))

    parent_dataset: Mapped["SourceDataset"] = relationship(back_populates="data")
