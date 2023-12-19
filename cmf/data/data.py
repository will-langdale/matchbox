from typing import List
from uuid import UUID

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmf.data.db import CMFBase
from cmf.data.mixin import SHA1Mixin, UUIDMixin


class SourceDataset(UUIDMixin, CMFBase):
    __tablename__ = "source_dataset"
    __table_args__ = (UniqueConstraint("db_schema", "db_table"),)

    db_schema: Mapped[str]
    db_id: Mapped[str]
    db_table: Mapped[str]

    data: Mapped[List["SourceData"]] = relationship(back_populates="parent_dataset")


class SourceData(SHA1Mixin, CMFBase):
    __tablename__ = "source_data"
    __table_args__ = (UniqueConstraint("sha1", "dataset"),)  # id array can change

    # Uses array as source data may have identical rows. We can't control this
    # Must be indexed or PostgreSQL incorrectly tries to use nested joins
    # when retrieving small datasets in query() -- extremely slow
    id: Mapped[List[str]] = mapped_column(ARRAY(String), index=True)
    dataset: Mapped[UUID] = mapped_column(ForeignKey("source_dataset.uuid"))

    parent_dataset: Mapped["SourceDataset"] = relationship(back_populates="data")
