from typing import TypeVar

from sqlalchemy import func

from matchbox.server.postgresql.db import MBDB

T = TypeVar("T")


class CountMixin:
    @classmethod
    def count(cls: type[T]) -> int:
        with MBDB.get_session() as session:
            return session.query(func.count()).select_from(cls).scalar()
