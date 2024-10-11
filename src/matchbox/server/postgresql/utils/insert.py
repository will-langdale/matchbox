from sqlalchemy import (
    Engine,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from matchbox.server.postgresql.models import Models, ModelsFrom
from matchbox.server.postgresql.utils.sha1 import (
    list_to_value_ordered_sha1,
    model_name_to_sha1,
)


def insert_deduper(model, deduplicates: str, description: str, engine: Engine) -> None:
    """Writes a deduper model to Matchbox."""
    with Session(engine) as session:
        # Construct model SHA1 from name and what it deduplicates
        model_sha1 = list_to_value_ordered_sha1([model, deduplicates])

        # Insert model
        model = Models(
            sha1=model_sha1,
            name=model,
            description=description,
            deduplicates=deduplicates,
        )

        session.merge(model)
        session.commit()

        # Insert reference to parent models


def insert_linker(
    model: str, left: str, right: str, description: str, engine: Engine
) -> None:
    """Writes a linker model to Matchbox."""
    with Session(engine) as session:
        # Construct model SHA1 from parent model SHA1s
        left_sha1 = model_name_to_sha1(left, engine=engine)
        right_sha1 = model_name_to_sha1(right, engine=engine)

        model_sha1 = list_to_value_ordered_sha1(
            [bytes(model, encoding="utf-8"), left_sha1, right_sha1]
        )

        # Insert model
        model = Models(
            sha1=model_sha1,
            name=model,
            description=description,
            deduplicates=None,
        )

        session.merge(model)
        session.commit()

        # Insert reference to parent models
        models_from_to_insert = [
            {"parent": model_sha1, "child": left_sha1},
            {"parent": model_sha1, "child": right_sha1},
        ]

        ins_stmt = insert(ModelsFrom)
        ins_stmt = ins_stmt.on_conflict_do_nothing(
            index_elements=[
                ModelsFrom.parent,
                ModelsFrom.child,
            ]
        )
        session.execute(ins_stmt, models_from_to_insert)
        session.commit()
