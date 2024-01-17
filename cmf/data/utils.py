import contextlib
import cProfile
import io
import pstats

from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.orm import Session

from cmf.data import ENGINE, SourceDataset

# Data conversion


def get_schema_table_names(full_name: str, validate: bool = False) -> (str, str):
    """
    Takes a string table name and returns the unquoted schema and
    table as a tuple. If you insert these into a query, you need to
    add double quotes in from statements, or single quotes in where.

    Parameters:
        full_name: A string indicating a Postgres table
        validate: Whether to error if both schema and table aren't
        detected

    Raises:
        ValueError: When the function can't detect either a
        schema.table or table format in the input
        ValidationError: If both schema and table can't be detected
        when the validate argument is True

    Returns:
        (schema, table): A tuple of schema and table name. If schema
        cannot be inferred, returns None.
    """

    schema_name_list = full_name.replace('"', "").split(".")

    if len(schema_name_list) == 1:
        schema = None
        table = schema_name_list[0]
    elif len(schema_name_list) == 2:
        schema = schema_name_list[0]
        table = schema_name_list[1]
    else:
        raise ValueError(
            f"""
            Could not identify schema and table in {full_name}.
        """
        )

    if validate and schema is None:
        raise ("Schema could not be detected and validation required.")

    return (schema, table)


def dataset_to_table(dataset: SourceDataset, engine: Engine = ENGINE) -> Table:
    with Session(engine) as session:
        source_schema = MetaData(schema=dataset.db_schema)
        source_table = Table(
            dataset.db_table,
            source_schema,
            schema=dataset.db_schema,
            autoload_with=session.get_bind(),
        )
    return source_table


def string_to_table(db_schema: str, db_table: str, engine: Engine = ENGINE) -> Table:
    with Session(engine) as session:
        source_schema = MetaData(schema=db_schema)
        source_table = Table(
            db_table,
            source_schema,
            schema=db_schema,
            autoload_with=session.get_bind(),
        )
    return source_table


def string_to_dataset(
    db_schema: str, db_table: str, engine: Engine = ENGINE
) -> SourceDataset:
    with Session(engine) as session:
        dataset = (
            session.query(SourceDataset)
            .filter_by(db_schema=db_schema, db_table=db_table)
            .first()
        )
    return dataset


# SQLAlchemy profiling


@contextlib.contextmanager
def sqa_profiled():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())
