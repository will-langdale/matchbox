import contextlib
import cProfile
import io
import pstats

from sqlalchemy import Engine, MetaData, Table
from sqlalchemy.orm import Session

from cmf.data import ENGINE, SourceDataset

# Data conversion


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
