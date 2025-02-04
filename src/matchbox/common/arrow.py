from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_MB_IDS = pa.schema([("id", pa.int64()), ("source_pk", pa.string())])


def table_to_buffer(table: pa.Table) -> BytesIO:
    sink = BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    return sink
