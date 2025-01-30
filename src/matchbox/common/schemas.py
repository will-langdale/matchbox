import pyarrow as pa

MB_IDS = pa.schema([("source_pk", pa.string()), ("cluster_id", pa.int64())])
