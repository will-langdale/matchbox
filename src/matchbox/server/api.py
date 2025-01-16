from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile

from matchbox.common.graph import ResolutionGraph
from matchbox.common.schemas import (
    BackendEntityType,
    CountResult,
    HealthCheck,
    ModelResultsType,
)
from matchbox.server.base import BackendManager, MatchboxDBAdapter

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


app = FastAPI(
    title="matchbox API",
    version="0.2.0",
)


def get_backend() -> MatchboxDBAdapter:
    return BackendManager.get_backend()


async def table_to_s3(client: S3Client, bucket: str, file: UploadFile) -> str:
    """Upload a PyArrow Table to S3 and return the key."""
    upload_id = str(uuid4())

    file_bytes = await file.read()
    reader = pa.BufferReader(file_bytes)
    table = pa.ipc.open_file(reader).read_all()

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)

    client.put_object(
        Bucket=bucket,
        Key=f"{upload_id}.parquet",
        Body=sink.getvalue().to_pybytes(),
    )

    return upload_id


async def s3_to_recordbatch(
    client: S3Client, bucket: str, key: str, batch_size: int = 1000
) -> AsyncGenerator[pa.RecordBatch, None]:
    """Download a PyArrow Table from S3 and stream it as RecordBatches."""
    response = client.get_object(Bucket=bucket, Key=key)
    buffer = pa.BufferReader(response["Body"].read())

    parquet_file = pq.ParquetFile(buffer)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch


@app.get("/health")
async def healthcheck() -> HealthCheck:
    """Perform a health check and return the status."""
    return HealthCheck(status="OK")


@app.get("/testing/count")
async def count_backend_items(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    entity: BackendEntityType | None = None,
) -> CountResult:
    def get_count(e: BackendEntityType) -> int:
        return getattr(backend, str(e)).count()

    if entity is not None:
        return CountResult(entities={str(entity): get_count(entity)})
    else:
        res = {str(e): get_count(e) for e in BackendEntityType}
        return CountResult(entities=res)


@app.post("/testing/clear")
async def clear_backend():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources")
async def list_sources():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources/{hash}")
async def get_source(hash: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/sources/{hash}")
async def add_source(hash: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models")
async def list_models():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}")
async def get_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}")
async def add_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.delete("/models/{name}")
async def delete_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/results")
async def get_results(name: str, result_type: ModelResultsType | None):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/results")
async def set_results(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/truth")
async def get_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/truth")
async def set_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors")
async def get_ancestors(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors_cache")
async def get_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/ancestors_cache")
async def set_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/query")
async def query():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/match")
async def match():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/resolutions")
async def get_resolutions(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
) -> ResolutionGraph:
    return backend.get_resolution_graph()
