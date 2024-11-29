from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="matchbox API",
    version="0.1.0",
)


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get("/health")
async def healthcheck() -> HealthCheck:
    return HealthCheck(status="OK")


@app.get("/testing/count")
async def count_backend_items(entity: str | None = None):
    raise HTTPException(status_code=501, detail="Not implemented")


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


@app.post("/models")
async def add_model():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.delete("/models/{name}")
async def delete_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/results")
async def get_results(name: str):
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


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/models")
async def get_model_subgraph():
    raise HTTPException(status_code=501, detail="Not implemented")
