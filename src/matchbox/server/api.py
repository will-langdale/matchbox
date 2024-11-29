from fastapi import FastAPI
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
    return HealthCheck(status="ok")
