import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.ingestion.structured import ingest_structured
from app.ingestion.unstructured import ingest_unstructured

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Audit AI Assistant…")
    ingest_structured()
    ingest_unstructured()
    logger.info("Ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Audit AI Assistant",
    description="RAG-based AI assistant over Northstar Robotics Q1 2026 audit data.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)
