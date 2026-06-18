from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.routes.metrics import router as metrics_router
from backend.app.routes.recommender import router as recommender_router
from backend.app.services.recommender_service import get_recommender_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the local artifact once and fail fast if it is unavailable.
    get_recommender_service()
    yield


app = FastAPI(
    title="Retail Recommender API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(recommender_router)
app.include_router(metrics_router)
