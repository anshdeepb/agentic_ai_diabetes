import logging
from contextlib import asynccontextmanager

import openai
from agents import AgentsException
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from memory.state import StateStore
from pipeline.preprocessing import get_dataframe
from routes.sessions import router as sessions_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    get_dataframe()
    app.state.store = StateStore(settings.db_path)
    yield


app = FastAPI(title="Agentic Diabetes Risk Assistant", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(AgentsException)
async def agents_error_handler(request: Request, exc: AgentsException) -> JSONResponse:
    logger.exception("agent run failed", exc_info=exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The assistant could not complete this request. Please try again."},
    )


@app.exception_handler(openai.APIError)
async def model_provider_error_handler(request: Request, exc: openai.APIError) -> JSONResponse:
    logger.exception("model provider error", exc_info=exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The model provider is unavailable right now. Please try again."},
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(sessions_router)
