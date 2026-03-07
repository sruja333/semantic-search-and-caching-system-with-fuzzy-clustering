from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import load_engine

STATIC_DIR = Path(__file__).resolve().parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, description="Natural language query.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = load_engine()
    yield


app = FastAPI(
    title="Trademarkia Semantic Cache API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def home():
    return {"message": "Semantic Cache API is running. Open /demo for UI or /docs for API docs."}


@app.get("/demo")
async def demo_page():
    return FileResponse(STATIC_DIR / "demo.html")


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return app.state.engine.query(query_text)


@app.get("/cache/stats")
async def cache_stats():
    return app.state.engine.cache.stats()


@app.delete("/cache")
async def clear_cache():
    app.state.engine.cache.flush()
    return {"status": "ok", "message": "Cache flushed and stats reset."}
