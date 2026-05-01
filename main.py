from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.database import init_db
from app.api.v1.router import api_router

app = FastAPI(
    title="AI Document Intelligence Platform",
    description=(
        "ADE-compatible schema-driven document extraction. "
        "Produces markdown, chunks, splits, grounding exactly like LandingAI ADE."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "parse_engine": "ADE-compatible (PyMuPDF + OCR fallback)"
    }


@app.get("/")
async def root():
    return {
        "message": "AI Document Intelligence Platform API — ADE Edition",
        "docs": "/docs",
        "endpoints": {
            "parse_sync": "POST /api/v1/parse",
            "parse_chunks": "GET /api/v1/parse/{id}/chunks",
            "parse_splits": "GET /api/v1/parse/{id}/splits",
            "parse_grounding": "GET /api/v1/parse/{id}/grounding",
            "parse_markdown": "GET /api/v1/parse/{id}/markdown",
            "documents": "/api/v1/documents",
            "extraction": "/api/v1/extraction",
            "schemas": "/api/v1/schemas",
        }
    }
