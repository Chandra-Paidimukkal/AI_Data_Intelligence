from fastapi import APIRouter
from app.api.v1.endpoints import documents, schemas, extraction, export, jobs, batch, compare, chat, intelligence, auth
from app.api.v1.endpoints import parse

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(schemas.router)
api_router.include_router(extraction.router)
api_router.include_router(export.router)
api_router.include_router(jobs.router)
api_router.include_router(parse.router)
api_router.include_router(batch.router)
api_router.include_router(compare.router)
api_router.include_router(chat.router)
api_router.include_router(intelligence.router)
