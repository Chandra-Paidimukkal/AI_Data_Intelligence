from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

_db_url = settings.DATABASE_URL
_is_sqlite = "sqlite" in _db_url

# Supabase / any external Postgres requires SSL
if not _is_sqlite and "sslmode" not in _db_url:
    _db_url = _db_url + ("&" if "?" in _db_url else "?") + "sslmode=require"

engine = create_engine(
    _db_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
    pool_recycle=300 if not _is_sqlite else -1,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import document, schema, job, user  # noqa — registers all models
    Base.metadata.create_all(bind=engine)
    _migrate_add_missing_columns()


def _migrate_add_missing_columns():
    """
    Safe migration: add any columns that exist in the model but not in the DB.
    This handles Railway/SQLite deployments where the DB was created before
    new columns were added to the models.
    """
    from sqlalchemy import text, inspect
    try:
        inspector = inspect(engine)

        # ── extraction_jobs migrations ────────────────────────────────────────
        if "extraction_jobs" in inspector.get_table_names():
            existing = {col["name"] for col in inspector.get_columns("extraction_jobs")}
            new_cols = {
                "sources":       "JSON",
                "evidence":      "JSON",
                "schema_fields": "JSON",
                "schema_name":   "VARCHAR",
                "schema_id":     "VARCHAR",
                "batch_id":      "VARCHAR",
                "model":         "VARCHAR",
                "user_id":       "VARCHAR",
                "error_message": "TEXT",
                "duration_seconds": "FLOAT",
            }
            with engine.connect() as conn:
                for col, col_type in new_cols.items():
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE extraction_jobs ADD COLUMN {col} {col_type}"))
                conn.commit()

        # ── documents migrations ──────────────────────────────────────────────
        if "documents" in inspector.get_table_names():
            existing = {col["name"] for col in inspector.get_columns("documents")}
            new_cols = {
                "user_id":   "VARCHAR",
                "file_path": "VARCHAR",
                "mime_type": "VARCHAR",
            }
            with engine.connect() as conn:
                for col, col_type in new_cols.items():
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col} {col_type}"))
                conn.commit()

    except Exception as e:
        from loguru import logger
        logger.warning(f"Migration warning (non-fatal): {e}")
