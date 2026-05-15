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
    Uses raw SQL to avoid SQLAlchemy ORM dependency issues.
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            # Check what tables exist
            if _is_sqlite:
                tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = {row[0] for row in tables_result}
            else:
                tables_result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
                tables = {row[0] for row in tables_result}

            # ── extraction_jobs migrations ─────────────────────────────────
            if "extraction_jobs" in tables:
                if _is_sqlite:
                    cols_result = conn.execute(text("PRAGMA table_info(extraction_jobs)"))
                    existing = {row[1] for row in cols_result}
                else:
                    cols_result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='extraction_jobs'"))
                    existing = {row[0] for row in cols_result}

                new_cols = [
                    ("sources",            "TEXT"),
                    ("evidence",           "TEXT"),
                    ("schema_fields",      "TEXT"),
                    ("schema_name",        "VARCHAR(255)"),
                    ("schema_id",          "VARCHAR(255)"),
                    ("batch_id",           "VARCHAR(255)"),
                    ("model",              "VARCHAR(255)"),
                    ("user_id",            "VARCHAR(255)"),
                    ("error_message",      "TEXT"),
                    ("duration_seconds",   "FLOAT"),
                    ("confidence",         "TEXT"),
                    ("validation_errors",  "TEXT"),
                    ("failure_log",        "TEXT"),
                    ("updated_at",         "DATETIME"),
                ]
                for col, col_type in new_cols:
                    if col not in existing:
                        try:
                            conn.execute(text(f"ALTER TABLE extraction_jobs ADD COLUMN {col} {col_type}"))
                            conn.commit()
                        except Exception as col_err:
                            pass  # Column may already exist in some edge cases

            # ── documents migrations ───────────────────────────────────────
            if "documents" in tables:
                if _is_sqlite:
                    cols_result = conn.execute(text("PRAGMA table_info(documents)"))
                    existing = {row[1] for row in cols_result}
                else:
                    cols_result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='documents'"))
                    existing = {row[0] for row in cols_result}

                new_cols = [
                    ("user_id",        "VARCHAR(255)"),
                    ("file_path",      "VARCHAR(500)"),
                    ("mime_type",      "VARCHAR(255)"),
                    ("error_message",  "TEXT"),
                    ("parsed_data",    "TEXT"),
                    ("page_count",     "INTEGER"),
                ]
                for col, col_type in new_cols:
                    if col not in existing:
                        try:
                            conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col} {col_type}"))
                            conn.commit()
                        except Exception:
                            pass

    except Exception as e:
        from loguru import logger
        logger.warning(f"Migration warning (non-fatal): {e}")
