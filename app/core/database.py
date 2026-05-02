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
