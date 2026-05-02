from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Build engine — SQLite needs check_same_thread=False, Postgres does not
_is_sqlite = "sqlite" in settings.DATABASE_URL

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # For Postgres: use a connection pool suitable for a web app
    pool_pre_ping=True,  # detect stale connections
    pool_recycle=300 if not _is_sqlite else -1,  # recycle every 5 min for Postgres
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
