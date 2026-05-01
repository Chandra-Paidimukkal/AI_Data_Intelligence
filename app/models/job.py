from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, Float
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class ExtractionJob(Base):
    __tablename__ = "extraction_jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=True, index=True)  # owner
    document_id = Column(String, nullable=False)
    schema_name = Column(String, nullable=True)
    schema_id = Column(String, nullable=True)
    batch_id = Column(String, nullable=True)
    status = Column(String, default="pending")
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    result = Column(JSON, nullable=True)
    confidence = Column(JSON, nullable=True)
    sources = Column(JSON, nullable=True)
    evidence = Column(JSON, nullable=True)
    validation_errors = Column(JSON, nullable=True)
    failure_log = Column(JSON, nullable=True)
    schema_fields = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ExtractionBatch(Base):
    __tablename__ = "extraction_batches"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=True, index=True)  # owner
    schema_id = Column(String, nullable=True)
    document_ids = Column(JSON, default=list)
    job_ids = Column(JSON, default=list)
    status = Column(String, default="pending")
    total = Column(Integer, default=0)
    completed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
