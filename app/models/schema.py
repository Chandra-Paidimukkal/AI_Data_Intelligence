from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class SchemaDefinition(Base):
    __tablename__ = "schema_definitions"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=True, index=True)  # owner (null = shared/system)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String, default="1.0")
    domain = Column(String, nullable=True)
    fields = Column(JSON, nullable=False, default=list)
    record_mode = Column(Boolean, default=False)
    record_anchor = Column(String, nullable=True)
    domain_keywords = Column(JSON, default=list)
    reject_domain_mismatch = Column(Boolean, default=False)
    raw_definition = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
