from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List
import os


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./doc_intel.db"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string-at-least-32-chars")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure upload dir exists
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
