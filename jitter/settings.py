"""App settings management"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """A pydantic class for config environment variables"""
    refresh_frequency: int = 60

settings = Settings()
