from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    refresh_frequency: int = 60


settings = Settings()
