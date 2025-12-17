"""Configuration for NBA Prediction Service."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    service_name: str = "nba-prediction-service"
    service_version: str = "5.0.0-beta"
    
    # Model paths
    models_dir: str = "/app/models"
    
    # API URLs
    feature_store_url: str = "http://localhost:8081"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
