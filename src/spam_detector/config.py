"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    app_name: str = Field(default="spam-detector", description="Application name")
    environment: Literal["development", "production", "test"] = Field(
        default="development", description="Environment"
    )

    # Model Settings
    model_path: Path = Field(
        default=Path("models/spam_detector_model.pkl"), description="Path to trained model"
    )
    vectorizer_path: Path = Field(
        default=Path("models/vectorizer.pkl"), description="Path to vectorizer"
    )
    model_version: str = Field(default="v1", description="Model version")

    # Training Settings
    train_data_path: Path = Field(
        default=Path("data/spam_emails.csv"), description="Path to training data"
    )
    test_size: float = Field(default=0.2, description="Test set size ratio", ge=0.0, le=1.0)
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    max_features: int = Field(
        default=5000, description="Maximum features for vectorization", gt=0
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port", gt=0, lt=65536)
    api_reload: bool = Field(default=True, description="Auto-reload for development")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Data Generation Settings
    generated_data_size: int = Field(
        default=5000, description="Number of samples to generate", gt=0
    )
    spam_ratio: float = Field(
        default=0.4, description="Ratio of spam emails", ge=0.0, le=1.0
    )


# Global settings instance
settings = Settings()
