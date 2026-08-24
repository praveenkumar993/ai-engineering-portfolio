"""
Central configuration for the app.

Why this exists:
    In notebook-style code, people scatter values like API keys, model names,
    chunk sizes etc. as random constants across files. That breaks the moment
    you need to run the same code in dev vs prod with different values.

    Here, every setting is declared ONCE, typed, and loaded from environment
    variables (or a local .env file). If a required value is missing, the app
    fails immediately at startup with a clear error -- not three steps deep
    into a pipeline run.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App identity ---
    app_name: str = "multi-source-rag"
    environment: str = "dev"  # dev | staging | prod

    # --- Logging ---
    log_level: str = "INFO"
     # --- LLM provider ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
        # --- Azure Blob Storage ---
    azure_storage_connection_string: str = ""
    azure_container_name: str = ""

    # We will add embedding/LLM/vector-db settings here in later steps,
    # ONLY when the code that needs them is being written. Not before.

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# A single shared instance, imported everywhere else in the app.
# This is the pattern: settings = Settings() lives in exactly one place.
settings = Settings()
