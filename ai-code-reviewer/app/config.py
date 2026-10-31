"""
Central configuration for the AI Code Reviewer.

Same pattern as multi-source-rag: every setting lives here, typed,
loaded from .env. If something required is missing the app fails
immediately at startup with a clear message -- not deep inside a
graph node after 30 seconds of work.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App identity ---
    app_name: str = "ai-code-reviewer"
    environment: str = "dev"

    # --- Logging ---
    log_level: str = "INFO"

    # --- GitHub ---
    # Personal access token with repo scope.
    # Get one at: github.com -> Settings -> Developer settings ->
    # Personal access tokens -> Tokens (classic)
    github_token: str = ""
    github_repo: str = ""   # e.g. "praveenkumar993/sample-python-api"

    # --- LLM (Groq, same as Project 1) ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # --- LangSmith (tracing -- optional but highly recommended) ---
    # Get a free key at: smith.langchain.com
    langsmith_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "ai-code-reviewer"

    # --- Agent behaviour ---
    # Max tool call rounds per agent before it gives up and reports
    # what it has. Prevents infinite loops if the LLM keeps calling
    # tools without reaching a conclusion.
    max_agent_iterations: int = 10

    # Human-in-the-loop: if True, the graph pauses before posting
    # the review and waits for your approval. Set to False to post
    # automatically (useful once you trust the agent's quality).
    require_human_approval: bool = True

    # Diffs longer than this are truncated before being sent to the LLM,
    # so one huge PR can't overflow the model's context window.
    max_diff_chars: int = 30000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()