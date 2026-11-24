"""
run.py -- starts the FastAPI server.

Separate from api.py so the app object stays importable
without starting a server (important for testing).

Usage:
    python run.py

The server runs on http://localhost:8000
API docs: http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",  # uvicorn's own logs -- ours come from structlog
    )