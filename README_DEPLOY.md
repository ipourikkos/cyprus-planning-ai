from pathlib import Path

from fastapi.staticfiles import StaticFiles

# Import the already-tested API and its lifespan/startup logic.
from planning_api import app

WEB_DIR = Path(__file__).resolve().parent / "web"

if not WEB_DIR.exists():
    raise RuntimeError(f"Web directory not found: {WEB_DIR}")

# API routes (/api/chat, /health, /docs) were registered before this mount,
# so they remain available. Everything else is served from /web.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
