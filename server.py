"""Server entry point — run with: python server.py"""

import uvicorn
from src.core.config import settings
from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
