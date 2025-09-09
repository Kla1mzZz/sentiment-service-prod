from pathlib import Path
from fastapi import FastAPI

from .routers.predict import router as predict_router


app = FastAPI(
    title="Sentiment Service",
    description="Sentiment Service",
    version="0.1.0",
)

app.include_router(predict_router)
