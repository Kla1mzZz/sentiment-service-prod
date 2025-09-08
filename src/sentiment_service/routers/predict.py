import joblib
from pathlib import Path
from fastapi import APIRouter

from ..schemas.input import SentimentAnalysisInput, SentimentAnalysisOutput


router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

pipline = joblib.load(MODEL_PATH)


@router.post("/predict", tags=["Predictions"], response_model=SentimentAnalysisOutput)
async def predict(input: SentimentAnalysisInput) -> SentimentAnalysisOutput:
    output = pipline.predict([input.text])
    return SentimentAnalysisOutput(prediction=output)
