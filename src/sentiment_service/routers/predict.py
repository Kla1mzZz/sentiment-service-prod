import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..schemas.input import SentimentAnalysisInput, SentimentAnalysisOutput


router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)


@router.post("/predict", tags=["Predictions"], response_model=SentimentAnalysisOutput)
async def predict(input: SentimentAnalysisInput) -> SentimentAnalysisOutput:
    try:
        output = pipeline.predict([input.text])[0]

        return SentimentAnalysisOutput(prediction=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
