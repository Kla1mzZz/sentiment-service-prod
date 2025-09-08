from pydantic import BaseModel


class SentimentAnalysisInput(BaseModel):
    text: str


class SentimentAnalysisOutput(BaseModel):
    prediction: int
