"""FastAPI application for spam detection inference."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spam_detector.config import settings
from spam_detector.predict import SpamDetectorPredictor

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: SpamDetectorPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, cleanup on shutdown."""
    global predictor
    logger.info("Starting up API server...")

    try:
        predictor = SpamDetectorPredictor(settings.model_path)
        predictor.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down API server...")


app = FastAPI(
    title="Spam Detector API",
    description="Email spam detection API using machine learning",
    version=settings.model_version,
    lifespan=lifespan,
)


class EmailRequest(BaseModel):
    """Request model for single email prediction."""

    text: str = Field(..., description="Email text to classify", min_length=1)


class EmailBatchRequest(BaseModel):
    """Request model for batch email prediction."""

    texts: list[str] = Field(..., description="List of email texts to classify", min_length=1)


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    text: str = Field(..., description="Input email text")
    prediction: str = Field(..., description="Predicted label (spam or ham)")
    confidence: float = Field(..., description="Prediction confidence score", ge=0.0, le=1.0)
    is_spam: bool = Field(..., description="Boolean indicator if email is spam")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of emails processed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status")
    model_version: str = Field(..., description="Model version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_version=settings.model_version,
        model_loaded=predictor is not None and predictor.pipeline is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_email(request: EmailRequest) -> PredictionResponse:
    """
    Predict whether a single email is spam or ham.

    Args:
        request: Email text to classify

    Returns:
        Prediction with confidence score
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction, confidence = predictor.predict(request.text)

        return PredictionResponse(
            text=request.text,
            prediction=prediction,
            confidence=confidence,
            is_spam=prediction == "spam",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_email_batch(request: EmailBatchRequest) -> BatchPredictionResponse:
    """
    Predict whether multiple emails are spam or ham.

    Args:
        request: List of email texts to classify

    Returns:
        List of predictions with confidence scores
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = predictor.predict_batch(request.texts)

        predictions = [
            PredictionResponse(
                text=text,
                prediction=pred,
                confidence=conf,
                is_spam=pred == "spam",
            )
            for text, (pred, conf) in zip(request.texts, results)
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


def main() -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "spam_detector.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
