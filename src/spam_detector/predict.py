"""Inference module for spam detection."""

import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from spam_detector.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class SpamDetectorPredictor:
    """Spam detection predictor using trained model."""

    def __init__(self, model_path: Path) -> None:
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model pipeline
        """
        self.model_path = model_path
        self.pipeline: Pipeline | None = None

    def load_model(self) -> None:
        """Load the trained model pipeline."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        logger.info("Model loaded successfully")

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict whether an email is spam or ham.

        Args:
            text: Email text to classify

        Returns:
            Tuple of (prediction, confidence)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Predict
        prediction = self.pipeline.predict([text])[0]

        # Get probability scores
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = max(probabilities)

        return prediction, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """
        Predict spam/ham for multiple emails.

        Args:
            texts: List of email texts to classify

        Returns:
            List of (prediction, confidence) tuples
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        confidences = [max(probs) for probs in probabilities]

        return list(zip(predictions, confidences))


def main() -> None:
    """Demo inference using the trained model."""
    predictor = SpamDetectorPredictor(settings.model_path)
    predictor.load_model()

    # Test examples
    test_emails = [
        "Hi team, the meeting is scheduled for tomorrow at 2pm. Please review the agenda.",
        "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize NOW!",
        "Following up on our conversation about the project proposal. Attached is the documentation.",
        "URGENT: Limited time offer! Get 90% off viagra. Buy now!",
    ]

    logger.info("Testing predictions:\n")
    for email in test_emails:
        prediction, confidence = predictor.predict(email)
        logger.info(f"Text: {email[:80]}...")
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.4f})\n")


if __name__ == "__main__":
    main()
