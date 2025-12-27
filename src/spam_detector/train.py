"""Model training module for spam detection."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spam_detector.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class SpamDetectorTrainer:
    """Train and evaluate spam detection model."""

    def __init__(
        self,
        max_features: int = 5000,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            max_features: Maximum number of features for TF-IDF vectorization
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
        """
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        self.pipeline: Pipeline | None = None

    def load_data(self, data_path: Path) -> tuple[pd.Series, pd.Series]:
        """
        Load training data from CSV file.

        Args:
            data_path: Path to CSV file with 'text' and 'label' columns

        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")

        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")

        return df["text"], df["label"]

    def train(
        self, texts: pd.Series, labels: pd.Series
    ) -> tuple[float, dict[str, float], list[list[int]]]:
        """
        Train the spam detection model.

        Args:
            texts: Email text data
            labels: Corresponding labels ('spam' or 'ham')

        Returns:
            Tuple of (accuracy, classification_report_dict, confusion_matrix)
        """
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )

        logger.info(
            f"Train size: {len(X_train)}, Test size: {len(X_test)}"
        )

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
        )

        # Create Logistic Regression model
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )

        # Create pipeline
        self.pipeline = Pipeline(
            [
                ("vectorizer", self.vectorizer),
                ("classifier", self.model),
            ]
        )

        logger.info("Training model...")
        self.pipeline.fit(X_train, y_train)

        logger.info("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        return accuracy, report, cm

    def save_model(self, model_path: Path, vectorizer_path: Path) -> None:
        """
        Save trained model and vectorizer.

        Args:
            model_path: Path to save the trained model
            vectorizer_path: Path to save the vectorizer
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.pipeline, model_path)

        logger.info(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(self.vectorizer, vectorizer_path)

        logger.info("Model and vectorizer saved successfully")


def main() -> None:
    """Train model using settings from environment."""
    trainer = SpamDetectorTrainer(
        max_features=settings.max_features,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    # Load data
    texts, labels = trainer.load_data(settings.train_data_path)

    # Train model
    accuracy, report, cm = trainer.train(texts, labels)

    # Save model
    trainer.save_model(settings.model_path, settings.vectorizer_path)

    logger.info(f"\nTraining complete! Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
