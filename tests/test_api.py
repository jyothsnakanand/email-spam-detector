"""Tests for API endpoints."""

import tempfile
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from spam_detector import api
from spam_detector.config import settings


@pytest.fixture(scope="function")
def mock_model() -> Path:
    """Create a mock trained model for testing."""
    # Create a simple mock model
    vectorizer = TfidfVectorizer(max_features=100)
    classifier = LogisticRegression(random_state=42)

    # Train on minimal data
    texts = [
        "win prize money click",
        "meeting tomorrow project",
        "free cash bonus urgent",
        "thanks for the report",
    ]
    labels = ["spam", "ham", "spam", "ham"]

    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
    pipeline.fit(texts, labels)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        joblib.dump(pipeline, f.name)
        return Path(f.name)


@pytest.fixture(scope="function")
def client(mock_model: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:  # type: ignore[misc]
    """Create test client with mock model."""
    # Patch settings before importing app
    monkeypatch.setattr(settings, "model_path", mock_model)

    # Clear the global predictor to force reload
    api.predictor = None

    # Create client which will trigger lifespan and load our mock model
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "model_loaded" in data


def test_predict_single_email(client: TestClient) -> None:
    """Test single email prediction."""
    response = client.post("/predict", json={"text": "win free money now click here"})

    assert response.status_code == 200

    data = response.json()
    assert "text" in data
    assert "prediction" in data
    assert "confidence" in data
    assert "is_spam" in data
    assert data["prediction"] in ["spam", "ham"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["is_spam"], bool)


def test_predict_empty_text(client: TestClient) -> None:
    """Test prediction with empty text."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Validation error


def test_predict_batch(client: TestClient) -> None:
    """Test batch email prediction."""
    emails = [
        "meeting scheduled for tomorrow",
        "congratulations you won a prize",
        "please review the attached document",
    ]

    response = client.post("/predict/batch", json={"texts": emails})

    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert "total_processed" in data
    assert data["total_processed"] == 3
    assert len(data["predictions"]) == 3

    for pred in data["predictions"]:
        assert "text" in pred
        assert "prediction" in pred
        assert "confidence" in pred
        assert "is_spam" in pred


def test_predict_batch_empty_list(client: TestClient) -> None:
    """Test batch prediction with empty list."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422  # Validation error


def test_invalid_request_format(client: TestClient) -> None:
    """Test invalid request format."""
    response = client.post("/predict", json={"invalid_field": "test"})
    assert response.status_code == 422
