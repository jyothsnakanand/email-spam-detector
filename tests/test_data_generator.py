"""Tests for data generation module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from spam_detector.data_generator import (
    generate_dataset,
    generate_legitimate_email,
    generate_spam_email,
)


def test_generate_spam_email() -> None:
    """Test spam email generation."""
    email = generate_spam_email()
    assert isinstance(email, str)
    assert len(email) > 0


def test_generate_legitimate_email() -> None:
    """Test legitimate email generation."""
    email = generate_legitimate_email()
    assert isinstance(email, str)
    assert len(email) > 0


def test_generate_dataset() -> None:
    """Test dataset generation."""
    df = generate_dataset(num_samples=100, spam_ratio=0.4)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "text" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({"spam", "ham"})

    spam_count = (df["label"] == "spam").sum()
    assert 35 <= spam_count <= 45  # Allow some variance due to rounding


def test_generate_dataset_with_file() -> None:
    """Test dataset generation with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_data.csv"
        df = generate_dataset(num_samples=50, spam_ratio=0.5, output_path=output_path)

        assert output_path.exists()
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 50
        assert loaded_df.equals(df)


def test_generate_dataset_spam_ratio() -> None:
    """Test different spam ratios."""
    for ratio in [0.0, 0.3, 0.5, 0.7, 1.0]:
        df = generate_dataset(num_samples=100, spam_ratio=ratio)
        spam_count = (df["label"] == "spam").sum()
        expected = int(100 * ratio)
        # Allow ±5 variance
        assert abs(spam_count - expected) <= 5
