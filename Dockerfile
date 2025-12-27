FROM python:3.12.3-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml .
COPY .python-version .

# Copy source code
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e .

# Copy models (if available)
COPY models/ models/ 2>/dev/null || true

# Create data directory
RUN mkdir -p data

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Run the API server
CMD ["python", "-m", "spam_detector.api"]
