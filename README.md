# Email Spam Detector

A complete machine learning project for email spam detection with training, inference, and deployment capabilities. Built with Python 3.12.3+, using modern tooling including uv, pyenv, and pre-commit hooks.

## Features

- **ML Pipeline**: Complete training and inference pipeline for spam detection
- **Data Generation**: Synthetic labeled data generation for training
- **API Server**: FastAPI-based REST API for real-time predictions
- **Jupyter Notebooks**: Interactive notebooks for training and inference
- **Kubernetes Support**: Ready-to-deploy K8s manifests for minikube
- **Code Quality**: Pre-commit hooks with mypy, ruff, black, and pytest
- **Environment Management**: Support for .env configuration files

## Project Structure

```
email-spam-detector/
├── src/spam_detector/          # Main package
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data_generator.py       # Synthetic data generation
│   ├── train.py                # Model training
│   ├── predict.py              # Inference
│   └── api.py                  # FastAPI application
├── notebooks/                  # Jupyter notebooks
│   ├── 01_training.ipynb       # Training notebook
│   └── 02_inference.ipynb      # Inference notebook
├── tests/                      # Test suite
│   ├── test_data_generator.py
│   └── test_api.py
├── k8s/                        # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── pvc.yaml
│   └── configmap.yaml
├── data/                       # Training data
├── models/                     # Trained models
├── Dockerfile                  # Container image
├── pyproject.toml              # Project dependencies
├── .python-version             # Python version for pyenv
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .env-example                # Environment variables template
└── README.md
```

## Prerequisites (macOS)

### 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python Version Manager (pyenv)

```bash
brew install pyenv

# Add to ~/.zshrc or ~/.bash_profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Restart shell
exec "$SHELL"

# Install Python 3.12.3
pyenv install 3.12.3
```

### 3. Install Java Version Manager (jenv)

```bash
brew install jenv

# Add to ~/.zshrc or ~/.bash_profile
echo 'export PATH="$HOME/.jenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(jenv init -)"' >> ~/.zshrc

# Restart shell
exec "$SHELL"

# Install Java (if needed)
brew install openjdk@17
jenv add /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
```

### 4. Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

### 5. Install GitHub CLI

```bash
brew install gh

# Authenticate
gh auth login
```

### 6. Install Minikube (for Kubernetes)

```bash
brew install minikube

# Start minikube
minikube start
```

### 7. Install Docker

```bash
brew install --cask docker
```

## Setup

### 1. Clone and Setup Project

```bash
# If using GitHub CLI to create a new repo
gh repo create email-spam-detector --private --clone

# Or clone existing repo
git clone <your-repo-url>
cd email-spam-detector

# Set Python version
pyenv local 3.12.3
```

### 2. Install Dependencies

```bash
# Install project dependencies with uv
uv pip install -e ".[dev,test,lint]"

# Or install all dev dependencies
uv sync
```

### 3. Setup Pre-commit Hooks

```bash
pre-commit install
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env-example .env

# Edit .env with your settings
nano .env
```

## Usage

### Generate Training Data

```bash
# Generate synthetic labeled email data
python -m spam_detector.data_generator

# This creates data/spam_emails.csv with 5000 samples (configurable via .env)
```

### Train the Model

```bash
# Train spam detection model
python -m spam_detector.train

# Models saved to:
# - models/spam_detector_model.pkl
# - models/vectorizer.pkl
```

### Run Inference

```bash
# Test predictions on sample emails
python -m spam_detector.predict
```

### Start API Server

```bash
# Start FastAPI server
python -m spam_detector.api

# Or with uvicorn directly
uvicorn spam_detector.api:app --reload

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# Predict single email
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "CONGRATULATIONS! You won $1,000,000!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Meeting scheduled for tomorrow at 2pm",
      "Click here to claim your FREE prize!",
      "Please review the attached documentation"
    ]
  }'
```

### Use Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks:
# - notebooks/01_training.ipynb - Complete training pipeline
# - notebooks/02_inference.ipynb - Inference and visualization
```

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spam_detector --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

### Code Quality Checks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
black .                  # Format code
ruff check .             # Lint code
mypy src/                # Type checking
pytest                   # Run tests
```

## Kubernetes Deployment

### Build and Deploy

```bash
# Use minikube's Docker daemon
eval $(minikube docker-env)

# Build Docker image
docker build -t spam-detector:latest .

# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=spam-detector
kubectl get svc spam-detector

# Get service URL
minikube service spam-detector --url
```

### Access the API

```bash
# Port forward
kubectl port-forward svc/spam-detector 8000:8000

# Test
curl http://localhost:8000/health
```

See [k8s/README.md](k8s/README.md) for detailed Kubernetes deployment instructions.

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/new-feature
```

### 2. Make Changes

```bash
# Edit code
# Pre-commit hooks run automatically on commit
git add .
git commit -m "Add new feature"
```

### 3. Push and Create PR

```bash
# Push to GitHub
git push origin feature/new-feature

# Create pull request with GitHub CLI
gh pr create --title "Add new feature" --body "Description of changes"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | `spam-detector` |
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `MODEL_PATH` | Path to trained model | `models/spam_detector_model.pkl` |
| `VECTORIZER_PATH` | Path to vectorizer | `models/vectorizer.pkl` |
| `TRAIN_DATA_PATH` | Path to training data | `data/spam_emails.csv` |
| `TEST_SIZE` | Test set ratio | `0.2` |
| `RANDOM_STATE` | Random seed | `42` |
| `MAX_FEATURES` | TF-IDF max features | `5000` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `GENERATED_DATA_SIZE` | Number of samples to generate | `5000` |
| `SPAM_RATIO` | Ratio of spam in generated data | `0.4` |

## Model Details

### Algorithm

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Max features: 5000
  - N-grams: 1-2 (unigrams and bigrams)
  - Stop words: English

- **Classifier**: Logistic Regression
  - Solver: lbfgs
  - Max iterations: 1000

### Performance

The model achieves high accuracy on the synthetic dataset. For production use, train on real email data for better performance.

## Troubleshooting

### Python Version Issues

```bash
# Ensure correct Python version
pyenv versions
pyenv local 3.12.3
python --version
```

### uv Installation Issues

```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or update
uv self update
```

### Pre-commit Hook Failures

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean

# Run manually
pre-commit run --all-files
```

### Minikube Issues

```bash
# Restart minikube
minikube stop
minikube start

# Delete and recreate
minikube delete
minikube start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with scikit-learn, FastAPI, and modern Python tooling
- Designed for educational purposes and production deployment
