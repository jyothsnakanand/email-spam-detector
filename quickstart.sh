#!/bin/bash
set -e

echo "🚀 Email Spam Detector - Quick Start"
echo "======================================"
echo ""

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv not found. Please install pyenv first."
    echo "   Run: brew install pyenv"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Set Python version
echo "📦 Setting Python version to 3.12.3..."
if ! pyenv versions | grep -q "3.12.3"; then
    echo "   Installing Python 3.12.3 with pyenv..."
    pyenv install 3.12.3
fi
pyenv local 3.12.3

# Create virtual environment
echo "🔨 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "   Virtual environment created at .venv"
else
    echo "   Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e ".[dev,test,lint]"

# Setup pre-commit
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env-example .env
fi

# Generate training data
echo "🎲 Generating synthetic training data..."
python -m spam_detector.data_generator

# Train model
echo "🤖 Training spam detection model..."
python -m spam_detector.train

# Run tests
echo "🧪 Running tests..."
pytest -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "⚠️  Don't forget to activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  • Start API server:    make api  (or python -m spam_detector.api)"
echo "  • Run inference:       make predict"
echo "  • Open notebooks:      make notebook"
echo "  • Run tests:           make test"
echo "  • See all commands:    make help"
echo ""
echo "API will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
