.PHONY: help install setup generate-data train predict api test lint format clean docker-build k8s-deploy

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies with uv"
	@echo "  make setup            - Setup pre-commit hooks"
	@echo "  make generate-data    - Generate synthetic training data"
	@echo "  make train            - Train the spam detection model"
	@echo "  make predict          - Run inference on test emails"
	@echo "  make api              - Start the FastAPI server"
	@echo "  make test             - Run tests with pytest"
	@echo "  make lint             - Run linting (ruff + mypy)"
	@echo "  make format           - Format code with black and ruff"
	@echo "  make clean            - Clean generated files and caches"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make k8s-deploy       - Deploy to Kubernetes"
	@echo "  make notebook         - Start Jupyter notebook server"

install:
	uv pip install -e ".[dev,test,lint]"

setup: install
	pre-commit install
	@echo "✅ Development environment setup complete!"

generate-data:
	python -m spam_detector.data_generator

train:
	python -m spam_detector.train

predict:
	python -m spam_detector.predict

api:
	python -m spam_detector.api

notebook:
	jupyter notebook

test:
	pytest -v --cov=spam_detector --cov-report=term-missing

test-quick:
	pytest -v

lint:
	ruff check .
	mypy src/

format:
	black .
	ruff check --fix .

pre-commit:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage dist/ build/

docker-build:
	eval $$(minikube docker-env) && docker build -t spam-detector:latest .

k8s-deploy:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/

k8s-logs:
	kubectl logs -l app=spam-detector --tail=100 -f

k8s-status:
	kubectl get pods -l app=spam-detector
	kubectl get svc spam-detector

minikube-url:
	minikube service spam-detector --url

all: setup generate-data train test
	@echo "✅ Full pipeline complete!"
