.PHONY: help install install-dev test test-cov lint format type-check security clean docker-build docker-run docker-stop demo train evaluate docs serve-docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Computational Pathology Research Framework$(NC)"
	@echo "$(BLUE)===========================================$(NC)"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make install        # Install dependencies"
	@echo "  make test           # Run tests"
	@echo "  make demo           # Run quick demo"
	@echo "  make docker-build   # Build Docker image"
	@echo ""

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	@echo "$(GREEN)✓ Development installation complete$(NC)"

# Testing targets
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-quick: ## Run quick tests (no slow tests)
	@echo "$(BLUE)Running quick tests...$(NC)"
	pytest tests/ -v -m "not slow"
	@echo "$(GREEN)✓ Quick tests complete$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	pytest-watch tests/ -v

# Code quality targets
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	@echo "$(YELLOW)Checking with flake8...$(NC)"
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "$(YELLOW)Checking with black...$(NC)"
	black --check src/ tests/
	@echo "$(YELLOW)Checking with isort...$(NC)"
	isort --check-only src/ tests/
	@echo "$(GREEN)✓ Linting checks complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/ --ignore-missing-imports --no-strict-optional
	@echo "$(GREEN)✓ Type checking complete$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src/ -f json -o bandit-report.json
	@echo "$(GREEN)✓ Security scan complete (see bandit-report.json)$(NC)"

check-all: lint type-check security ## Run all code quality checks
	@echo "$(GREEN)✓ All checks passed$(NC)"

# Demo targets
demo: ## Run quick demo
	@echo "$(BLUE)Running quick demo...$(NC)"
	python run_quick_demo.py
	@echo "$(GREEN)✓ Demo complete - check results/quick_demo/$(NC)"

demo-missing: ## Run missing modality demo
	@echo "$(BLUE)Running missing modality demo...$(NC)"
	python run_missing_modality_demo.py
	@echo "$(GREEN)✓ Demo complete - check results/missing_modality_demo/$(NC)"

demo-temporal: ## Run temporal reasoning demo
	@echo "$(BLUE)Running temporal reasoning demo...$(NC)"
	python run_temporal_demo.py
	@echo "$(GREEN)✓ Demo complete - check results/temporal_demo/$(NC)"

demo-all: demo demo-missing demo-temporal ## Run all demos
	@echo "$(GREEN)✓ All demos complete$(NC)"

# Training targets
train: ## Train model with default config
	@echo "$(BLUE)Training model...$(NC)"
	python experiments/train.py --config-name default
	@echo "$(GREEN)✓ Training complete$(NC)"

train-quick: ## Train with quick demo config
	@echo "$(BLUE)Training with quick config...$(NC)"
	python experiments/train.py --config-name quick_demo
	@echo "$(GREEN)✓ Quick training complete$(NC)"

train-full: ## Train with full production config
	@echo "$(BLUE)Training with full config...$(NC)"
	python experiments/train.py --config-name full_training
	@echo "$(GREEN)✓ Full training complete$(NC)"

# Evaluation targets
evaluate: ## Evaluate trained model
	@echo "$(BLUE)Evaluating model...$(NC)"
	python experiments/evaluate.py \
		--checkpoint checkpoints/best_model.pth \
		--data-dir ./data \
		--output-dir ./results/evaluation
	@echo "$(GREEN)✓ Evaluation complete - check results/evaluation/$(NC)"

evaluate-ablation: ## Evaluate with ablation study
	@echo "$(BLUE)Running ablation study...$(NC)"
	python experiments/evaluate.py \
		--checkpoint checkpoints/best_model.pth \
		--data-dir ./data \
		--run-ablation \
		--test-missing-modalities \
		--output-dir ./results/ablation
	@echo "$(GREEN)✓ Ablation study complete$(NC)"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t pathology-api:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Starting Docker container...$(NC)"
	docker-compose up -d api
	@echo "$(GREEN)✓ Container running at http://localhost:8000$(NC)"
	@echo "$(YELLOW)View logs: make docker-logs$(NC)"
	@echo "$(YELLOW)Stop container: make docker-stop$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Container stopped$(NC)"

docker-logs: ## View Docker container logs
	docker-compose logs -f api

docker-test: ## Test Docker image
	@echo "$(BLUE)Testing Docker image...$(NC)"
	bash test_docker.sh
	@echo "$(GREEN)✓ Docker tests passed$(NC)"

docker-shell: ## Open shell in Docker container
	@echo "$(BLUE)Opening shell in container...$(NC)"
	docker-compose exec api bash

# Jupyter targets
jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook notebooks/
	@echo "$(GREEN)✓ Jupyter server started$(NC)"

jupyter-docker: ## Start Jupyter in Docker
	@echo "$(BLUE)Starting Jupyter in Docker...$(NC)"
	docker-compose up -d notebook
	@echo "$(GREEN)✓ Jupyter running at http://localhost:8888$(NC)"

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation is in markdown format$(NC)"
	@echo "$(GREEN)✓ See README.md and docs/ directory$(NC)"

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	python -m http.server 8080 --directory .
	@echo "$(GREEN)✓ Documentation served at http://localhost:8080$(NC)"

# Cleaning targets
clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	rm -f bandit-report.json 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-results: ## Clean result files
	@echo "$(BLUE)Cleaning result files...$(NC)"
	rm -rf results/quick_demo/* 2>/dev/null || true
	rm -rf results/missing_modality_demo/* 2>/dev/null || true
	rm -rf results/temporal_demo/* 2>/dev/null || true
	rm -rf results/evaluation/* 2>/dev/null || true
	@echo "$(GREEN)✓ Results cleaned$(NC)"

clean-models: ## Clean saved models
	@echo "$(BLUE)Cleaning saved models...$(NC)"
	@echo "$(RED)Warning: This will delete all saved models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/*.pth 2>/dev/null || true; \
		rm -rf checkpoints/* 2>/dev/null || true; \
		echo "$(GREEN)✓ Models cleaned$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-all: clean clean-results ## Clean everything (except models)
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

# Development workflow targets
dev-setup: install-dev ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	pre-commit install 2>/dev/null || echo "$(YELLOW)pre-commit not available$(NC)"
	@echo "$(GREEN)✓ Development environment ready$(NC)"

dev-check: format lint type-check test ## Run all development checks
	@echo "$(GREEN)✓ All development checks passed$(NC)"

# CI simulation
ci: ## Simulate CI pipeline locally
	@echo "$(BLUE)Simulating CI pipeline...$(NC)"
	@echo "$(YELLOW)Step 1: Linting$(NC)"
	@make lint
	@echo ""
	@echo "$(YELLOW)Step 2: Type checking$(NC)"
	@make type-check
	@echo ""
	@echo "$(YELLOW)Step 3: Security scan$(NC)"
	@make security
	@echo ""
	@echo "$(YELLOW)Step 4: Tests$(NC)"
	@make test-cov
	@echo ""
	@echo "$(YELLOW)Step 5: Docker build$(NC)"
	@make docker-build
	@echo ""
	@echo "$(YELLOW)Step 6: Quick demo$(NC)"
	@make demo
	@echo ""
	@echo "$(GREEN)✓ CI simulation complete$(NC)"

# Release targets
release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	@echo "$(YELLOW)Running all checks...$(NC)"
	@make check-all
	@make test-cov
	@echo "$(YELLOW)Checking version...$(NC)"
	@python -c "import src; print('Version:', getattr(src, '__version__', 'unknown'))"
	@echo "$(GREEN)✓ Release checks complete$(NC)"

# Utility targets
version: ## Show version information
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import src; print(f'Package: {getattr(src, \"__version__\", \"unknown\")}')"

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "$(BLUE)==================$(NC)"
	@echo "Name: Computational Pathology Research Framework"
	@echo "Description: Multimodal fusion for computational pathology"
	@echo ""
	@echo "$(YELLOW)Directories:$(NC)"
	@echo "  Source code: src/"
	@echo "  Tests: tests/"
	@echo "  Experiments: experiments/"
	@echo "  Results: results/"
	@echo "  Models: models/"
	@echo "  Notebooks: notebooks/"
	@echo ""
	@echo "$(YELLOW)Key files:$(NC)"
	@echo "  Training: experiments/train.py"
	@echo "  Evaluation: experiments/evaluate.py"
	@echo "  Configs: experiments/configs/"
	@echo "  Docker: Dockerfile, docker-compose.yml"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  Main: README.md"
	@echo "  Architecture: ARCHITECTURE.md"
	@echo "  Performance: PERFORMANCE.md"
	@echo "  Docker: DOCKER.md"
	@echo "  Portfolio: PORTFOLIO_SUMMARY.md"

status: ## Show project status
	@echo "$(BLUE)Project Status$(NC)"
	@echo "$(BLUE)==============$(NC)"
	@echo ""
	@echo "$(YELLOW)Git status:$(NC)"
	@git status --short 2>/dev/null || echo "Not a git repository"
	@echo ""
	@echo "$(YELLOW)Recent commits:$(NC)"
	@git log --oneline -5 2>/dev/null || echo "No git history"
	@echo ""
	@echo "$(YELLOW)Docker containers:$(NC)"
	@docker ps --filter "name=pathology" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"
	@echo ""
	@echo "$(YELLOW)Disk usage:$(NC)"
	@du -sh results/ models/ checkpoints/ 2>/dev/null || echo "Directories not found"

# Quick workflows
quick-start: install demo ## Quick start: install and run demo
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - Check results in results/quick_demo/"
	@echo "  - Run 'make train' to train a model"
	@echo "  - Run 'make help' to see all available commands"

full-workflow: install-dev dev-check demo-all ## Full workflow: setup, check, and demo
	@echo "$(GREEN)✓ Full workflow complete!$(NC)"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@echo "$(YELLOW)This may take several minutes$(NC)"
	python -m pytest tests/ -v --benchmark-only 2>/dev/null || echo "$(YELLOW)Benchmark tests not available$(NC)"
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

# Database/data management (if applicable)
data-check: ## Check data availability
	@echo "$(BLUE)Checking data...$(NC)"
	@if [ -d "data/raw" ]; then \
		echo "$(GREEN)✓ data/raw exists$(NC)"; \
		ls -lh data/raw/ | tail -n +2 | wc -l | xargs -I {} echo "  Files: {}"; \
	else \
		echo "$(RED)✗ data/raw not found$(NC)"; \
	fi
	@if [ -d "data/processed" ]; then \
		echo "$(GREEN)✓ data/processed exists$(NC)"; \
		ls -lh data/processed/ | tail -n +2 | wc -l | xargs -I {} echo "  Files: {}"; \
	else \
		echo "$(RED)✗ data/processed not found$(NC)"; \
	fi

# Monitoring
monitor-training: ## Monitor training progress (TensorBoard)
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	tensorboard --logdir=logs/ --port=6006
	@echo "$(GREEN)✓ TensorBoard running at http://localhost:6006$(NC)"

# Export targets
export-model: ## Export model to ONNX format
	@echo "$(BLUE)Exporting model to ONNX...$(NC)"
	python -c "print('$(YELLOW)ONNX export not yet implemented$(NC)')"
	@echo "$(YELLOW)TODO: Implement ONNX export$(NC)"

# Help for specific topics
help-docker: ## Show Docker-specific help
	@echo "$(BLUE)Docker Commands$(NC)"
	@echo "$(BLUE)===============$(NC)"
	@echo ""
	@echo "$(GREEN)make docker-build$(NC)     - Build Docker image"
	@echo "$(GREEN)make docker-run$(NC)       - Start API container"
	@echo "$(GREEN)make docker-stop$(NC)      - Stop containers"
	@echo "$(GREEN)make docker-logs$(NC)      - View container logs"
	@echo "$(GREEN)make docker-shell$(NC)     - Open shell in container"
	@echo "$(GREEN)make docker-test$(NC)      - Test Docker setup"
	@echo "$(GREEN)make jupyter-docker$(NC)   - Start Jupyter in Docker"

help-training: ## Show training-specific help
	@echo "$(BLUE)Training Commands$(NC)"
	@echo "$(BLUE)=================$(NC)"
	@echo ""
	@echo "$(GREEN)make train$(NC)            - Train with default config"
	@echo "$(GREEN)make train-quick$(NC)      - Quick training (5 epochs)"
	@echo "$(GREEN)make train-full$(NC)       - Full training (200 epochs)"
	@echo "$(GREEN)make evaluate$(NC)         - Evaluate trained model"
	@echo "$(GREEN)make evaluate-ablation$(NC) - Run ablation study"
	@echo "$(GREEN)make monitor-training$(NC) - Start TensorBoard"
	@echo ""
	@echo "$(YELLOW)Custom training:$(NC)"
	@echo "  python experiments/train.py --config-name <config>"
	@echo ""
	@echo "$(YELLOW)Available configs:$(NC)"
	@ls experiments/configs/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml//' | sed 's/^/  - /'
