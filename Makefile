default: help;

setup: ## Build and install the dependencies
	uv sync
.PHONY: setup

format: ## Format all files with black & isort
	uv run black ./src
	uv run isort ./src
.PHONY: format

check_format: ## Ensure format for all files with black & isort
	uv run black ./src --check
	uv run isort ./src --check
.PHONY: check_format
