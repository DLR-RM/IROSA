.PHONY: install install-dev test lint format type commit-checks clean

install:
	pip install -e .

install-dev:
	pip install -e ".[tests,sim]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check irosa/ tests/ examples/

format:
	ruff format irosa/ tests/ examples/
	black irosa/ tests/ examples/

type:
	mypy irosa/

commit-checks: format type lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info build dist .mypy_cache .pytest_cache
