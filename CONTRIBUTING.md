# Contributing to IROSA

Thank you for your interest in contributing to IROSA! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork and install development dependencies:

```bash
git clone https://github.com/<your-username>/IROSA.git
cd IROSA
pip install -e ".[tests,sim]"
```

## Development Workflow

Before submitting a pull request, run the full check suite:

```bash
make commit-checks   # format + type check + lint
make test            # run tests
```

### Code Style

- Code is formatted with [ruff](https://docs.astral.sh/ruff/) and [black](https://black.readthedocs.io/)
- Type annotations are checked with [mypy](https://mypy-lang.org/)
- Linting is done with ruff

### Running Individual Checks

```bash
make format   # auto-format code
make type     # run mypy type checks
make lint     # run ruff linter
make test     # run pytest
```

## Submitting Changes

1. Create a new branch for your changes
2. Make your changes and ensure all checks pass
3. Write clear commit messages
4. Open a pull request against `main`

## Reporting Issues

Please use [GitHub Issues](https://github.com/DLR-RM/IROSA/issues) to report bugs or request features.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
