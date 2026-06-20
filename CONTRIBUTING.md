# Contributing to JasWolf

Thank you for wanting to contribute! 🐺

## Code of Conduct

Be kind, be constructive, and remember that JasWolf was built with love for companionship.

## How to Contribute

1. **Fork** the repo
2. **Create a branch**: `git checkout -b feature/your-feature`
3. **Make your changes**
4. **Run tests**: `python -m pytest tests/ -q`
5. **Run ruff**: `ruff check src/`
6. **Commit**: Use clear, descriptive commit messages
7. **Push**: `git push origin feature/your-feature`
8. **Open a Pull Request**

## Development Setup

```bash
git clone https://github.com/iamvinay5555/jaswolf.git
cd jaswolf
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Guidelines

- Keep the code simple — JasWolf's strength is its focused design
- Add tests for new features
- Write clear docstrings
- Don't add dependencies without good reason
- Respect the existing API surface
