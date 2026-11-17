# Development guide

## Set up a development environment

The project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
git clone https://github.com/kitzeslab/bioacoustics-model-zoo
cd bioacoustics-model-zoo

# core + dev dependencies (pytest, black, tensorflow, ...)
uv sync

# also install optional model dependencies
uv sync --extra all                    # everything
uv sync --extra birdnet --extra hawkears   # or pick specific extras
```

Run commands inside the environment with `uv run <cmd>`, or activate `.venv` directly.

Without uv, a plain pip editable install also works:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]" pytest black requests-mock
```

Notes:
- Supported Python versions are 3.10-3.13 (`requires-python` in `pyproject.toml`).
- `tensorflow-metal` in the dev group is for Apple silicon only; remove it locally if it fails to resolve on another platform.
- Optional dependencies are declared under `[project.optional-dependencies]`; dev-only tools live in `[dependency-groups]`. Add dependencies with `uv add <pkg>` (core), `uv add --optional <extra> <pkg>`, or `uv add --dev <pkg>`.

## Code style

Code is formatted with [Black](https://black.readthedocs.io/):

```bash
uv run black .            # format
uv run black --check .    # check only
```

## Run tests

```bash
uv run pytest                      # all tests
uv run pytest tests/test_birdnet.py   # one model
uv run pytest -m "not slow"        # skip slow full-model tests
```

Tests for a model are skipped automatically when its optional dependencies are not installed, so install the relevant extras to actually exercise a model. Slow tests are marked with `@pytest.mark.slow`.

## Release procedure

1. **Update the version** in `pyproject.toml` (`version = "x.y.z"`) and commit it.

2. **Run the full test suite** with all extras installed, including slow tests:
   ```bash
   uv sync --extra all
   uv run black --check .
   uv run pytest
   ```

3. **Build** the sdist and wheel from a clean `dist/`:
   ```bash
   rm -rf dist/
   uv build
   ```
   This produces `dist/bioacoustics_model_zoo-x.y.z.tar.gz` and `dist/bioacoustics_model_zoo-x.y.z-py3-none-any.whl`.

4. **Check the artifacts**:
   ```bash
   uvx twine check dist/*
   ```
   Optionally, install the wheel in a fresh virtual environment and confirm `import bioacoustics_model_zoo` and `bmz.list_models()` work, and that the BirdSet config YAMLs are included (`unzip -l dist/*.whl | grep configs`).

5. **(Optional) Upload to TestPyPI first**:
   ```bash
   uvx twine upload --repository testpypi dist/*
   ```

6. **Publish to PyPI**:
   ```bash
   uvx twine upload dist/*
   ```
   Authenticate with a PyPI API token: username `__token__`, password the token (`pypi-...`). Provide it interactively, via `TWINE_USERNAME` / `TWINE_PASSWORD`, or in `~/.pypirc`. Never commit tokens.

7. **Tag the release** and push:
   ```bash
   git tag x.y.z
   git push origin main --tags
   ```
   Then create a GitHub release for the tag, if desired.

Uploaded versions cannot be overwritten or reused on PyPI, so if something is wrong, bump the version and release again.
