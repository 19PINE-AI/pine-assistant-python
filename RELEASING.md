# Releasing pine-assistant (Python)

This package uses **PyPI trusted publishing** (OIDC) — no PyPI tokens are stored in GitHub.

## One-time setup

### Register the trusted publisher on PyPI

1. Go to <https://pypi.org/manage/project/pine-assistant/settings/publishing/> (or add a **pending publisher** at <https://pypi.org/manage/account/publishing/> if the project doesn't exist on PyPI yet).
2. Under **Add a new publisher**, fill in:
   - **Owner:** `pineai`
   - **Repository:** `pine-python`
   - **Workflow name:** `publish.yml`
3. Save.

## CI

CI runs automatically on every push to `main` and on pull requests.
It installs, imports, builds, and verifies the package across Python 3.10 – 3.13.

## Publishing a new version

1. Bump the version in **both** places:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `src/pine_assistant/__init__.py` → `__version__ = "X.Y.Z"`
2. Commit, tag, and push:
   ```bash
   git add -A && git commit -m "release: vX.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```
3. The `publish.yml` workflow triggers on the `v*` tag, runs CI first, then publishes to PyPI.
