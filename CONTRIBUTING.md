# Contributing

Thanks for your interest in improving GraphRAG! This is a small project, so the
process is deliberately lightweight.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your GEMINI_API_KEY
```

You need a running Neo4j (5.x) for the full app; `docker-compose up -d` starts
one locally. The unit tests do **not** require Neo4j or a Gemini key — the
external services are mocked, and the Neo4j integration tests auto-skip when
`NEO4J_URI` is unset.

## Before opening a pull request

Run the same two checks CI runs:

```bash
ruff check .      # lint
pytest -q         # tests
```

Both must pass. If you touch the Gemini call sites, please also run a manual
smoke test against the live API, since the unit tests mock the SDK.

## Style

- Format and lint with [ruff](https://docs.astral.sh/ruff/) (config in
  `pyproject.toml`, line length 100).
- Type hints on public functions; docstrings in the Google style already used
  throughout the codebase.
- Keep pull requests focused — one logical change per PR, with a descriptive
  commit message (the history uses Conventional Commits, e.g. `feat:`, `fix:`,
  `docs:`, `test:`, `chore:`).

## Reporting bugs

Open an issue describing what you did, what you expected, and what happened.
A minimal reproduction (a short document + the query and mode) helps a lot.
