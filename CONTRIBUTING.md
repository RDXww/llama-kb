# Contributing to llama-kb

Thanks for considering a contribution! Issues, documentation fixes, and pull
requests are all welcome.

## Development setup

```bash
git clone https://github.com/RDXww/llama-kb.git
cd llama-kb
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[devdocs]"
```

Optional quality tools:

```bash
pip install ruff
ruff check llama_kb tests
```

## Running tests

```bash
python -m unittest discover -s tests -v
```

These 45 unit tests do not require a model or network access. Two optional
suites exercise real models:

```bash
python -m tests.selftest                                # embedding model + cross-lingual cases
KB_MODEL=/path/to/model.gguf python -m tests.e2e_check  # full stack, ports 8180/8181
```

The end-to-end test builds a disposable index from `examples/docs`.

## Pull request guidelines

- Keep changes focused; one topic per PR.
- Add or update tests for behavior changes.
- Update both `README.md` and `README.zh-CN.md` when user-facing behavior,
  defaults, or commands change. English is the primary documentation.
- Run `python -m compileall -q llama_kb tests` and the unit suite before
  opening the PR.
- Describe the motivation, the behavior change, and how you tested it.

## Licensing and content rules

- **Do not commit third-party documentation content** (DevDocs exports, API
  manuals, books, paid material, or content copied from another repository).
  Use original sample documents under `examples/` instead.
- Do not commit models, vector indexes, logs, or personal data.
- By submitting a pull request you agree that your contribution is licensed
  under the repository's MIT license.

## Architecture pointers

- Retrieval and evidence gating: `llama_kb/retriever.py`
- Prompting and citations: `llama_kb/prompt.py`
- Indexing and manifests: `llama_kb/store.py`
- HTTP surface: `llama_kb/server.py`
- Design rationale: [docs/architecture.md](docs/architecture.md)

If you plan a larger change (reranker, OCR, web UI, auth), open an issue first
so the interface can be discussed before implementation.
