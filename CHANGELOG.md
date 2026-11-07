# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-07

Initial open-source release.

### Added

- Group-aware hybrid retrieval over Chroma vectors and SQLite FTS5, with dense
  ranking and a bounded lexical boost.
- Catalog-driven routing across document groups, versions, aliases, and
  unsupported-group declarations.
- OpenAI-compatible RAG proxy with streaming, `X-KB-Status`, `X-KB-Sources`,
  `X-KB-Mode`, and `X-KB-Doc`.
- Terminal chat client with citations, source listing, group pinning, and
  retrieval-only search.
- Incremental, checkpointed index builds with SHA-256 file tracking, index
  fingerprints, and resumable builds.
- Bilingual prompting with cross-lingual retrieval, language enforcement,
  citation renumbering, and untrusted-document hardening.
- Two-tier evidence gates (`no_evidence` instead of fabricated grounding).
- CLI: `fetch-model`, `build`, `search`, `stats`, `groups`, `rebuild-fts`.
- Optional DevDocs JSON importer and generic catalog generation.
- Original demo corpus, architecture documentation, and 45 unit tests.
