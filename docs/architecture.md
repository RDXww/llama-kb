# Architecture

llama-kb is a retrieval-augmented generation (RAG) layer that sits between an
OpenAI-compatible client and an OpenAI-compatible model server. It owns the
document index, query routing, prompting, and evidence policy; the model server
owns token generation.

```
terminal chat / OpenAI SDK / editor
              │
              ▼
      RAG proxy (llama_kb.server, :8081)
        ├─ group router        catalog → groups
        ├─ hybrid retriever    Chroma dense + SQLite FTS5
        ├─ evidence gate       inject / refuse / no_evidence
        └─ prompt builder      citations, language, scope
              │  OpenAI /v1/chat/completions (SSE passthrough)
              ▼
      llama.cpp :8080 or any OpenAI-compatible backend
```

## Design goals

1. **Grounding over fluency.** Answers should cite retrieved material and say
   when the material is missing, instead of silently improvising.
2. **Local-first.** No service dependency beyond the model server. Index and
   documents stay on disk; the proxy binds to loopback by default.
3. **Bilingual by default.** Mixed Chinese/English corpora and cross-lingual
   questions should work without maintaining two indexes.
4. **Group isolation.** Similar names in unrelated domains (C vs C++, two
   Docker versions, product manual vs runbook) must not pollute each other.
5. **Operational at scale.** A corpus with hundreds of thousands of chunks
   must build incrementally, survive interruption, and not consume gigabytes
   of RAM for keyword search.

Non-goals for v0.1: multi-user authorization, a web UI, OCR, reranking, and
distributed indexing.

## Components

| Module | Responsibility |
| --- | --- |
| `config.py` | defaults, `KB_*` environment overrides, index fingerprint inputs |
| `text.py` | file scanning, loaders, heading-aware chunking, language detection, bilingual tokenization |
| `catalog.py` | groups, aliases, versions, source prefixes, unsupported groups |
| `router.py` | alias/version routing, follow-up inheritance, description-embedding fallback |
| `embedder.py` | Qwen3-Embedding-0.6B loading, query instruction, batching, fingerprint |
| `lexical.py` | SQLite FTS5 schema and queries; CJK bigrams and English normalization |
| `bm25.py` | small pure-Python BM25 used in tests and for corpus inspection |
| `store.py` | Chroma collection, FTS5 synchronization, manifest, locks, checkpoints |
| `retriever.py` | hybrid candidate merge, diversification, multi-turn rewrite, gates |
| `prompt.py` | bilingual system prompts, reference blocks, citations, source headers |
| `server.py` | HTTP server, request validation, SSE streaming, upstream error handling |
| `chat.py` | terminal client: streaming, commands, source display |
| `cli.py` | build/search/stats/groups commands |
| `importers/devdocs.py` | optional DevDocs JSON → Markdown + catalog importer |

## Indexing

### Loading and chunking

Supported inputs are `.txt`, `.md`, `.markdown`, and `.pdf` (text layer only).
Markdown headings are tracked so every chunk carries its heading path. PDF
pages are extracted individually and carry a 1-based page number.

Chunks target 700 characters with a hard maximum of 900. Splitting prefers
paragraph boundaries, then sentence boundaries, then a 100-character overlap
window for unbroken text. Chunks shorter than 20 characters are dropped.
Each chunk stores:

```
source, title, heading, page, chunk_index, language, chars, content_hash,
content, embed_text, group metadata (id/name/type/version/parent/label)
```

The chunk ID is derived from source, page, index, and content hash, so
identical content in the same file position keeps a stable identity.

### Incremental synchronization

The manifest tracks, for every indexed file, its SHA-256, size, mtime,
chunk count, group, and language. A build:

1. scans the docs directory;
2. computes the set of added/changed/deleted files;
3. groups changed files by catalog group;
4. processes them in batches of 500 files by default;
5. atomically replaces rows for a batch in both Chroma and FTS5;
6. checkpoints the manifest after each batch;
7. removes rows for deleted files.

An interrupted build resumes with `build --if-needed`. A full rebuild is only
required when an index fingerprint changes: manifest version, chunker version,
chunk sizes, embedding model/config, or catalog fingerprint. Changing
`top_n`, gates, or BM25 bonus does **not** require a rebuild.

### Storage layout

```
~/.llama-kb/
  docs/                                   # your documents
  catalog.json                            # group definitions
  index/
    chroma.sqlite3                        # Chroma persistence
    <collection>_manifest.json            # incremental state + fingerprints
    <collection>_fts.sqlite3              # FTS5 index
    <collection>.lock                     # single-writer build lock
  logs/
```

Chroma holds normalized vectors and metadata; FTS5 holds token text plus a
companion `chunks` table with source, heading, group, and document text. The
keyword index is never loaded wholesale into RAM.

## Retrieval

### Query rewriting across turns

If the current question is shorter than 40 characters or contains a referential
term (`它`, `这个`, `上述`, `its`, `this`, ...), the previous user question is
prepended to the retrieval query. This is retrieval-only; the model still sees
the original conversation.

### Routing

Routing order:

1. explicit pinned groups (`doc` field / `X-KB-Doc` / `/doc`);
2. catalog aliases and version combinations, longest match first;
3. inheritance from the previous turn for follow-ups;
4. cosine similarity between the question and group descriptions;
5. fallback to all preferred groups.

If a question names a group listed in `unsupported`, the proxy refuses without
calling the model. If the router is confident about several groups, it may
search them together and label the answer per group.

### Hybrid ranking

Dense top-20 and FTS5 top-20 are merged by chunk ID. The final score is:

```
score = dense_cosine + bm25_bonus * (bm25_score / max_bm25_score)
```

with `bm25_bonus = 0.10` by default. Dense retrieval ranks; lexical matches can
promote an exact hit but cannot dominate the list. This matters in
cross-lingual search, where pure reciprocal-rank fusion can promote a document
that shares one incidental English word but is semantically unrelated.

Results are diversified: at most 2 chunks per source before other sources are
considered, then the list is topped up if needed. The default injection is 6
chunks, capped at 4,200 characters; each chunk is capped at 1,200 characters in
the prompt.

### Evidence gates

Two thresholds separate "no material" from "weakly related material":

- a pinned-group search requires `top_dense >= KB_MIN_DENSE_SCORE` (0.26)
  **or** a lexical hit;
- a global fallback requires `top_dense >= KB_MIN_DENSE_SCORE_GLOBAL` (0.42).

A global search spans every group, so a single noisy keyword is weak evidence;
hence the higher bar. Below the gate the request is marked `no_evidence`. The
model may answer from general knowledge, but the system prompt requires the
first line to carry the disclaimer, and the response header lets clients style
or suppress the answer.

## Prompting and responses

Each request injects only the references for the current turn. Citations in
older assistant messages are stripped so numbering cannot collide. References
are renumbered contiguously after budget trimming.

The system prompt:

- requires answers in the language of the latest question;
- forbids mixing APIs, commands, versions, or conclusions across groups;
- requires `[n]` citations from this turn only;
- requires refusal when the references are insufficient;
- treats retrieved text as untrusted data and ignores instructions inside it;
- forbids inventing versions, ports, URLs, commands, or steps.

The proxy supports both streaming and non-streaming responses. It returns:

| Header | Meaning |
| --- | --- |
| `X-KB-Status` | `ok`, `no_evidence`, `no_result`, `unsupported`, `smalltalk`, `degraded`, `skipped` |
| `X-KB-Sources` | base64 JSON source list, trimmed to fit header limits |

If retrieval raises an exception, chat still works and the request is marked
`degraded`. If the upstream server is unreachable, the proxy returns a 502 with
a structured error.

Prompt history is capped at 8 messages by default. The proxy trims old history
from the front until the estimated prompt fits `KB_MAX_PROMPT_TOKENS`. The
output cap must exceed any reasoning budget on the model server, otherwise a
reasoning model can spend the entire budget thinking and emit no answer.

## Security model

- The proxy binds to `127.0.0.1` by default and warns loudly otherwise.
- No CORS headers and no arbitrary URL forwarding; the upstream is fixed by
  configuration.
- Request bodies are capped (2 MB by default) and validated.
- Retrieved documents are untrusted input; the prompt explicitly separates
  data from instructions.
- There is **no authentication**. Do not expose the proxy to a network without
  a reverse proxy that adds authentication, rate limiting, and TLS.
- The model server itself is also unauthenticated in the default launcher.

## Scale notes

The reference deployment (a single laptop GPU) indexes 73,898 files and
895,910 chunks across 74 groups. An interrupted GPU build was resumed to
completion with `build --if-needed`; retrieval after model load is typically
under one second, while generation dominates latency.

Embeddings are built on GPU when available, but the serving process keeps them
on CPU to leave VRAM for the generation model. On the reference machine,
fp32 CPU embeddings were significantly faster than bf16, which is why CPU
serving defaults to `float32`.

## Why these choices?

- **Chroma + FTS5 instead of one engine:** Chroma gives a small, well-understood
  persistent vector store; FTS5 gives durable keyword search with no extra
  service. Both are local files.
- **Dense-first fusion instead of RRF:** RRF ignores score magnitude, which
  hurt cross-lingual cases where a shared incidental word produced a lexical
  hit in the wrong language. A bounded BM25 boost preserved exact-term recall
  without overriding semantic ranking.
- **Group routing before retrieval:** filtering by metadata after a global
  search is cheap, but the global candidate pool can be dominated by the wrong
  group for very large corpora. Routing first keeps the candidate pool focused
  and makes explicit user intent reliable.
- **Two gates instead of one:** a pinned group is a strong user signal; a global
  fallback is not. Sharing one threshold either admits noise globally or makes
  pinned searches too strict.

## Future work

Reranking, neighboring-chunk expansion, OCR, evaluation harnesses, optional
auth, and a web UI are natural extensions. Components are deliberately
separated so a reranker slots between `retriever.py` and `prompt.py`, and an
OCR step slots into `text.py` without touching routing or the API.
