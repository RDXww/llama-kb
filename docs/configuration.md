# Configuration

All settings are environment variables. Unset variables use the defaults
below; nothing needs to be configured for a first local run.

## Paths

| Variable | Default | Description |
| --- | --- | --- |
| `KB_DATA_DIR` | `~/.llama-kb` | Root used to derive the defaults below |
| `KB_DOCS_DIR` | `$KB_DATA_DIR/docs` | Documents to index (`.md`, `.markdown`, `.txt`, `.pdf`) |
| `KB_CATALOG` | `$KB_DATA_DIR/catalog.json` | Group definitions |
| `KB_DB_DIR` | `$KB_DATA_DIR/index` | Chroma, FTS5, manifest, lock |
| `KB_LOG_DIR` | `$KB_DATA_DIR/logs` | Launcher and service logs |
| `KB_EMBED_CACHE` | `~/.cache/llama-kb/embedding` | Downloaded embedding model |
| `KB_FTS_PATH` | derived | Override the FTS5 database path |

## Embedding

| Variable | Default | Description |
| --- | --- | --- |
| `KB_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Sentence-transformers model ID |
| `KB_EMBED_DEVICE` | `cpu` | Build device; `cuda` is recommended for large builds |
| `KB_EMBED_DTYPE` | `float32` | `float16` is faster on GPU; `float32` was faster on the reference CPU |
| `KB_EMBED_BATCH` | `8` | Encoding batch size in the serving process |
| `KB_EMBED_THREADS` | `8` | Torch CPU threads |
| `KB_EMBED_MAX_SEQ` | `1024` | Maximum sequence length |
| `KB_CHUNKER_VERSION` | `v2` | Bump when chunk logic changes |
| `KB_CHUNK_SIZE` | `700` | Target chunk size (characters) |
| `KB_CHUNK_MAX` | `900` | Hard chunk limit |
| `KB_CHUNK_OVERLAP` | `100` | Overlap for text without sentence boundaries |
| `KB_MIN_CHUNK_CHARS` | `20` | Drop chunks below this size |
| `KB_BUILD_BATCH` | `16` | Embedding batch size during builds |
| `KB_BUILD_CHECKPOINT_FILES` | `500` | Files per checkpoint batch |
| `KB_COLLECTION` | `llamakb_v1_qwen3_0_6b` | Index namespace; changing it creates a new collection |

Changing the model, chunk size, chunker version, or collection name requires a
rebuild. The manifest fingerprints these values and refuses to serve a
mismatched index.

## Retrieval

| Variable | Default | Description |
| --- | --- | --- |
| `KB_DENSE_K` | `20` | Dense candidates |
| `KB_BM25_K` | `20` | FTS5 candidates |
| `KB_BM25_BONUS` | `0.10` | Maximum normalized lexical boost |
| `KB_TOP_N` | `6` | Chunks injected into the answer |
| `KB_MAX_PER_SOURCE` | `2` | Source diversity cap before topping up |
| `KB_MAX_CONTEXT_CHARS` | `4200` | Total prompt budget for references |
| `KB_MAX_CHUNK_CHARS` | `1200` | Per-chunk character cap in the prompt |
| `KB_MIN_DENSE_SCORE` | `0.26` | Gate when a group is pinned |
| `KB_MIN_DENSE_SCORE_GLOBAL` | `0.42` | Gate for global fallback |
| `KB_ROUTER` | `1` | Enable group routing |
| `KB_ROUTER_MIN_SCORE` | unset | Force a minimum description score for routing |
| `KB_ROUTER_MAX_GROUPS` | `3` | Maximum groups searched together |

Tuning:

- Too many unrelated citations → raise `KB_MIN_DENSE_SCORE_GLOBAL` slightly.
- Missing answers that exist in the docs → lower `KB_MIN_DENSE_SCORE` and/or
  raise `KB_TOP_N`.
- Exact terms (ports, flags, API names) still missed → raise `KB_BM25_K` or
  `KB_BM25_BONUS`.
- Keyword noise dominating → lower `KB_BM25_BONUS`.

Retrieval settings do not require a rebuild.

## History and output

| Variable | Default | Description |
| --- | --- | --- |
| `KB_MAX_HISTORY` | `8` | Prior messages kept |
| `KB_MAX_QUESTION_CHARS` | `2000` | Maximum question length |
| `KB_MAX_PROMPT_TOKENS` | `5600` | Estimated prompt budget |
| `KB_MAX_OUTPUT_TOKENS` | `768` | Output cap including reasoning |
| `KB_HARD_MAX_OUTPUT_TOKENS` | `1536` | Client-requested cap |
| `KB_TEMPERATURE` | `0.3` | Default temperature |
| `KB_STRICT` | `1` | Strict grounding mode |

If you use a reasoning model (DeepSeek-R1 style), make sure the model server's
reasoning budget is smaller than the output cap. Otherwise the model can spend
all tokens thinking and never emit `content`.

## Proxy and upstream

| Variable | Default | Description |
| --- | --- | --- |
| `KB_LLAMA_URL` | `http://127.0.0.1:8080` | OpenAI-compatible upstream base URL |
| `KB_PROXY_HOST` | `127.0.0.1` | Bind address; changing this exposes the proxy |
| `KB_PROXY_PORT` | `8081` | Bind port |
| `KB_UPSTREAM_TIMEOUT` | `900` | Upstream request timeout (seconds) |
| `KB_REQUEST_BODY_LIMIT` | `2097152` | Request body limit (bytes) |
| `KB_SOURCES_HEADER_LIMIT` | `6000` | `X-KB-Sources` size budget |
| `KB_DEBUG` | `0` | Verbose logging |

## Launcher (`scripts/start.sh`)

| Variable | Default | Description |
| --- | --- | --- |
| `KB_MODEL` | unset | GGUF model path; omit when attaching to a backend |
| `KB_PYTHON` | `python3` | Interpreter used for all services |
| `KB_NGL` | auto | GPU layers; 99 when `nvidia-smi` works, otherwise 0 |
| `KB_CTX` | `8192` | Context size |
| `KB_LLAMA_PORT` | `8080` | llama-server port |
| `KB_LLAMA_EXTRA_ARGS` | unset | Extra llama-server flags (for example `--jinja`) |
| `KB_ALIAS` | `llama-kb` | Model alias advertised upstream |
| `KB_SERVE_EMBED_DEVICE` | `cpu` | Embedding device while serving |
| `KB_SERVE_EMBED_DTYPE` | `float32` | Embedding dtype while serving |
| `KB_ALLOW_CPU_BUILD` | `1` | Set to 0 to abort when no CUDA is found |
| `KB_NO_SEED` | `0` | Set to 1 to stop first-run demo document seeding |
| `LLAMA_KB_USE_HF_MIRROR` | `0` | Set to 1 to download through hf-mirror.com |

Flags: `--model PATH`, `--docs PATH`, `--catalog PATH`, `--rebuild`,
`--no-build`, `--no-chat`, `--no-llama`.

## Catalog fields

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Stable identifier used by `doc` / `X-KB-Doc` |
| `name` | recommended | Display name |
| `type` | no | Free-form type (`programming_language`, `product`, `runbook`, ...) |
| `sources` | yes | Paths or prefixes relative to the docs directory |
| `aliases` | recommended | Explicit names and versions; matched before embeddings |
| `description` | recommended | Used for embedding-based routing |
| `version` | no | Version label; combined aliases are generated by importers |
| `parent` | no | Parent group id for version families |
| `preferred` | no | Whether the group participates in global fallback (default true) |
| `embed_prefix` | no | Prefix prepended to group metadata for description embeddings |
| `attributes` | no | Importer-specific metadata |

The catalog also accepts `default_group`, `unsupported` (name + aliases +
optional message), and `settings` (`router_min_score`,
`router_scope_min_score`, `router_max_groups`, `router_multi_margin`).

## Required rebuilds

Rebuild (`build --rebuild`) when changing:

- the embedding model, dtype used for stored vectors, or `KB_COLLECTION`;
- chunk size/max/overlap or `KB_CHUNKER_VERSION`;
- a group's `sources` or `embed_prefix` (changes the catalog fingerprint);
- the manifest schema (maintainers only).

No rebuild is needed for retrieval gates, `top_n`, BM25 weights, prompts,
history limits, ports, or model-server settings.
