"""Runtime configuration for llama-kb.

All paths live under ``~/.llama-kb`` by default and can be overridden with
``KB_*`` environment variables. This keeps the package independent of the
caller's working directory and makes it easy to run several isolated indexes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_DATA_DIR = Path.home() / ".llama-kb"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llama-kb"


@dataclass
class Config:
    """Runtime settings.

    Changing path/chunking/embedding fields may invalidate an existing index;
    keep the fingerprint logic in ``store`` in sync when adding fields.
    """

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    docs_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR / "docs")
    db_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR / "index")
    embed_cache_dir: Path = field(
        default_factory=lambda: DEFAULT_CACHE_DIR / "embedding"
    )
    log_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR / "logs")
    catalog_path: Path = field(
        default_factory=lambda: DEFAULT_DATA_DIR / "catalog.json"
    )
    fts_path_override: Path | None = None

    # The collection name encodes the embedding model and index schema so a
    # stale collection is never mixed with fresh vectors.
    collection_name: str = "llamakb_v1_qwen3_0_6b"
    embed_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_device: str = "cpu"
    embed_dtype: str = "float32"
    embed_batch_size: int = 8
    embed_threads: int = 8
    embed_max_seq_length: int = 1024
    # Bump when the chunking strategy or metadata changes.
    chunker_version: str = "v2"

    chunk_size: int = 700
    chunk_max: int = 900
    chunk_overlap: int = 100
    min_chunk_chars: int = 20
    build_batch_size: int = 16
    build_checkpoint_files: int = 500

    dense_k: int = 20
    bm25_k: int = 20
    bm25_bonus: float = 0.10
    top_n: int = 6
    max_per_source: int = 2
    max_context_chars: int = 4200
    max_chunk_chars_in_prompt: int = 1200
    # Group-aware routing.
    router_enabled: bool = True
    router_min_score: float | None = None
    router_max_groups: int = 3
    # Gate: require both a low vector score and no lexical hits before
    # declaring "no evidence" inside a pinned group.
    min_dense_score: float = 0.26
    # A global fallback search spans every group, so a single noisy keyword is
    # not enough evidence; require a higher semantic score before injecting.
    min_dense_score_global: float = 0.42

    max_history_messages: int = 8
    max_question_chars: int = 2000
    max_prompt_tokens: int = 5600
    # Controlled by llama-server's --reasoning-budget (default 384).
    # The output limit must exceed the reasoning budget.
    default_max_tokens: int = 768
    hard_max_tokens: int = 1536
    default_temperature: float = 0.3

    llama_url: str = "http://127.0.0.1:8080"
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8081
    upstream_timeout: float = 900.0
    request_body_limit: int = 2 * 1024 * 1024
    max_sources_header_bytes: int = 6000

    strict_grounding: bool = True
    debug: bool = False

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        data_dir = Path(_env("KB_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
        cfg = cls(
            data_dir=data_dir,
            docs_dir=Path(_env("KB_DOCS_DIR", str(data_dir / "docs"))).expanduser(),
            db_dir=Path(_env("KB_DB_DIR", str(data_dir / "index"))).expanduser(),
            embed_cache_dir=Path(
                _env("KB_EMBED_CACHE", str(DEFAULT_CACHE_DIR / "embedding"))
            ).expanduser(),
            log_dir=Path(_env("KB_LOG_DIR", str(data_dir / "logs"))).expanduser(),
            catalog_path=Path(
                _env("KB_CATALOG", str(data_dir / "catalog.json"))
            ).expanduser(),
            fts_path_override=(
                Path(os.environ["KB_FTS_PATH"]).expanduser()
                if os.environ.get("KB_FTS_PATH")
                else None
            ),
            collection_name=_env("KB_COLLECTION", cls.collection_name),
            embed_model_name=_env("KB_EMBED_MODEL", cls.embed_model_name),
            embed_device=_env("KB_EMBED_DEVICE", cls.embed_device),
            embed_dtype=_env("KB_EMBED_DTYPE", cls.embed_dtype),
            embed_batch_size=_env_int("KB_EMBED_BATCH", cls.embed_batch_size),
            embed_threads=_env_int("KB_EMBED_THREADS", cls.embed_threads),
            embed_max_seq_length=_env_int("KB_EMBED_MAX_SEQ", cls.embed_max_seq_length),
            chunker_version=_env("KB_CHUNKER_VERSION", cls.chunker_version),
            chunk_size=_env_int("KB_CHUNK_SIZE", cls.chunk_size),
            chunk_max=_env_int("KB_CHUNK_MAX", cls.chunk_max),
            chunk_overlap=_env_int("KB_CHUNK_OVERLAP", cls.chunk_overlap),
            min_chunk_chars=_env_int("KB_MIN_CHUNK_CHARS", cls.min_chunk_chars),
            build_batch_size=_env_int("KB_BUILD_BATCH", cls.build_batch_size),
            build_checkpoint_files=_env_int(
                "KB_BUILD_CHECKPOINT_FILES", cls.build_checkpoint_files
            ),
            dense_k=_env_int("KB_DENSE_K", cls.dense_k),
            bm25_k=_env_int("KB_BM25_K", cls.bm25_k),
            bm25_bonus=_env_float("KB_BM25_BONUS", cls.bm25_bonus),
            top_n=_env_int("KB_TOP_N", cls.top_n),
            max_per_source=_env_int("KB_MAX_PER_SOURCE", cls.max_per_source),
            max_context_chars=_env_int("KB_MAX_CONTEXT_CHARS", cls.max_context_chars),
            max_chunk_chars_in_prompt=_env_int(
                "KB_MAX_CHUNK_CHARS", cls.max_chunk_chars_in_prompt
            ),
            min_dense_score=_env_float("KB_MIN_DENSE_SCORE", cls.min_dense_score),
            min_dense_score_global=_env_float(
                "KB_MIN_DENSE_SCORE_GLOBAL", cls.min_dense_score_global
            ),
            router_enabled=_env_bool("KB_ROUTER", cls.router_enabled),
            router_min_score=(
                _env_float("KB_ROUTER_MIN_SCORE", 0.0)
                if os.environ.get("KB_ROUTER_MIN_SCORE")
                else None
            ),
            router_max_groups=_env_int("KB_ROUTER_MAX_GROUPS", cls.router_max_groups),
            max_history_messages=_env_int("KB_MAX_HISTORY", cls.max_history_messages),
            max_question_chars=_env_int("KB_MAX_QUESTION_CHARS", cls.max_question_chars),
            max_prompt_tokens=_env_int("KB_MAX_PROMPT_TOKENS", cls.max_prompt_tokens),
            default_max_tokens=_env_int("KB_MAX_OUTPUT_TOKENS", cls.default_max_tokens),
            hard_max_tokens=_env_int("KB_HARD_MAX_OUTPUT_TOKENS", cls.hard_max_tokens),
            default_temperature=_env_float("KB_TEMPERATURE", cls.default_temperature),
            llama_url=_env("KB_LLAMA_URL", cls.llama_url).rstrip("/"),
            proxy_host=_env("KB_PROXY_HOST", cls.proxy_host),
            proxy_port=_env_int("KB_PROXY_PORT", cls.proxy_port),
            upstream_timeout=_env_float("KB_UPSTREAM_TIMEOUT", cls.upstream_timeout),
            request_body_limit=_env_int(
                "KB_REQUEST_BODY_LIMIT", cls.request_body_limit
            ),
            max_sources_header_bytes=_env_int(
                "KB_SOURCES_HEADER_LIMIT", cls.max_sources_header_bytes
            ),
            strict_grounding=_env_bool("KB_STRICT", cls.strict_grounding),
            debug=_env_bool("KB_DEBUG", cls.debug),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown config field: {key}")
            setattr(cfg, key, value)
        return cfg

    @property
    def manifest_path(self) -> Path:
        return self.db_dir / f"{self.collection_name}_manifest.json"

    @property
    def lock_path(self) -> Path:
        return self.db_dir / f"{self.collection_name}.lock"

    @property
    def fts_path(self) -> Path:
        if self.fts_path_override is not None:
            return self.fts_path_override
        return self.db_dir / f"{self.collection_name}_fts.sqlite3"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.embed_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
