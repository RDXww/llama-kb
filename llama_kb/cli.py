"""Command-line interface for llama-kb.

Examples:
    python -m llama_kb.cli fetch-model          # download the embedding model
    python -m llama_kb.cli build                # build or update the index
    python -m llama_kb.cli build --if-needed    # skip when nothing changed
    python -m llama_kb.cli search "how to deploy"  # retrieval only
    python -m llama_kb.cli stats                # inspect the index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import Config
from .embedder import Embedder
from .retriever import Retriever
from .store import KBStore


def _config_with_docs(
    docs: str | None,
    device: str | None = None,
    dtype: str | None = None,
) -> Config:
    overrides = {}
    if docs:
        overrides["docs_dir"] = Path(docs).expanduser().resolve()
    if device:
        overrides["embed_device"] = device
    if dtype:
        overrides["embed_dtype"] = dtype
    return Config.from_env(**overrides)


def cmd_fetch_model(args: argparse.Namespace) -> int:
    config = _config_with_docs(None)
    embedder = Embedder(config)
    if args.check:
        if embedder.is_available():
            snapshot = embedder.snapshot_path
            print(f"Embedding model ready: {snapshot}")
            print(f"Cache directory: {config.embed_cache_dir}")
            return 0
        print(
            f"Embedding model not downloaded: {config.embed_model_name}",
            file=sys.stderr,
        )
        return 1
    print(f"Downloading {config.embed_model_name} -> {config.embed_cache_dir}")
    if os.environ.get("LLAMA_KB_USE_HF_MIRROR") != "1":
        print("Using the official Hugging Face endpoint. Set "
              "LLAMA_KB_USE_HF_MIRROR=1 if it is unreachable.", flush=True)
    else:
        print("Using https://hf-mirror.com (LLAMA_KB_USE_HF_MIRROR=1).", flush=True)
    path = embedder.fetch(force=args.force)
    print(f"Done: {path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    config = _config_with_docs(args.docs, args.device, args.dtype)
    store = KBStore(config)
    embedder = Embedder(config)

    if not config.docs_dir.is_dir():
        print(f"Documents directory not found: {config.docs_dir}", file=sys.stderr)
        return 1

    if args.if_needed and not args.rebuild and not store.needs_build(embedder):
        print("Index is up to date; skipping build.")
        return 0

    try:
        report = store.sync(
            embedder,
            if_needed=args.if_needed,
            rebuild=args.rebuild,
            allow_download=not args.offline,
            show_progress=True,
        )
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1

    print(report.summary())
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    print(f"Collection: {config.collection_name}, chunks: {store.count()}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    config = _config_with_docs(args.docs)
    store = KBStore(config)
    embedder = Embedder(config)
    forced = None
    if args.doc:
        forced, unknown = store.catalog.resolve(args.doc)
        if unknown:
            print(
                "Unknown group(s): " + ", ".join(unknown) + " (see `groups`)",
                file=sys.stderr,
            )
            return 1
    if not embedder.is_available():
        print(
            f"Embedding model not downloaded: {config.embed_model_name}\n"
            "Run: python -m llama_kb.cli fetch-model",
            file=sys.stderr,
        )
        return 1
    try:
        embedder.load(allow_download=False)
        store.validate(embedder)
        retriever = Retriever(config, store, embedder)
    except Exception as exc:
        print(f"Cannot load the knowledge base: {exc}", file=sys.stderr)
        return 1

    query = " ".join(args.query).strip()
    try:
        result = retriever.search(query, limit=args.k, forced_groups=forced or None)
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    print(f"Query: {query}")
    if result.routing is not None:
        print(
            f"Route: {result.routing.mode}, scope: {result.routing.scope_text}, "
            f"reason: {result.routing.reason}"
        )
        if result.routing.scores:
            top = sorted(result.routing.scores.items(), key=lambda item: -item[1])[:5]
            print("  Group scores: " + ", ".join(
                f"{key}={value:.3f}" for key, value in top
            ))
    print(
        f"Result: {result.reason}, top_dense={result.top_dense:.4f}, "
        f"lexical_hits={result.lexical_hits}"
    )
    if result.unsupported:
        print("Unsupported groups: " + ", ".join(result.unsupported))
    if not result.chunks:
        print("No relevant documents found.")
        return 0
    for chunk in result.chunks:
        metadata = chunk.metadata
        print(
            f"\n[{chunk.citation}] {metadata.get('group_label') or '-'} | "
            f"{metadata.get('source')} | "
            f"{metadata.get('heading') or '-'} | "
            f"page={metadata.get('page')} | lang={metadata.get('language')}"
        )
        print(
            f"    dense={chunk.dense_score:.4f} bm25={chunk.bm25_score:.3f} "
            f"score={chunk.score:.6f}"
        )
        print("    " + " ".join(chunk.document.split())[:500])
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    config = _config_with_docs(args.docs)
    store = KBStore(config)
    try:
        data = store.stats()
    except Exception as exc:
        print(f"Cannot read the index: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def cmd_groups(args: argparse.Namespace) -> int:
    config = _config_with_docs(None)
    store = KBStore(config)
    payload = {
        "catalog": str(config.catalog_path),
        "default": store.catalog.default_group.id,
        "groups": store.catalog.groups_payload(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_rebuild_fts(args: argparse.Namespace) -> int:
    config = _config_with_docs(args.docs)
    store = KBStore(config)
    print(f"Rebuilding FTS5: {config.fts_path}", flush=True)
    try:
        count = store.rebuild_lexical(show_progress=True)
    except Exception as exc:
        print(f"Rebuild failed: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {count} chunks in FTS5.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llama_kb.cli",
        description="Local bilingual knowledge base for llama.cpp",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch-model", help="download or check the multilingual embedding model"
    )
    fetch.add_argument("--check", action="store_true", help="only check the cache")
    fetch.add_argument("--force", action="store_true", help="re-download")
    fetch.set_defaults(func=cmd_fetch_model)

    build = sub.add_parser("build", help="build or incrementally update the index")
    build.add_argument(
        "--if-needed", action="store_true", help="skip when nothing changed"
    )
    build.add_argument("--rebuild", action="store_true", help="full rebuild")
    build.add_argument(
        "--offline", action="store_true", help="never download the model"
    )
    build.add_argument(
        "--docs", default=None, help="documents directory (default: ~/.llama-kb/docs)"
    )
    build.add_argument(
        "--device", default=None, help="embedding device for the build, e.g. cuda/cpu"
    )
    build.add_argument(
        "--dtype", default=None, help="embedding dtype for the build, e.g. float16"
    )
    build.set_defaults(func=cmd_build)

    search = sub.add_parser("search", help="retrieve only; do not call the LLM")
    search.add_argument("query", nargs="+", help="query text")
    search.add_argument("-k", type=int, default=4, help="number of results")
    search.add_argument(
        "--doc", default=None, help="pin group id/name(s), comma separated"
    )
    search.add_argument("--docs", default=None, help="documents directory")
    search.set_defaults(func=cmd_search)

    stats = sub.add_parser("stats", help="show index statistics")
    stats.add_argument("--docs", default=None, help="documents directory")
    stats.set_defaults(func=cmd_stats)

    groups = sub.add_parser("groups", help="list document groups from the catalog")
    groups.set_defaults(func=cmd_groups)

    fts = sub.add_parser("rebuild-fts", help="rebuild only the FTS5 keyword index")
    fts.add_argument("--docs", default=None, help="documents directory")
    fts.set_defaults(func=cmd_rebuild_fts)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
