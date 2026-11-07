"""Chroma vector storage, FTS5 keyword index, and resumable builds.

Design notes:
- groups are defined by the catalog; the store is agnostic to document type;
- files are updated incrementally by SHA-256, with batched checkpoints so a
  long build can resume after interruption;
- vectors live in Chroma and keyword search in SQLite FTS5, which avoids a
  multi-gigabyte in-memory BM25 for large corpora;
- embeddings may be built on GPU while the server keeps them on CPU.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import chromadb
from chromadb.errors import NotFoundError

from .catalog import Catalog, DocGroup
from .config import Config
from .embedder import Embedder
from .lexical import LexicalIndex
from .text import (
    Chunk,
    SourceFile,
    chunk_document,
    load_document,
    scan_sources,
)

MANIFEST_VERSION = 3
CHROMA_WRITE_BATCH = 256


class IndexOutOfDate(RuntimeError):
    """The index does not match the current configuration or documents."""


@dataclass
class BuildReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    deleted: int = 0
    chunks: int = 0
    skipped: bool = False
    full_rebuild: bool = False
    fts_rebuilt: bool = False
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped:
            return "Index is up to date; build skipped."
        action = "Full rebuild" if self.full_rebuild else "Incremental update"
        fts = ", FTS5 rebuilt" if self.fts_rebuilt else ""
        return (
            f"{action} done: scanned {self.scanned} files, "
            f"added {self.added}, updated {self.updated}, deleted {self.deleted}, "
            f"wrote {self.chunks} chunks in {self.elapsed:.1f}s{fts}."
        )


class KBStore:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.ensure_dirs()
        self.catalog = Catalog.load(config.catalog_path)
        settings = chromadb.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(
            path=str(self.config.db_dir), settings=settings
        )
        self._collection = None
        self._lexical: LexicalIndex | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 集合
    # ------------------------------------------------------------------
    @property
    def collection(self):
        with self._lock:
            if self._collection is None:
                self._collection = self._get_collection(create=False)
        return self._collection

    def _get_collection(self, create: bool = False):
        try:
            collection = self.client.get_collection(self.config.collection_name)
            self._collection = collection
            return collection
        except NotFoundError:
            if not create:
                return None
        collection = self.client.create_collection(
            name=self.config.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "purpose": "general_document_knowledge_base",
                "embed_model": self.config.embed_model_name,
                "chunker_version": self.config.chunker_version,
            },
        )
        self._collection = collection
        return collection

    def _drop_collection(self) -> None:
        try:
            self.client.delete_collection(self.config.collection_name)
        except NotFoundError:
            pass
        finally:
            self._collection = None

    def count(self) -> int:
        collection = self.collection
        return collection.count() if collection is not None else 0

    # ------------------------------------------------------------------
    # FTS5
    # ------------------------------------------------------------------
    def open_lexical(self, *, create: bool = False) -> LexicalIndex:
        if self._lexical is None:
            self._lexical = LexicalIndex(self.config.fts_path, create=create)
        return self._lexical

    def close_lexical(self) -> None:
        if self._lexical is not None:
            self._lexical.close()
            self._lexical = None

    def fts_count(self) -> int:
        try:
            return self.open_lexical(create=False).count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def load_manifest(self) -> dict[str, Any]:
        path = self.config.manifest_path
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _manifest_base(self, embedder: Embedder | None = None) -> dict[str, Any]:
        base: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "collection": self.config.collection_name,
            "chunker_version": self.config.chunker_version,
            "chunk_size": self.config.chunk_size,
            "chunk_max": self.config.chunk_max,
            "chunk_overlap": self.config.chunk_overlap,
            "catalog_path": str(self.config.catalog_path),
            "catalog_fingerprint": self.catalog.index_fingerprint(),
            "embed_model": self.config.embed_model_name,
            "fts_path": str(self.config.fts_path),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if embedder is not None:
            base.update(
                {
                    "embed_config_fingerprint": embedder.config_fingerprint,
                    "embed_fingerprint": embedder.fingerprint,
                    "embed_dimension": embedder.dimension,
                    "embed_snapshot": embedder.snapshot_name,
                }
            )
        return base

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.config.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    @contextmanager
    def _build_lock(self) -> Iterator[None]:
        self.config.ensure_dirs()
        with self.config.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _full_rebuild_required(
        self,
        manifest: dict[str, Any],
        embedder: Embedder,
    ) -> bool:
        collection = self.collection
        if collection is None or collection.count() == 0:
            return True
        if not manifest:
            return True
        checks = {
            "manifest_version": MANIFEST_VERSION,
            "collection": self.config.collection_name,
            "chunker_version": self.config.chunker_version,
            "chunk_size": self.config.chunk_size,
            "chunk_max": self.config.chunk_max,
            "chunk_overlap": self.config.chunk_overlap,
            "embed_model": self.config.embed_model_name,
            "embed_config_fingerprint": embedder.config_fingerprint,
            "catalog_fingerprint": self.catalog.index_fingerprint(),
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                return True
        current_snapshot = embedder.snapshot_name
        if current_snapshot and manifest.get("embed_snapshot") not in ("", current_snapshot):
            return True
        if not isinstance(manifest.get("files"), dict):
            return True
        return False

    def needs_build(self, embedder: Embedder) -> bool:
        """无需加载模型即可判断是否需要运行构建。"""
        if not embedder.is_available():
            return True
        sources = {
            source.rel_path: source for source in scan_sources(self.config.docs_dir)
        }
        if not sources:
            return True
        manifest = self.load_manifest()
        if self._full_rebuild_required(manifest, embedder):
            return True
        files = manifest.get("files", {})
        if set(files) != set(sources):
            return True
        for rel, source in sources.items():
            if files.get(rel, {}).get("sha256") != source.sha256:
                return True
        try:
            return self.fts_count() == 0
        except Exception:
            return True

    def _augment_chunks(self, chunks: Iterable[Chunk], group: DocGroup) -> list[Chunk]:
        metadata = group.metadata()
        prefix = group.embed_prefix.strip()
        result: list[Chunk] = []
        for chunk in chunks:
            chunk.metadata.update(metadata)
            chunk.metadata["group_preferred"] = bool(group.preferred)
            if prefix:
                chunk.embed_text = f"{prefix}\n{chunk.embed_text}"
            result.append(chunk)
        return result

    def _load_file_chunks(
        self, source: SourceFile, group: DocGroup
    ) -> tuple[Chunk, ...]:
        document = load_document(source, self.config.docs_dir)
        if document is None:
            return ()
        chunks = chunk_document(document, self.config)
        return tuple(self._augment_chunks(chunks, group))

    def _add_chunks(self, collection, chunks: list[Chunk], vectors) -> None:
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.document for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        for start in range(0, len(ids), CHROMA_WRITE_BATCH):
            batch_vectors = vectors[start : start + CHROMA_WRITE_BATCH]
            if hasattr(batch_vectors, "tolist"):
                batch_vectors = batch_vectors.tolist()
            collection.add(
                ids=ids[start : start + CHROMA_WRITE_BATCH],
                embeddings=batch_vectors,
                documents=documents[start : start + CHROMA_WRITE_BATCH],
                metadatas=metadatas[start : start + CHROMA_WRITE_BATCH],
            )

    def _checkpoint(
        self,
        embedder: Embedder,
        written_files: dict[str, Any],
        collection,
        lex: LexicalIndex,
    ) -> None:
        manifest = self._manifest_base(embedder)
        manifest["files"] = written_files
        manifest["total_chunks"] = collection.count()
        manifest["fts"] = {
            "path": str(self.config.fts_path),
            "chunks": lex.count(),
            "sources": lex.source_count(),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        manifest["groups"] = lex.group_counts()
        self._write_manifest(manifest)

    def sync(
        self,
        embedder: Embedder,
        if_needed: bool = False,
        rebuild: bool = False,
        allow_download: bool = True,
        show_progress: bool = True,
    ) -> BuildReport:
        started = time.time()
        report = BuildReport()

        with self._build_lock():
            sources = {
                source.rel_path: source
                for source in scan_sources(self.config.docs_dir)
            }
            report.scanned = len(sources)
            if not sources:
                raise RuntimeError(
                    f"No indexable documents in {self.config.docs_dir}\n"
                    "Add .txt / .md / .pdf files and try again."
                )

            manifest = self.load_manifest()
            full_rebuild = rebuild or self._full_rebuild_required(manifest, embedder)
            previous_files = (
                manifest.get("files", {})
                if isinstance(manifest.get("files"), dict)
                else {}
            )

            if full_rebuild:
                print(f"Full rebuild: {self.config.collection_name}", flush=True)
                self._drop_collection()
                collection = self._get_collection(create=True)
                lex = self.open_lexical(create=True)
                lex.rebuild()
                changed = list(sources.values())
                unchanged_files: dict[str, Any] = {}
                deleted: list[str] = []
                report.full_rebuild = True
            else:
                collection = self.collection
                if collection is None:
                    collection = self._get_collection(create=True)
                lex = self.open_lexical(create=True)
                # 索引存在但 FTS 缺失/不一致时，从 Markdown 重建 FTS，不重新嵌入。
                if collection.count() > 0 and lex.count() != collection.count():
                    print("FTS5 index missing or out of sync; rebuilding from documents ...", flush=True)
                    self.rebuild_lexical(show_progress=show_progress)
                    lex = self.open_lexical(create=False)
                    report.fts_rebuilt = True

                changed: list[SourceFile] = []
                unchanged_files = {}
                for rel, source in sources.items():
                    previous = previous_files.get(rel)
                    if previous and previous.get("sha256") == source.sha256:
                        unchanged_files[rel] = previous
                    else:
                        changed.append(source)
                deleted = sorted(set(previous_files) - set(sources))

                if if_needed and not changed and not deleted and not report.fts_rebuilt:
                    report.skipped = True
                    report.elapsed = time.time() - started
                    return report

            if show_progress:
                print(
                    f"Pending: {len(changed)} to add/update, {len(deleted)} to delete.",
                    flush=True,
                )

            embedder.load(allow_download=allow_download)
            written_files: dict[str, Any] = dict(unchanged_files)
            written_chunks = 0

            for rel in deleted:
                collection.delete(where={"source": rel})
                lex.delete_source(rel)
                written_files.pop(rel, None)
                report.deleted += 1
                if show_progress:
                    print(f"  delete: {rel}", flush=True)

            grouped: dict[str, list[SourceFile]] = {}
            for source in changed:
                group = self.catalog.match_source(source.rel_path)
                grouped.setdefault(group.id, []).append(source)

            total_files = len(changed)
            processed_files = 0
            checkpoint_files = max(1, int(self.config.build_checkpoint_files))
            for group_id, group_sources in grouped.items():
                group = self.catalog.group_by_id.get(
                    group_id,
                    self.catalog.default_group,
                )
                if show_progress:
                    print(
                        f"Group [{group.label}]: {len(group_sources)} file(s)",
                        flush=True,
                    )
                for batch_start in range(0, len(group_sources), checkpoint_files):
                    batch = group_sources[batch_start : batch_start + checkpoint_files]
                    file_chunks: dict[str, tuple[Chunk, ...]] = {}
                    file_titles: dict[str, str] = {}
                    all_chunks: list[Chunk] = []
                    for source in batch:
                        chunks = self._load_file_chunks(source, group)
                        file_chunks[source.rel_path] = chunks
                        document_title = source.path.stem
                        if chunks:
                            document_title = str(chunks[0].metadata.get("title") or document_title)
                            all_chunks.extend(chunks)
                        file_titles[source.rel_path] = document_title

                    vectors = None
                    if all_chunks:
                        if show_progress:
                            print(
                                f"  embedding {len(all_chunks)} chunks ...",
                                flush=True,
                            )
                        vectors = embedder.encode_documents_numpy(
                            [chunk.embed_text for chunk in all_chunks],
                            show_progress=False,
                            batch_size=self.config.build_batch_size,
                        )
                        if len(vectors) != len(all_chunks):
                            raise RuntimeError(
                                f"embedding count mismatch: {len(vectors)} != {len(all_chunks)}"
                            )

                    # 整批删除旧块/旧 FTS 行，再整批写入，避免 7 万个文件
                    # 各自触发一次 Chroma 事务造成大量开销。
                    batch_sources = [source.rel_path for source in batch]
                    if batch_sources:
                        try:
                            collection.delete(
                                where={"source": {"$in": batch_sources}}
                            )
                        except Exception:
                            for rel in batch_sources:
                                collection.delete(where={"source": rel})
                        lex.delete_sources(batch_sources, commit=False)
                    if all_chunks:
                        assert vectors is not None
                        self._add_chunks(collection, all_chunks, vectors)
                        lex.add_chunks(all_chunks, commit=False)
                        written_chunks += len(all_chunks)

                    for source in batch:
                        rel = source.rel_path
                        chunks = list(file_chunks.get(rel, ()))
                        is_update = (not full_rebuild) and rel in previous_files
                        written_files[rel] = {
                            "sha256": source.sha256,
                            "size": source.size,
                            "mtime_ns": source.mtime_ns,
                            "chunks": len(chunks),
                            "title": file_titles.get(rel, source.path.stem),
                            "group_id": group.id,
                        }
                        if is_update:
                            report.updated += 1
                        else:
                            report.added += 1
                        processed_files += 1

                    lex.commit()
                    self._checkpoint(embedder, written_files, collection, lex)
                    if show_progress:
                        print(
                            f"  progress: {processed_files}/{total_files} files, "
                            f"{written_chunks} chunks written, checkpoint saved.",
                            flush=True,
                        )

            report.chunks = written_chunks
            self._checkpoint(embedder, written_files, collection, lex)
            if lex.count() != collection.count():
                report.warnings.append(
                    f"FTS chunks ({lex.count()}) and vector chunks ({collection.count()}) differ; "
                    "run: python -m llama_kb.cli rebuild-fts"
                )
            report.elapsed = time.time() - started
            return report

    def rebuild_lexical(self, show_progress: bool = True) -> int:
        """Rebuild FTS5 from existing documents without re-embedding."""
        sources = scan_sources(self.config.docs_dir)
        lex = self.open_lexical(create=True)
        lex.rebuild()
        written = 0
        for position, source in enumerate(sources, start=1):
            group = self.catalog.match_source(source.rel_path)
            chunks = self._load_file_chunks(source, group)
            if chunks:
                lex.add_chunks(chunks, commit=False)
                written += len(chunks)
            if show_progress and position % 2000 == 0:
                print(
                    f"  FTS5: {position}/{len(sources)} files, {written} chunks",
                    flush=True,
                )
        lex.commit()
        manifest = self.load_manifest()
        manifest["fts"] = {
            "path": str(self.config.fts_path),
            "chunks": lex.count(),
            "sources": lex.source_count(),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        manifest["groups"] = lex.group_counts()
        if manifest:
            self._write_manifest(manifest)
        return lex.count()

    # ------------------------------------------------------------------
    # 查询 / 校验
    # ------------------------------------------------------------------
    def validate(self, embedder: Embedder) -> None:
        """Validate index, catalog, manifest, and FTS consistency before serving."""
        collection = self.collection
        if collection is None:
            raise IndexOutOfDate(
                "Knowledge base index not found. Run: python -m llama_kb.cli build"
            )
        if collection.count() == 0:
            raise IndexOutOfDate(
                "Knowledge base index is empty. Add documents and run: "
                "python -m llama_kb.cli build"
            )
        manifest = self.load_manifest()
        if not manifest:
            raise IndexOutOfDate(
                "Index manifest is missing. Run: python -m llama_kb.cli build --rebuild"
            )
        expected = {
            "manifest_version": MANIFEST_VERSION,
            "collection": self.config.collection_name,
            "chunker_version": self.config.chunker_version,
            "chunk_size": self.config.chunk_size,
            "chunk_max": self.config.chunk_max,
            "chunk_overlap": self.config.chunk_overlap,
            "embed_fingerprint": embedder.fingerprint,
            "catalog_fingerprint": self.catalog.index_fingerprint(),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise IndexOutOfDate(
                    f"Index configuration changed ({key} mismatch). Run:\n"
                    "  python -m llama_kb.cli build --rebuild"
                )
        current_snapshot = embedder.snapshot_name
        if current_snapshot and manifest.get("embed_snapshot") not in ("", current_snapshot):
            raise IndexOutOfDate(
                "Embedding model snapshot changed. Run:\n"
                "  python -m llama_kb.cli build --rebuild"
            )
        fts = self.open_lexical(create=False)
        try:
            fts_chunks = fts.count()
        except Exception:
            raise IndexOutOfDate(
                "FTS5 index is missing or corrupted. Run:\n"
                "  python -m llama_kb.cli rebuild-fts"
            ) from None
        if fts_chunks == 0:
            raise IndexOutOfDate(
                "FTS5 index is empty. Run:\n"
                "  python -m llama_kb.cli rebuild-fts"
            )
        if fts_chunks != collection.count():
            raise IndexOutOfDate(
                f"FTS5 chunks ({fts_chunks}) and vector chunks ({collection.count()}) differ. "
                "Run:\n  python -m llama_kb.cli rebuild-fts"
            )

    def query_dense(
        self,
        vector: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self.collection
        if collection is None:
            return []
        total = collection.count()
        if total <= 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": min(max(1, k), total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        with self._lock:
            result = collection.query(**kwargs)
        ids = result.get("ids", [[]])[0] or []
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        records: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            records.append(
                {
                    "id": chunk_id,
                    "document": document or "",
                    "metadata": metadata or {},
                    # 余弦距离 -> 相似度
                    "dense_score": float(1.0 - float(distance)),
                }
            )
        return records

    def all_records(self) -> list[dict[str, Any]]:
        """兼容旧接口；大规模索引下不建议使用。"""
        collection = self.collection
        if collection is None or collection.count() == 0:
            return []
        with self._lock:
            result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            {"id": chunk_id, "document": document or "", "metadata": metadata or {}}
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]

    def source_count(self) -> int:
        try:
            return self.open_lexical(create=False).source_count()
        except Exception:
            return len(self.load_manifest().get("files", {}))

    def stats(self) -> dict[str, Any]:
        collection = self.collection
        manifest = self.load_manifest()
        total = collection.count() if collection is not None else 0
        groups: dict[str, int] = {}
        languages: dict[str, int] = {}
        sources: list[str] = []
        if total:
            try:
                lex = self.open_lexical(create=False)
                groups = lex.group_counts()
                languages = lex.language_counts()
                sources = lex.sources()
            except Exception:
                pass
        files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
        group_files: dict[str, int] = {}
        for entry in files.values():
            group_id = str(entry.get("group_id", "") or "")
            group_files[group_id] = group_files.get(group_id, 0) + 1
        return {
            "collection": self.config.collection_name,
            "total_chunks": total,
            "total_sources": len(sources) or len(files),
            "sources_truncated": len(sources) > 500,
            "sources": sources[:500],
            "groups": {
                group_id: {
                    "label": self.catalog.label_for(group_id),
                    "chunks": chunks,
                    "files": group_files.get(group_id, 0),
                }
                for group_id, chunks in sorted(groups.items())
            },
            "languages": languages,
            "manifest": {
                "built_at": manifest.get("built_at"),
                "embed_model": manifest.get("embed_model"),
                "embed_dimension": manifest.get("embed_dimension"),
                "embed_snapshot": manifest.get("embed_snapshot"),
                "chunker_version": manifest.get("chunker_version"),
                "catalog_fingerprint": manifest.get("catalog_fingerprint"),
                "fts": manifest.get("fts"),
            },
        }
