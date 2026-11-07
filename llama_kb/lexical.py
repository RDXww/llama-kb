"""SQLite FTS5 keyword index.

Replaces the in-memory pure-Python BM25 for production-sized corpora: at
hundreds of thousands of chunks the in-memory index costs gigabytes and slows
startup, while FTS5 lives on disk and supports stable incremental maintenance.

The FTS content table stores token text; the companion chunks table stores
group, source, metadata, and body text, so a query reads only matching rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from .text import Chunk, tokenize_lexical

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    tokens TEXT NOT NULL,
    metadata TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_group ON chunks(group_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens,
    content='chunks',
    content_rowid='rowid',
    tokenize="unicode61 tokenchars '_.'"
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, tokens) VALUES (new.rowid, new.tokens);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, tokens) VALUES ('delete', old.rowid, old.tokens);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, tokens) VALUES ('delete', old.rowid, old.tokens);
    INSERT INTO chunks_fts(rowid, tokens) VALUES (new.rowid, new.tokens);
END;
"""


def _chunk_tokens(chunk: Chunk) -> str:
    text = "\n".join(
        part
        for part in (
            str(chunk.metadata.get("title", "")),
            str(chunk.metadata.get("heading", "")),
            chunk.document,
        )
        if part
    )
    return " ".join(tokenize_lexical(text))


def _fts_query(tokens: Sequence[str], limit: int = 96) -> str:
    """把 token 组成 FTS5 OR 查询；每个 token 用引号包裹，避免语法字符干扰。"""
    seen: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    if not seen:
        return ""
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in seen)


class LexicalIndex:
    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.path),
            timeout=60.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        if create:
            self._ensure_schema()
        else:
            self._enable_pragmas()

    def _enable_pragmas(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

    def _ensure_schema(self) -> None:
        self._enable_pragmas()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                DROP TRIGGER IF EXISTS chunks_ai;
                DROP TRIGGER IF EXISTS chunks_ad;
                DROP TRIGGER IF EXISTS chunks_au;
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunks;
                """
            )
            self._ensure_schema()

    def add_chunks(self, chunks: Iterable[Chunk], *, commit: bool = True) -> int:
        rows = [
            (
                chunk.id,
                str(chunk.metadata.get("group_id", "")),
                chunk.source,
                str(chunk.metadata.get("language", "")),
                _chunk_tokens(chunk),
                json.dumps(chunk.metadata, ensure_ascii=False, separators=(",", ":")),
                chunk.document,
            )
            for chunk in chunks
        ]
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT INTO chunks(id, group_id, source, language, tokens, metadata, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            if commit:
                self.conn.commit()
        return len(rows)

    def delete_source(self, source: str, *, commit: bool = True) -> int:
        with self._lock:
            cursor = self.conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            if commit:
                self.conn.commit()
            return int(cursor.rowcount or 0)

    def delete_sources(self, sources: Sequence[str], *, commit: bool = True) -> int:
        items = [str(source) for source in sources if source]
        if not items:
            return 0
        deleted = 0
        with self._lock:
            # SQLite 默认变量上限 999；按 400 分批，避免大检查点时报错。
            for start in range(0, len(items), 400):
                batch = items[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                cursor = self.conn.execute(
                    f"DELETE FROM chunks WHERE source IN ({placeholders})",
                    batch,
                )
                deleted += int(cursor.rowcount or 0)
            if commit:
                self.conn.commit()
        return deleted

    def delete_group(self, group_id: str) -> int:
        with self._lock:
            cursor = self.conn.execute("DELETE FROM chunks WHERE group_id = ?", (group_id,))
            self.conn.commit()
            return int(cursor.rowcount or 0)

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def search(
        self,
        tokens: Sequence[str],
        k: int = 20,
        group_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = _fts_query(tokens)
        if not query:
            return []
        k = max(1, min(int(k), 200))
        params: list[Any] = [query]
        group_clause = ""
        if group_ids is not None:
            ids = [str(item) for item in group_ids if item]
            if not ids:
                return []
            group_clause = " AND c.group_id IN (" + ",".join("?" for _ in ids) + ")"
            params.extend(ids)
        params.append(k)
        sql = (
            "SELECT c.id, c.metadata, c.body, c.language, "
            "bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?"
            + group_clause
            + " ORDER BY rank LIMIT ?"
        )
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            rank = float(row["rank"] or 0.0)
            results.append(
                {
                    "id": row["id"],
                    "document": row["body"] or "",
                    "metadata": metadata,
                    # FTS5 的 bm25 越小越好，转成正向分数便于融合。
                    "bm25_score": max(0.0, -rank),
                }
            )
        return results

    def get_metadata(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, metadata, body FROM chunks WHERE id IN ({placeholders})",
                [str(item) for item in chunk_ids],
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            result[str(row["id"])] = {
                "metadata": metadata,
                "document": row["body"] or "",
            }
        return result

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"] if row else 0)

    def source_count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(DISTINCT source) AS n FROM chunks"
            ).fetchone()
        return int(row["n"] if row else 0)

    def group_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT group_id, COUNT(*) AS n FROM chunks GROUP BY group_id"
            ).fetchall()
        return {str(row["group_id"] or ""): int(row["n"]) for row in rows}

    def language_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT language, COUNT(*) AS n FROM chunks GROUP BY language"
            ).fetchall()
        return {str(row["language"] or "unknown"): int(row["n"]) for row in rows}

    def sources(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT source FROM chunks ORDER BY source"
            ).fetchall()
        return [str(row["source"]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
