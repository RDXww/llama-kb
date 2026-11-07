"""Hybrid retrieval: group routing + dense ranking + FTS5 lexical boost.

Routing is catalog-driven. Explicit aliases win; otherwise group-description
embeddings decide; finally the search falls back to all preferred groups.
This handles both "C vs C++" style manual questions and entirely unrelated
document sets such as an employee handbook and an API reference.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .catalog import Catalog, DocGroup
from .config import Config
from .embedder import Embedder
from .lexical import LexicalIndex
from .prompt import build_retrieval_query, last_user_message
from .router import DocRouter, RoutingDecision
from .store import KBStore
from .text import is_smalltalk, tokenize_lexical


@dataclass
class Candidate:
    id: str
    document: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    score: float = 0.0
    citation: int = 0
    expanded: bool = False

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def group_id(self) -> str:
        return str(self.metadata.get("group_id", ""))


@dataclass
class RetrievalResult:
    query: str = ""
    chunks: list[Candidate] = field(default_factory=list)
    skipped: bool = False
    smalltalk: bool = False
    reason: str = ""
    top_dense: float = 0.0
    lexical_hits: int = 0
    routing: RoutingDecision | None = None
    groups: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)


class Retriever:
    def __init__(
        self,
        config: Config,
        store: KBStore,
        embedder: Embedder,
        *,
        lexical: LexicalIndex | None = None,
        router: DocRouter | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.catalog: Catalog = getattr(store, "catalog", None) or Catalog()
        self.lexical = lexical
        if self.lexical is None and hasattr(store, "open_lexical"):
            try:
                self.lexical = store.open_lexical(create=False)
            except Exception:
                self.lexical = None
        self._embed_lock = threading.Lock()
        self.router = router or DocRouter(
            self.catalog,
            embedder,
            min_score=config.router_min_score,
            max_groups=config.router_max_groups,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def retrieve(
        self,
        messages: list[dict[str, Any]],
        forced_groups: Iterable[DocGroup] | None = None,
    ) -> RetrievalResult:
        question = last_user_message(messages)
        if not question:
            return RetrievalResult(skipped=True, reason="no_user")
        if is_smalltalk(question):
            return RetrievalResult(
                query=question, skipped=True, smalltalk=True, reason="smalltalk"
            )
        query = build_retrieval_query(messages, max_chars=1200)
        return self.retrieve_text(
            query,
            forced_groups=forced_groups,
            messages=messages,
        )

    def retrieve_text(
        self,
        query: str,
        limit: int | None = None,
        forced_groups: Iterable[DocGroup] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        query = (query or "").strip()[: self.config.max_question_chars]
        if not query:
            return RetrievalResult(query=query, skipped=True, reason="empty_query")

        with self._embed_lock:
            vector = self.embedder.encode_query(query)

        if self.config.router_enabled:
            routing = self.router.route(
                query,
                query_vector=vector,
                forced_groups=forced_groups,
                messages=messages,
            )
        else:
            routing = RoutingDecision(
                mode="multi" if forced_groups else "global",
                groups=list(forced_groups or []),
                reason="router_disabled",
                explicit=bool(forced_groups),
            )

        if routing.mode == "unsupported":
            return RetrievalResult(
                query=query,
                skipped=True,
                reason="unsupported_doc",
                routing=routing,
                unsupported=list(routing.unsupported),
            )

        group_ids = list(routing.group_ids)
        if not group_ids and routing.mode == "global":
            group_ids = list(self.catalog.preferred_ids)
        where = {"group_id": {"$in": group_ids}} if group_ids else None

        try:
            dense_hits = self.store.query_dense(
                vector, self.config.dense_k, where=where
            )
        except TypeError:
            # 兼容旧测试替身/旧存储实现
            dense_hits = self.store.query_dense(vector, self.config.dense_k)

        tokens = tokenize_lexical(query)
        fts_hits: list[dict[str, Any]] = []
        if self.lexical is not None and tokens:
            try:
                fts_hits = self.lexical.search(
                    tokens,
                    self.config.bm25_k,
                    group_ids=group_ids or None,
                )
            except Exception:
                fts_hits = []

        fused: dict[str, Candidate] = {}

        for hit in dense_hits:
            record_id = str(hit.get("id", ""))
            if not record_id:
                continue
            candidate = fused.get(record_id)
            if candidate is None:
                candidate = Candidate(
                    id=record_id,
                    document=str(hit.get("document", "")),
                    metadata=dict(hit.get("metadata") or {}),
                )
                fused[record_id] = candidate
            candidate.dense_score = max(
                candidate.dense_score, float(hit.get("dense_score", 0.0))
            )

        for hit in fts_hits:
            record_id = str(hit.get("id", ""))
            if not record_id:
                continue
            candidate = fused.get(record_id)
            if candidate is None:
                candidate = Candidate(
                    id=record_id,
                    document=str(hit.get("document", "")),
                    metadata=dict(hit.get("metadata") or {}),
                )
                fused[record_id] = candidate
            candidate.bm25_score = max(
                candidate.bm25_score, float(hit.get("bm25_score", 0.0))
            )

        if not fused:
            return RetrievalResult(
                query=query,
                skipped=True,
                reason="no_candidates",
                top_dense=0.0,
                routing=routing,
                groups=group_ids,
                unsupported=list(routing.unsupported),
            )

        max_bm25 = max((item.bm25_score for item in fused.values()), default=0.0)
        bonus = max(0.0, float(self.config.bm25_bonus))
        for candidate in fused.values():
            lexical_boost = (
                candidate.bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            )
            candidate.score = candidate.dense_score + bonus * lexical_boost

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item.score,
                -item.dense_score,
                -item.bm25_score,
                item.id,
            ),
        )

        top_dense = max((item.dense_score for item in ranked), default=0.0)
        lexical_hits = len(fts_hits)

        # 双弱门控：语义相似度低且没有任何关键词命中，则认为资料不相关。
        # 全局兜底检索覆盖全部默认分组，单个噪声关键词命中不足以作为依据，
        # 因此全局路由要求更高的语义相似度。
        global_route = getattr(routing, "mode", "") == "global"
        threshold = float(self.config.min_dense_score)
        if global_route:
            threshold = max(
                threshold,
                float(
                    getattr(
                        self.config,
                        "min_dense_score_global",
                        threshold,
                    )
                ),
            )
        if top_dense < threshold and (lexical_hits == 0 or global_route):
            return RetrievalResult(
                query=query,
                skipped=True,
                reason="low_relevance",
                top_dense=top_dense,
                lexical_hits=lexical_hits,
                routing=routing,
                groups=group_ids,
                unsupported=list(routing.unsupported),
            )

        top_n = max(1, int(limit)) if limit is not None else self.config.top_n
        selected = self._select(ranked, top_n, group_ids)
        for index, candidate in enumerate(selected, start=1):
            candidate.citation = index

        return RetrievalResult(
            query=query,
            chunks=selected,
            reason="ok",
            top_dense=top_dense,
            lexical_hits=lexical_hits,
            routing=routing,
            groups=group_ids,
            unsupported=list(routing.unsupported),
        )

    def _select(
        self,
        ranked: list[Candidate],
        limit: int,
        group_ids: Sequence[str],
    ) -> list[Candidate]:
        """多分组时先每组各取一块，再按来源多样性补足。"""
        selected: list[Candidate] = []
        seen: set[str] = set()
        source_counts: Counter[str] = Counter()

        if len(group_ids) > 1:
            for group_id in group_ids:
                if len(selected) >= limit:
                    break
                for candidate in ranked:
                    if candidate.id in seen:
                        continue
                    if candidate.group_id != group_id:
                        continue
                    if source_counts[candidate.source] >= self.config.max_per_source:
                        continue
                    selected.append(candidate)
                    seen.add(candidate.id)
                    source_counts[candidate.source] += 1
                    break

        for candidate in ranked:
            if len(selected) >= limit:
                break
            if candidate.id in seen:
                continue
            if source_counts[candidate.source] >= self.config.max_per_source:
                continue
            selected.append(candidate)
            seen.add(candidate.id)
            source_counts[candidate.source] += 1

        if len(selected) < limit:
            for candidate in ranked:
                if len(selected) >= limit:
                    break
                if candidate.id in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.id)

        return selected

    def search(
        self,
        query: str,
        limit: int | None = None,
        forced_groups: Iterable[DocGroup] | None = None,
    ) -> RetrievalResult:
        return self.retrieve_text(
            query, limit=limit, forced_groups=forced_groups
        )

    def stats(self) -> dict[str, Any]:
        return {
            "fts_documents": self.lexical.count() if self.lexical is not None else 0,
            "groups": len(self.catalog.groups),
            "preferred_groups": len(self.catalog.preferred_groups),
            "top_dense": self.config.min_dense_score,
            "router_min_score": self.router.min_score,
        }
