"""Group router driven entirely by the catalog.

The router does not assume that a group is a programming language:
- explicit alias matching (longest first, version-aware)
- follow-up questions inherit the previous turn's group
- description embeddings as a fallback signal
- detection of groups that are explicitly unsupported
- fall back to all preferred groups when no group is obvious
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .catalog import Catalog, DocGroup
from .text import has_referential_term

_COMPARISON_RE = re.compile(
    r"(区别|差异|对比|比较|不同|相比|分别|和|与|及|vs\.?|versus|compare|difference|between)",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    mode: str = "global"  # forced | scoped | multi | global | unsupported | unknown_group | off
    groups: list[DocGroup] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    reason: str = ""
    explicit: bool = False
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def group_ids(self) -> list[str]:
        return [group.id for group in self.groups]

    @property
    def labels(self) -> list[str]:
        return [group.label for group in self.groups]

    @property
    def scope_text(self) -> str:
        if self.mode == "unsupported":
            return "未收录：" + "、".join(self.unsupported)
        if self.groups:
            return "、".join(self.labels)
        return "全部文档"

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "groups": [
                {"id": group.id, "label": group.label, "type": group.type}
                for group in self.groups
            ],
            "unsupported": list(self.unsupported),
            "reason": self.reason,
            "explicit": self.explicit,
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
        }


class DocRouter:
    def __init__(
        self,
        catalog: Catalog,
        embedder: Any | None = None,
        *,
        min_score: float | None = None,
        max_groups: int | None = None,
        multi_margin: float | None = None,
        scope_min_score: float | None = None,
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        settings = catalog.settings or {}
        self.min_score = float(
            settings.get("router_min_score", 0.30) if min_score is None else min_score
        )
        self.max_groups = int(
            settings.get("router_max_groups", 3) if max_groups is None else max_groups
        )
        self.multi_margin = float(
            settings.get("router_multi_margin", 0.02)
            if multi_margin is None
            else multi_margin
        )
        # 只有把握较大时才锁定单个分组；否则走全局检索，让 FTS/向量共同决定。
        self.scope_min_score = float(
            settings.get("router_scope_min_score", 0.42)
            if scope_min_score is None
            else scope_min_score
        )
        self._summary_groups: list[DocGroup] = []
        self._summary_vectors: list[list[float]] | None = None

    # ------------------------------------------------------------------
    # 预计算分组描述向量
    # ------------------------------------------------------------------
    def precompute(self) -> None:
        if self._summary_vectors is not None:
            return
        if self.embedder is None or not hasattr(self.embedder, "encode_documents"):
            return
        groups = list(self.catalog.preferred_groups)
        if not groups:
            return
        texts = [group.summary_text() for group in groups]
        try:
            vectors = self.embedder.encode_documents(texts, show_progress=False)
        except Exception:
            return
        self._summary_groups = groups
        self._summary_vectors = [list(vector) for vector in vectors]

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def route(
        self,
        text: str,
        *,
        query_vector: Sequence[float] | None = None,
        forced_groups: Iterable[DocGroup] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> RoutingDecision:
        text = (text or "").strip()
        forced = list(forced_groups or [])
        if forced:
            # 用户固定了分组时仍然检测问题里是否点名了未收录分组，
            # 让提示词能明确提醒“Java 等分组未收录”，而不是混入 C 的答案。
            forced_match = self.catalog.match_text(text)
            return RoutingDecision(
                mode="multi" if len(forced) > 1 else "forced",
                groups=forced,
                unsupported=[item.name for item in forced_match.unsupported],
                reason="forced_by_user",
                explicit=True,
            )

        match = self.catalog.match_text(text)
        groups = list(match.groups)
        unsupported = [item.name for item in match.unsupported]
        explicit = match.explicit

        # 多轮追问：当前问题没有显式分组时，引用式/短追问继承上一轮。
        if not groups and not unsupported and messages:
            inherited = self._inherit_groups(messages, text)
            if inherited:
                groups = inherited
                explicit = True

        if groups:
            mode = "multi" if len(groups) > 1 else "scoped"
            reason = "explicit_alias" if match.explicit else "history_inherited"
            return RoutingDecision(
                mode=mode,
                groups=groups,
                unsupported=unsupported,
                reason=reason,
                explicit=explicit,
            )

        if unsupported:
            return RoutingDecision(
                mode="unsupported",
                unsupported=unsupported,
                reason="explicit_unsupported",
                explicit=True,
            )

        # 没有显式别名：用分组描述向量兜底。
        routed = self._route_by_embedding(text, query_vector)
        if routed is not None:
            return routed

        return RoutingDecision(
            mode="global",
            reason="fallback_global",
        )

    def _inherit_groups(
        self, messages: list[dict[str, Any]], text: str
    ) -> list[DocGroup]:
        if not (has_referential_term(text) or len(text) <= 16):
            return []
        for message in reversed(messages):
            if str(message.get("role", "")) != "user":
                continue
            previous = str(message.get("content", "") or "")
            if not previous or previous == text:
                continue
            found = self.catalog.match_text(previous)
            if found.groups:
                return list(found.groups)
        return []

    def _route_by_embedding(
        self, text: str, query_vector: Sequence[float] | None
    ) -> RoutingDecision | None:
        self.precompute()
        if not self._summary_vectors or not self._summary_groups:
            return None
        if query_vector is None:
            if self.embedder is None or not hasattr(self.embedder, "encode_query"):
                return None
            try:
                query_vector = self.embedder.encode_query(text)
            except Exception:
                return None
        query = list(query_vector)
        scored: list[tuple[float, DocGroup]] = []
        for group, vector in zip(self._summary_groups, self._summary_vectors):
            scored.append((_cosine(query, vector), group))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        if not scored:
            return None
        scores = {group.id: score for score, group in scored}
        top_score, top_group = scored[0]
        if top_score < self.min_score:
            return RoutingDecision(
                mode="global",
                reason="router_low_score",
                scores=scores,
            )
        # 默认分组（未分类资料）不作为具体路由目标，直接全局检索。
        if top_group.id == self.catalog.default_group.id:
            return RoutingDecision(
                mode="global",
                reason="router_default_group",
                scores=scores,
            )
        if top_score < self.scope_min_score:
            return RoutingDecision(
                mode="global",
                reason="router_low_confidence",
                scores=scores,
            )
        if len(scored) == 1:
            return RoutingDecision(
                mode="scoped",
                groups=[top_group],
                reason="embedding_route",
                scores=scores,
            )
        second_score, second_group = scored[1]
        wants_multi = bool(_COMPARISON_RE.search(text))
        close = (
            second_score >= self.min_score
            and (top_score - second_score) <= self.multi_margin
        )
        if close:
            if wants_multi:
                groups = [group for _, group in scored[: self.max_groups]]
                return RoutingDecision(
                    mode="multi",
                    groups=groups,
                    reason="embedding_route_multi",
                    scores=scores,
                )
            # 没有显式分组且前两名接近时，不冒险锁定某一个分组。
            # 全局检索会让 FTS 精确词和向量共同决定，避免把 strtok_s
            # 这类同时出现在多个参考手册里的符号错误地限定到 C++。
            return RoutingDecision(
                mode="global",
                reason="router_ambiguous",
                scores=scores,
            )
        return RoutingDecision(
            mode="scoped",
            groups=[top_group],
            reason="embedding_route",
            scores=scores,
        )

    # ------------------------------------------------------------------
    # 未收录 / 可用范围
    # ------------------------------------------------------------------
    def unsupported_answer(self, names: list[str]) -> str:
        details = []
        for name in names:
            entry = next(
                (item for item in self.catalog.unsupported if item.name == name),
                None,
            )
            details.append(entry.message if entry and entry.message else name)
        available = "、".join(group.label for group in self.catalog.preferred_groups[:16])
        suffix = f" 当前可检索的分组包括：{available}。" if available else ""
        return (
            "根据现有知识库无法回答该问题：未收录 "
            + "、".join(details)
            + " 的分组。"
            + suffix
            + "（可以换一个已收录的分组，或把对应文档加入知识库后重新构建索引。）"
        )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
