"""Pure-Python BM25 with the same tokenizer for Chinese and English.

No extra dependencies (rank_bm25, jieba, ...): Chinese uses character
bigrams, English uses words with light stemming. BM25 provides exact-term
recall; dense retrieval provides semantics and cross-lingual recall.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

from .text import tokenize_lexical


class BM25Index:
    def __init__(
        self,
        documents: Mapping[str, str] | Iterable[tuple[str, str]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if isinstance(documents, Mapping):
            items = documents.items()
        else:
            items = documents

        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter[str] = Counter()

        for doc_id, text in items:
            tokens = tokenize_lexical(text)
            if not tokens:
                continue
            index = len(self.ids)
            self.ids.append(doc_id)
            self.doc_len.append(len(tokens))
            tf = Counter(tokens)
            for term, count in tf.items():
                self.postings[term].append((index, count))
                df[term] += 1

        self.count = len(self.ids)
        self.avgdl = sum(self.doc_len) / self.count if self.count else 0.0
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            # BM25 的平滑 idf，始终为正
            self.idf[term] = math.log(1.0 + (self.count - freq + 0.5) / (freq + 0.5))

    def search(self, tokens: Sequence[str], k: int = 20) -> list[tuple[str, float]]:
        if not self.count or not tokens or k <= 0:
            return []

        query_tf = Counter(tokens)
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in query_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            qtf_weight = min(qtf, 3)
            for index, tf in self.postings.get(term, ()):
                doc_len = self.doc_len[index] or 1
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0)
                )
                scores[index] += idf * qtf_weight * tf * (self.k1 + 1.0) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.ids[index], score) for index, score in ranked[:k]]
