from __future__ import annotations

import unittest

from llama_kb.config import Config
from llama_kb.retriever import Retriever
from llama_kb.text import tokenize_lexical


class FakeStore:
    def __init__(self, records, dense_hits=None):
        self._records = records
        self._dense_hits = dense_hits or []

    def all_records(self):
        return self._records

    def query_dense(self, vector, k, where=None):
        return self._dense_hits[:k]


class FakeLexical:
    def __init__(self, records):
        self.records = {record["id"]: record for record in records}

    def search(self, tokens, k=20, group_ids=None):
        query = set(tokens)
        hits = []
        for record_id, record in self.records.items():
            text_tokens = set(tokenize_lexical(record["document"]))
            overlap = len(query & text_tokens)
            if overlap:
                hits.append(
                    {
                        "id": record_id,
                        "document": record["document"],
                        "metadata": record["metadata"],
                        "bm25_score": float(overlap),
                    }
                )
        hits.sort(key=lambda item: -item["bm25_score"])
        return hits[:k]

    def count(self):
        return len(self.records)


class FakeEmbedder:
    def encode_query(self, text):
        return [1.0, 0.0]


def record(chunk_id: str, text: str, source: str, index: int = 0):
    return {
        "id": chunk_id,
        "document": text,
        "metadata": {
            "source": source,
            "title": source,
            "heading": "",
            "page": 0,
            "chunk_index": index,
            "language": "zh",
        },
    }


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.from_env(
            top_n=2,
            max_per_source=1,
            min_dense_score=0.26,
            dense_k=10,
            bm25_k=10,
        )
        self.records = [
            record("a", "alpha only", "src-a"),
            record("b", "alpha beta shared", "src-b"),
            record("c", "beta gamma", "src-c"),
        ]
        self.dense_hits = [
            {"id": "b", "dense_score": 0.8},
            {"id": "a", "dense_score": 0.6},
            {"id": "c", "dense_score": 0.4},
        ]
        self.store = FakeStore(self.records, self.dense_hits)
        self.retriever = Retriever(
            self.config, self.store, FakeEmbedder(), lexical=FakeLexical(self.records)
        )

    def test_dense_primary_with_bm25_bonus(self) -> None:
        result = self.retriever.retrieve_text("beta")
        self.assertEqual(result.reason, "ok")
        self.assertEqual(result.chunks[0].id, "b")
        self.assertEqual(result.chunks[0].citation, 1)
        self.assertTrue(result.chunks[0].bm25_score > 0)

    def test_source_diversity(self) -> None:
        self.dense_hits = [
            {"id": "b", "dense_score": 0.8},
            {"id": "b", "dense_score": 0.8},
        ]
        self.store = FakeStore(self.records, self.dense_hits)
        retriever = Retriever(
            self.config, self.store, FakeEmbedder(), lexical=FakeLexical(self.records)
        )
        result = retriever.retrieve_text("alpha")
        sources = [chunk.source for chunk in result.chunks]
        self.assertEqual(len(sources), len(set(sources)))

    def test_low_relevance_gate(self) -> None:
        self.dense_hits = [{"id": "a", "dense_score": 0.10}]
        self.store = FakeStore(self.records, self.dense_hits)
        retriever = Retriever(
            self.config, self.store, FakeEmbedder(), lexical=FakeLexical(self.records)
        )
        result = retriever.retrieve_text("完全无关的词汇")
        self.assertFalse(result.chunks)
        self.assertEqual(result.reason, "low_relevance")

    def test_global_route_requires_stronger_dense_score(self) -> None:
        # 0.40 高于局部阈值 0.26，但低于全局阈值 0.42；
        # 即使 FTS 有噪声命中也应判定为无依据，不注入上下文。
        self.dense_hits = [{"id": "a", "dense_score": 0.40}]
        self.store = FakeStore(self.records, self.dense_hits)
        retriever = Retriever(
            self.config, self.store, FakeEmbedder(), lexical=FakeLexical(self.records)
        )
        result = retriever.retrieve_text("alpha")
        self.assertGreater(result.lexical_hits, 0)
        self.assertFalse(result.chunks)
        self.assertEqual(result.reason, "low_relevance")

    def test_global_route_keeps_strong_dense_hits(self) -> None:
        self.dense_hits = [{"id": "a", "dense_score": 0.55}]
        self.store = FakeStore(self.records, self.dense_hits)
        retriever = Retriever(
            self.config, self.store, FakeEmbedder(), lexical=FakeLexical(self.records)
        )
        result = retriever.retrieve_text("alpha")
        self.assertTrue(result.chunks)
        self.assertEqual(result.reason, "ok")

    def test_smalltalk_skips_retrieval(self) -> None:
        result = self.retriever.retrieve([{"role": "user", "content": "你好"}])
        self.assertTrue(result.smalltalk)
        self.assertFalse(result.chunks)

    def test_multiturn_reference_rewrite(self) -> None:
        messages = [
            {"role": "user", "content": "星火网关的管理端口是多少？"},
            {"role": "assistant", "content": "是 9100 [1]。"},
            {"role": "user", "content": "它的备份呢？"},
        ]
        result = self.retriever.retrieve(messages)
        self.assertIn("管理端口", result.query)
        self.assertIn("备份", result.query)

    def test_empty_query(self) -> None:
        result = self.retriever.retrieve_text("   ")
        self.assertTrue(result.skipped)


if __name__ == "__main__":
    unittest.main()
