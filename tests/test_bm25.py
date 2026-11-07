from __future__ import annotations

import unittest

from llama_kb.bm25 import BM25Index
from llama_kb.text import tokenize_lexical


class BM25Tests(unittest.TestCase):
    def test_chinese_bigram_ranking(self) -> None:
        index = BM25Index(
            {
                "a": "星火网关的管理端口是 9100，仅限内网访问。",
                "b": "数据库备份每天执行一次，保留十四天。",
                "c": "星火网关支持多租户和审计日志。",
            }
        )
        hits = index.search(tokenize_lexical("管理端口是多少"), k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "a")

    def test_english_stem_and_exact_terms(self) -> None:
        index = BM25Index(
            {
                "en1": "AuroraDB stores metadata in catalog.db.",
                "en2": "The backup interval is six hours and snapshots are kept for 30 days.",
                "en3": "The management gateway listens on port 9100.",
            }
        )
        hits = index.search(tokenize_lexical("How often are backups taken?"), k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "en2")

        hits = index.search(tokenize_lexical("catalog.db"), k=3)
        self.assertEqual(hits[0][0], "en1")

    def test_empty_index_and_empty_query(self) -> None:
        index = BM25Index({})
        self.assertEqual(index.search(["中国"], k=3), [])
        index = BM25Index({"a": "hello world"})
        self.assertEqual(index.search([], k=3), [])

    def test_scores_are_positive(self) -> None:
        index = BM25Index({"a": "alpha beta", "b": "beta gamma"})
        hits = dict(index.search(tokenize_lexical("beta"), k=2))
        self.assertTrue(all(score > 0 for score in hits.values()))
        self.assertGreater(hits["a"], 0)
        self.assertGreater(hits["b"], 0)


if __name__ == "__main__":
    unittest.main()
