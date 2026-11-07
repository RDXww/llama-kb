from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llama_kb.lexical import LexicalIndex
from llama_kb.text import Chunk, tokenize_lexical


def make_chunk(chunk_id: str, text: str, source: str, group: str = "c") -> Chunk:
    return Chunk(
        id=chunk_id,
        document=text,
        embed_text=text,
        metadata={
            "source": source,
            "title": text.splitlines()[0][:40],
            "heading": "",
            "page": 0,
            "chunk_index": 0,
            "language": "en",
            "group_id": group,
            "group_name": group.upper(),
            "group_label": group.upper(),
            "group_type": "document",
            "group_version": "",
            "group_parent": "",
            "group_preferred": True,
        },
        language="en",
    )


class LexicalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "fts.sqlite3"
        self.index = LexicalIndex(self.path, create=True)
        self.index.rebuild()
        self.index.add_chunks(
            [
                make_chunk("c1", "strtok_s splits a string with strtok.", "c/string/strtok.md"),
                make_chunk("c2", "malloc allocates memory.", "c/memory/malloc.md"),
                make_chunk("cpp1", "std::vector is a C++ container.", "cpp/vector.md", "cpp"),
            ]
        )

    def tearDown(self) -> None:
        self.index.close()

    def test_search_exact_identifier(self) -> None:
        hits = self.index.search(tokenize_lexical("strtok_s"), 5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["id"], "c1")
        self.assertGreater(hits[0]["bm25_score"], 0)

    def test_group_filter(self) -> None:
        hits = self.index.search(tokenize_lexical("vector"), 5, group_ids=["cpp"])
        self.assertEqual([hit["id"] for hit in hits], ["cpp1"])
        hits = self.index.search(tokenize_lexical("vector"), 5, group_ids=["c"])
        self.assertEqual(hits, [])

    def test_delete_source_updates_fts(self) -> None:
        self.index.delete_source("c/string/strtok.md")
        self.assertEqual(self.index.count(), 2)
        hits = self.index.search(tokenize_lexical("strtok_s"), 5)
        self.assertEqual(hits, [])

    def test_batch_delete_sources(self) -> None:
        self.index.delete_sources(["c/memory/malloc.md", "cpp/vector.md"])
        self.assertEqual(self.index.count(), 1)
        self.assertEqual(self.index.group_counts(), {"c": 1})

    def test_stats(self) -> None:
        self.assertEqual(self.index.count(), 3)
        self.assertEqual(self.index.source_count(), 3)
        self.assertEqual(self.index.group_counts(), {"c": 2, "cpp": 1})
        self.assertEqual(self.index.language_counts(), {"en": 3})


if __name__ == "__main__":
    unittest.main()
