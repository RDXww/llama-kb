from __future__ import annotations

import unittest

from llama_kb.config import Config
from llama_kb.text import (
    LoadedDoc,
    Page,
    _hard_split,
    chunk_document,
    detect_language,
    estimate_tokens,
    has_referential_term,
    is_smalltalk,
    query_language,
    tokenize_lexical,
)


class TextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.from_env(chunk_size=200, chunk_max=280, chunk_overlap=40)
        self.doc = LoadedDoc(
            path="dummy.md",  # type: ignore[arg-type]
            rel_path="manual.md",
            title="运维手册",
            is_markdown=True,
            pages=[
                Page(
                    text=(
                        "# 运维手册\n\n"
                        "## 管理端口\n\n"
                        "管理服务使用 9100 端口。"
                        "只有内网可以访问，请勿暴露到公网。\n\n"
                        "## 备份策略\n\n"
                        "系统每天凌晨 2 点自动备份，保留 14 天。"
                    ),
                    page=0,
                )
            ],
        )

    def test_markdown_chunking_and_heading(self) -> None:
        chunks = chunk_document(self.doc, self.config)
        self.assertGreaterEqual(len(chunks), 2)
        headings = [chunk.heading for chunk in chunks]
        self.assertTrue(any("管理端口" in heading for heading in headings))
        self.assertTrue(any("备份策略" in heading for heading in headings))
        for chunk in chunks:
            self.assertLessEqual(len(chunk.document), self.config.chunk_max + 20)
            self.assertTrue(chunk.id)
            self.assertEqual(chunk.metadata["source"], "manual.md")
            self.assertEqual(chunk.metadata["language"], "zh")
            self.assertNotIn("# 运维手册", chunk.document)

    def test_hard_split_overlap(self) -> None:
        text = "a" * 1000
        pieces = _hard_split(text, size=200, overlap=50)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 200 for piece in pieces))
        self.assertEqual(pieces[0][-50:], pieces[1][:50])

    def test_long_paragraph_respects_max(self) -> None:
        long_text = "这是一个很长的句子。" * 100
        doc = LoadedDoc(
            path="dummy.txt",  # type: ignore[arg-type]
            rel_path="long.txt",
            title="长文",
            pages=[Page(text=long_text, page=0)],
        )
        chunks = chunk_document(doc, self.config)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.document), self.config.chunk_max)

    def test_language_detection(self) -> None:
        self.assertEqual(detect_language("这是中文文档，包含 English terms."), "en")
        self.assertEqual(detect_language("这是一段很长的中文内容，用来测试语言识别。"), "zh")
        self.assertEqual(detect_language("This is an English document about ports."), "en")
        self.assertEqual(query_language("AuroraDB 的备份间隔是多久？"), "zh")
        self.assertEqual(query_language("How often are backups taken?"), "en")

    def test_tokenizer_bilingual(self) -> None:
        tokens = tokenize_lexical("智能助手 Pro 支持 tools and ports v2.5.1")
        self.assertIn("智能", tokens)
        self.assertIn("助手", tokens)
        self.assertIn("pro", tokens)
        self.assertIn("tools", tokens)
        self.assertIn("tool", tokens)
        self.assertIn("ports", tokens)
        self.assertIn("port", tokens)
        self.assertIn("v2.5.1", tokens)

    def test_smalltalk_and_reference(self) -> None:
        self.assertTrue(is_smalltalk("你好！"))
        self.assertTrue(is_smalltalk("Hello"))
        self.assertFalse(is_smalltalk("你好，如何部署？"))
        self.assertTrue(has_referential_term("它的端口是多少"))
        self.assertTrue(has_referential_term("What about its port?"))
        self.assertFalse(has_referential_term("How do I deploy the product?"))

    def test_estimate_tokens(self) -> None:
        self.assertGreater(estimate_tokens("这是一段中文"), 0)
        self.assertGreater(estimate_tokens("hello world"), 0)


if __name__ == "__main__":
    unittest.main()
