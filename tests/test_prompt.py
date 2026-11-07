from __future__ import annotations

import json
import unittest

from llama_kb.config import Config
from llama_kb.prompt import (
    SYSTEM_EN,
    SYSTEM_NO_EVIDENCE_EN,
    SYSTEM_ZH,
    compose_messages,
    compose_no_evidence_messages,
    compose_smalltalk_messages,
    estimate_prompt_tokens,
    last_user_message,
    no_result_answer,
    sources_payload,
)
from llama_kb.retriever import Candidate, RetrievalResult


def candidate(index: int, text: str, source: str = "manual.md") -> Candidate:
    return Candidate(
        id=f"id{index}",
        document=text,
        metadata={
            "source": source,
            "title": "手册",
            "heading": "端口",
            "page": 1 if index % 2 else 0,
            "language": "zh",
        },
        dense_score=0.6,
        bm25_score=1.2,
        score=0.03,
        citation=index,
    )


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.from_env(max_context_chars=500, max_history_messages=4)

    def test_chinese_grounded_prompt(self) -> None:
        messages = [{"role": "user", "content": "管理端口是多少？"}]
        result = RetrievalResult(
            query="管理端口",
            chunks=[candidate(1, "管理端口是 9100。")],
        )
        composed, used = compose_messages(messages, result, self.config)
        self.assertEqual(composed[0]["role"], "system")
        self.assertIn("知识库助手", composed[0]["content"])
        self.assertEqual(composed[0]["content"], SYSTEM_ZH)
        self.assertEqual(composed[-1]["role"], "user")
        self.assertIn("【知识库参考资料】", composed[-1]["content"])
        self.assertIn("[1]", composed[-1]["content"])
        self.assertIn("9100", composed[-1]["content"])
        self.assertEqual(len(used), 1)
        self.assertEqual(used[0].citation, 1)

    def test_english_prompt_and_language(self) -> None:
        messages = [{"role": "user", "content": "What is the management port?"}]
        result = RetrievalResult(
            query="management port",
            chunks=[candidate(1, "The management port is 9100.")],
        )
        composed, _ = compose_messages(messages, result, self.config)
        self.assertEqual(composed[0]["content"], SYSTEM_EN)
        self.assertIn("[Knowledge base references]", composed[-1]["content"])
        self.assertIn("[Question]", composed[-1]["content"])
        self.assertIn("Answer in English", composed[-1]["content"])

    def test_history_citation_stripped_and_trimmed(self) -> None:
        messages = [
            {"role": "user", "content": "旧问题一"},
            {"role": "assistant", "content": "旧回答一 [1]"},
            {"role": "user", "content": "旧问题二"},
            {"role": "assistant", "content": "旧回答二 [2]"},
            {"role": "user", "content": "它的端口是多少？"},
        ]
        result = RetrievalResult(
            query="它的端口",
            chunks=[candidate(1, "端口是 9100。")],
        )
        composed, _ = compose_messages(messages, result, self.config)
        assistant_texts = [
            item["content"] for item in composed if item["role"] == "assistant"
        ]
        self.assertTrue(all("[1]" not in text and "[2]" not in text for text in assistant_texts))
        self.assertLessEqual(len(composed), 1 + self.config.max_history_messages + 1)

    def test_context_budget_and_renumbering(self) -> None:
        messages = [{"role": "user", "content": "端口？"}]
        chunks = [candidate(i, "端口信息 " * 100) for i in range(1, 5)]
        result = RetrievalResult(query="端口", chunks=chunks)
        composed, used = compose_messages(messages, result, self.config)
        self.assertLessEqual(len(used), 4)
        self.assertEqual([item.citation for item in used], list(range(1, len(used) + 1)))
        self.assertLessEqual(
            len(composed[-1]["content"]),
            self.config.max_context_chars + 500,
        )

    def test_smalltalk_has_no_context(self) -> None:
        messages = [{"role": "user", "content": "你好"}]
        composed = compose_smalltalk_messages(messages, self.config)
        self.assertTrue(composed[-1]["content"].startswith("你好"))
        self.assertIn("中文回答", composed[-1]["content"])
        self.assertNotIn("知识库参考资料", composed[-1]["content"])

    def test_no_evidence_prompt_uses_general_knowledge_with_note(self) -> None:
        messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答 [1]"},
            {"role": "user", "content": "法国的首都是哪里？"},
        ]
        composed = compose_no_evidence_messages(messages, self.config)
        self.assertEqual(composed[0]["role"], "system")
        self.assertIn("无明显知识依据", composed[0]["content"])
        self.assertEqual(composed[-1]["role"], "user")
        self.assertIn("法国的首都是哪里", composed[-1]["content"])
        self.assertIn("无明显知识依据", composed[-1]["content"])
        self.assertNotIn("知识库参考资料", composed[-1]["content"])
        assistant_texts = [
            item["content"] for item in composed if item["role"] == "assistant"
        ]
        self.assertTrue(all("[1]" not in text for text in assistant_texts))

    def test_no_evidence_prompt_english(self) -> None:
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        composed = compose_no_evidence_messages(messages, self.config)
        self.assertEqual(composed[0]["content"], SYSTEM_NO_EVIDENCE_EN)
        self.assertIn("No clear knowledge basis", composed[-1]["content"])

    def test_refusal_and_tokens(self) -> None:
        self.assertIn("无法回答", no_result_answer("zh"))
        self.assertIn("cannot answer", no_result_answer("en").lower())
        self.assertGreater(
            estimate_prompt_tokens([{"role": "user", "content": "这是中文问题"}]), 0
        )
        self.assertEqual(last_user_message([{"role": "assistant", "content": "x"}]), "")

    def test_sources_payload_fits(self) -> None:
        chunks = [candidate(i, "很长的内容 " * 100) for i in range(1, 5)]
        payload = sources_payload(chunks, max_bytes=500)
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.assertLessEqual(len(raw), 700)
        self.assertEqual([item["n"] for item in payload], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
