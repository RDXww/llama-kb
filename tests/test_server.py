from __future__ import annotations

import base64
import http.client
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_kb.config import Config
from llama_kb.retriever import Candidate, RetrievalResult
from llama_kb.server import Handler, KBServer


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []

    def log_message(self, fmt, *args):  # noqa: N802
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).captured.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        data = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result

    def retrieve(self, messages):
        return self.result


class FakeStore:
    def count(self):
        return 1

    def all_records(self):
        return []


class FakeEmbedder:
    dimension = 1024


class FakeService:
    def __init__(self, config: Config, result: RetrievalResult):
        self.config = config
        self.retriever = FakeRetriever(result)
        self.store = FakeStore()
        self.embedder = FakeEmbedder()
        self.started_at = time.time()


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        cls.upstream.daemon_threads = True
        cls.upstream_thread = threading.Thread(
            target=cls.upstream.serve_forever, daemon=True
        )
        cls.upstream_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def start_proxy(self, result: RetrievalResult):
        upstream_port = self.upstream.server_address[1]
        config = Config.from_env(
            llama_url=f"http://127.0.0.1:{upstream_port}",
            max_prompt_tokens=5600,
        )
        server = KBServer(("127.0.0.1", 0), Handler)
        server.service = FakeService(config, result)  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server.server_address[1]

    def post(self, port: int, payload: dict) -> tuple[int, dict, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            body = json.dumps(payload).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            data = response.read()
            headers = {key.lower(): value for key, value in response.getheaders()}
            return response.status, headers, json.loads(data.decode("utf-8"))
        finally:
            conn.close()

    def chunk(self) -> Candidate:
        return Candidate(
            id="c1",
            document="管理端口是 9100。",
            metadata={
                "source": "manual.md",
                "title": "手册",
                "heading": "端口",
                "page": 0,
                "language": "zh",
            },
            dense_score=0.7,
            bm25_score=1.0,
            score=0.03,
            citation=1,
        )

    def test_grounded_non_stream(self) -> None:
        FakeUpstreamHandler.captured.clear()
        result = RetrievalResult(
            query="管理端口",
            chunks=[self.chunk()],
            reason="ok",
            top_dense=0.7,
            lexical_hits=2,
        )
        port = self.start_proxy(result)
        status, headers, data = self.post(
            port,
            {"messages": [{"role": "user", "content": "管理端口是多少？"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["x-kb-status"], "ok")
        self.assertEqual(data["choices"][0]["message"]["content"], "answer")
        sources = json.loads(
            base64.b64decode(headers["x-kb-sources"]).decode("utf-8")
        )
        self.assertEqual(sources[0]["source"], "manual.md")
        self.assertTrue(FakeUpstreamHandler.captured)
        sent = FakeUpstreamHandler.captured[-1]
        self.assertIn("知识库参考资料", sent["messages"][-1]["content"])
        self.assertIn("[1]", sent["messages"][-1]["content"])
        self.assertEqual(sent["max_tokens"], 768)

    def test_no_evidence_answered_freely_with_note(self) -> None:
        FakeUpstreamHandler.captured.clear()
        result = RetrievalResult(
            query="无关", skipped=True, reason="low_relevance", top_dense=0.1
        )
        port = self.start_proxy(result)
        status, headers, data = self.post(
            port,
            {"messages": [{"role": "user", "content": "法国的首都是哪里？"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["x-kb-status"], "no_evidence")
        self.assertEqual(data["choices"][0]["message"]["content"], "answer")
        self.assertTrue(FakeUpstreamHandler.captured)
        sent = FakeUpstreamHandler.captured[-1]
        self.assertIn("无明显知识依据", sent["messages"][0]["content"])
        self.assertNotIn("知识库参考资料", sent["messages"][-1]["content"])

    def test_stream_has_sources_header(self) -> None:
        result = RetrievalResult(
            query="管理端口",
            chunks=[self.chunk()],
            reason="ok",
            top_dense=0.7,
            lexical_hits=2,
        )
        port = self.start_proxy(result)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            body = json.dumps(
                {
                    "stream": True,
                    "messages": [{"role": "user", "content": "管理端口是多少？"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("X-KB-Status"), "ok")
            self.assertTrue(response.getheader("X-KB-Sources"))
            data = response.read().decode("utf-8")
            self.assertIn("data: [DONE]", data)
            self.assertIn("answer", data)
        finally:
            conn.close()

    def test_smalltalk_forwards_without_context(self) -> None:
        FakeUpstreamHandler.captured.clear()
        result = RetrievalResult(
            query="你好", skipped=True, smalltalk=True, reason="smalltalk"
        )
        port = self.start_proxy(result)
        status, headers, data = self.post(
            port,
            {"messages": [{"role": "user", "content": "你好"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["x-kb-status"], "smalltalk")
        self.assertNotIn("知识库参考资料", FakeUpstreamHandler.captured[-1]["messages"][-1]["content"])
        self.assertEqual(data["choices"][0]["message"]["content"], "answer")


if __name__ == "__main__":
    unittest.main()
