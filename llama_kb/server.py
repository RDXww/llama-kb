"""OpenAI-compatible RAG proxy.

client -> 127.0.0.1:8081 (this proxy)
       -> hybrid retrieval over the local index
       -> 127.0.0.1:8080 (llama.cpp or any OpenAI-compatible backend)

Listens on the loopback interface by default, never forwards arbitrary URLs,
and does not enable CORS.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import Config
from .embedder import Embedder
from .prompt import (
    compose_messages,
    compose_no_evidence_messages,
    compose_smalltalk_messages,
    estimate_prompt_tokens,
    last_user_message,
    no_result_answer,
    sources_payload,
    unsupported_answer,
)
from .retriever import RetrievalResult, Retriever
from .store import KBStore
from .text import query_language

log = logging.getLogger("llama_kb.server")


class KBService:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.started_at = time.time()
        self.store = KBStore(config)
        print(f"Loading embedding model ({config.embed_model_name}, device={config.embed_device}) ...", flush=True)
        self.embedder = Embedder(config).load(allow_download=False)
        self.store.validate(self.embedder)
        print("Loading FTS5 keyword index and group router ...", flush=True)
        self.catalog = self.store.catalog
        self.retriever = Retriever(config, self.store, self.embedder)
        self.retriever.router.precompute()
        # 索引在服务运行期间只读，来源数量缓存起来，避免每次健康检查都全量取数据。
        self._source_count = self.store.source_count()

    def health(self) -> dict[str, Any]:
        chunks = self.store.count()
        return {
            "status": "ok",
            "collection": self.config.collection_name,
            "chunks": chunks,
            "fts_chunks": self.store.fts_count(),
            "sources": self._source_count,
            "groups": len(self.catalog.groups),
            "preferred_groups": len(self.catalog.preferred_groups),
            "embed_model": self.config.embed_model_name,
            "embed_dimension": self.embedder.dimension,
            "embed_device": self.config.embed_device,
            "llama_url": self.config.llama_url,
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }

    def stats(self) -> dict[str, Any]:
        data = self.store.stats()
        data["retriever"] = self.retriever.stats()
        data["groups_catalog"] = {
            "total": len(self.catalog.groups),
            "preferred": len(self.catalog.preferred_groups),
            "default": self.catalog.default_group.id,
        }
        data["config"] = {
            "top_n": self.config.top_n,
            "dense_k": self.config.dense_k,
            "bm25_k": self.config.bm25_k,
            "min_dense_score": self.config.min_dense_score,
            "max_context_chars": self.config.max_context_chars,
            "max_history_messages": self.config.max_history_messages,
            "router_enabled": self.config.router_enabled,
            "router_max_groups": self.config.router_max_groups,
        }
        return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kb-rag/1.0"
    sys_version = ""

    # ------------------------------------------------------------------
    # HTTP 基础
    # ------------------------------------------------------------------
    @property
    def service(self) -> KBService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.service.config.debug:
            sys.stderr.write(
                f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}\n"
            )

    def _read_json(self) -> dict[str, Any] | None:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_error(411, "missing Content-Length")
            return None
        try:
            length = int(length_header)
        except ValueError:
            self._send_error(400, "invalid Content-Length")
            return None
        limit = self.service.config.request_body_limit
        if length > limit:
            self._send_error(413, f"request body too large (>{limit} bytes)")
            return None
        if length <= 0:
            self._send_error(400, "empty request body")
            return None
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_error(400, "invalid JSON body")
            return None
        if not isinstance(payload, dict):
            self._send_error(400, "request body must be a JSON object")
            return None
        return payload

    def _send_json(
        self,
        status: int,
        payload: Any,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error(self, status: int, message: str) -> None:
        self.close_connection = True
        self._send_json(
            status,
            {"error": {"message": message, "type": "invalid_request_error"}},
        )

    def _send_sse_headers(self, headers: dict[str, str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/kb/health":
            self._send_json(200, self.service.health())
            return
        if path == "/kb/stats":
            self._send_json(200, self.service.stats())
            return
        if path == "/kb/groups":
            self._send_json(
                200,
                {
                    "default": self.service.catalog.default_group.id,
                    "groups": self.service.catalog.groups_payload(),
                },
            )
            return
        if path in {"/", "/kb"}:
            self._send_json(
                200,
                {
                    "name": "local-general-kb",
                    "endpoints": [
                        "GET /kb/health",
                        "GET /kb/stats",
                        "GET /kb/groups",
                        "POST /kb/search",
                        "POST /v1/chat/completions",
                        "GET /v1/models",
                    ],
                },
            )
            return
        if path == "/v1/models":
            self._proxy_get(path)
            return
        self._send_error(404, "unknown path")

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------
    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/kb/search":
            self._handle_search()
            return
        if path == "/v1/chat/completions":
            self._handle_chat()
            return
        self._send_error(404, "unknown path")

    def _handle_search(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            self._send_error(400, "missing non-empty query field")
            return
        try:
            limit = int(payload.get("k", self.service.config.top_n))
        except (TypeError, ValueError):
            limit = self.service.config.top_n
        limit = max(1, min(limit, 20))
        doc_hint = payload.get("doc") or payload.get("groups")
        if doc_hint is None:
            doc_hint = self.headers.get("X-KB-Doc")
        forced = None
        if doc_hint:
            if not isinstance(doc_hint, str):
                doc_hint = ",".join(str(item) for item in doc_hint)
            forced, unknown = self.service.catalog.resolve(doc_hint)
            if unknown:
                self._send_error(
                    400,
                    "Unknown group(s): " + ", ".join(unknown) + " (see GET /kb/groups)",
                )
                return
        try:
            if forced:
                result = self.service.retriever.search(
                    query, limit=limit, forced_groups=forced
                )
            else:
                result = self.service.retriever.search(query, limit=limit)
        except Exception as exc:
            log.exception("retrieval failed")
            self._send_error(500, f"retrieval failed: {exc}")
            return
        results = []
        for chunk in result.chunks:
            metadata = chunk.metadata
            results.append(
                {
                    "n": chunk.citation,
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "heading": metadata.get("heading", ""),
                    "page": metadata.get("page", 0),
                    "language": metadata.get("language", ""),
                    "group_id": metadata.get("group_id", ""),
                    "group": metadata.get("group_label") or metadata.get("group_name", ""),
                    "score": round(chunk.score, 6),
                    "dense": round(chunk.dense_score, 4),
                    "bm25": round(chunk.bm25_score, 4),
                    "snippet": " ".join(chunk.document.split())[:400],
                }
            )
        self._send_json(
            200,
            {
                "query": result.query,
                "reason": result.reason,
                "top_dense": round(result.top_dense, 4),
                "lexical_hits": result.lexical_hits,
                "groups": result.groups,
                "unsupported": result.unsupported,
                "routing": result.routing.summary() if result.routing else None,
                "results": results,
            },
        )

    def _handle_chat(self) -> None:
        payload = self._read_json()
        if payload is None:
            return
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_error(400, "missing non-empty messages array")
            return
        if not all(isinstance(item, dict) for item in messages):
            self._send_error(400, "every item in messages must be an object")
            return

        stream = bool(payload.get("stream", False))
        kb_mode = (self.headers.get("X-KB-Mode") or "").strip().lower()
        if kb_mode == "off":
            self._forward(payload, stream, status="off", sources=[])
            return

        doc_hint = self.headers.get("X-KB-Doc") or payload.get("doc")
        forced = None
        if doc_hint:
            if not isinstance(doc_hint, str):
                doc_hint = ",".join(str(item) for item in doc_hint)
            forced, unknown = self.service.catalog.resolve(doc_hint)
            if unknown:
                self._send_error(
                    400,
                    "Unknown group(s): " + ", ".join(unknown) + " (see GET /kb/groups)",
                )
                return
            if not forced:
                forced = None

        result: RetrievalResult | None = None
        retrieval_error = ""
        try:
            if forced:
                result = self.service.retriever.retrieve(
                    messages, forced_groups=forced
                )
            else:
                result = self.service.retriever.retrieve(messages)
        except Exception as exc:
            retrieval_error = str(exc)
            log.exception("retrieval failed; degrading to plain chat")

        question = last_user_message(messages)
        language = query_language(question)

        if result is not None and result.smalltalk:
            composed = compose_smalltalk_messages(messages, self.service.config)
            self._forward(
                payload,
                stream,
                status="smalltalk",
                sources=[],
                messages_override=composed,
            )
            return

        if result is not None and result.reason == "unsupported_doc":
            content = unsupported_answer(
                language,
                result.unsupported,
                [
                    group.label
                    for group in self.service.catalog.preferred_groups[:16]
                ],
            )
            self._send_no_result(
                payload,
                stream,
                language,
                content=content,
                status="unsupported",
            )
            return

        if result is not None and not result.has_context and result.reason not in (
            "no_user",
            "empty_query",
        ):
            # 知识库没有明显依据：不注入弱相关资料，允许模型用通用知识回答，
            # 并通过 no_evidence 状态要求客户端注明“无明显知识依据”。
            composed = compose_no_evidence_messages(messages, self.service.config)
            composed = self._fit_prompt_budget(composed)
            self._forward(
                payload,
                stream,
                status="no_evidence",
                sources=[],
                messages_override=composed,
            )
            return

        if result is None:
            # 检索异常时仍然保证聊天可用，明确标记 degraded。
            self._forward(
                payload,
                stream,
                status="degraded",
                sources=[],
                extra_headers={"X-KB-Error": base64.b64encode(
                    retrieval_error.encode("utf-8")
                ).decode("ascii")[:800]},
            )
            return

        if not result.has_context:
            self._forward(payload, stream, status="skipped", sources=[])
            return

        composed, used = compose_messages(
            messages, result, self.service.config
        )
        composed = self._fit_prompt_budget(composed)
        sources = sources_payload(
            used,
            max_bytes=max(512, int(self.service.config.max_sources_header_bytes * 0.65)),
        )
        if self.service.config.debug:
            log.info(
                "retrieval: %d references, top_dense=%.3f, lexical_hits=%d",
                len(used),
                result.top_dense,
                result.lexical_hits,
            )
        self._forward(
            payload,
            stream,
            status="ok",
            sources=sources,
            messages_override=composed,
        )

    # ------------------------------------------------------------------
    # 上游转发
    # ------------------------------------------------------------------
    def _normalized_body(
        self,
        payload: dict[str, Any],
        messages_override: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        body = {
            key: value
            for key, value in payload.items()
            if not str(key).lower().startswith("x_kb")
        }
        if messages_override is not None:
            body["messages"] = messages_override
        body["model"] = body.get("model") or "ds-kb"
        if "temperature" not in body:
            body["temperature"] = self.service.config.default_temperature

        value = body.pop("max_completion_tokens", None)
        if value is None:
            value = body.get("max_tokens")
        if value is not None:
            try:
                body["max_tokens"] = max(
                    1, min(int(value), self.service.config.hard_max_tokens)
                )
            except (TypeError, ValueError):
                body["max_tokens"] = self.service.config.default_max_tokens
        else:
            body["max_tokens"] = self.service.config.default_max_tokens
        return body

    def _forward(
        self,
        payload: dict[str, Any],
        stream: bool,
        status: str,
        sources: list[dict[str, Any]],
        messages_override: list[dict[str, Any]] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = self._normalized_body(payload, messages_override)
        body["stream"] = stream
        url = self.service.config.llama_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response_headers = self._kb_headers(status, sources)
        response_headers.update(extra_headers or {})

        try:
            upstream = urllib.request.urlopen(
                request, timeout=self.service.config.upstream_timeout
            )
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            self._relay_upstream_error(exc.code, error_body, response_headers)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._send_json(
                502,
                {
                    "error": {
                        "message": f"cannot reach the upstream model server: {exc}",
                        "type": "upstream_error",
                    }
                },
                extra_headers=response_headers,
            )
            return

        if not stream:
            upstream_status = upstream.status
            try:
                data = upstream.read()
            finally:
                upstream.close()
            self.send_response(upstream_status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # 流式：先拿到上游响应头，再开始向客户端写 SSE。
        try:
            self._send_sse_headers(response_headers)
            while True:
                line = upstream.readline()
                if not line:
                    break
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            upstream.close()

    def _relay_upstream_error(
        self,
        code: int,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        # 尽量保留上游的错误 JSON；否则包装成标准错误。
        payload: Any
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {
                "error": {
                    "message": body.decode("utf-8", "ignore")[:1000]
                    or "upstream model server returned an error",
                    "type": "upstream_error",
                }
            }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _proxy_get(self, path: str) -> None:
        url = self.service.config.llama_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=10) as upstream:
                data = upstream.read()
                self.send_response(upstream.status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self._relay_upstream_error(exc.code, data, {})
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._send_json(
                502,
                {"error": {"message": f"cannot reach the upstream model server: {exc}", "type": "upstream_error"}},
            )

    # ------------------------------------------------------------------
    # 头部 / 拒答 / 预算
    # ------------------------------------------------------------------
    def _kb_headers(self, status: str, sources: list[dict[str, Any]]) -> dict[str, str]:
        raw = json.dumps(sources, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        encoded = base64.b64encode(raw).decode("ascii")
        limit = self.service.config.max_sources_header_bytes
        if len(encoded) > limit:
            slim = [
                {"n": item.get("n"), "source": item.get("source", "")}
                for item in sources
            ]
            raw = json.dumps(slim, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            encoded = base64.b64encode(raw).decode("ascii")
        if len(encoded) > limit:
            # 绝不能截断 base64，否则客户端解码失败；宁可不带来源。
            encoded = base64.b64encode(b"[]").decode("ascii")
        return {
            "X-KB-Status": status,
            "X-KB-Sources-Encoding": "base64-json",
            "X-KB-Sources": encoded,
        }

    def _send_no_result(
        self,
        payload: dict[str, Any],
        stream: bool,
        language: str,
        *,
        content: str | None = None,
        status: str = "no_result",
    ) -> None:
        content = content or no_result_answer(language)
        headers = self._kb_headers(status, [])
        model = str(payload.get("model") or "ds-kb")
        created = int(time.time())
        completion_id = f"kb-{time.time_ns()}"

        if not stream:
            self._send_json(
                200,
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                extra_headers=headers,
            )
            return

        self._send_sse_headers(headers)
        base = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
        try:
            for delta, finish in (
                ({"role": "assistant"}, None),
                ({"content": content}, None),
                ({}, "stop"),
            ):
                chunk = {
                    **base,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}
                    ],
                }
                self.wfile.write(
                    ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode(
                        "utf-8"
                    )
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _fit_prompt_budget(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        budget = self.service.config.max_prompt_tokens
        items = list(messages)
        # 保留 system(0) 和最后一条当前问题；从旧到新删除历史。
        while len(items) > 2 and estimate_prompt_tokens(items) > budget:
            del items[1]
        return items


class KBServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llama-kb RAG proxy")
    parser.add_argument("--host", default=None, help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="listen port (default: 8081)")
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    config = Config.from_env()
    if args.host:
        config.proxy_host = args.host
    if args.port:
        config.proxy_port = args.port
    if args.debug:
        config.debug = True

    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config.proxy_host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"warning: binding to {config.proxy_host}; the knowledge base and "
            "chat traffic may be exposed to the network.",
            file=sys.stderr,
        )

    try:
        service = KBService(config)
    except Exception as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        return 1

    server = KBServer((config.proxy_host, config.proxy_port), Handler)
    server.service = service  # type: ignore[attr-defined]
    print(
        f"llama-kb proxy listening on http://{config.proxy_host}:{config.proxy_port}\n"
        f"  index: {service.store.count()} chunks, embedding: {config.embed_model_name}\n"
        f"  upstream: {config.llama_url}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping the RAG proxy ...", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
