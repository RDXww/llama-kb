"""Terminal chat client for llama-kb.

Talks to the local RAG proxy and supports streaming answers, an optional
reasoning trace, source listings, and /search, /health, /stats commands.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import sys
import urllib.parse
from typing import Any, Iterator

try:  # 有 readline 时支持方向键和行编辑
    import readline  # noqa: F401
except ImportError:  # pragma: no cover
    readline = None

DIM = "\033[90m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}" if _supports_color() else text


def _yellow(text: str) -> str:
    return f"{YELLOW}{text}{RESET}" if _supports_color() else text


def _green(text: str) -> str:
    return f"{GREEN}{text}{RESET}" if _supports_color() else text


class ChatClient:
    def __init__(self, base_url: str, show_reasoning: bool = False) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid base URL: {base_url}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.timeout = 900.0
        self.messages: list[dict[str, str]] = []
        self.last_sources: list[dict[str, Any]] = []
        self.show_reasoning = show_reasoning
        self.pinned_groups: list[str] = []
        self.groups_cache: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _connect(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout
            )
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        conn = self._connect()
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, self.base_path + path, body=body, headers=headers)
            response = conn.getresponse()
        except Exception:
            conn.close()
            raise
        return conn, response

    def health(self) -> dict[str, Any]:
        conn, response = self._request("GET", "/kb/health")
        try:
            data = response.read()
        finally:
            conn.close()
        if response.status != 200:
            raise RuntimeError(data.decode("utf-8", "ignore"))
        return json.loads(data.decode("utf-8"))

    def stats(self) -> dict[str, Any]:
        conn, response = self._request("GET", "/kb/stats")
        try:
            data = response.read()
        finally:
            conn.close()
        if response.status != 200:
            raise RuntimeError(data.decode("utf-8", "ignore"))
        return json.loads(data.decode("utf-8"))

    def search(self, query: str, k: int = 4) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "k": k}
        if self.pinned_groups:
            payload["doc"] = ",".join(self.pinned_groups)
        conn, response = self._request("POST", "/kb/search", payload)
        try:
            data = response.read()
        finally:
            conn.close()
        if response.status != 200:
            raise RuntimeError(data.decode("utf-8", "ignore"))
        return json.loads(data.decode("utf-8"))

    def groups(self) -> list[dict[str, Any]]:
        conn, response = self._request("GET", "/kb/groups")
        try:
            data = response.read()
        finally:
            conn.close()
        if response.status != 200:
            raise RuntimeError(data.decode("utf-8", "ignore"))
        payload = json.loads(data.decode("utf-8"))
        groups = payload.get("groups") if isinstance(payload, dict) else None
        self.groups_cache = groups if isinstance(groups, list) else []
        return self.groups_cache

    # ------------------------------------------------------------------
    # 聊天
    # ------------------------------------------------------------------
    def _decode_sources(self, response: http.client.HTTPResponse) -> list[dict[str, Any]]:
        encoded = response.getheader("X-KB-Sources") or ""
        if not encoded:
            return []
        try:
            raw = base64.b64decode(encoded.encode("ascii"))
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    @staticmethod
    def _iter_sse(response: http.client.HTTPResponse) -> Iterator[str]:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[5:].strip()

    def ask(self, user_text: str) -> str:
        payload = {
            "model": "ds-kb",
            "messages": [*self.messages, {"role": "user", "content": user_text}],
            "stream": True,
        }
        extra_headers = (
            {"X-KB-Doc": ",".join(self.pinned_groups)}
            if self.pinned_groups
            else None
        )
        conn, response = self._request(
            "POST",
            "/v1/chat/completions",
            payload,
            stream=True,
            extra_headers=extra_headers,
        )
        try:
            if response.status != 200:
                error_body = response.read().decode("utf-8", "ignore")
                raise RuntimeError(f"HTTP {response.status}: {error_body}")
            self.last_sources = self._decode_sources(response)
            kb_status = (response.getheader("X-KB-Status") or "").strip()
            if kb_status and kb_status not in {
                "ok",
                "no_result",
                "no_evidence",
                "smalltalk",
            }:
                print(_yellow(f"[{kb_status}]"), flush=True)

            answer_parts: list[str] = []
            reasoning_started = False
            answer_started = False
            got_done = False

            for data in self._iter_sse(response):
                if data == "[DONE]":
                    got_done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                content = delta.get("content")

                if reasoning:
                    if self.show_reasoning:
                        if not reasoning_started:
                            print(_dim("Thinking: "), end="", flush=True)
                            reasoning_started = True
                        print(_dim(reasoning), end="", flush=True)
                    elif not reasoning_started:
                        print(_dim("[model is thinking...]"), flush=True)
                        reasoning_started = True

                if content:
                    if not answer_started:
                        if reasoning_started and not self.show_reasoning:
                            pass
                        elif reasoning_started and self.show_reasoning:
                            print()
                        print(_green("Assistant: "), end="", flush=True)
                        answer_started = True
                    print(content, end="", flush=True)
                    answer_parts.append(content)

            if answer_started:
                print()
            elif reasoning_started:
                print(
                    _yellow(
                        "(The reasoning budget was exhausted before the final "
                        "answer. Increase KB_MAX_OUTPUT_TOKENS or decrease "
                        "KB_REASONING_BUDGET.)"
                    )
                )
            else:
                print(_yellow("(the model returned no text)"))
            if kb_status == "no_result":
                print(_dim("(no relevant knowledge-base documents; request refused)"))
            elif kb_status == "no_evidence":
                print(
                    _dim(
                        "(no clear knowledge basis; the answer above comes from "
                        "the model's general knowledge)"
                    )
                )
            elif kb_status == "unsupported":
                print(_dim("(group not indexed; the model was not called; see /docs)"))
            self._print_sources()
            if not got_done and answer_started:
                print(_dim("(stream ended early)"))

            answer = "".join(answer_parts).strip()
            if answer:
                self.messages.extend(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": answer},
                    ]
                )
                # 客户端也限制一下历史长度，服务端还会再裁剪。
                if len(self.messages) > 20:
                    self.messages = self.messages[-20:]
            return answer
        finally:
            conn.close()

    def _print_sources(self) -> None:
        if not self.last_sources:
            return
        print(_dim("Sources:"))
        for item in self.last_sources:
            number = item.get("n", "?")
            source = item.get("source", "")
            group = item.get("group") or ""
            heading = item.get("heading") or ""
            page = item.get("page") or 0
            dense = item.get("dense")
            extras = []
            if group:
                extras.append(str(group))
            if heading:
                extras.append(str(heading))
            if page:
                extras.append(f"p.{page}")
            if isinstance(dense, (int, float)):
                extras.append(f"score {dense:.3f}")
            suffix = f" · {' · '.join(extras)}" if extras else ""
            print(_dim(f"  [{number}] {source}{suffix}"))

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def print_help(self) -> None:
        print(
            "\nCommands:\n"
            "  /exit, /quit      quit\n"
            "  /clear            clear the conversation\n"
            "  /docs [keyword]   list document groups\n"
            "  /doc [group]      pin a group; /doc off restores auto routing\n"
            "  /sources          show sources of the last answer\n"
            "  /search <query>   retrieve only (respects /doc)\n"
            "  /health           show index status\n"
            "  /stats            show index statistics\n"
            "  /reasoning        toggle the reasoning trace\n"
            "  /help             show this help\n"
        )

    def print_sources_detail(self) -> None:
        if not self.last_sources:
            print("The previous answer had no sources.")
            return
        for item in self.last_sources:
            group = item.get("group") or ""
            print(
                f"[{item.get('n')}] {group + ' · ' if group else ''}"
                f"{item.get('source', '')} "
                f"{item.get('heading', '')} p.{item.get('page', 0)}"
            )
            snippet = item.get("snippet")
            if snippet:
                print(f"    {snippet}")
            print(
                f"    dense={item.get('dense')} bm25={item.get('bm25')} "
                f"score={item.get('score')}"
            )

    def show_groups(self, keyword: str = "") -> None:
        try:
            groups = self.groups()
        except Exception as exc:
            print(_yellow(f"Cannot fetch groups: {exc}"))
            return
        lowered = keyword.casefold()
        shown = 0
        for group in groups:
            haystack = " ".join(
                [
                    str(group.get("id", "")),
                    str(group.get("name", "")),
                    str(group.get("label", "")),
                    str(group.get("type", "")),
                    " ".join(str(item) for item in group.get("aliases") or []),
                ]
            ).casefold()
            if lowered and lowered not in haystack:
                continue
            preferred = "★" if group.get("preferred") else " "
            print(
                f"{preferred} {str(group.get('id', '')):<26} "
                f"{group.get('label', '')} [{group.get('type', '')}]"
            )
            shown += 1
        print(_dim(f"{shown} groups; ★ marks the default group."))
        if self.pinned_groups:
            print(_dim("Pinned groups: " + ", ".join(self.pinned_groups)))

    def set_doc(self, argument: str) -> None:
        if not argument or argument.lower() in {"show", "status"}:
            print(
                "Pinned groups: "
                + (", ".join(self.pinned_groups) if self.pinned_groups else "automatic")
            )
            return
        if argument.lower() in {"off", "auto", "none", "clear", "*"}:
            self.pinned_groups = []
            print("Automatic group routing restored.")
            return
        try:
            groups = self.groups()
        except Exception as exc:
            print(_yellow(f"Cannot fetch groups: {exc}"))
            return
        queries = (
            [item.strip() for item in argument.split(",") if item.strip()]
            if "," in argument
            else [argument.strip()]
        )
        selected: list[str] = []
        unknown: list[str] = []
        for query in queries:
            lowered = query.casefold()
            matches = [
                group
                for group in groups
                if lowered
                in {
                    str(group.get("id", "")).casefold(),
                    str(group.get("name", "")).casefold(),
                    str(group.get("label", "")).casefold(),
                    *(str(alias).casefold() for alias in group.get("aliases") or []),
                }
            ]
            if len(matches) == 1:
                selected.append(str(matches[0].get("id")))
            elif len(matches) > 1:
                exact = [
                    group
                    for group in matches
                    if str(group.get("id", "")).casefold() == lowered
                    or str(group.get("label", "")).casefold() == lowered
                ]
                if len(exact) == 1:
                    selected.append(str(exact[0].get("id")))
                else:
                    print(
                        _yellow(
                            f"Group {query!r} is ambiguous: "
                            + "、".join(str(item.get("label")) for item in matches)
                        )
                    )
            else:
                unknown.append(query)
        if unknown:
            print(_yellow("Unknown group(s): " + ", ".join(unknown) + " (see /docs)"))
        if selected:
            self.pinned_groups = selected
            print("Pinned: " + ", ".join(self.pinned_groups))

    def command(self, line: str) -> bool:
        stripped = line.strip()
        if stripped in {"/exit", "/quit", "exit", "quit"}:
            return False
        if stripped == "/help":
            self.print_help()
            return True
        if stripped == "/clear":
            self.messages.clear()
            self.last_sources = []
            print("Conversation history cleared.")
            return True
        if stripped == "/sources":
            self.print_sources_detail()
            return True
        if stripped == "/reasoning":
            self.show_reasoning = not self.show_reasoning
            print("Reasoning display: " + ("on" if self.show_reasoning else "off"))
            return True
        if stripped == "/health":
            try:
                print(json.dumps(self.health(), ensure_ascii=False, indent=2))
            except Exception as exc:
                print(_yellow(f"Cannot fetch status: {exc}"))
            return True
        if stripped == "/stats":
            try:
                print(json.dumps(self.stats(), ensure_ascii=False, indent=2))
            except Exception as exc:
                print(_yellow(f"Cannot fetch statistics: {exc}"))
            return True
        if stripped.startswith("/docs"):
            keyword = stripped[len("/docs") :].strip()
            self.show_groups(keyword)
            return True
        if stripped.startswith("/doc"):
            argument = stripped[len("/doc") :].strip()
            self.set_doc(argument)
            return True
        if stripped.startswith("/search"):
            query = stripped[len("/search") :].strip()
            if not query:
                print("Usage: /search <query>")
                return True
            try:
                result = self.search(query)
                for item in result.get("results", []):
                    print(
                        f"[{item.get('n')}] {item.get('group', '')} · "
                        f"{item.get('source')} "
                        f"{item.get('heading', '')} "
                        f"dense={item.get('dense')} bm25={item.get('bm25')}"
                    )
                    print(f"    {item.get('snippet', '')}")
                if not result.get("results"):
                    print("No relevant documents found.")
            except Exception as exc:
                print(_yellow(f"Search failed: {exc}"))
            return True
        if stripped.startswith("/"):
            print("Unknown command. Type /help for help.")
            return True

        try:
            self.ask(stripped)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as exc:
            print(_yellow(f"Request failed: {exc}"))
            print(_dim("Check that llama-server and the llama-kb proxy are still running."))
        return True

    def run(self) -> int:
        try:
            health = self.health()
        except Exception as exc:
            print(f"Cannot connect to the llama-kb proxy: {exc}", file=sys.stderr)
            print(
                "Start the stack with scripts/start.sh, or make sure "
                "llama_kb.server is listening on the configured proxy port.",
                file=sys.stderr,
            )
            return 1

        print("=" * 64)
        print(_green("llama-kb chat (Chinese / English)"))
        print(
            f"Index: {health.get('chunks', 0)} chunks, "
            f"{health.get('sources', 0)} sources, "
            f"{health.get('groups', 0)} groups · "
            f"embedding: {health.get('embed_model', '')}"
        )
        print("Ask a question to begin. /help commands, /docs groups, /exit quit.")
        print("=" * 64)

        while True:
            try:
                line = input(_green("You: "))
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            line = line.strip()
            if not line:
                continue
            if not self.command(line):
                break
        print("Bye.")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llama-kb terminal chat client")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8081",
        help="RAG proxy URL (default: http://127.0.0.1:8081)",
    )
    parser.add_argument(
        "--reasoning", action="store_true", help="show the reasoning trace by default"
    )
    args = parser.parse_args(argv)

    try:
        client = ChatClient(args.base_url, show_reasoning=args.reasoning)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return client.run()


if __name__ == "__main__":
    raise SystemExit(main())
