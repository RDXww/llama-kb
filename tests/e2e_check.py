#!/usr/bin/env python3
"""End-to-end smoke test against a real model and the demo knowledge base.

It launches ``scripts/start.sh --no-chat`` with a temporary data directory,
builds a small index from ``examples/docs``, and exercises:

1. ``/kb/health`` and ``/kb/search``
2. grounded non-streaming answer with citations
3. streaming answer
4. cross-lingual retrieval (Chinese question, English document)
5. the terminal chat client
6. the ``no_evidence`` path for an unrelated question

Requirements:
- ``KB_MODEL`` points to a .gguf file readable by ``llama-server``, or a
  backend is already running on port 8180.
- Ports 8180 and 8181 are free.
- The embedding model is cached (``python -m llama_kb.cli fetch-model``).

The index is built in a temporary directory and removed afterwards. This test
takes several minutes on a laptop GPU.

Usage:
    KB_MODEL=/path/to/model.gguf python -m tests.e2e_check
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start.sh"
LLAMA_PORT = 8180
PROXY_PORT = 8181
PROXY = ("127.0.0.1", PROXY_PORT)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_health(timeout: int = 900) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PROXY_PORT}/kb/health", timeout=3
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
                print(f"health: {data}", flush=True)
                return data
        except Exception as exc:  # expected while the model loads
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"timed out waiting for /kb/health: {last_error}")


def post_json(path: str, payload: dict[str, Any], timeout: int = 600):
    conn = http.client.HTTPConnection(*PROXY, timeout=timeout)
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        conn.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        data = response.read()
        headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, headers, data
    finally:
        conn.close()


def decode_sources(headers: dict[str, str]) -> list[dict[str, Any]]:
    encoded = headers.get("x-kb-sources")
    if not encoded:
        return []
    try:
        return json.loads(base64.b64decode(encoded).decode("utf-8"))
    except Exception:
        return []


def chat_nonstream(question: str, *, doc: str | None = None, max_tokens: int = 768):
    started = time.time()
    payload: dict[str, Any] = {
        "model": "llama-kb",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if doc:
        payload["doc"] = doc
    status, headers, data = post_json("/v1/chat/completions", payload)
    body = json.loads(data.decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return status, headers, content, time.time() - started


def chat_stream(question: str, *, doc: str | None = None, max_tokens: int = 768):
    started = time.time()
    conn = http.client.HTTPConnection(*PROXY, timeout=900)
    try:
        payload: dict[str, Any] = {
            "model": "llama-kb",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "stream": True,
        }
        if doc:
            payload["doc"] = doc
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        response = conn.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        content_parts: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
        return response.status, headers, "".join(content_parts), time.time() - started
    finally:
        conn.close()


def run_chat_client(question: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "llama_kb.chat",
            "--base-url",
            f"http://127.0.0.1:{PROXY_PORT}",
        ],
        input=f"{question}\n/exit\n",
        text=True,
        capture_output=True,
        timeout=600,
        cwd=str(ROOT),
        env=env,
    )


def has_source(headers: dict[str, str], filename: str) -> bool:
    return any(
        str(item.get("source", "")).endswith(filename)
        for item in decode_sources(headers)
    )


def tail(path: Path, limit: int = 5000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-limit:]
    except OSError:
        return "<unreadable>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="llama-kb end-to-end smoke test")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary data directory after the test",
    )
    args = parser.parse_args(argv)

    model = os.environ.get("KB_MODEL", "")
    backend_ready = port_open(LLAMA_PORT)
    if not model and not backend_ready:
        print(
            "Set KB_MODEL=/path/to/model.gguf, or start a backend on "
            f"127.0.0.1:{LLAMA_PORT}.",
            file=sys.stderr,
        )
        return 2
    if port_open(PROXY_PORT):
        print(f"Port {PROXY_PORT} is already in use; refusing to run.", file=sys.stderr)
        return 2

    data_dir = Path(tempfile.mkdtemp(prefix="llamakb-e2e-"))
    shutil.copytree(ROOT / "examples" / "docs", data_dir / "docs")
    shutil.copy(ROOT / "examples" / "catalog.json", data_dir / "catalog.json")
    log_dir = data_dir / "logs"
    log_dir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "KB_MODEL": model,
            "KB_PYTHON": sys.executable,
            "KB_DATA_DIR": str(data_dir),
            "KB_DOCS_DIR": str(data_dir / "docs"),
            "KB_CATALOG": str(data_dir / "catalog.json"),
            "KB_DB_DIR": str(data_dir / "index"),
            "KB_LOG_DIR": str(log_dir),
            "KB_LLAMA_PORT": str(LLAMA_PORT),
            "KB_PROXY_PORT": str(PROXY_PORT),
            "KB_LLAMA_URL": f"http://127.0.0.1:{LLAMA_PORT}",
            "KB_NO_SEED": "1",
            "PYTHONPATH": str(ROOT),
        }
    )
    launcher_log = (log_dir / "e2e-launcher.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(LAUNCHER), "--no-chat"],
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    try:
        health = wait_health()
        assert health.get("status") == "ok", health
        assert health.get("embed_dimension") == 1024, health
        assert health.get("chunks", 0) >= 4, health

        status, _, data = post_json(
            "/kb/search",
            {"query": "How long are audit logs retained?", "k": 3, "doc": "ops"},
        )
        result = json.loads(data.decode("utf-8"))
        print(f"search: {json.dumps(result, ensure_ascii=False)[:600]}", flush=True)
        assert status == 200, status
        assert result["results"], result
        assert result["results"][0]["source"].endswith("gateway-runbook.md"), result

        status, headers, content, elapsed = chat_nonstream(
            "According to the product manual, what should I check if the model "
            "cannot be connected?",
            doc="product",
        )
        print(
            f"\n[non-stream] status={status} kb={headers.get('x-kb-status')} "
            f"elapsed={elapsed:.1f}s\n{content}",
            flush=True,
        )
        assert status == 200, status
        assert headers.get("x-kb-status") == "ok", headers
        assert "8080" in content, content
        assert has_source(headers, "product-manual.md"), headers

        status, headers, content, elapsed = chat_stream(
            "What happens when port 8080 is already in use?", doc="product"
        )
        print(
            f"\n[stream] status={status} kb={headers.get('x-kb-status')} "
            f"elapsed={elapsed:.1f}s\n{content}",
            flush=True,
        )
        assert status == 200, status
        assert headers.get("x-kb-status") == "ok", headers
        assert has_source(headers, "product-manual.md"), headers

        status, headers, content, elapsed = chat_nonstream(
            "星火网关的管理端口是多少？", doc="ops"
        )
        print(
            f"\n[cross-lingual] status={status} kb={headers.get('x-kb-status')} "
            f"elapsed={elapsed:.1f}s\n{content}",
            flush=True,
        )
        assert status == 200, status
        assert headers.get("x-kb-status") == "ok", headers
        assert "9100" in content, content
        assert re.search(r"[一-鿿]", content), content

        client = run_chat_client("How often does AuroraDB back up?")
        print(
            f"\n[terminal client] rc={client.returncode}\n{client.stdout[-1500:]}",
            flush=True,
        )
        if client.stderr:
            print(f"client stderr: {client.stderr[-800:]}", flush=True)
        assert client.returncode == 0, client.returncode
        assert "Sources" in client.stdout, client.stdout

        started = time.time()
        status, headers, data = post_json(
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "stream": False,
            },
        )
        body = json.loads(data.decode("utf-8"))
        elapsed = time.time() - started
        answer = body["choices"][0]["message"]["content"]
        print(
            f"\n[no evidence] status={status} kb={headers.get('x-kb-status')} "
            f"elapsed={elapsed:.2f}s\n{answer}",
            flush=True,
        )
        assert headers.get("x-kb-status") == "no_evidence", headers
        assert "No clear knowledge basis" in answer, answer

        print("\nE2E PASS")
        return 0
    except Exception:
        traceback.print_exc()
        print("\n--- launcher log ---", flush=True)
        print(tail(data_dir / "logs" / "e2e-launcher.log"), flush=True)
        print("\n--- proxy log ---", flush=True)
        print(tail(data_dir / "logs" / "kb-server.log"), flush=True)
        return 1
    finally:
        print("stopping services ...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        launcher_log.close()
        time.sleep(1)
        if args.keep:
            print(f"kept: {data_dir}")
        else:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
