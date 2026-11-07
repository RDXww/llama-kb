#!/usr/bin/env python3
"""Cross-lingual retrieval self-test with the real embedding model.

Run:
    python -m tests.selftest

Uses temporary directories and a temporary collection; the real index is
never modified.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from llama_kb.config import Config
from llama_kb.embedder import Embedder
from llama_kb.retriever import Retriever
from llama_kb.store import KBStore

ZH_DOC = """# 星火网关运维手册

## 管理端口

星火网关的管理服务监听 9100 端口，仅允许内网访问。
如果管理页面打不开，请先检查防火墙是否放行 9100。

## 备份策略

星火网关每天凌晨 2 点自动备份配置，备份文件保留 14 天。

## 审计日志

所有管理操作都会写入审计日志，审计日志默认保留 30 天。
"""

EN_DOC = """# AuroraDB Operations Guide

## Metadata storage

AuroraDB stores all metadata in a single SQLite file named catalog.db.
Back up this file together with the data directory.

## Backup schedule

The backup interval is every six hours, and snapshots are retained for 30 days.

## Administration endpoint

The administration endpoint listens on port 9200 by default and should only be
reachable from the internal network.
"""

CASES = [
    ("星火网关的管理端口是多少？", "gateway.md", "中文问题 -> 中文文档"),
    ("How often does AuroraDB back up?", "aurora.md", "English query -> English doc"),
    ("How long are audit logs retained?", "gateway.md", "English query -> 中文文档（跨语言）"),
    ("元数据存放在哪个文件里？", "aurora.md", "中文问题 -> English doc（跨语言）"),
    ("catalog.db", "aurora.md", "精确术语命中"),
]


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="kb-selftest-"))
    docs = root / "docs"
    docs.mkdir()
    (docs / "gateway.md").write_text(ZH_DOC, encoding="utf-8")
    (docs / "aurora.md").write_text(EN_DOC, encoding="utf-8")

    config = Config.from_env(
        docs_dir=docs,
        db_dir=root / "db",
        log_dir=root / "logs",
        collection_name="llamakb_selftest",
        catalog_path=root / "catalog.json",
        top_n=3,
        max_per_source=2,
        min_dense_score=0.26,
    )
    failures: list[str] = []
    try:
        print("加载嵌入模型 ...", flush=True)
        embedder = Embedder(config)
        started = time.time()
        embedder.load(allow_download=False)
        print(
            f"模型加载完成：{config.embed_model_name} "
            f"dim={embedder.dimension}，用时 {time.time() - started:.1f}s",
            flush=True,
        )
        if embedder.dimension != 1024:
            raise AssertionError(f"维度异常：{embedder.dimension}")

        store = KBStore(config)
        print("\n构建临时索引 ...", flush=True)
        started = time.time()
        report = store.sync(embedder, rebuild=True, allow_download=False, show_progress=True)
        print(f"{report.summary()}（{time.time() - started:.1f}s）")
        retriever = Retriever(config, store, embedder)

        for query, expected, label in CASES:
            result = retriever.search(query, limit=3)
            sources = [chunk.source for chunk in result.chunks]
            print(f"\n[{label}] {query}")
            print(
                f"  reason={result.reason} top_dense={result.top_dense:.4f} "
                f"lexical={result.lexical_hits} sources={sources}"
            )
            for chunk in result.chunks:
                print(
                    f"    - {chunk.source} dense={chunk.dense_score:.4f} "
                    f"bm25={chunk.bm25_score:.3f} {chunk.metadata.get('heading', '')}"
                )
            if expected not in sources[:2]:
                failures.append(
                    f"{label}: 期望 {expected} 出现在前二，实际 {sources}"
                )

        unrelated = retriever.search("法国的首都是哪里？", limit=3)
        print("\n[无关问题] 法国的首都是哪里？")
        print(
            f"  reason={unrelated.reason} top_dense={unrelated.top_dense:.4f} "
            f"sources={[chunk.source for chunk in unrelated.chunks]}"
        )
        if unrelated.chunks:
            failures.append(
                "无关问题仍命中资料；请检查 min_dense_score 或提示词拒答逻辑"
            )

        # 增量构建检查：文档未变化时应跳过
        report2 = store.sync(embedder, if_needed=True, allow_download=False, show_progress=False)
        print(f"\n增量检查：skipped={report2.skipped}")
        if not report2.skipped:
            failures.append("文档未变化但未跳过构建")
    except Exception:
        traceback.print_exc()
        failures.append("自测执行异常")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print("自测失败：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("全部通过：中文、英文、跨语言检索与无关问题拒答均正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
