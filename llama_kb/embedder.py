"""Multilingual embedding model wrapper.

Uses Qwen3-Embedding-0.6B (1024 dimensions) by default. It runs on CPU so it
does not compete with the generation model for VRAM. The model is downloaded
once into the local cache and reused offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from .config import Config

# 模型自带 config_sentence_transformers.json 中的 query prompt。
# 如果 SentenceTransformers 版本不支持 prompt_name，则手动拼接。
FALLBACK_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)

PROMPT_VERSION = "query-v1"


class Embedder:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: Any = None
        self.dimension: int | None = None
        self.loaded_path: str | None = None
        self._torch_configured = False

    # ------------------------------------------------------------------
    # 模型文件
    # ------------------------------------------------------------------
    @property
    def repo_cache_name(self) -> str:
        return "models--" + self.config.embed_model_name.replace("/", "--")

    def _snapshots_dir(self) -> Path:
        return self.config.embed_cache_dir / self.repo_cache_name / "snapshots"

    def is_available(self) -> bool:
        return self._find_cached_snapshot() is not None

    @property
    def snapshot_path(self) -> Path | None:
        return self._find_cached_snapshot()

    @property
    def snapshot_name(self) -> str:
        snapshot = self._find_cached_snapshot()
        return snapshot.name if snapshot is not None else ""

    def _find_cached_snapshot(self) -> Path | None:
        snapshots = self._snapshots_dir()
        if not snapshots.is_dir():
            return None
        candidates = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in candidates:
            has_config = (path / "config.json").is_file()
            has_weights = any(path.glob("*.safetensors")) or (
                path / "pytorch_model.bin"
            ).is_file()
            if has_config and has_weights:
                return path
        return None

    def fetch(self, force: bool = False) -> Path:
        """Download the embedding model into the local cache."""
        cached = self._find_cached_snapshot()
        if cached is not None and not force:
            return cached

        if os.environ.get("LLAMA_KB_USE_HF_MIRROR") == "1":
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

        from huggingface_hub import snapshot_download

        self.config.embed_cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.config.embed_model_name,
            cache_dir=str(self.config.embed_cache_dir),
            max_workers=4,
        )
        cached = self._find_cached_snapshot()
        if cached is None:
            raise RuntimeError(
                f"模型已下载但未找到有效快照：{self.config.embed_model_name}。"
                "请检查磁盘空间或网络。"
            )
        return cached

    # ------------------------------------------------------------------
    # 加载与编码
    # ------------------------------------------------------------------
    def _configure_torch(self) -> None:
        if self._torch_configured:
            return
        try:
            import torch

            threads = max(1, int(self.config.embed_threads))
            torch.set_num_threads(threads)
        except Exception:
            pass
        self._torch_configured = True

    def load(self, allow_download: bool = False) -> "Embedder":
        if self.model is not None:
            return self

        snapshot = self._find_cached_snapshot()
        if snapshot is None:
            if not allow_download:
                raise FileNotFoundError(
                    f"Embedding model not found: {self.config.embed_model_name}.\n"
                    "Download it first:\n"
                    "  python -m llama_kb.cli fetch-model\n"
                    "or rerun the launcher (it downloads on first run)."
                )
            snapshot = self.fetch()

        self._configure_torch()
        import torch
        from sentence_transformers import SentenceTransformer

        # CPU 上 fp32 明显更快；GPU 构建时可用 fp16 提速（KB_EMBED_DTYPE）。
        dtype_name = (self.config.embed_dtype or "float32").strip().lower()
        use_fp16 = dtype_name in {"float16", "fp16", "half"} and str(
            self.config.embed_device
        ).startswith("cuda")
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        try:
            self.model = SentenceTransformer(
                str(snapshot),
                device=self.config.embed_device,
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"dtype": torch_dtype},
            )
        except TypeError:
            self.model = SentenceTransformer(
                str(snapshot),
                device=self.config.embed_device,
                local_files_only=True,
                trust_remote_code=False,
            )
        self.model.max_seq_length = min(
            int(self.config.embed_max_seq_length),
            int(getattr(self.model, "max_seq_length", 512) or 512),
        )
        self.model.eval()
        self.loaded_path = str(snapshot)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = int(self.model.get_embedding_dimension())
        else:  # 兼容旧版 sentence-transformers
            self.dimension = int(self.model.get_sentence_embedding_dimension())
        return self

    @property
    def config_fingerprint(self) -> str:
        """不加载模型即可计算的配置指纹，用于跳过无变化的构建。"""
        return (
            f"{self.config.embed_model_name}|"
            f"seq={self.config.embed_max_seq_length}|prompt={PROMPT_VERSION}"
        )

    @property
    def fingerprint(self) -> str:
        """完整指纹；加载模型后包含实际维度。"""
        dim = self.dimension if self.dimension is not None else "unknown"
        return (
            f"{self.config.embed_model_name}|dim={dim}|"
            f"seq={self.config.embed_max_seq_length}|prompt={PROMPT_VERSION}"
        )

    def encode_documents_numpy(
        self,
        texts: Iterable[str],
        show_progress: bool = False,
        batch_size: int | None = None,
    ):
        """返回 np.ndarray，供大规模构建直接分批写入，避免 Python list 开销。"""
        import numpy as np

        self.load(allow_download=False)
        items = [text if text else " " for text in texts]
        if not items:
            return np.zeros((0, self.dimension or 0), dtype="float32")
        vectors = self.model.encode(
            items,
            batch_size=max(
                1,
                int(
                    batch_size
                    if batch_size is not None
                    else self.config.build_batch_size
                ),
            ),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            device=self.config.embed_device,
        )
        return vectors.astype("float32")

    def encode_documents(
        self, texts: Iterable[str], show_progress: bool = False
    ) -> list[list[float]]:
        return self.encode_documents_numpy(
            texts, show_progress=show_progress, batch_size=self.config.embed_batch_size
        ).tolist()

    def encode_query(self, text: str) -> list[float]:
        self.load(allow_download=False)
        query = (text or "").strip() or " "
        try:
            vector = self.model.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.config.embed_device,
            )
        except (KeyError, ValueError):
            vector = self.model.encode(
                [FALLBACK_QUERY_PROMPT + query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.config.embed_device,
            )
        return vector.astype("float32")[0].tolist()

    def info(self) -> dict[str, Any]:
        return {
            "model": self.config.embed_model_name,
            "device": self.config.embed_device,
            "dtype": self.config.embed_dtype,
            "dimension": self.dimension,
            "max_seq_length": self.config.embed_max_seq_length,
            "path": self.loaded_path,
            "fingerprint": self.fingerprint,
        }
