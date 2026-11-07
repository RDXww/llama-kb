"""Document loading, bilingual-aware chunking, language detection, tokenization."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import Config

log = logging.getLogger("kb.text")

SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf"}

_CJK_IDEOGRAPH_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_CJK_RUN_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]+"
)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/+][a-z0-9]+)*")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

GREETING_RE = re.compile(
    r"^\s*(你好|您好|嗨|哈喽|早上好|下午好|晚上好|谢谢|多谢|感谢|"
    r"hi|hello|hey|thanks|thank\s+you|thx|bye|goodbye|再见)"
    r"[\s!！。,.，?？~～:：]*$",
    re.IGNORECASE,
)
META_QUESTION_RE = re.compile(
    r"^\s*(你是谁|你是什么|你能做什么|你会什么|介绍一下你自己|"
    r"who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|"
    r"help|帮助|使用说明)[\s?？!！。.,，]*$",
    re.IGNORECASE,
)

_REFERENTIAL_TERMS = (
    "它",
    "这个",
    "那个",
    "上述",
    "该",
    "前者",
    "后者",
    "这些",
    "那些",
    "其",
    "此",
    "这种",
    "那里",
    "这里",
    "it",
    "this",
    "that",
    "those",
    "these",
    "they",
    "its",
    "above",
    "former",
    "latter",
    "there",
)


@dataclass(slots=True)
class SourceFile:
    """一个待索引的源文件。"""

    path: Path
    rel_path: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass(slots=True)
class Page:
    """文档的一页；非 PDF 文档 page=0。"""

    text: str
    page: int = 0


@dataclass(slots=True)
class LoadedDoc:
    path: Path
    rel_path: str
    title: str
    pages: list[Page]
    is_markdown: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    id: str
    document: str
    embed_text: str
    metadata: dict[str, Any]
    language: str

    @property
    def source(self) -> str:
        return str(self.metadata["source"])

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """统一换行、去掉 BOM 和不可见字符，压缩过多空行。"""
    if not text:
        return ""
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")
    text = text.replace("\x00", "")
    # 超过两个连续空行压成一个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def scan_sources(docs_dir: Path) -> list[SourceFile]:
    """递归扫描知识库目录，返回带哈希的源文件列表。"""
    docs_dir = docs_dir.expanduser()
    if not docs_dir.is_dir():
        return []
    sources: list[SourceFile] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            stat = path.stat()
            rel = str(path.relative_to(docs_dir))
            sources.append(
                SourceFile(
                    path=path,
                    rel_path=rel,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=file_sha256(path),
                )
            )
        except OSError as exc:
            log.warning("cannot read %s: %s", path, exc)
    return sources


def detect_language(text: str) -> str:
    """粗略识别中英文；混合文本返回 mixed。"""
    cjk = len(_CJK_IDEOGRAPH_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    if cjk == 0 and latin == 0:
        return "other"
    if cjk == 0:
        return "en"
    if latin == 0:
        return "zh"
    if cjk >= 10 and cjk >= latin * 0.15:
        return "zh"
    if latin >= cjk * 0.5:
        return "en"
    return "mixed"


def query_language(text: str) -> str:
    """面向提示词选择：只要出现足量中文就按中文问题处理。"""
    cjk = len(_CJK_IDEOGRAPH_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    if cjk > 0 and cjk >= latin * 0.2:
        return "zh"
    return "en"


def is_smalltalk(text: str) -> bool:
    text = (text or "").strip()
    return bool(GREETING_RE.match(text) or META_QUESTION_RE.match(text))


def has_referential_term(text: str) -> bool:
    lowered = (text or "").lower()
    for term in _REFERENTIAL_TERMS:
        if term.isascii():
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return True
        elif term in lowered:
            return True
    return False


def estimate_tokens(text: str) -> int:
    """保守估计 token 数，用于控制注入上下文和历史长度。"""
    if not text:
        return 0
    cjk = len(_CJK_IDEOGRAPH_RE.findall(text))
    latin_words = len(_LATIN_WORD_RE.findall(text))
    other = max(0, len(text) - cjk)
    return int(cjk * 1.0 + latin_words * 1.4 + other * 0.3) + 16


# ---------------------------------------------------------------------------
# 分词（BM25 用）
# ---------------------------------------------------------------------------

def _english_stem_variants(word: str) -> list[str]:
    """非常轻量的英文词形归一，覆盖常见复数/进行时。"""
    variants: list[str] = []
    if not word.isalpha() or len(word) < 4:
        return variants
    if word.endswith("ies") and len(word) > 4:
        variants.append(word[:-3] + "y")
    elif word.endswith("es") and len(word) > 3:
        variants.append(word[:-2])
    elif word.endswith("s") and not word.endswith("ss"):
        variants.append(word[:-1])
    if word.endswith("ing") and len(word) > 5:
        variants.append(word[:-3])
        variants.append(word[:-3] + "e")
    if word.endswith("ed") and len(word) > 4:
        variants.append(word[:-2])
        variants.append(word[:-1])
    return [v for v in variants if v]


def tokenize_lexical(text: str) -> list[str]:
    """中英混排分词：英文按单词，中文按二元组，便于 BM25 精确匹配。"""
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []

    for match in _LATIN_TOKEN_RE.finditer(normalized):
        word = match.group(0)
        tokens.append(word)
        tokens.extend(_english_stem_variants(word))

    for run in _CJK_RUN_RE.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
        else:
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])

    return tokens


# ---------------------------------------------------------------------------
# 文档加载
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            raise
    return path.read_text(encoding="utf-8", errors="ignore")


def _first_markdown_title(text: str) -> str:
    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == 1:
            title = match.group(2).strip()
            if title:
                return title
    return ""


def _is_cjk_char(ch: str) -> bool:
    return bool(ch) and bool(_CJK_IDEOGRAPH_RE.match(ch))


def _join_wrapped_lines(lines: list[str]) -> str:
    """把 PDF / 纯文本里的软换行合成段落。"""
    if not lines:
        return ""
    parts: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not parts:
            parts.append(line)
            continue
        prev = parts[-1]
        if prev.endswith(("。", "！", "？", "；", ".", "!", "?", ";", ":", "：")) or _is_cjk_char(prev[-1]):
            parts.append(line)
        else:
            parts.append(" " + line)
    return "".join(parts)


def _reflow_plain_paragraphs(text: str) -> str:
    """按空行分段，段内合并软换行；中英混排都适用。"""
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            buffer.append(line)
        else:
            if buffer:
                paragraphs.append(_join_wrapped_lines(buffer))
                buffer = []
    if buffer:
        paragraphs.append(_join_wrapped_lines(buffer))
    return "\n\n".join(p for p in paragraphs if p.strip())


def _load_pdf(path: Path) -> tuple[list[Page], list[str], str]:
    warnings: list[str] = []
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is required: pip install pypdf")
    reader = PdfReader(str(path))
    pages: list[Page] = []
    title = ""
    meta = reader.metadata
    if meta and meta.title:
        title = str(meta.title).strip()
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf 对损坏页面可能抛异常
            warnings.append(f"failed to parse page {number}: {exc}")
            text = ""
        text = normalize_text(text)
        if text:
            text = _reflow_plain_paragraphs(text)
            pages.append(Page(text=text, page=number))
    if not pages:
        warnings.append("PDF has no extractable text layer (likely scanned; OCR is not supported)")
    return pages, warnings, title


def load_document(source: SourceFile, docs_dir: Path) -> LoadedDoc | None:
    """加载单个文件，返回按页组织的文本。"""
    path = source.path
    ext = path.suffix.lower()
    title_fallback = path.stem
    try:
        if ext == ".pdf":
            pages, warnings, meta_title = _load_pdf(path)
            return LoadedDoc(
                path=path,
                rel_path=source.rel_path,
                title=(meta_title or title_fallback).strip(),
                pages=pages,
                is_markdown=False,
                warnings=warnings,
            )

        raw = normalize_text(_read_text_file(path))
        if not raw:
            return LoadedDoc(
                path=path,
                rel_path=source.rel_path,
                title=title_fallback,
                pages=[],
                is_markdown=ext in {".md", ".markdown"},
                warnings=["文件内容为空"],
            )
        is_markdown = ext in {".md", ".markdown"}
        title = (_first_markdown_title(raw) if is_markdown else "") or title_fallback
        return LoadedDoc(
            path=path,
            rel_path=source.rel_path,
            title=title.strip(),
            pages=[Page(text=raw, page=0)],
            is_markdown=is_markdown,
        )
    except Exception as exc:
        log.warning("加载失败 %s: %s", path, exc)
        return LoadedDoc(
            path=path,
            rel_path=source.rel_path,
            title=title_fallback,
            pages=[],
            is_markdown=ext in {".md", ".markdown"},
            warnings=[f"加载失败：{exc}"],
        )


# ---------------------------------------------------------------------------
# 分块
# ---------------------------------------------------------------------------

def _iter_blocks(
    text: str, markdown: bool
) -> Iterator[tuple[str, str]]:
    """产出 (标题路径, 段落文本)。Markdown 识别标题和代码块。"""
    if not markdown:
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if paragraph:
                yield "", paragraph
        return

    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False
    fence_lines: list[str] = []

    def heading_path() -> str:
        return " > ".join(title for _, title in heading_stack)

    def flush() -> Iterator[tuple[str, str]]:
        nonlocal buffer
        paragraph = "\n".join(buffer).strip()
        buffer = []
        if paragraph:
            yield heading_path(), paragraph

    for line in text.split("\n"):
        fence = _FENCE_RE.match(line)
        if fence:
            if not in_fence:
                yield from flush()
                in_fence = True
                fence_lines = [line]
            else:
                fence_lines.append(line)
                in_fence = False
                block = "\n".join(fence_lines).strip()
                fence_lines = []
                if block:
                    yield heading_path(), block
            continue

        if in_fence:
            fence_lines.append(line)
            continue

        match = _HEADING_RE.match(line.strip())
        if match:
            yield from flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            if title:
                heading_stack.append((level, title))
            continue

        if not line.strip():
            yield from flush()
            continue

        buffer.append(line)

    if in_fence and fence_lines:
        block = "\n".join(fence_lines).strip()
        if block:
            yield heading_path(), block
    yield from flush()


def _split_sentences(text: str) -> list[str]:
    """中英文句子切分；对无标点长文本返回原文。"""
    parts = re.split(
        r"(?<=[。！？；!?;])\s*|(?<=[.!?])\s+(?=[A-Z0-9\"'（(])",
        text,
    )
    parts = [p for p in (part.strip() for part in parts) if p]
    return parts or [text.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """无标点长文本按字符窗口切分，带重叠。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - max(0, overlap))
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return [p.strip() for p in pieces if p.strip()]


def _split_long_paragraph(text: str, cfg: Config) -> list[str]:
    """把过长的段落按句子边界切开；单句过长时按字符窗口切开。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= cfg.chunk_max:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > cfg.chunk_max:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, cfg.chunk_size, cfg.chunk_overlap))
            continue
        if current and len(current) + len(sentence) + 1 > cfg.chunk_size:
            pieces.append(current)
            current = sentence
        else:
            current = sentence if not current else current + sentence
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def _make_chunk(
    doc: LoadedDoc,
    body: str,
    page: int,
    index: int,
    heading: str,
) -> Chunk:
    body = body.strip()
    content_hash = hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
    raw_id = f"{doc.rel_path}\0{page}\0{index}\0{content_hash}".encode("utf-8")
    chunk_id = hashlib.sha1(raw_id).hexdigest()[:32]
    embed_parts = [doc.title, heading, body]
    embed_text = "\n".join(part for part in embed_parts if part)
    metadata = {
        "source": doc.rel_path,
        "title": doc.title,
        "heading": heading,
        "page": int(page),
        "chunk_index": int(index),
        "language": detect_language(body),
        "chars": len(body),
        "content_hash": content_hash,
    }
    return Chunk(
        id=chunk_id,
        document=body,
        embed_text=embed_text,
        metadata=metadata,
        language=metadata["language"],
    )


def chunk_document(doc: LoadedDoc, cfg: Config) -> list[Chunk]:
    """把文档切成带标题、页码和来源元数据的中英兼容块。"""
    chunks: list[Chunk] = []
    next_index = 0

    for page in doc.pages:
        current_parts: list[str] = []
        current_len = 0
        current_heading = ""

        def flush() -> None:
            nonlocal current_parts, current_len, next_index
            body = "\n\n".join(current_parts).strip()
            current_parts = []
            current_len = 0
            if not body:
                return
            chunks.append(
                _make_chunk(doc, body, page.page, next_index, current_heading)
            )
            next_index += 1

        for heading, paragraph in _iter_blocks(page.text, doc.is_markdown):
            if heading != current_heading and current_parts:
                flush()
            current_heading = heading or current_heading
            for piece in _split_long_paragraph(paragraph, cfg):
                if not piece:
                    continue
                if current_parts and current_len + len(piece) + 2 > cfg.chunk_size:
                    flush()
                    current_heading = heading or current_heading
                current_parts.append(piece)
                current_len += len(piece) + 2
                if current_len >= cfg.chunk_size:
                    flush()
                    current_heading = heading or current_heading
        flush()

    # 合并过短且属于同一页的尾块，避免产生只有几个字的噪声块。
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk.document) < cfg.min_chunk_chars
            and merged[-1].page == chunk.page
            and merged[-1].source == chunk.source
            and len(merged[-1].document) + len(chunk.document) + 2 <= cfg.chunk_max
        ):
            previous = merged[-1]
            body = (previous.document + "\n\n" + chunk.document).strip()
            replacement = _make_chunk(
                doc,
                body,
                previous.page,
                previous.chunk_index,
                previous.heading,
            )
            merged[-1] = replacement
        else:
            merged.append(chunk)

    return merged
