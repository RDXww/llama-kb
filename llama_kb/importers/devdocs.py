"""DevDocs 离线 JSON 导入器。

它只负责两件事：
1. 把 DevDocs 的 HTML 页面转换成带代码围栏的 Markdown；
2. 生成通用 kb_catalog.json（分组、别名、版本、默认分组）。

知识库检索层不依赖 DevDocs 的任何字段，因此其他来源（PDF、TXT、
内部手册）可以复用同一个 catalog 结构。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from bs4 import BeautifulSoup
from markdownify import markdownify as markdownify_html

# ---------------------------------------------------------------------------
# 文档名称 / 类型 / 别名
# ---------------------------------------------------------------------------

BASE_META: dict[str, dict[str, Any]] = {
    "apache_http_server": {
        "name": "Apache HTTP Server",
        "type": "server",
        "aliases": ["Apache", "httpd", "Apache HTTP Server"],
        "description": "Apache HTTP Server 配置、模块与指令参考",
    },
    "c": {
        "name": "C",
        "type": "programming_language",
        "aliases": [
            "C", "C语言", "C 语言", "ANSI C", "ISO C", "标准C",
            "C89", "C90", "C95", "C99", "C11", "C17", "C18", "C23",
        ],
        "description": "C 语言标准、关键字与标准库参考（C89/C99/C11/C17/C23）",
    },
    "cpp": {
        "name": "C++",
        "type": "programming_language",
        "aliases": [
            "C++", "C＋＋", "cpp", "CPP", "C++11", "C++14", "C++17",
            "C++20", "C++23", "STL", "C++ 标准库",
        ],
        "description": "C++ 语言、标准库（STL）与示例参考",
    },
    "rust": {
        "name": "Rust",
        "type": "programming_language",
        "aliases": ["Rust", "Rust语言", "cargo", "rustc"],
        "description": "Rust 语言与标准库参考",
    },
    "go": {
        "name": "Go",
        "type": "programming_language",
        "aliases": ["Go", "Golang", "Go语言", "Go 语言"],
        "description": "Go 语言标准库与语言参考",
    },
    "python": {
        "name": "Python",
        "type": "programming_language",
        "aliases": ["Python", "Python3", "Python 3", "py"],
        "description": "Python 语言与标准库参考",
    },
    "javascript": {
        "name": "JavaScript",
        "type": "programming_language",
        "aliases": ["JavaScript", "JS", "ECMAScript", "ES6", "ES2015", "ES2020"],
        "description": "JavaScript 语言与内置对象参考",
    },
    "typescript": {
        "name": "TypeScript",
        "type": "programming_language",
        "aliases": ["TypeScript", "TS"],
        "description": "TypeScript 语言与类型系统参考",
    },
    "php": {
        "name": "PHP",
        "type": "programming_language",
        "aliases": ["PHP", "PHP语言"],
        "description": "PHP 语言与函数参考",
    },
    "ruby": {
        "name": "Ruby",
        "type": "programming_language",
        "aliases": ["Ruby", "Ruby语言"],
        "description": "Ruby 语言、核心类与标准库参考",
    },
    "r": {
        "name": "R",
        "type": "programming_language",
        "aliases": ["R", "R语言", "R 语言", "Rscript"],
        "description": "R 语言与统计函数参考",
    },
    "powershell": {
        "name": "PowerShell",
        "type": "programming_language",
        "aliases": ["PowerShell", "pwsh", "PS"],
        "description": "PowerShell 命令、语法与模块参考",
    },
    "html": {
        "name": "HTML",
        "type": "web_language",
        "aliases": ["HTML", "HTML5"],
        "description": "HTML 元素与属性参考",
    },
    "css": {
        "name": "CSS",
        "type": "web_language",
        "aliases": ["CSS", "CSS3"],
        "description": "CSS 属性、选择器与布局参考",
    },
    "dom": {
        "name": "DOM",
        "type": "web_api",
        "aliases": ["DOM", "Document Object Model", "浏览器 DOM"],
        "description": "Web DOM API 参考",
    },
    "svg": {
        "name": "SVG",
        "type": "format",
        "aliases": ["SVG", "可缩放矢量图形"],
        "description": "SVG 元素与属性参考",
    },
    "http": {
        "name": "HTTP",
        "type": "protocol",
        "aliases": ["HTTP", "HTTPS", "HTTP/1.1", "HTTP/2", "HTTP/3", "超文本传输协议"],
        "description": "HTTP 协议、头部、方法与状态码参考",
    },
    "web_extensions": {
        "name": "WebExtensions",
        "type": "web_api",
        "aliases": ["WebExtensions", "Web Extension", "浏览器扩展"],
        "description": "浏览器扩展 API 参考",
    },
    "sqlite": {
        "name": "SQLite",
        "type": "database",
        "aliases": ["SQLite", "SQLite3"],
        "description": "SQLite 数据库与 SQL 语法参考",
    },
    "mariadb": {
        "name": "MariaDB",
        "type": "database",
        "aliases": ["MariaDB", "MySQL", "MySQL/MariaDB"],
        "description": "MariaDB 数据库、SQL 语句与函数参考（MySQL 兼容）",
    },
    "postgresql": {
        "name": "PostgreSQL",
        "type": "database",
        "aliases": ["PostgreSQL", "Postgres", "PG"],
        "description": "PostgreSQL 数据库、SQL 语句与函数参考",
    },
    "redis": {
        "name": "Redis",
        "type": "database",
        "aliases": ["Redis"],
        "description": "Redis 命令、数据结构与配置参考",
    },
    "git": {
        "name": "Git",
        "type": "tool",
        "aliases": ["Git", "git"],
        "description": "Git 命令与版本控制参考",
    },
    "docker": {
        "name": "Docker",
        "type": "tool",
        "aliases": ["Docker", "docker-compose"],
        "description": "Docker 命令行、镜像、容器、Compose 与配置参考",
    },
    "cmake": {
        "name": "CMake",
        "type": "tool",
        "aliases": ["CMake", "cmake"],
        "description": "CMake 构建系统、命令与变量参考",
    },
    "nginx": {
        "name": "Nginx",
        "type": "server",
        "aliases": ["Nginx", "NGINX"],
        "description": "Nginx 配置、模块与指令参考",
    },
    "nginx_lua_module": {
        "name": "Nginx Lua",
        "type": "module",
        "aliases": ["Nginx Lua", "ngx_lua", "OpenResty"],
        "description": "Nginx Lua 模块 API 参考",
    },
    "npm": {
        "name": "npm",
        "type": "tool",
        "aliases": ["npm", "Node 包管理器"],
        "description": "npm 命令行与 package.json 参考",
    },
    "node": {
        "name": "Node.js",
        "type": "runtime",
        "aliases": ["Node.js", "NodeJS", "Node"],
        "description": "Node.js 运行时与 API 参考",
    },
    "react": {
        "name": "React",
        "type": "framework",
        "aliases": ["React", "React.js", "ReactJS"],
        "description": "React 组件、Hooks 与 API 参考",
    },
    "react_bootstrap": {
        "name": "React Bootstrap",
        "type": "library",
        "aliases": ["React Bootstrap", "React-Bootstrap"],
        "description": "React Bootstrap 组件参考",
    },
    "vue": {
        "name": "Vue",
        "type": "framework",
        "aliases": ["Vue", "Vue.js", "Vue3", "Vue 3"],
        "description": "Vue 3 框架 API 与组件参考",
    },
    "nextjs": {
        "name": "Next.js",
        "type": "framework",
        "aliases": ["Next.js", "NextJS"],
        "description": "Next.js 框架、路由与 API 参考",
    },
    "django": {
        "name": "Django",
        "type": "framework",
        "aliases": ["Django"],
        "description": "Django Web 框架参考",
    },
    "flask": {
        "name": "Flask",
        "type": "framework",
        "aliases": ["Flask"],
        "description": "Flask Web 框架参考",
    },
    "fastapi": {
        "name": "FastAPI",
        "type": "framework",
        "aliases": ["FastAPI"],
        "description": "FastAPI Web 框架参考",
    },
    "spring_boot": {
        "name": "Spring Boot",
        "type": "framework",
        "aliases": ["Spring Boot", "SpringBoot", "Spring"],
        "description": "Spring Boot 框架参考",
    },
    "jquery": {
        "name": "jQuery",
        "type": "library",
        "aliases": ["jQuery", "JQuery"],
        "description": "jQuery API 参考",
    },
    "numpy": {
        "name": "NumPy",
        "type": "library",
        "aliases": ["NumPy", "Numpy", "np"],
        "description": "NumPy 数组、函数与 API 参考",
    },
    "wordpress": {
        "name": "WordPress",
        "type": "application",
        "aliases": ["WordPress", "WP"],
        "description": "WordPress 函数、钩子与主题/插件 API 参考",
    },
    "latex": {
        "name": "LaTeX",
        "type": "format",
        "aliases": ["LaTeX", "LaTex", "TeX"],
        "description": "LaTeX 排版语法与命令参考",
    },
    "markdown": {
        "name": "Markdown",
        "type": "format",
        "aliases": ["Markdown", "MD"],
        "description": "Markdown 语法参考",
    },
    "opengl": {
        "name": "OpenGL",
        "type": "api",
        "aliases": ["OpenGL", "GL"],
        "description": "OpenGL API 参考",
    },
}

UNSUPPORTED: list[dict[str, Any]] = [
    {"name": "Java", "aliases": ["Java", "Java语言", "JVM", "JDK"]},
    {"name": "C#", "aliases": ["C#", "CSharp", ".NET", "dotnet"]},
    {"name": "Swift", "aliases": ["Swift", "Swift语言"]},
    {"name": "Kotlin", "aliases": ["Kotlin"]},
    {"name": "Scala", "aliases": ["Scala"]},
    {"name": "Haskell", "aliases": ["Haskell"]},
    {"name": "Lua", "aliases": ["Lua", "Lua语言"]},
    {"name": "Perl", "aliases": ["Perl"]},
    {"name": "Dart", "aliases": ["Dart", "Flutter"]},
    {"name": "Elixir", "aliases": ["Elixir"]},
    {"name": "Objective-C", "aliases": ["Objective-C", "ObjC"]},
    {"name": "MATLAB", "aliases": ["MATLAB"]},
    {"name": "Fortran", "aliases": ["Fortran"]},
    {"name": "COBOL", "aliases": ["COBOL"]},
    {"name": "Groovy", "aliases": ["Groovy"]},
    {"name": "Julia", "aliases": ["Julia"]},
    {"name": "Bash", "aliases": ["Bash", "Shell", "shell 脚本"]},
    {"name": "Linux", "aliases": ["Linux 命令", "Linux命令", "systemd"]},
    {"name": "Kubernetes", "aliases": ["Kubernetes", "K8s", "kubectl"]},
    {"name": "MongoDB", "aliases": ["MongoDB", "Mongo"]},
    {"name": "AWS", "aliases": ["AWS", "Amazon Web Services"]},
    {"name": "Azure", "aliases": ["Azure"]},
    {"name": "GCP", "aliases": ["GCP", "Google Cloud"]},
    {"name": "Vim", "aliases": ["Vim", "Neovim"]},
    {"name": "Emacs", "aliases": ["Emacs"]},
]

# ---------------------------------------------------------------------------
# HTML -> Markdown
# ---------------------------------------------------------------------------


def extract_title(soup: BeautifulSoup) -> str:
    element = soup.find("h1") or soup.find("h2")
    if element:
        return element.get_text(" ", strip=True)
    return ""


def make_fence(code: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


def html_to_markdown(html: str, drop_attribution: bool = True) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)

    if drop_attribution:
        for element in soup.select("div._attribution"):
            element.decompose()

    code_blocks: list[tuple[str, str]] = []
    # 关键：占位符必须是纯字母数字，markdownify 不会把下划线转义成 \_。
    for index, pre in enumerate(soup.find_all("pre")):
        code = pre.get_text()
        language = pre.get("data-language") or ""
        if not language:
            code_el = pre.find("code")
            if code_el:
                for cls in code_el.get("class", []):
                    if cls.startswith("language-"):
                        language = cls[len("language-") :]
                        break
        code = code.rstrip("\n")
        placeholder = f"ZZZDEVDOCSCODEBLOCK{index:05d}ZZZ"
        pre.replace_with(soup.new_string(f"\n\n{placeholder}\n\n"))
        code_blocks.append((language, code))

    text = markdownify_html(str(soup), heading_style="ATX", bullets="-")

    for index, (language, code) in enumerate(code_blocks):
        placeholder = f"ZZZDEVDOCSCODEBLOCK{index:05d}ZZZ"
        fence = make_fence(code)
        block = f"{fence}{language}\n{code}\n{fence}"
        text = text.replace(placeholder, block, 1)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text:
        text += "\n"
    return text, title


def safe_rel_path(key: str) -> Path:
    key = key.replace("\\", "/")
    parts: list[str] = []
    for part in key.split("/"):
        part = part.strip()
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"非法路径片段 '..': {key!r}")
        part = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part)
        parts.append(part or "_")
    if not parts:
        parts = ["index"]
    name = parts[-1]
    if not name.endswith(".md"):
        name += ".md"
    return Path(*parts[:-1], name)


def write_index(base_dir: Path, doc_slug: str, links: list[tuple[str, Path]], titles: dict[str, str]) -> None:
    lines = [f"# {doc_slug}", ""]
    for key, rel in sorted(links, key=lambda item: item[0]):
        title = titles.get(key) or key
        lines.append(f"- [{title}]({quote(rel.as_posix(), safe='/')})")
    (base_dir / "_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 转换
# ---------------------------------------------------------------------------


def convert_docs(
    data: dict[str, Any],
    output_root: Path,
    *,
    only: set[str] | None = None,
    write_indexes: bool = False,
    drop_attribution: bool = True,
    progress: bool = True,
) -> tuple[int, int]:
    docs = data.get("docs")
    if not isinstance(docs, list):
        raise ValueError("JSON 中缺少 docs 数组")
    output_root.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    total_files = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        slug = str(doc.get("slug") or "doc")
        if only and slug not in only:
            continue
        db = doc.get("db") or {}
        if not isinstance(db, dict):
            continue
        base_dir = output_root / slug
        links: list[tuple[str, Path]] = []
        titles: dict[str, str] = {}
        for key, html in db.items():
            key = str(key)
            try:
                rel = safe_rel_path(key)
            except ValueError:
                continue
            destination = base_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            text, title = html_to_markdown(str(html), drop_attribution=drop_attribution)
            destination.write_text(text, encoding="utf-8")
            links.append((key, rel))
            titles[key] = title or key
            total_pages += 1
            total_files += 1
        if write_indexes:
            write_index(base_dir, slug, links, titles)
        if progress:
            print(f"  转换：{slug}（{len(db)} 页）", flush=True)
    return total_pages, total_files


# ---------------------------------------------------------------------------
# Catalog 生成
# ---------------------------------------------------------------------------


def _version_key(version: str) -> tuple:
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    numbers = tuple(int(part) if part else 0 for part in (match.groups() if match else ()))
    return numbers + (version.lower(),)


def build_catalog(
    data: dict[str, Any],
    *,
    docs_root_name: str = "devdocs",
    name: str = "llama-kb",
) -> dict[str, Any]:
    docs = [doc for doc in (data.get("docs") or []) if isinstance(doc, dict)]
    entries: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        slug = str(doc.get("slug") or "doc")
        base = slug.split("~", 1)[0]
        version = slug.split("~", 1)[1] if "~" in slug else ""
        entries.setdefault(base, []).append({"slug": slug, "version": version})

    # 同名产品如果存在无版本条目，优先无版本；否则选版本号最高的一个。
    preferred_slugs: dict[str, str] = {}
    for base, items in entries.items():
        plain = [item for item in items if not item["version"]]
        if plain:
            preferred_slugs[base] = plain[0]["slug"]
        else:
            items.sort(key=lambda item: _version_key(item["version"]), reverse=True)
            preferred_slugs[base] = items[0]["slug"]

    groups: list[dict[str, Any]] = []
    for doc in docs:
        slug = str(doc.get("slug") or "doc")
        base = slug.split("~", 1)[0]
        version = slug.split("~", 1)[1] if "~" in slug else ""
        meta = BASE_META.get(base)
        if meta is None:
            meta = {
                "name": base,
                "type": "document",
                "aliases": [base],
                "description": f"{base} 参考文档",
            }
        label = f"{meta['name']} {version}".strip()
        base_aliases = list(meta.get("aliases") or [])
        has_unversioned = any(not item["version"] for item in entries[base])
        # 同族存在无版本条目时，只有无版本条目携带“Docker”“Node.js”这类
        # 通用别名；历史版本分组只带版本组合别名，避免把普通问题路由到旧版本。
        # 同族没有无版本条目时，由 preferred 的那个版本分组携带通用别名。
        if version and has_unversioned:
            aliases: list[str] = []
        elif version and slug != preferred_slugs.get(base):
            aliases = []
        else:
            aliases = list(base_aliases)
        if version:
            for alias in base_aliases:
                aliases.extend(
                    [
                        f"{alias} {version}",
                        f"{alias} v{version}",
                        f"{alias}{version}",
                    ]
                )
            aliases.append(slug)
        groups.append(
            {
                "id": slug,
                "name": meta["name"],
                "type": meta["type"],
                "version": version,
                "parent": base if version else "",
                "preferred": slug == preferred_slugs.get(base),
                "sources": [f"{docs_root_name}/{slug}/"],
                "aliases": sorted(set(aliases)),
                "description": meta.get("description", ""),
                "embed_prefix": label,
                "attributes": {"devdocs_slug": slug},
            }
        )

    default_group = {
        "id": "general",
        "name": "General",
        "type": "general",
        "version": "",
        "parent": "",
        "preferred": True,
        "sources": [],
        "aliases": ["general", "docs", "其他资料"],
        "description": "Documents that do not belong to any other group",
        "embed_prefix": "",
    }
    return {
        "version": 1,
        "name": name,
        "default_group": default_group,
        "groups": groups,
        "unsupported": UNSUPPORTED,
        "settings": {
            "router_min_score": 0.30,
            "router_scope_min_score": 0.42,
            "router_max_groups": 3,
            "router_multi_margin": 0.02,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def merge_catalog(
    generated: dict[str, Any],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    """重新导入 DevDocs 时保留用户手工加入的其他分组。"""
    if not isinstance(existing, dict):
        return generated
    generated_ids = {str(group.get("id")) for group in generated.get("groups", [])}
    merged_groups = list(generated.get("groups", []))
    for group in existing.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id") or "")
        if group_id and group_id not in generated_ids:
            merged_groups.append(group)
            generated_ids.add(group_id)
    merged_unsupported = list(generated.get("unsupported", []))
    existing_names = {
        str(item.get("name"))
        for item in merged_unsupported
        if isinstance(item, dict)
    }
    for item in existing.get("unsupported", []) or []:
        if isinstance(item, dict) and str(item.get("name")) not in existing_names:
            merged_unsupported.append(item)
            existing_names.add(str(item.get("name")))
    settings = dict(generated.get("settings") or {})
    settings.update(existing.get("settings") or {})
    result = dict(generated)
    result["groups"] = merged_groups
    result["unsupported"] = merged_unsupported
    result["settings"] = settings
    if existing.get("name"):
        result["name"] = existing.get("name")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m llama_kb.importers.devdocs",
        description="Convert a devdocs-offline JSON export to Markdown and a llama-kb catalog",
    )
    parser.add_argument("input", help="devdocs-offline JSON file")
    parser.add_argument(
        "-o", "--output", default=str(Path.home() / ".llama-kb" / "docs" / "devdocs"),
        help="Markdown output root (default: ~/.llama-kb/docs/devdocs)",
    )
    parser.add_argument(
        "--catalog", default=str(Path.home() / ".llama-kb" / "catalog.json"),
        help="catalog output path (default: ~/.llama-kb/catalog.json)",
    )
    parser.add_argument(
        "--only", default="", help="only convert these slugs (comma separated)",
    )
    parser.add_argument("--indexes", action="store_true", help="also write _index.md files")
    parser.add_argument(
        "--keep-attribution", action="store_true", help="keep div._attribution license blocks",
    )
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args(argv)

    source = Path(args.input)
    if not source.is_file():
        print(f"error: input file not found: {source}", file=sys.stderr)
        return 1
    try:
        data = json.loads(source.read_text(encoding=args.encoding))
    except Exception as exc:
        print(f"error: cannot read or parse JSON: {exc}", file=sys.stderr)
        return 1
    if data.get("type") != "devdocs-offline":
        print("warning: JSON type is not 'devdocs-offline'; trying anyway.", file=sys.stderr)

    only = {item.strip() for item in args.only.split(",") if item.strip()} or None
    output = Path(args.output).expanduser()
    catalog_path = Path(args.catalog).expanduser()
    pages, files = convert_docs(
        data,
        output,
        only=only,
        write_indexes=args.indexes,
        drop_attribution=not args.keep_attribution,
    )
    catalog = build_catalog(data)
    existing_catalog: dict[str, Any] | None = None
    if catalog_path.is_file():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_catalog = loaded
        except (OSError, json.JSONDecodeError):
            existing_catalog = None
    catalog = merge_catalog(catalog, existing_catalog)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"done: {pages} pages, Markdown root {output}, catalog {catalog_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
