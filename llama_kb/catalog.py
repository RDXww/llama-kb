"""Generic document-group catalog.

The knowledge base does not assume that documents are programming manuals.
Any corpus can be described with a group:

    {
      "id": "employee_handbook",
      "name": "Employee Handbook",
      "type": "handbook",
      "sources": ["handbook/"],
      "aliases": ["handbook", "onboarding", "leave policy"],
      "description": "Company policies and onboarding material",
      "version": "",
      "parent": "",
      "preferred": true
    }

Retrieval only depends on these generic fields. Importers such as the DevDocs
converter are responsible for producing groups and Markdown; they never need
to touch retrieval logic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_GROUP_ID = "general"


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


@dataclass(frozen=True)
class DocGroup:
    """一个可独立检索/路由的文档分组。"""

    id: str
    name: str
    type: str = "document"
    version: str = ""
    parent: str = ""
    preferred: bool = True
    sources: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    description: str = ""
    embed_prefix: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.version and self.version not in self.name:
            return f"{self.name} {self.version}"
        return self.name

    def metadata(self) -> dict[str, Any]:
        """写入 Chroma / FTS 的通用分组元数据。"""
        return {
            "group_id": self.id,
            "group_name": self.name,
            "group_type": self.type,
            "group_version": self.version or "",
            "group_parent": self.parent or "",
            "group_label": self.label,
        }

    def summary_text(self) -> str:
        parts = [self.label, self.type, self.description]
        if self.aliases:
            parts.append("别名：" + "、".join(self.aliases))
        return " ｜ ".join(part for part in parts if part)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, default: bool = False) -> "DocGroup":
        return cls(
            id=str(data.get("id") or (DEFAULT_GROUP_ID if default else "")).strip(),
            name=str(data.get("name") or data.get("id") or "未命名分组").strip(),
            type=str(data.get("type") or "document").strip(),
            version=str(data.get("version") or "").strip(),
            parent=str(data.get("parent") or "").strip(),
            preferred=bool(data.get("preferred", True)),
            sources=_as_tuple(data.get("sources")),
            aliases=_as_tuple(data.get("aliases")),
            description=str(data.get("description") or "").strip(),
            embed_prefix=str(data.get("embed_prefix") or "").strip(),
            attributes=dict(data.get("attributes") or {}),
        )


@dataclass(frozen=True)
class UnsupportedEntry:
    name: str
    aliases: tuple[str, ...] = ()
    message: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UnsupportedEntry":
        name = str(data.get("name") or "").strip()
        return cls(
            name=name,
            aliases=_as_tuple(data.get("aliases")) or (name,),
            message=str(data.get("message") or "").strip(),
        )


@dataclass
class CatalogMatch:
    groups: list[DocGroup] = field(default_factory=list)
    unsupported: list[UnsupportedEntry] = field(default_factory=list)
    explicit: bool = False
    aliases: list[str] = field(default_factory=list)


class Catalog:
    """分组目录：路径归属、别名匹配、版本解析、默认范围。"""

    def __init__(
        self,
        groups: Iterable[DocGroup] = (),
        *,
        default_group: DocGroup | None = None,
        unsupported: Iterable[UnsupportedEntry] = (),
        settings: Mapping[str, Any] | None = None,
        name: str = "local-kb",
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.settings = dict(settings or {})
        self.default_group = default_group or DocGroup(
            id=DEFAULT_GROUP_ID,
            name="通用文档",
            type="general",
            preferred=True,
        )
        self.groups: list[DocGroup] = list(groups)
        if all(group.id != self.default_group.id for group in self.groups):
            self.groups.append(self.default_group)
        self.unsupported: list[UnsupportedEntry] = list(unsupported)
        self.group_by_id: dict[str, DocGroup] = {
            group.id: group for group in self.groups
        }
        # 最长路径前缀优先，避免 devdocs/c 与 devdocs/cpp 互相误匹配。
        self._source_groups: list[tuple[str, DocGroup]] = sorted(
            (
                (source.replace("\\", "/").lstrip("/"), group)
                for group in self.groups
                for source in group.sources
                if source
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self._alias_entries: list[tuple[str, DocGroup | None, UnsupportedEntry | None, int]] = []
        self._build_alias_entries()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None) -> "Catalog":
        if not path:
            return cls()
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            return cls(path=file_path)
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path=file_path)
        if not isinstance(data, dict):
            return cls(path=file_path)
        default_group = DocGroup.from_mapping(data.get("default_group") or {}, default=True)
        groups = [
            DocGroup.from_mapping(item)
            for item in (data.get("groups") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        unsupported = [
            UnsupportedEntry.from_mapping(item)
            for item in (data.get("unsupported") or [])
            if isinstance(item, dict) and item.get("name")
        ]
        return cls(
            groups,
            default_group=default_group,
            unsupported=unsupported,
            settings=data.get("settings") or {},
            name=str(data.get("name") or "local-kb"),
            path=file_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "default_group": self.default_group.__dict__,
            "groups": [group.__dict__ for group in self.groups],
            "unsupported": [item.__dict__ for item in self.unsupported],
            "settings": self.settings,
        }

    # ------------------------------------------------------------------
    # 路径归属
    # ------------------------------------------------------------------
    def match_source(self, rel_path: str) -> DocGroup:
        normalized = rel_path.replace("\\", "/").lstrip("/")
        for source, group in self._source_groups:
            if normalized == source or normalized.startswith(source):
                return group
        return self.default_group

    @property
    def preferred_groups(self) -> list[DocGroup]:
        preferred = [group for group in self.groups if group.preferred]
        return preferred or list(self.groups)

    @property
    def preferred_ids(self) -> list[str]:
        return [group.id for group in self.preferred_groups]

    # ------------------------------------------------------------------
    # 别名 / 版本
    # ------------------------------------------------------------------
    def _build_alias_entries(self) -> None:
        entries: dict[tuple[str, str | None, str | None], tuple[str, DocGroup | None, UnsupportedEntry | None, int]] = {}
        # 先建立父分组别名，供版本分组生成 "docker 1.13" 一类组合别名。
        parent_aliases: dict[str, list[str]] = {}
        for group in self.groups:
            base = group.parent or group.id.split("~", 1)[0]
            parent = self.group_by_id.get(base)
            if parent is not None:
                names = [parent.name, parent.label, *parent.aliases]
            else:
                names = []
            parent_aliases.setdefault(base, [])
            for name in [group.name, group.label, *group.aliases, *names]:
                if name:
                    parent_aliases[base].append(name)

        for group in self.groups:
            names: list[str] = [group.id, group.label, *group.aliases]
            # 版本分组只有在“同族没有无版本条目、且自己就是 preferred”时
            # 才携带通用名称；否则通用名称属于无版本/首选分组。
            if not group.version or group.preferred:
                names.append(group.name)
            if group.version:
                bases = {group.name, group.label}
                base_id = group.parent or group.id.split("~", 1)[0]
                bases.update(parent_aliases.get(base_id, []))
                variants = {group.version}
                # Node.js 这类 24_lts 版本，用户通常只说 24。
                if "_" in group.version:
                    head = group.version.split("_", 1)[0]
                    variants.update({head, f"{head} LTS", f"{head} lts"})
                for base in bases:
                    if not base:
                        continue
                    for version in variants:
                        names.extend(
                            [
                                f"{base} {version}",
                                f"{base} v{version}",
                                f"{base}{version}",
                                f"{base} version {version}",
                                f"{base} 版本 {version}",
                            ]
                        )
            for name in names:
                key = (name.strip(), group.id, None)
                if not key[0]:
                    continue
                entries[key] = (key[0], group, None, len(key[0]))
        for item in self.unsupported:
            for alias in [item.name, *item.aliases]:
                key = (alias.strip(), None, item.name)
                if not key[0]:
                    continue
                entries[key] = (key[0], None, item, len(key[0]))
        self._alias_entries = sorted(
            entries.values(),
            key=lambda item: (item[3], item[0].lower()),
            reverse=True,
        )

    @staticmethod
    def _matched_pattern(alias: str) -> re.Pattern[str]:
        escaped = re.escape(alias)
        if alias.isascii():
            # C 不能命中 C++/CSS/C#；C++、C# 等更长别名会先被消费。
            return re.compile(
                rf"(?<![A-Za-z0-9_+#]){escaped}(?![A-Za-z0-9_+#])",
                re.IGNORECASE,
            )
        return re.compile(escaped, re.IGNORECASE)

    def match_text(self, text: str) -> CatalogMatch:
        """按最长别名优先匹配；匹配后从文本中扣除，避免子串重复命中。"""
        result = CatalogMatch()
        if not text:
            return result
        remaining = text
        seen_groups: list[DocGroup] = []
        seen_unsupported: list[UnsupportedEntry] = []
        seen_aliases: list[str] = []
        for alias, group, unsupported, _ in self._alias_entries:
            if not alias:
                continue
            pattern = self._matched_pattern(alias)
            if not pattern.search(remaining):
                continue
            if group is not None and group.id not in {item.id for item in seen_groups}:
                seen_groups.append(group)
                seen_aliases.append(alias)
            if unsupported is not None and unsupported.name not in {
                item.name for item in seen_unsupported
            }:
                seen_unsupported.append(unsupported)
                seen_aliases.append(alias)
            remaining = pattern.sub(lambda match: " " * len(match.group(0)), remaining)

        # 版本修正：问题里出现多个版本号（如 docker 1.13 和 19）时，
        # 把同族中所有被提到的版本分组一起加入，支持版本对比。
        if seen_groups:
            adjusted: list[DocGroup] = []
            ids: set[str] = set()
            for group in seen_groups:
                if group.id not in ids:
                    adjusted.append(group)
                    ids.add(group.id)
                base_id = group.parent or group.id.split("~", 1)[0]
                family = [
                    candidate
                    for candidate in self.groups
                    if candidate.id == base_id
                    or candidate.parent == base_id
                    or (
                        not candidate.parent
                        and candidate.id.split("~", 1)[0] == base_id
                        and not candidate.version
                    )
                ]
                for candidate in family:
                    if not candidate.version or candidate.id in ids:
                        continue
                    variants = {candidate.version}
                    if "_" in candidate.version:
                        head = candidate.version.split("_", 1)[0]
                        variants.add(head)
                    for variant in variants:
                        if re.search(
                            rf"(?<![0-9.])v?{re.escape(variant)}(?![0-9])",
                            text,
                            re.IGNORECASE,
                        ):
                            adjusted.append(candidate)
                            ids.add(candidate.id)
                            break
            seen_groups = adjusted

        result.groups = seen_groups
        result.unsupported = seen_unsupported
        result.explicit = bool(seen_groups or seen_unsupported)
        result.aliases = seen_aliases
        return result

    def resolve(self, token: str) -> tuple[list[DocGroup], list[str]]:
        """把 /doc 或 X-KB-Doc 的值解析为分组；支持逗号分隔。"""
        groups: list[DocGroup] = []
        unknown: list[str] = []
        for raw in re.split(r"[,，]", token or ""):
            item = raw.strip()
            if not item:
                continue
            if item.lower() in {"off", "none", "auto", "*"}:
                continue
            found = self.group_by_id.get(item)
            if found is None:
                lowered = item.casefold()
                matches = [
                    group
                    for group in self.groups
                    if lowered
                    in {
                        group.id.casefold(),
                        group.name.casefold(),
                        group.label.casefold(),
                        *(alias.casefold() for alias in group.aliases),
                    }
                ]
                if len(matches) == 1:
                    found = matches[0]
                elif len(matches) > 1:
                    # 名称有歧义时优先精确 id / label，其余交给调用方报错。
                    found = next(
                        (
                            group
                            for group in matches
                            if group.id.casefold() == lowered
                            or group.label.casefold() == lowered
                        ),
                        None,
                    )
            if found is None:
                unknown.append(item)
            elif found.id not in {group.id for group in groups}:
                groups.append(found)
        return groups, unknown

    # ------------------------------------------------------------------
    # 统计 / 指纹
    # ------------------------------------------------------------------
    def index_fingerprint(self) -> str:
        """只有影响分组归属/嵌入前缀的字段变化时才需要重建索引。"""
        payload = {
            "default": {
                "id": self.default_group.id,
                "name": self.default_group.name,
                "embed_prefix": self.default_group.embed_prefix,
            },
            "groups": sorted(
                (
                    {
                        "id": group.id,
                        "name": group.name,
                        "type": group.type,
                        "version": group.version,
                        "parent": group.parent,
                        "sources": sorted(group.sources),
                        "embed_prefix": group.embed_prefix,
                    }
                    for group in self.groups
                ),
                key=lambda item: item["id"],
            ),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def groups_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "id": group.id,
                "name": group.name,
                "label": group.label,
                "type": group.type,
                "version": group.version,
                "parent": group.parent,
                "preferred": group.preferred,
                "description": group.description,
                "aliases": list(group.aliases),
            }
            for group in self.groups
        ]

    def summary_items(self) -> list[tuple[DocGroup, str]]:
        return [(group, group.summary_text()) for group in self.groups]

    def label_for(self, group_id: str) -> str:
        group = self.group_by_id.get(group_id)
        return group.label if group else group_id
