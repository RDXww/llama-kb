from __future__ import annotations

import unittest

from llama_kb.catalog import Catalog, DocGroup, UnsupportedEntry
from llama_kb.router import DocRouter


def build_catalog() -> Catalog:
    groups = [
        DocGroup(
            id="c",
            name="C",
            type="programming_language",
            aliases=("C", "C语言", "C11", "C17", "C23"),
            sources=("devdocs/c/",),
        ),
        DocGroup(
            id="cpp",
            name="C++",
            type="programming_language",
            aliases=("C++", "cpp", "STL"),
            sources=("devdocs/cpp/",),
        ),
        DocGroup(
            id="css",
            name="CSS",
            type="web_language",
            aliases=("CSS",),
            sources=("devdocs/css/",),
        ),
        DocGroup(
            id="docker",
            name="Docker",
            type="tool",
            aliases=("Docker",),
            sources=("devdocs/docker/",),
        ),
        DocGroup(
            id="docker~1.13",
            name="Docker",
            type="tool",
            version="1.13",
            parent="docker",
            preferred=False,
            sources=("devdocs/docker~1.13/",),
        ),
        DocGroup(
            id="docker~19",
            name="Docker",
            type="tool",
            version="19",
            parent="docker",
            preferred=False,
            sources=("devdocs/docker~19/",),
        ),
        DocGroup(
            id="node",
            name="Node.js",
            type="runtime",
            aliases=("Node.js", "NodeJS", "Node"),
            sources=("devdocs/node/",),
        ),
        DocGroup(
            id="node~24_lts",
            name="Node.js",
            type="runtime",
            version="24_lts",
            parent="node",
            preferred=False,
            sources=("devdocs/node~24_lts/",),
        ),
        DocGroup(
            id="handbook",
            name="员工手册",
            type="handbook",
            aliases=("员工手册", "入职", "请假"),
            sources=("handbook/",),
        ),
    ]
    return Catalog(
        groups,
        default_group=DocGroup(id="general", name="通用文档", preferred=True),
        unsupported=[
            UnsupportedEntry(name="Java", aliases=("Java", "JVM")),
            UnsupportedEntry(name="C#", aliases=("C#", ".NET")),
        ],
    )


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_catalog()

    def test_source_mapping(self) -> None:
        self.assertEqual(self.catalog.match_source("devdocs/c/memory/malloc.md").id, "c")
        self.assertEqual(self.catalog.match_source("devdocs/cpp/vector.md").id, "cpp")
        self.assertEqual(self.catalog.match_source("handbook/leave.md").id, "handbook")
        self.assertEqual(self.catalog.match_source("other/file.md").id, "general")

    def test_c_does_not_match_cpp_or_csharp(self) -> None:
        self.assertEqual([g.id for g in self.catalog.match_text("C 的 malloc").groups], ["c"])
        self.assertEqual([g.id for g in self.catalog.match_text("C++ 的 vector").groups], ["cpp"])
        self.assertEqual(
            [g.id for g in self.catalog.match_text("C# 的 LINQ").groups], []
        )
        self.assertEqual(
            [u.name for u in self.catalog.match_text("C# 的 LINQ").unsupported], ["C#"]
        )
        self.assertEqual([g.id for g in self.catalog.match_text("CSS 布局").groups], ["css"])

    def test_multi_group_comparison(self) -> None:
        match = self.catalog.match_text("C 和 C++ 的指针区别")
        self.assertEqual({group.id for group in match.groups}, {"c", "cpp"})

    def test_version_siblings(self) -> None:
        match = self.catalog.match_text("docker 1.13 和 19 的区别")
        self.assertEqual({group.id for group in match.groups}, {"docker~1.13", "docker~19"})
        match = self.catalog.match_text("Node 24 的 API")
        self.assertEqual([group.id for group in match.groups], ["node~24_lts"])

    def test_resolve(self) -> None:
        groups, unknown = self.catalog.resolve("C++,员工手册")
        self.assertEqual([group.id for group in groups], ["cpp", "handbook"])
        self.assertEqual(unknown, [])
        _, unknown = self.catalog.resolve("不存在的分组")
        self.assertEqual(unknown, ["不存在的分组"])


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_catalog()
        self.router = DocRouter(self.catalog, None)

    def test_scoped_multi_global_unsupported(self) -> None:
        self.assertEqual(self.router.route("C 的 malloc").group_ids, ["c"])
        self.assertEqual(
            set(self.router.route("C 和 C++ 的区别").group_ids), {"c", "cpp"}
        )
        self.assertEqual(self.router.route("如何定义数组").mode, "global")
        decision = self.router.route("Java 的 HashMap")
        self.assertEqual(decision.mode, "unsupported")
        self.assertEqual(decision.unsupported, ["Java"])
        self.assertIn("Java", self.router.unsupported_answer(["Java"]))

    def test_forced_groups(self) -> None:
        decision = self.router.route(
            "它的参数是什么",
            forced_groups=[self.catalog.group_by_id["c"]],
        )
        self.assertEqual(decision.mode, "forced")
        self.assertEqual(decision.group_ids, ["c"])

    def test_history_inheritance(self) -> None:
        messages = [
            {"role": "user", "content": "C 的 malloc 怎么用"},
            {"role": "assistant", "content": "略"},
            {"role": "user", "content": "它的参数呢"},
        ]
        decision = self.router.route("它的参数呢", messages=messages)
        self.assertEqual(decision.group_ids, ["c"])


if __name__ == "__main__":
    unittest.main()
