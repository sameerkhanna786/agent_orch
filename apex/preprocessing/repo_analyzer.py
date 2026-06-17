"""
Repository preprocessing and issue-scoped context utilities.
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("apex.preprocessing")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "when",
    "with",
}

_KEYWORD_NOISE_TOKENS = {
    "py",
    "pytest",
    "python",
    "python3",
    "pythonpath",
    "continue-on-collection-errors",
    "json-report",
    "json-report-file",
    "virtual_env",
    "venv",
}

_AFFINITY_TOKEN_NOISE = {
    "test",
    "tests",
    "unit",
    "units",
    "integration",
    "spec",
    "specs",
    "src",
    "lib",
    "pkg",
    "module",
    "modules",
    "plugin",
    "plugins",
    "target",
    "targets",
    "python",
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "main",
    "base",
    "common",
    "shared",
    "util",
    "utils",
    "helper",
    "helpers",
    "index",
    "__init__",
}

_HOST_PATH_PREFIXES = (
    "/usr/",
    "/opt/",
    "/System/",
    "/Library/",
    "/Applications/",
    "/private/",
    "/var/",
)

_HOST_PATH_MARKERS = (
    "site-packages/",
    "dist-packages/",
    ".venv/",
    "venv/",
    ".tox/",
    "__pycache__/",
)

_APEX_HARNESS_BASENAMES = frozenset(
    {
        "_apex_run_expected_ids.py",
        "_apex_expected_ids_filter.py",
        ".apex_expected_test_ids.txt",
        "_apex_expected_test_ids.txt",
    }
)

_PATH_PATTERN_METACHARS = frozenset("*?[")

_RATIONALE_COMMENT_PREFIXES = (
    "# NOTE:",
    "# IMPORTANT:",
    "# WHY:",
    "# RATIONALE:",
    "# HACK:",
    "# TODO:",
    "# FIXME:",
)

_KNOWLEDGE_GRAPH_EDGE_TYPES = frozenset(
    {
        "imports",
        "uses",
        "inherits",
        "references",
        "rationale_for",
    }
)

_PYTHON_BUILTIN_CALLS = frozenset(dir(builtins))


def _has_unresolved_path_syntax(value: str) -> bool:
    if any(char in value for char in _PATH_PATTERN_METACHARS):
        return True
    return "$" in value


def _is_apex_harness_path_hint(value: str) -> bool:
    return Path(value).name in _APEX_HARNESS_BASENAMES


def _is_host_path_hint(value: str) -> bool:
    return value.startswith(_HOST_PATH_PREFIXES) or any(
        marker in value for marker in _HOST_PATH_MARKERS
    )


@dataclass
class SymbolInfo:
    """Information about a code symbol."""

    name: str
    kind: str
    file_path: str
    line_number: int
    signature: str
    docstring: Optional[str] = None
    parent_class: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent_class": self.parent_class,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolInfo":
        payload = dict(data)
        if "file" in payload and "file_path" not in payload:
            payload["file_path"] = payload.pop("file")
        if "line" in payload and "line_number" not in payload:
            payload["line_number"] = payload.pop("line")
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in valid_keys})


@dataclass
class FileInfo:
    """Information about one source file."""

    path: str
    size_bytes: int
    line_count: int
    language: str
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "language": self.language,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "imports": list(self.imports),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileInfo":
        return cls(
            path=data["path"],
            size_bytes=data["size_bytes"],
            line_count=data["line_count"],
            language=data["language"],
            symbols=[SymbolInfo.from_dict(item) for item in data.get("symbols", [])],
            imports=list(data.get("imports", [])),
        )


@dataclass
class GraphNode:
    """Node in the structure-augmented repository graph."""

    id: str
    node_type: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    code: str
    docstring: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "docstring": self.docstring,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNode":
        payload = dict(data)
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass
class GraphEdge:
    """Directed graph edge."""

    source_id: str
    target_id: str
    edge_type: str
    confidence: str = "EXTRACTED"
    weight: float = 1.0
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "confidence": self.confidence,
            "weight": self.weight,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "context": self.context,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEdge":
        payload = dict(data)
        payload.setdefault("confidence", "EXTRACTED")
        payload.setdefault("weight", 1.0)
        payload.setdefault("source_file", None)
        payload.setdefault("line_number", None)
        payload.setdefault("context", None)
        payload.setdefault("metadata", {})
        return cls(**payload)


class RepoGraph:
    """Repository graph with contain and use edges."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._reverse_adjacency: dict[str, list[str]] = defaultdict(list)
        self._edge_keys: set[tuple[str, str, str]] = set()

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        confidence: str = "EXTRACTED",
        weight: float = 1.0,
        source_file: Optional[str] = None,
        line_number: Optional[int] = None,
        context: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        key = (source_id, target_id, edge_type)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                confidence=confidence,
                weight=weight,
                source_file=source_file,
                line_number=line_number,
                context=context,
                metadata=dict(metadata or {}),
            )
        )
        self._adjacency[source_id].append(target_id)
        self._reverse_adjacency[target_id].append(source_id)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def neighbors(self, node_id: str, edge_type: Optional[str] = None) -> list[GraphNode]:
        results = []
        for target_id in self._adjacency.get(node_id, []):
            node = self.nodes.get(target_id)
            if not node:
                continue
            if edge_type and not self._has_edge(node_id, target_id, edge_type):
                continue
            results.append(node)
        return results

    def reverse_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> list[GraphNode]:
        results = []
        for source_id in self._reverse_adjacency.get(node_id, []):
            node = self.nodes.get(source_id)
            if not node:
                continue
            if edge_type and not self._has_edge(source_id, node_id, edge_type):
                continue
            results.append(node)
        return results

    def find_entities(self, name: str) -> list[GraphNode]:
        lowered = name.lower()
        results = []
        for node in self.nodes.values():
            if node.node_type == "file":
                continue
            if node.name.lower() == lowered or node.name.lower().endswith(f".{lowered}"):
                results.append(node)
        return sorted(results, key=lambda item: (item.file_path, item.start_line, item.name))

    def edge_records(
        self,
        node_id: str,
        *,
        edge_types: Optional[Iterable[str]] = None,
        direction: str = "both",
    ) -> list[GraphEdge]:
        allowed = set(edge_types or [])
        records: list[GraphEdge] = []
        for edge in self.edges:
            if allowed and edge.edge_type not in allowed:
                continue
            if direction in {"out", "both"} and edge.source_id == node_id:
                records.append(edge)
            elif direction in {"in", "both"} and edge.target_id == node_id:
                records.append(edge)
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoGraph":
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(GraphNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            edge = GraphEdge.from_dict(edge_data)
            graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                confidence=edge.confidence,
                weight=edge.weight,
                source_file=edge.source_file,
                line_number=edge.line_number,
                context=edge.context,
                metadata=edge.metadata,
            )
        return graph

    def _has_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        return (source_id, target_id, edge_type) in self._edge_keys


@dataclass
class RepoContext:
    """Repository context shared by all rollouts."""

    repo_path: str
    repo_tree: str
    files: list[FileInfo] = field(default_factory=list)
    symbol_index: dict[str, list[SymbolInfo]] = field(default_factory=dict)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    repo_graph: RepoGraph = field(default_factory=RepoGraph)
    _file_index: dict[str, FileInfo] = field(default_factory=dict, init=False, repr=False)
    _affinity_cache: dict[tuple[str, ...], dict[str, dict[str, float]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _partition_interface_cache: dict[tuple[tuple[str, ...], ...], dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._file_index = {file_info.path: file_info for file_info in self.files}

    def add_file(self, file_info: FileInfo) -> None:
        self.files.append(file_info)
        self._file_index[file_info.path] = file_info
        self._affinity_cache.clear()
        self._partition_interface_cache.clear()
        self.dependency_graph[file_info.path] = list(file_info.imports)
        for symbol in file_info.symbols:
            self.symbol_index.setdefault(symbol.name, []).append(symbol)

    def get_file_info(self, path: str) -> Optional[FileInfo]:
        return self._file_index.get(path)

    def normalize_repo_path_candidate(self, value: str) -> Optional[str]:
        text = str(value or "").split("::", 1)[0].strip()
        if not text:
            return None
        return self._normalize_repo_path_candidate(text)

    def get_repo_map(
        self,
        max_symbols_per_file: int = 8,
        focus_files: Optional[Iterable[str]] = None,
    ) -> str:
        if focus_files:
            allowed = set(focus_files)
            file_infos = [file for file in self.files if file.path in allowed]
        else:
            file_infos = list(self.files)

        lines = ["# Repository Map", ""]
        for file_info in sorted(file_infos, key=lambda item: item.path):
            lines.append(f"## {file_info.path} ({file_info.line_count} lines)")
            for symbol in file_info.symbols[:max_symbols_per_file]:
                prefix = f"{symbol.parent_class}." if symbol.parent_class else ""
                doc = ""
                if symbol.docstring:
                    doc_lines = [
                        line.strip() for line in symbol.docstring.splitlines() if line.strip()
                    ]
                    if doc_lines:
                        doc = f" - {doc_lines[0][:80]}"
                lines.append(f"- {prefix}{symbol.signature}{doc}")
            if len(file_info.symbols) > max_symbols_per_file:
                remaining = len(file_info.symbols) - max_symbols_per_file
                lines.append(f"- ... and {remaining} more symbols")
            lines.append("")
        return "\n".join(lines).strip()

    def build_focus_map(
        self,
        focus_files: Iterable[str],
        max_symbols_per_file: int = 10,
        include_neighbors: bool = True,
    ) -> str:
        focus = list(dict.fromkeys(focus_files))
        if include_neighbors:
            focus.extend(self.get_dependency_neighbors(focus, max_neighbors=6))
        return self.get_repo_map(
            max_symbols_per_file=max_symbols_per_file,
            focus_files=list(dict.fromkeys(focus)),
        )

    def build_context_pack(
        self,
        focus_files: Iterable[str],
        max_symbols_per_file: int = 8,
        max_files: int = 14,
        include_neighbors: bool = True,
        include_related_tests: bool = True,
        include_knowledge_graph: bool = True,
        seed_symbols: Optional[Iterable[str]] = None,
    ) -> str:
        """Build a compact Aider-style structural map for agent prompts.

        The legacy focus map only listed signatures. This pack keeps the same
        compact shape while adding the signals that help agents navigate larger
        repositories: imports, direct call edges, dependency neighbors, and
        related tests. It is a context/ranking aid only; it is not an edit
        boundary.
        """

        normalized_focus = [
            path
            for value in focus_files
            if (path := self.normalize_repo_path_candidate(str(value or "")))
        ]
        focus = list(dict.fromkeys(normalized_focus))
        neighbors = self.get_dependency_neighbors(focus, max_neighbors=8) if include_neighbors else []
        related_tests = (
            self.find_related_tests(focus, max_files=6)
            if include_related_tests and focus
            else []
        )
        ordered_files = list(dict.fromkeys(focus + related_tests + neighbors))[: max(1, max_files)]
        if not ordered_files:
            return self.get_repo_map(max_symbols_per_file=max_symbols_per_file)

        focus_set = set(focus)
        related_test_set = set(related_tests)
        neighbor_set = set(neighbors)
        lines = [
            "# Repository Map",
            "Context pack: ranked files with signatures, imports, call edges, and related tests.",
            "",
            "## Ranked Files",
        ]
        for path in ordered_files:
            role = "focus"
            if path in related_test_set and path not in focus_set:
                role = "related-test"
            elif path in neighbor_set and path not in focus_set:
                role = "dependency-neighbor"
            lines.append(f"- {path} ({role})")
        lines.append("")

        if include_knowledge_graph:
            graph_pack = self.build_knowledge_graph_pack(
                ordered_files,
                seed_symbols=seed_symbols,
                max_nodes=14,
                max_edges=18,
            )
            if graph_pack:
                lines.extend([graph_pack, ""])

        for path in ordered_files:
            file_info = self.get_file_info(path)
            if file_info is None:
                continue
            lines.append(f"## {file_info.path} ({file_info.line_count} lines, {file_info.language})")
            if file_info.imports:
                imports = ", ".join(file_info.imports[:6])
                suffix = f", ... +{len(file_info.imports) - 6}" if len(file_info.imports) > 6 else ""
                lines.append(f"imports: {imports}{suffix}")
            for symbol in file_info.symbols[:max_symbols_per_file]:
                prefix = f"{symbol.parent_class}." if symbol.parent_class else ""
                doc = ""
                if symbol.docstring:
                    doc_lines = [
                        line.strip() for line in symbol.docstring.splitlines() if line.strip()
                    ]
                    if doc_lines:
                        doc = f" - {doc_lines[0][:80]}"
                lines.append(f"- {prefix}{symbol.signature}{doc}")
            if len(file_info.symbols) > max_symbols_per_file:
                lines.append(f"- ... and {len(file_info.symbols) - max_symbols_per_file} more symbols")
            relationship_lines = self._context_pack_relationship_lines(path, max_lines=5)
            if relationship_lines:
                lines.append("relationships:")
                lines.extend(f"- {line}" for line in relationship_lines)
            lines.append("")
        return "\n".join(lines).strip()

    def build_knowledge_graph_pack(
        self,
        focus_files: Iterable[str],
        *,
        seed_symbols: Optional[Iterable[str]] = None,
        max_nodes: int = 14,
        max_edges: int = 18,
    ) -> str:
        """Render a compact Graphify-style structural neighborhood.

        This is a navigation signal for rollout creation, not a validity rule.
        It keeps source-grounded relationships explicit so agents can make
        surgical edits without treating localization as a hard boundary.
        """

        focus = [
            path
            for value in focus_files
            if (path := self.normalize_repo_path_candidate(str(value or "")))
        ]
        focus = list(dict.fromkeys(focus))
        symbols = [
            str(symbol or "").strip()
            for symbol in list(seed_symbols or [])
            if str(symbol or "").strip()
        ]
        if not focus and not symbols:
            return ""

        node_scores: dict[str, float] = {}
        seed_node_ids: set[str] = set()

        def add_node(node: Optional[GraphNode], score: float) -> None:
            if node is None:
                return
            node_scores[node.id] = max(score, node_scores.get(node.id, 0.0))

        for path in focus:
            file_node = self.repo_graph.get_node(f"file:{path}")
            add_node(file_node, 18.0)
            if file_node is None:
                continue
            seed_node_ids.add(file_node.id)
            for child in self.repo_graph.neighbors(file_node.id, edge_type="contains"):
                if child.node_type == "rationale":
                    add_node(child, 5.0)
                else:
                    add_node(child, 14.0)
                    seed_node_ids.add(child.id)

        for symbol in symbols:
            for node in self.lookup_definition(symbol)[:4]:
                add_node(node, 20.0)
                seed_node_ids.add(node.id)
                file_node = self.repo_graph.get_node(f"file:{node.file_path}")
                add_node(file_node, 12.0)

        for node_id in list(seed_node_ids):
            for edge in self.repo_graph.edge_records(
                node_id,
                edge_types=_KNOWLEDGE_GRAPH_EDGE_TYPES,
                direction="both",
            ):
                source = self.repo_graph.get_node(edge.source_id)
                target = self.repo_graph.get_node(edge.target_id)
                relation_weight = {
                    "imports": 7.0,
                    "uses": 6.0,
                    "inherits": 6.5,
                    "references": 4.5,
                    "rationale_for": 3.5,
                }.get(edge.edge_type, 2.0)
                add_node(source, relation_weight)
                add_node(target, relation_weight)
                if source is not None:
                    add_node(self.repo_graph.get_node(f"file:{source.file_path}"), 3.0)
                if target is not None:
                    add_node(self.repo_graph.get_node(f"file:{target.file_path}"), 3.0)

        if not node_scores:
            return ""

        ranked_node_ids = [
            node_id
            for node_id, _ in sorted(
                node_scores.items(),
                key=lambda item: (-item[1], self.repo_graph.nodes[item[0]].file_path, item[0]),
            )
        ]
        selected_nodes = [
            self.repo_graph.get_node(node_id)
            for node_id in ranked_node_ids[: max(1, max_nodes)]
        ]
        selected_nodes = [node for node in selected_nodes if node is not None]
        selected_ids = {node.id for node in selected_nodes}
        focus_set = set(focus)

        core_nodes = [
            self.repo_graph.nodes[node_id]
            for node_id in ranked_node_ids
            if self.repo_graph.nodes[node_id].node_type not in {"file", "rationale"}
        ][:8]
        rationale_nodes = [
            self.repo_graph.nodes[node_id]
            for node_id in ranked_node_ids
            if self.repo_graph.nodes[node_id].node_type == "rationale"
        ][:4]
        edge_lines = self._knowledge_graph_edge_lines(
            selected_ids | seed_node_ids,
            focus_set=focus_set,
            max_edges=max_edges,
        )

        lines = [
            "## Code Knowledge Graph",
            (
                "Structural relationships are extracted from source and used as "
                "context only; they are not edit boundaries."
            ),
        ]
        if core_nodes:
            lines.append("Core symbols:")
            for node in core_nodes:
                detail = self._knowledge_graph_node_detail(node)
                lines.append(f"- {self._format_graph_node_ref(node)}{detail}")
        if edge_lines:
            lines.append("Structural edges:")
            lines.extend(f"- {line}" for line in edge_lines)
        if rationale_nodes:
            lines.append("Source rationale:")
            for node in rationale_nodes:
                snippet = " ".join(str(node.code or node.name).split())
                lines.append(f"- {node.file_path}:{node.start_line} {snippet[:140]}")
        return "\n".join(lines).strip()

    def to_knowledge_graph_dict(self) -> dict[str, Any]:
        """Return a Graphify-compatible node/edge payload for diagnostics."""

        relation_map = {"uses": "calls"}
        nodes = []
        for node in self.repo_graph.nodes.values():
            file_type = "rationale" if node.node_type == "rationale" else "code"
            nodes.append(
                {
                    "id": node.id,
                    "label": node.name,
                    "file_type": file_type,
                    "node_type": node.node_type,
                    "source_file": node.file_path,
                    "source_location": f"L{node.start_line}",
                    "end_location": f"L{node.end_line}",
                    "metadata": dict(node.metadata),
                }
            )
        edges = []
        for edge in self.repo_graph.edges:
            edges.append(
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "relation": relation_map.get(edge.edge_type, edge.edge_type),
                    "confidence": edge.confidence,
                    "confidence_score": edge.weight,
                    "source_file": edge.source_file,
                    "source_location": f"L{edge.line_number}" if edge.line_number else None,
                    "context": edge.context,
                    "metadata": dict(edge.metadata),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def _knowledge_graph_node_detail(self, node: GraphNode) -> str:
        calls = self._context_pack_node_refs(
            self.repo_graph.neighbors(node.id, edge_type="uses"),
            exclude_file="",
            limit=2,
        )
        callers = self._context_pack_node_refs(
            self.repo_graph.reverse_neighbors(node.id, edge_type="uses"),
            exclude_file="",
            limit=2,
        )
        inherits = self._context_pack_node_refs(
            self.repo_graph.neighbors(node.id, edge_type="inherits"),
            exclude_file="",
            limit=2,
        )
        parts = []
        if inherits:
            parts.append("inherits " + ", ".join(inherits))
        if calls:
            parts.append("calls " + ", ".join(calls))
        if callers:
            parts.append("called by " + ", ".join(callers))
        return f" ({'; '.join(parts)})" if parts else ""

    def _knowledge_graph_edge_lines(
        self,
        selected_ids: set[str],
        *,
        focus_set: set[str],
        max_edges: int,
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        relation_weight = {
            "imports": 8.0,
            "uses": 7.0,
            "inherits": 7.0,
            "references": 5.0,
            "rationale_for": 3.5,
        }
        for edge in self.repo_graph.edges:
            if edge.edge_type not in _KNOWLEDGE_GRAPH_EDGE_TYPES:
                continue
            source = self.repo_graph.get_node(edge.source_id)
            target = self.repo_graph.get_node(edge.target_id)
            if source is None or target is None:
                continue
            if edge.source_id not in selected_ids and edge.target_id not in selected_ids:
                continue
            if edge.edge_type == "imports" and source.file_path not in focus_set:
                continue
            if edge.edge_type == "rationale_for":
                continue
            score = relation_weight.get(edge.edge_type, 1.0)
            if source.file_path in focus_set or target.file_path in focus_set:
                score += 2.0
            if source.file_path != target.file_path:
                score += 1.5
            confidence = f" [{edge.confidence}]" if edge.confidence != "EXTRACTED" else ""
            location = f":{edge.line_number}" if edge.line_number else ""
            scored.append(
                (
                    score,
                    (
                        f"{self._format_graph_node_ref(source)} --{edge.edge_type}--> "
                        f"{self._format_graph_node_ref(target)}{confidence}{location}"
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [line for _, line in scored[:max_edges]]

    @staticmethod
    def _format_graph_node_ref(node: GraphNode) -> str:
        if node.node_type == "file":
            return node.file_path
        return f"{node.name} @ {node.file_path}:{node.start_line}"

    def _context_pack_relationship_lines(self, path: str, *, max_lines: int) -> list[str]:
        nodes = sorted(
            [
                node
                for node in self.repo_graph.nodes.values()
                if node.file_path == path and node.node_type != "file"
            ],
            key=lambda item: (item.start_line, item.name),
        )
        lines: list[str] = []
        for node in nodes:
            callees = self._context_pack_node_refs(
                self.repo_graph.neighbors(node.id, edge_type="uses"),
                exclude_file=path,
                limit=3,
            )
            callers = self._context_pack_node_refs(
                self.repo_graph.reverse_neighbors(node.id, edge_type="uses"),
                exclude_file=path,
                limit=3,
            )
            if callees:
                lines.append(f"{node.name} calls " + ", ".join(callees))
            if callers:
                lines.append(f"{node.name} called by " + ", ".join(callers))
            if len(lines) >= max_lines:
                break
        return lines[:max_lines]

    @staticmethod
    def _context_pack_node_refs(
        nodes: Iterable[GraphNode],
        *,
        exclude_file: str,
        limit: int,
    ) -> list[str]:
        refs: list[str] = []
        for node in sorted(nodes, key=lambda item: (item.file_path, item.start_line, item.name)):
            if node.file_path == exclude_file:
                continue
            refs.append(f"{node.name} @ {node.file_path}:{node.start_line}")
            if len(refs) >= limit:
                break
        return refs

    def get_dependency_neighbors(
        self,
        seed_files: Iterable[str],
        max_neighbors: int = 10,
    ) -> list[str]:
        seeds = [item for item in seed_files if item]
        if not seeds:
            return []

        reverse_graph: dict[str, list[str]] = {}
        for file_path, imports in self.dependency_graph.items():
            for imported in imports:
                reverse_graph.setdefault(imported, []).append(file_path)

        scored: list[tuple[int, str]] = []
        seed_set = set(seeds)
        for file_info in self.files:
            if file_info.path in seed_set:
                continue
            score = 0
            imports_blob = " ".join(file_info.imports)
            for seed in seeds:
                seed_stem = Path(seed).stem
                if seed_stem and seed_stem in imports_blob:
                    score += 3
                if seed_stem and any(seed_stem in imported for imported in file_info.imports):
                    score += 2
                if file_info.path in reverse_graph.get(seed_stem, []):
                    score += 1
                if Path(file_info.path).parent == Path(seed).parent:
                    score += 1
            if score > 0:
                scored.append((score, file_info.path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:max_neighbors]]

    def find_related_tests(
        self,
        seed_files: Iterable[str],
        *,
        seed_symbols: Optional[Iterable[str]] = None,
        issue_description: str = "",
        max_files: int = 6,
    ) -> list[str]:
        normalized_seed_files = list(
            dict.fromkeys(
                path for value in seed_files if (path := self.normalize_repo_path_candidate(value))
            )
        )
        normalized_seed_symbols = [
            str(value or "").strip()
            for value in list(seed_symbols or [])
            if str(value or "").strip()
        ]
        issue_paths = self._extract_issue_repo_path_mentions(issue_description)
        issue_keywords = self.extract_issue_keywords(issue_description, max_keywords=12)
        if (
            not normalized_seed_files
            and not normalized_seed_symbols
            and not issue_paths
            and not issue_keywords
        ):
            return []

        seed_symbol_tokens: set[str] = set()
        seed_path_tokens: dict[str, set[str]] = {}
        seed_file_symbols: dict[str, set[str]] = {}
        seed_module_aliases: dict[str, list[str]] = {}
        for seed in normalized_seed_files:
            seed_path_tokens[seed] = self._path_affinity_tokens(seed)
            seed_file_symbols[seed] = self._file_symbol_tokens(seed)
            seed_module_aliases[seed] = self._module_aliases_for_path(seed)
            seed_symbol_tokens.update(seed_path_tokens[seed])
            seed_symbol_tokens.update(seed_file_symbols[seed])
        for symbol in normalized_seed_symbols:
            tail = str(symbol).split("::")[-1]
            for chunk in re.split(r"[^A-Za-z0-9_.]+", tail):
                piece = chunk.split(".")[-1].strip()
                if not piece:
                    continue
                seed_symbol_tokens.update(self._symbol_name_candidates(piece))

        scored: list[tuple[float, str]] = []
        for file_info in self.files:
            if not self._looks_like_test_path(file_info.path):
                continue
            candidate_tokens = self._path_affinity_tokens(
                file_info.path
            ) | self._file_symbol_tokens(file_info.path)
            imports_text = " ".join(file_info.imports).lower()
            symbol_text = " ".join(symbol.name.lower() for symbol in file_info.symbols)
            score = 0.0

            if file_info.path in normalized_seed_files:
                score += 12.0

            for seed in normalized_seed_files:
                if file_info.path == seed:
                    continue
                shared_stem_weight = self._test_source_link_weight(file_info.path, seed)
                if shared_stem_weight > 0:
                    score += 2.5 * shared_stem_weight
                shared_tokens = candidate_tokens.intersection(
                    seed_path_tokens.get(seed, set()) | seed_file_symbols.get(seed, set())
                )
                if shared_tokens:
                    score += min(4.0, 1.2 * len(shared_tokens))
                if Path(file_info.path).parent == Path(seed).parent:
                    score += 0.6
                if any(
                    alias.lower() in imports_text or alias.lower() in symbol_text
                    for alias in seed_module_aliases.get(seed, [])
                    if alias
                ):
                    score += 1.5

            if seed_symbol_tokens:
                shared_seed_symbols = candidate_tokens.intersection(seed_symbol_tokens)
                if shared_seed_symbols:
                    score += min(4.0, 1.3 * len(shared_seed_symbols))

            for issue_path in issue_paths:
                if file_info.path == issue_path:
                    score += 8.0
                shared_issue_tokens = candidate_tokens.intersection(
                    self._path_affinity_tokens(issue_path)
                )
                if shared_issue_tokens:
                    score += min(2.0, 0.8 * len(shared_issue_tokens))

            path_lower = file_info.path.lower()
            for keyword in issue_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in path_lower:
                    score += 0.8
                if keyword_lower in symbol_text:
                    score += 0.4

            if score > 0:
                scored.append((score, file_info.path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:max_files]]

    def build_file_affinity_graph(
        self,
        candidate_files: Iterable[str],
    ) -> dict[str, dict[str, float]]:
        files = [
            path
            for path in dict.fromkeys(str(item) for item in candidate_files if item)
            if self.get_file_info(path) is not None
        ]
        adjacency: dict[str, dict[str, float]] = {path: {} for path in files}
        if not files:
            return adjacency
        cache_key = tuple(sorted(files))
        cached = self._affinity_cache.get(cache_key)
        if cached is not None:
            return cached

        candidate_set = set(files)
        module_aliases = self._candidate_module_aliases(files)
        symbol_tokens = {path: self._file_symbol_tokens(path) for path in files}
        path_tokens = {path: self._path_affinity_tokens(path) for path in files}

        def connect(left: str, right: str, weight: float) -> None:
            if left == right or weight <= 0:
                return
            adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
            adjacency[right][left] = adjacency[right].get(left, 0.0) + weight

        for path in files:
            file_info = self.get_file_info(path)
            if file_info is None:
                continue
            for imported in file_info.imports or []:
                resolved = self._resolve_import_to_candidate_file(imported, module_aliases)
                if resolved and resolved in candidate_set and resolved != path:
                    connect(path, resolved, 3.0)

        for edge in self.repo_graph.edges:
            if edge.edge_type != "uses":
                continue
            source = self.repo_graph.get_node(edge.source_id)
            target = self.repo_graph.get_node(edge.target_id)
            if source is None or target is None or source.file_path == target.file_path:
                continue
            if source.file_path in candidate_set and target.file_path in candidate_set:
                connect(source.file_path, target.file_path, 4.0)

        ordered_files = list(files)
        for index, left in enumerate(ordered_files):
            left_path = Path(left)
            for right in ordered_files[index + 1 :]:
                right_path = Path(right)
                heuristic_weight = 0.0
                if left_path.parent == right_path.parent:
                    heuristic_weight += 0.35
                elif left_path.parts[:1] and left_path.parts[:1] == right_path.parts[:1]:
                    heuristic_weight += 0.15
                shared_symbols = symbol_tokens.get(left, set()) & symbol_tokens.get(right, set())
                if shared_symbols:
                    heuristic_weight += min(1.8, 0.30 * len(shared_symbols))
                shared_path_tokens = path_tokens.get(left, set()) & path_tokens.get(right, set())
                if shared_path_tokens:
                    heuristic_weight += min(0.9, 0.12 * len(shared_path_tokens))
                heuristic_weight += self._test_source_link_weight(left, right)
                heuristic_weight += self._build_adjacency_weight(left, right)
                if heuristic_weight > 0:
                    connect(left, right, heuristic_weight)

        self._affinity_cache[cache_key] = adjacency
        return adjacency

    def describe_partition_interfaces(
        self,
        clusters: Iterable[Iterable[str]],
    ) -> dict[str, Any]:
        normalized_clusters = [
            [
                path
                for path in dict.fromkeys(
                    str(item).strip() for item in cluster if str(item).strip()
                )
                if self.get_file_info(path) is not None
            ]
            for cluster in clusters
        ]
        normalized_clusters = [cluster for cluster in normalized_clusters if cluster]
        cache_key = tuple(tuple(cluster) for cluster in normalized_clusters)
        cached = self._partition_interface_cache.get(cache_key)
        if cached is not None:
            return cached
        cluster_index = {
            path: index for index, cluster in enumerate(normalized_clusters) for path in cluster
        }
        if not cluster_index:
            result: dict[str, Any] = {
                "bridge_files": [],
                "interface_symbols": [],
                "cluster_hints": [],
            }
            self._partition_interface_cache[cache_key] = result
            return result

        all_files = list(cluster_index)
        module_aliases = self._candidate_module_aliases(all_files)
        symbol_names: dict[str, set[str]] = {}
        for path in all_files:
            file_info = self.get_file_info(path)
            symbol_names[path] = {
                symbol.name.lower()
                for symbol in (file_info.symbols if file_info is not None else [])
                if symbol.name
            }
        bridge_files: set[str] = set()
        interface_symbols: Counter[str] = Counter()
        cluster_bridge_files: list[set[str]] = [set() for _ in normalized_clusters]
        cluster_interface_symbols: list[Counter[str]] = [Counter() for _ in normalized_clusters]

        def record_bridge(left: str, right: str, *tokens: str) -> None:
            left_index = cluster_index.get(left)
            right_index = cluster_index.get(right)
            if left_index is None or right_index is None or left_index == right_index:
                return
            bridge_files.update({left, right})
            cluster_bridge_files[left_index].update({left, right})
            cluster_bridge_files[right_index].update({left, right})
            for token in tokens:
                for normalized in self._symbol_name_candidates(token):
                    interface_symbols[normalized] += 1
                    cluster_interface_symbols[left_index][normalized] += 1
                    cluster_interface_symbols[right_index][normalized] += 1

        for edge in self.repo_graph.edges:
            if edge.edge_type != "uses":
                continue
            source = self.repo_graph.get_node(edge.source_id)
            target = self.repo_graph.get_node(edge.target_id)
            if source is None or target is None or source.file_path == target.file_path:
                continue
            record_bridge(source.file_path, target.file_path, source.name, target.name)

        for path in all_files:
            file_info = self.get_file_info(path)
            if file_info is None:
                continue
            for imported in file_info.imports or []:
                resolved = self._resolve_import_to_candidate_file(imported, module_aliases)
                if resolved and resolved != path:
                    record_bridge(path, resolved, imported)

        ordered_files = list(all_files)
        for index, left in enumerate(ordered_files):
            for right in ordered_files[index + 1 :]:
                shared_symbol_names = symbol_names.get(left, set()) & symbol_names.get(right, set())
                if shared_symbol_names:
                    record_bridge(left, right, *sorted(shared_symbol_names)[:4])
                shared_test_source_stem = self._shared_test_source_stem(left, right)
                if shared_test_source_stem:
                    record_bridge(left, right, shared_test_source_stem)

        cluster_hints = [
            {
                "bridge_files": sorted(paths),
                "interface_symbols": [token for token, _ in counter.most_common(6)],
                "peer_files": sorted(
                    {
                        path
                        for peer_index, cluster in enumerate(normalized_clusters)
                        if peer_index != index
                        for path in cluster
                    }
                ),
            }
            for index, (paths, counter) in enumerate(
                zip(cluster_bridge_files, cluster_interface_symbols, strict=False)
            )
        ]
        result = {
            "bridge_files": sorted(bridge_files),
            "interface_symbols": [token for token, _ in interface_symbols.most_common(8)],
            "cluster_hints": cluster_hints,
        }
        self._partition_interface_cache[cache_key] = result
        return result

    def _candidate_module_aliases(
        self,
        candidate_files: Iterable[str],
    ) -> dict[str, Optional[str]]:
        aliases: dict[str, Optional[str]] = {}

        def register(alias: str, path: str) -> None:
            normalized = str(alias or "").strip(". ")
            if not normalized:
                return
            existing = aliases.get(normalized)
            if existing is None and normalized in aliases:
                return
            if existing is not None and existing != path:
                aliases[normalized] = None
                return
            aliases[normalized] = path

        for path in candidate_files:
            rel_path = Path(path)
            # `Path('/').with_suffix("")` raises `ValueError: PosixPath('/')
            # has an empty name`. Guard against root / empty inputs.
            if not rel_path.name:
                continue
            module_parts = list(rel_path.with_suffix("").parts)
            if not module_parts:
                continue
            if module_parts[-1] == "__init__":
                package_parts = module_parts[:-1]
                if package_parts:
                    register(".".join(package_parts), path)
                    register(package_parts[-1], path)
                continue
            register(".".join(module_parts), path)
            register(module_parts[-1], path)
            if len(module_parts) >= 2:
                register(".".join(module_parts[-2:]), path)
        return aliases

    def _resolve_import_to_candidate_file(
        self,
        imported: str,
        module_aliases: dict[str, Optional[str]],
    ) -> Optional[str]:
        parts = [part for part in str(imported or "").split(".") if part]
        if not parts:
            return None

        for width in range(len(parts), 0, -1):
            candidate = ".".join(parts[:width])
            if candidate in module_aliases:
                return module_aliases[candidate]
        for start in range(1, len(parts)):
            candidate = ".".join(parts[start:])
            if candidate in module_aliases:
                return module_aliases[candidate]
        return None

    def _file_symbol_tokens(self, path: str) -> set[str]:
        file_info = self.get_file_info(path)
        tokens: set[str] = set()
        if file_info is None:
            return tokens
        for symbol in file_info.symbols:
            tokens.update(self._symbol_name_candidates(symbol.name))
            if symbol.parent_class:
                tokens.update(self._symbol_name_candidates(symbol.parent_class))
        return tokens

    def _path_affinity_tokens(self, path: str) -> set[str]:
        tokens: set[str] = set()
        target = Path(path)
        if not target.name:
            return tokens
        for part in target.with_suffix("").parts:
            tokens.update(self._split_identifier_tokens(part))
        return {token for token in tokens if token not in _AFFINITY_TOKEN_NOISE}

    def _build_adjacency_weight(self, left: str, right: str) -> float:
        left_path = Path(left)
        right_path = Path(right)
        weight = 0.0
        if left_path.name == "__init__.py" and left_path.parent == right_path.parent:
            weight += 0.9
        if right_path.name == "__init__.py" and right_path.parent == left_path.parent:
            weight += 0.9
        build_files = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "package.json",
            "cargo.toml",
            "go.mod",
            "makefile",
        }
        left_build = left_path.name.lower() in build_files
        right_build = right_path.name.lower() in build_files
        if left_build != right_build:
            build_path = left_path if left_build else right_path
            other_path = right_path if left_build else left_path
            if build_path.parent == other_path.parent or build_path.parent in other_path.parents:
                weight += 0.65
            elif not build_path.parent.parts:
                weight += 0.35
        return weight

    def _looks_like_test_path(self, path: str) -> bool:
        normalized = str(path or "").strip().replace("\\", "/")
        name = Path(normalized).name.lower()
        return (
            "tests" in {part.lower() for part in Path(normalized).parts}
            or "test" in {part.lower() for part in Path(normalized).parts}
            or "spec" in {part.lower() for part in Path(normalized).parts}
            or "__tests__" in {part.lower() for part in Path(normalized).parts}
            or "testdata" in {part.lower() for part in Path(normalized).parts}
            or name == "conftest.py"
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith("_test.go")
            or name.endswith("_test.rs")
            or name.endswith("_spec.rb")
            or ".test." in name
            or ".spec." in name
            or re.search(r"(?:^|[._-])test[s]?\.(?:java|kt|kts|scala|cs)$", name) is not None
            or re.search(r"(?:^|[._-])spec\.(?:java|kt|kts|scala|cs)$", name) is not None
        )

    def _shared_test_source_stem(self, left: str, right: str) -> Optional[str]:
        left_is_test = self._looks_like_test_path(left)
        right_is_test = self._looks_like_test_path(right)
        if left_is_test == right_is_test:
            return None
        test_path = Path(left if left_is_test else right)
        source_path = Path(right if left_is_test else left)
        test_tokens = self._split_identifier_tokens(test_path.stem)
        source_tokens = self._split_identifier_tokens(source_path.stem)
        stripped_test_tokens = {
            token for token in test_tokens if token not in {"test", "tests", "spec", "specs"}
        }
        shared = sorted(stripped_test_tokens & source_tokens)
        if shared:
            return shared[0]
        normalized_test = re.sub(r"^(test_|spec_)", "", test_path.stem.lower())
        normalized_test = re.sub(r"(_test|_spec)$", "", normalized_test)
        source_stem = source_path.stem.lower()
        if normalized_test and normalized_test == source_stem:
            return normalized_test
        return None

    def _test_source_link_weight(self, left: str, right: str) -> float:
        shared_stem = self._shared_test_source_stem(left, right)
        if not shared_stem:
            return 0.0
        return 2.4 if len(shared_stem) >= 4 else 1.6

    def _normalize_repo_path_candidate(self, value: str) -> Optional[str]:
        text = str(value or "").strip().strip("`'\"").replace("\\", "/")
        if not text:
            return None
        if _has_unresolved_path_syntax(text):
            return None
        if any(part == ".." for part in Path(text).parts):
            return None
        path = Path(text)
        if path.is_absolute():
            repo_root = Path(self.repo_path).resolve()
            resolved = path.resolve(strict=False)
            try:
                text = resolved.relative_to(repo_root).as_posix()
            except ValueError:
                if _is_host_path_hint(text):
                    return None
        else:
            while text.startswith("./"):
                text = text[2:]
        if _is_apex_harness_path_hint(text):
            return None
        if _is_host_path_hint(text):
            return None
        if text in self._file_index:
            return text
        for known_path in self._file_index:
            if text.endswith(known_path) or known_path.endswith(text):
                return known_path
        return None

    def _extract_issue_repo_path_mentions(self, issue_description: str) -> list[str]:
        matches = re.findall(
            r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+",
            str(issue_description or ""),
        )
        normalized: list[str] = []
        for match in matches:
            path = self._normalize_repo_path_candidate(match)
            if path:
                normalized.append(path)
        return list(dict.fromkeys(normalized))

    def _module_aliases_for_path(self, path: str) -> list[str]:
        target = Path(path)
        if not target.name:
            return []
        stem = target.stem
        module_parts = list(target.with_suffix("").parts)
        aliases: list[str] = []
        if module_parts:
            if module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            if module_parts:
                aliases.append(".".join(module_parts))
                aliases.append(module_parts[-1])
        if stem and stem != "__init__":
            aliases.append(stem)
        return list(dict.fromkeys(alias for alias in aliases if alias))

    def _split_identifier_tokens(self, text: str) -> set[str]:
        raw = str(text or "").strip().replace("-", "_")
        if not raw:
            return set()
        snake_tokens = [token for token in re.split(r"[^A-Za-z0-9]+", raw) if token]
        tokens: set[str] = set()
        for token in snake_tokens:
            camel_tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", token)
            for item in camel_tokens or [token]:
                normalized = item.lower()
                if len(normalized) >= 3 and normalized not in _AFFINITY_TOKEN_NOISE:
                    tokens.add(normalized)
        return tokens

    def _symbol_name_candidates(self, value: str) -> set[str]:
        tokens = self._split_identifier_tokens(str(value or "").split(".")[-1])
        lowered = str(value or "").strip().split(".")[-1].lower()
        if len(lowered) >= 3 and lowered not in _AFFINITY_TOKEN_NOISE:
            tokens.add(lowered)
        return tokens

    def extract_issue_keywords(self, issue_description: str, max_keywords: int = 12) -> list[str]:
        counts: dict[str, int] = {}
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", issue_description):
            normalized = token.lower().strip(".,:;()[]{}")
            if (
                normalized in _STOP_WORDS
                or normalized in _KEYWORD_NOISE_TOKENS
                or normalized.isdigit()
                or normalized.startswith("$")
                or normalized.startswith("--")
                or normalized.startswith("./")
                or normalized.startswith("../")
                or normalized.startswith("/")
                or "/.venv/" in normalized
                or normalized.endswith("/python")
                or normalized.endswith("/python3")
                or normalized.endswith(".json")
                or normalized.endswith(".txt")
            ):
                continue
            counts[normalized] = counts.get(normalized, 0) + 1

        explicit_keywords: list[str] = []
        for path in self._extract_issue_repo_path_mentions(issue_description):
            explicit_keywords.append(path)
            explicit_keywords.extend(self._module_aliases_for_path(path))

        ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        ordered: list[str] = []
        seen: set[str] = set()
        for token in explicit_keywords + [token for token, _ in ranked]:
            if not token or token in seen:
                continue
            ordered.append(token)
            seen.add(token)
            if len(ordered) >= max_keywords:
                break
        return ordered

    def get_relevant_files(self, keywords: list[str], max_files: int = 20) -> list[str]:
        scored: list[tuple[float, str]] = []
        explicit_paths = [
            path for keyword in keywords if (path := self._normalize_repo_path_candidate(keyword))
        ]
        explicit_path_signals = {
            path: {
                "stem_tokens": self._split_identifier_tokens(Path(path).stem),
                "module_aliases": self._module_aliases_for_path(path),
            }
            for path in explicit_paths
        }

        for file_info in self.files:
            path_lower = file_info.path.lower()
            symbol_chunks = [
                " ".join(
                    part
                    for part in [
                        symbol.name.lower(),
                        symbol.signature.lower(),
                        (symbol.docstring or "").lower(),
                    ]
                    if part
                )
                for symbol in file_info.symbols
            ]
            symbol_text = " ".join(symbol_chunks)
            imports_text = " ".join(file_info.imports).lower()
            graph_bonus = self._graph_keyword_score(file_info.path, keywords)
            score = 0.0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in path_lower:
                    score += 4.0
                if keyword_lower in symbol_text:
                    score += 2.0
                if keyword_lower in " ".join(file_info.imports).lower():
                    score += 1.0
                if "test" in path_lower and keyword_lower in path_lower:
                    score += 1.0
            for explicit_path, signal in explicit_path_signals.items():
                if file_info.path == explicit_path:
                    score += 12.0
                    continue
                shared_stem = self._shared_test_source_stem(file_info.path, explicit_path)
                if shared_stem:
                    score += 6.0 if len(shared_stem) >= 4 else 4.0
                stem_tokens = set(signal["stem_tokens"])
                if stem_tokens and stem_tokens.intersection(
                    self._path_affinity_tokens(file_info.path)
                ):
                    score += min(
                        3.0,
                        1.5
                        * len(stem_tokens.intersection(self._path_affinity_tokens(file_info.path))),
                    )
                if any(alias.lower() in imports_text for alias in signal["module_aliases"]):
                    score += 2.0
            score += graph_bonus
            if score > 0:
                scored.append((score, file_info.path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:max_files]]

    def lookup_definition(self, symbol_name: str) -> list[GraphNode]:
        return self.repo_graph.find_entities(symbol_name)

    def trace_callers(self, symbol_name: str) -> list[GraphNode]:
        callers: list[GraphNode] = []
        seen: set[str] = set()
        for target in self.lookup_definition(symbol_name):
            for caller in self.repo_graph.reverse_neighbors(target.id, edge_type="uses"):
                if caller.id not in seen:
                    callers.append(caller)
                    seen.add(caller.id)
        return sorted(callers, key=lambda item: (item.file_path, item.start_line, item.name))

    def trace_callees(self, symbol_name: str) -> list[GraphNode]:
        callees: list[GraphNode] = []
        seen: set[str] = set()
        for source in self.lookup_definition(symbol_name):
            for callee in self.repo_graph.neighbors(source.id, edge_type="uses"):
                if callee.id not in seen:
                    callees.append(callee)
                    seen.add(callee.id)
        return sorted(callees, key=lambda item: (item.file_path, item.start_line, item.name))

    def get_entity_context(self, entity_name: str) -> dict[str, Any]:
        matches = self.lookup_definition(entity_name)
        if not matches:
            return {}

        entity = matches[0]
        container = None
        parents = self.repo_graph.reverse_neighbors(entity.id, edge_type="contains")
        if parents:
            container = parents[0]

        siblings: list[GraphNode] = []
        if container:
            siblings = [
                node
                for node in self.repo_graph.neighbors(container.id, edge_type="contains")
                if node.id != entity.id
            ]

        return {
            "entity": entity.to_dict(),
            "container": container.to_dict() if container else None,
            "siblings": [node.to_dict() for node in siblings],
            "callers": [
                node.to_dict()
                for node in self.repo_graph.reverse_neighbors(entity.id, edge_type="uses")
            ],
            "callees": [
                node.to_dict() for node in self.repo_graph.neighbors(entity.id, edge_type="uses")
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "repo_path": self.repo_path,
                "repo_tree": self.repo_tree,
                "files": [file_info.to_dict() for file_info in self.files],
                "dependency_graph": self.dependency_graph,
                "repo_graph": self.repo_graph.to_dict(),
            },
            indent=2,
        )

    def save(self, path: str | Path) -> None:
        from apex.evaluation.checkpointing import atomic_write_text

        atomic_write_text(Path(path), self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "RepoContext":
        data = json.loads(Path(path).read_text())
        context = cls(
            repo_path=data["repo_path"],
            repo_tree=data["repo_tree"],
            dependency_graph=data.get("dependency_graph", {}),
            repo_graph=RepoGraph.from_dict(data.get("repo_graph", {})),
        )
        for file_payload in data.get("files", []):
            context.add_file(FileInfo.from_dict(file_payload))
        return context

    def _graph_keyword_score(self, file_path: str, keywords: list[str]) -> float:
        file_node = self.repo_graph.get_node(f"file:{file_path}")
        if not file_node:
            return 0.0
        score = 0.0
        for entity in self.repo_graph.neighbors(file_node.id, edge_type="contains"):
            entity_text = " ".join(
                part
                for part in [
                    entity.name.lower(),
                    (entity.docstring or "").lower(),
                    entity.code.lower(),
                ]
                if part
            )
            for keyword in keywords:
                if keyword.lower() in entity_text:
                    score += 0.5
        return min(score, 4.0)


class _PythonSymbolCollector(ast.NodeVisitor):
    """Collect top-level functions, classes, and methods with class context."""

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.symbols: list[SymbolInfo] = []
        self.imports: list[str] = []
        self._class_stack: list[str] = []

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = "." * int(getattr(node, "level", 0) or 0) + (node.module or "")
        for alias in node.names:
            if module:
                self.imports.append(f"{module}.{alias.name}")
            else:
                self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.symbols.append(
            SymbolInfo(
                name=node.name,
                kind="class",
                file_path=self.rel_path,
                line_number=node.lineno,
                signature=f"class {node.name}",
                docstring=ast.get_docstring(node),
                parent_class=self._class_stack[-1] if self._class_stack else None,
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._add_function_symbol(node, async_function=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._add_function_symbol(node, async_function=True)
        self.generic_visit(node)

    def _add_function_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        async_function: bool,
    ) -> None:
        args = ", ".join(argument.arg for argument in node.args.args)
        if self._class_stack:
            kind = "async_method" if async_function else "method"
        else:
            kind = "async_function" if async_function else "function"
        prefix = "async def" if async_function else "def"
        self.symbols.append(
            SymbolInfo(
                name=node.name,
                kind=kind,
                file_path=self.rel_path,
                line_number=node.lineno,
                signature=f"{prefix} {node.name}({args})",
                docstring=ast.get_docstring(node),
                parent_class=self._class_stack[-1] if self._class_stack else None,
            )
        )


class RepoAnalyzer:
    """Analyze a repository into a reusable RepoContext."""

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
    }

    IGNORE_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".tox",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
        ".idea",
        ".vscode",
    }

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def analyze(self) -> RepoContext:
        logger.info("Analyzing repository: %s", self.repo_path)
        context = RepoContext(repo_path=str(self.repo_path), repo_tree=self._build_tree())
        analyzed_files: dict[str, tuple[FileInfo, str, Optional[ast.Module]]] = {}
        parsed_python_files: dict[str, tuple[str, ast.Module]] = {}

        for file_path in self._iter_source_files():
            analyzed = self._analyze_file(file_path)
            if analyzed is None:
                continue
            file_info, content, tree = analyzed
            context.add_file(file_info)
            analyzed_files[file_info.path] = (file_info, content, tree)
            if tree is not None:
                parsed_python_files[file_info.path] = (content, tree)

        for rel_path, (file_info, content, tree) in analyzed_files.items():
            if tree is not None:
                self._add_graph_nodes(context, rel_path, content, tree)
                self._add_python_rationale_nodes(context, rel_path, content)
            else:
                self._add_generic_graph_nodes(context, file_info, content)

        self._add_graph_import_edges(context)

        for rel_path, (_, tree) in parsed_python_files.items():
            self._add_graph_inheritance_edges(context, rel_path, tree)
            self._add_graph_reference_edges(context, rel_path, tree)

        for rel_path, (_, tree) in parsed_python_files.items():
            self._add_graph_use_edges(context, rel_path, tree)

        logger.info(
            "Analysis complete: %s files, %s symbols, %s graph nodes",
            len(context.files),
            sum(len(file_info.symbols) for file_info in context.files),
            len(context.repo_graph.nodes),
        )
        return context

    def _build_tree(self, max_depth: int = 4) -> str:
        lines: list[str] = []
        self._tree_recursive(self.repo_path, lines, prefix="", max_depth=max_depth, depth=0)
        return "\n".join(lines)

    def _tree_recursive(
        self,
        path: Path,
        lines: list[str],
        prefix: str,
        max_depth: int,
        depth: int,
    ) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name != ".github":
                continue
            if entry.is_dir():
                if entry.name in self.IGNORE_DIRS:
                    continue
                lines.append(f"{prefix}{entry.name}/")
                self._tree_recursive(
                    entry,
                    lines,
                    prefix=prefix + "  ",
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            else:
                lines.append(f"{prefix}{entry.name}")

    def _iter_source_files(self) -> Iterable[Path]:
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [directory for directory in dirs if directory not in self.IGNORE_DIRS]
            for filename in files:
                if _is_apex_harness_path_hint(filename):
                    continue
                suffix = Path(filename).suffix
                if suffix in self.LANGUAGE_MAP:
                    yield Path(root) / filename

    def _extract_non_python_symbols(
        self,
        *,
        rel_path: str,
        language: str,
        content: str,
    ) -> list[SymbolInfo]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        if language == "go":
            patterns = [
                ("type", re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
                (
                    "function",
                    re.compile(r"^\s*func\s+(?:\(\s*[^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                ),
            ]
        elif language in {"javascript", "typescript"}:
            patterns = [
                (
                    "class",
                    re.compile(
                        r"^\s*(?:export\s+default\s+|export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"
                    ),
                ),
                (
                    "function",
                    re.compile(
                        r"^\s*(?:export\s+(?:default\s+)?)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
                    ),
                ),
                (
                    "function",
                    re.compile(
                        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][A-Za-z0-9_$]*\s*=>)"
                    ),
                ),
                (
                    "interface",
                    re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
                ),
                (
                    "type",
                    re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
                ),
            ]

        if not patterns:
            return []

        symbols: list[SymbolInfo] = []
        seen: set[tuple[str, int]] = set()
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            for kind, pattern in patterns:
                match = pattern.search(raw_line)
                if not match:
                    continue
                name = str(match.group(1) or "").strip()
                if not name or (name, line_number) in seen:
                    continue
                seen.add((name, line_number))
                symbols.append(
                    SymbolInfo(
                        name=name,
                        kind=kind,
                        file_path=rel_path,
                        line_number=line_number,
                        signature=stripped[:160],
                    )
                )
        return symbols

    def _extract_non_python_imports(
        self,
        *,
        language: str,
        content: str,
    ) -> list[str]:
        imports: list[str] = []
        if language == "go":
            in_import_block = False
            for raw_line in content.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if stripped.startswith("import ("):
                    in_import_block = True
                    continue
                if in_import_block:
                    if stripped == ")":
                        in_import_block = False
                        continue
                    match = re.search(r'"([^"]+)"', stripped)
                    if match:
                        imports.append(match.group(1))
                    continue
                if stripped.startswith("import "):
                    match = re.search(r'"([^"]+)"', stripped)
                    if match:
                        imports.append(match.group(1))
            return list(dict.fromkeys(imports))

        if language in {"javascript", "typescript"}:
            imports.extend(
                match.group(1) for match in re.finditer(r"\bfrom\s+['\"]([^'\"]+)['\"]", content)
            )
            imports.extend(
                match.group(1)
                for match in re.finditer(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)", content)
            )
            imports.extend(
                match.group(1)
                for match in re.finditer(r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)", content)
            )
        return list(dict.fromkeys(imports))

    def _analyze_file(
        self,
        file_path: Path,
    ) -> Optional[tuple[FileInfo, str, Optional[ast.Module]]]:
        try:
            content = file_path.read_text(errors="replace")
        except Exception:
            return None

        rel_path = str(file_path.relative_to(self.repo_path))
        language = self.LANGUAGE_MAP.get(file_path.suffix, "unknown")
        file_info = FileInfo(
            path=rel_path,
            size_bytes=file_path.stat().st_size,
            line_count=len(content.splitlines()),
            language=language,
        )

        if language != "python":
            file_info.symbols = self._extract_non_python_symbols(
                rel_path=rel_path,
                language=language,
                content=content,
            )
            file_info.imports = self._extract_non_python_imports(
                language=language,
                content=content,
            )
            return file_info, content, None

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return file_info, content, None

        collector = _PythonSymbolCollector(rel_path)
        collector.visit(tree)
        file_info.symbols = collector.symbols
        file_info.imports = collector.imports
        return file_info, content, tree

    def _add_generic_graph_nodes(
        self,
        context: RepoContext,
        file_info: FileInfo,
        content: str,
    ) -> None:
        file_node = GraphNode(
            id=f"file:{file_info.path}",
            node_type="file",
            name=file_info.path,
            file_path=file_info.path,
            start_line=1,
            end_line=max(file_info.line_count, 1),
            code=content,
            metadata={"language": file_info.language},
        )
        context.repo_graph.add_node(file_node)
        for symbol in file_info.symbols:
            qualified_name = (
                f"{symbol.parent_class}.{symbol.name}" if symbol.parent_class else symbol.name
            )
            node = GraphNode(
                id=f"entity:{file_info.path}:{qualified_name}",
                node_type=symbol.kind,
                name=qualified_name,
                file_path=file_info.path,
                start_line=max(int(symbol.line_number or 1), 1),
                end_line=max(int(symbol.line_number or 1), 1),
                code=symbol.signature,
                docstring=symbol.docstring,
                metadata={"language": file_info.language},
            )
            context.repo_graph.add_node(node)
            context.repo_graph.add_edge(
                file_node.id,
                node.id,
                "contains",
                source_file=file_info.path,
                line_number=symbol.line_number,
            )

    def _add_graph_import_edges(self, context: RepoContext) -> None:
        module_aliases = self._repo_module_aliases(context)
        for file_info in context.files:
            source_id = f"file:{file_info.path}"
            if source_id not in context.repo_graph.nodes:
                continue
            for imported in file_info.imports:
                target_path = self._resolve_import_to_repo_file(
                    context,
                    importer_path=file_info.path,
                    imported=imported,
                    module_aliases=module_aliases,
                )
                if not target_path or target_path == file_info.path:
                    continue
                target_id = f"file:{target_path}"
                context.repo_graph.add_edge(
                    source_id,
                    target_id,
                    "imports",
                    confidence="EXTRACTED",
                    source_file=file_info.path,
                    context=imported,
                )

    def _repo_module_aliases(self, context: RepoContext) -> dict[str, Optional[str]]:
        aliases: dict[str, Optional[str]] = {}

        def register(alias: str, path: str) -> None:
            normalized = str(alias or "").strip(". ")
            if not normalized:
                return
            existing = aliases.get(normalized)
            if existing is None and normalized in aliases:
                return
            if existing is not None and existing != path:
                aliases[normalized] = None
                return
            aliases[normalized] = path

        for file_info in context.files:
            rel_path = Path(file_info.path)
            if not rel_path.name:
                continue
            module_parts = list(rel_path.with_suffix("").parts)
            if not module_parts:
                continue
            if module_parts[-1] == "__init__":
                package_parts = module_parts[:-1]
                if package_parts:
                    register(".".join(package_parts), file_info.path)
                    register(package_parts[-1], file_info.path)
                continue
            register(".".join(module_parts), file_info.path)
            register(module_parts[-1], file_info.path)
            if len(module_parts) >= 2:
                register(".".join(module_parts[-2:]), file_info.path)
        return aliases

    def _resolve_import_to_repo_file(
        self,
        context: RepoContext,
        *,
        importer_path: str,
        imported: str,
        module_aliases: dict[str, Optional[str]],
    ) -> Optional[str]:
        text = str(imported or "").strip().strip("'\"")
        if not text:
            return None

        path_target = self._resolve_path_like_import(context, importer_path, text)
        if path_target:
            return path_target

        parts = [part for part in text.strip(".").split(".") if part]
        if not parts:
            return None
        for width in range(len(parts), 0, -1):
            candidate = ".".join(parts[:width])
            if candidate in module_aliases:
                return module_aliases[candidate]
        for start in range(1, len(parts)):
            candidate = ".".join(parts[start:])
            if candidate in module_aliases:
                return module_aliases[candidate]
        return None

    def _resolve_path_like_import(
        self,
        context: RepoContext,
        importer_path: str,
        imported: str,
    ) -> Optional[str]:
        if not (imported.startswith(".") or imported.startswith("/")):
            return None
        python_module_target = self._resolve_python_relative_module_import(
            context,
            importer_path=importer_path,
            imported=imported,
        )
        if python_module_target:
            return python_module_target
        importer_dir = Path(importer_path).parent
        candidate = Path(imported)
        if not candidate.is_absolute():
            candidate = importer_dir / candidate
        normalized = candidate.as_posix()
        known_paths = {file_info.path for file_info in context.files}

        def match(path: Path) -> Optional[str]:
            raw = path.as_posix().lstrip("./")
            if raw in known_paths:
                return raw
            for suffix in self.LANGUAGE_MAP:
                with_suffix = f"{raw}{suffix}"
                if with_suffix in known_paths:
                    return with_suffix
            for index_name in (
                "__init__.py",
                "index.ts",
                "index.tsx",
                "index.js",
                "index.jsx",
                "index.mjs",
            ):
                indexed = f"{raw.rstrip('/')}/{index_name}"
                if indexed in known_paths:
                    return indexed
            return None

        return match(Path(os.path.normpath(normalized)))

    def _resolve_python_relative_module_import(
        self,
        context: RepoContext,
        *,
        importer_path: str,
        imported: str,
    ) -> Optional[str]:
        if "/" in imported or "\\" in imported or not imported.startswith("."):
            return None
        leading = len(imported) - len(imported.lstrip("."))
        if leading <= 0:
            return None
        module_tail = imported[leading:]
        importer_dir = Path(importer_path).parent
        base = importer_dir
        for _ in range(max(0, leading - 1)):
            base = base.parent
        parts = [part for part in module_tail.split(".") if part]
        if not parts:
            return None
        known_paths = {file_info.path for file_info in context.files}
        for width in range(len(parts), 0, -1):
            candidate = base.joinpath(*parts[:width]).as_posix().lstrip("./")
            for raw in (
                f"{candidate}.py",
                f"{candidate}/__init__.py",
            ):
                if raw in known_paths:
                    return raw
        return None

    def _add_graph_nodes(
        self,
        context: RepoContext,
        rel_path: str,
        content: str,
        tree: ast.Module,
    ) -> None:
        total_lines = len(content.splitlines())
        file_node = GraphNode(
            id=f"file:{rel_path}",
            node_type="file",
            name=rel_path,
            file_path=rel_path,
            start_line=1,
            end_line=max(total_lines, 1),
            code=content,
            docstring=ast.get_docstring(tree),
            metadata={"language": "python"},
        )
        context.repo_graph.add_node(file_node)

        def walk_body(body: list[ast.stmt], parent_id: str, class_stack: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualified_name = (
                        ".".join(class_stack + [node.name]) if class_stack else node.name
                    )
                    class_node = GraphNode(
                        id=f"entity:{rel_path}:{qualified_name}",
                        node_type="class",
                        name=qualified_name,
                        file_path=rel_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        code=self._source_segment(content, node),
                        docstring=ast.get_docstring(node),
                        metadata={"language": "python"},
                    )
                    context.repo_graph.add_node(class_node)
                    context.repo_graph.add_edge(parent_id, class_node.id, "contains")
                    walk_body(node.body, class_node.id, class_stack + [node.name])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified_name = (
                        ".".join(class_stack + [node.name]) if class_stack else node.name
                    )
                    node_type = "method" if class_stack else "function"
                    function_node = GraphNode(
                        id=f"entity:{rel_path}:{qualified_name}",
                        node_type=node_type,
                        name=qualified_name,
                        file_path=rel_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        code=self._source_segment(content, node),
                        docstring=ast.get_docstring(node),
                        metadata={"language": "python", "async": isinstance(node, ast.AsyncFunctionDef)},
                    )
                    context.repo_graph.add_node(function_node)
                    context.repo_graph.add_edge(parent_id, function_node.id, "contains")

        walk_body(tree.body, file_node.id, [])

    def _add_python_rationale_nodes(
        self,
        context: RepoContext,
        rel_path: str,
        content: str,
    ) -> None:
        file_node = context.repo_graph.get_node(f"file:{rel_path}")
        if file_node is None:
            return
        entity_nodes = [
            node
            for node in context.repo_graph.nodes.values()
            if node.file_path == rel_path and node.node_type not in {"file", "rationale"}
        ]
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped.startswith(_RATIONALE_COMMENT_PREFIXES):
                continue
            parent = self._smallest_containing_entity(entity_nodes, line_number) or file_node
            rationale_node = GraphNode(
                id=f"rationale:{rel_path}:{line_number}",
                node_type="rationale",
                name=stripped[:80],
                file_path=rel_path,
                start_line=line_number,
                end_line=line_number,
                code=stripped,
                metadata={"language": "python"},
            )
            context.repo_graph.add_node(rationale_node)
            context.repo_graph.add_edge(
                file_node.id,
                rationale_node.id,
                "contains",
                source_file=rel_path,
                line_number=line_number,
            )
            context.repo_graph.add_edge(
                rationale_node.id,
                parent.id,
                "rationale_for",
                source_file=rel_path,
                line_number=line_number,
            )

    @staticmethod
    def _smallest_containing_entity(
        nodes: Iterable[GraphNode],
        line_number: int,
    ) -> Optional[GraphNode]:
        candidates = [
            node for node in nodes if node.start_line <= line_number <= max(node.end_line, node.start_line)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda node: (node.end_line - node.start_line, node.name))

    def _add_graph_inheritance_edges(
        self,
        context: RepoContext,
        rel_path: str,
        tree: ast.Module,
    ) -> None:
        def walk_body(body: list[ast.stmt], class_stack: list[str]) -> None:
            for node in body:
                if not isinstance(node, ast.ClassDef):
                    continue
                qualified_name = ".".join(class_stack + [node.name]) if class_stack else node.name
                class_id = f"entity:{rel_path}:{qualified_name}"
                for base in node.bases:
                    base_name = self._extract_annotation_name(base)
                    if not base_name:
                        continue
                    for target_id in self._resolve_call_targets(
                        context,
                        rel_path,
                        class_stack,
                        base_name,
                    ):
                        if target_id == class_id:
                            continue
                        target = context.repo_graph.get_node(target_id)
                        confidence = (
                            "EXTRACTED"
                            if target is not None and target.file_path == rel_path
                            else "INFERRED"
                        )
                        context.repo_graph.add_edge(
                            class_id,
                            target_id,
                            "inherits",
                            confidence=confidence,
                            source_file=rel_path,
                            line_number=getattr(node, "lineno", None),
                            context=base_name,
                        )
                walk_body(node.body, class_stack + [node.name])

        walk_body(tree.body, [])

    def _add_graph_reference_edges(
        self,
        context: RepoContext,
        rel_path: str,
        tree: ast.Module,
    ) -> None:
        def add_annotation_refs(source_id: str, annotation: Optional[ast.AST]) -> None:
            if annotation is None:
                return
            for name in self._annotation_names(annotation):
                for target_id in self._resolve_call_targets(context, rel_path, [], name):
                    if target_id == source_id:
                        continue
                    target = context.repo_graph.get_node(target_id)
                    confidence = (
                        "EXTRACTED"
                        if target is not None and target.file_path == rel_path
                        else "INFERRED"
                    )
                    context.repo_graph.add_edge(
                        source_id,
                        target_id,
                        "references",
                        confidence=confidence,
                        source_file=rel_path,
                        line_number=getattr(annotation, "lineno", None),
                        context=name,
                    )

        def walk_body(body: list[ast.stmt], class_stack: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualified_name = (
                        ".".join(class_stack + [node.name]) if class_stack else node.name
                    )
                    class_id = f"entity:{rel_path}:{qualified_name}"
                    for stmt in node.body:
                        if isinstance(stmt, ast.AnnAssign):
                            add_annotation_refs(class_id, stmt.annotation)
                    walk_body(node.body, class_stack + [node.name])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified_name = (
                        ".".join(class_stack + [node.name]) if class_stack else node.name
                    )
                    function_id = f"entity:{rel_path}:{qualified_name}"
                    for argument in (
                        list(node.args.posonlyargs)
                        + list(node.args.args)
                        + list(node.args.kwonlyargs)
                    ):
                        add_annotation_refs(function_id, argument.annotation)
                    if node.args.vararg is not None:
                        add_annotation_refs(function_id, node.args.vararg.annotation)
                    if node.args.kwarg is not None:
                        add_annotation_refs(function_id, node.args.kwarg.annotation)
                    add_annotation_refs(function_id, node.returns)

        walk_body(tree.body, [])

    def _add_graph_use_edges(
        self,
        context: RepoContext,
        rel_path: str,
        tree: ast.Module,
    ) -> None:
        def walk_body(body: list[ast.stmt], class_stack: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    walk_body(node.body, class_stack + [node.name])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified_name = (
                        ".".join(class_stack + [node.name]) if class_stack else node.name
                    )
                    caller_id = f"entity:{rel_path}:{qualified_name}"
                    if caller_id not in context.repo_graph.nodes:
                        continue
                    for call in ast.walk(node):
                        if not isinstance(call, ast.Call):
                            continue
                        called_name = self._extract_call_name(call)
                        if not called_name:
                            continue
                        for target_id in self._resolve_call_targets(
                            context, rel_path, class_stack, called_name
                        ):
                            if target_id != caller_id:
                                target = context.repo_graph.get_node(target_id)
                                confidence = (
                                    "EXTRACTED"
                                    if target is not None and target.file_path == rel_path
                                    else "INFERRED"
                                )
                                context.repo_graph.add_edge(
                                    caller_id,
                                    target_id,
                                    "uses",
                                    confidence=confidence,
                                    source_file=rel_path,
                                    line_number=getattr(call, "lineno", None),
                                    context=called_name,
                                )

        walk_body(tree.body, [])

    def _resolve_call_targets(
        self,
        context: RepoContext,
        rel_path: str,
        class_stack: list[str],
        called_name: str,
    ) -> list[str]:
        candidates: list[str] = []
        if called_name.startswith("self.") or called_name.startswith("cls."):
            method_name = called_name.split(".", 1)[1]
            if class_stack:
                local_id = f"entity:{rel_path}:{'.'.join(class_stack)}.{method_name}"
                if local_id in context.repo_graph.nodes:
                    candidates.append(local_id)
            called_name = method_name

        if "." in called_name:
            return list(dict.fromkeys(candidates))
        short_name = called_name

        same_file_candidates: list[str] = []
        repo_candidates: list[str] = []

        for symbol in context.symbol_index.get(short_name, []):
            parent_prefix = f"{symbol.parent_class}." if symbol.parent_class else ""
            target_id = f"entity:{symbol.file_path}:{parent_prefix}{short_name}"
            if target_id in context.repo_graph.nodes:
                if symbol.file_path == rel_path:
                    same_file_candidates.append(target_id)
                else:
                    repo_candidates.append(target_id)

        if same_file_candidates:
            candidates.extend(same_file_candidates)
            return list(dict.fromkeys(candidates))
        if short_name in _PYTHON_BUILTIN_CALLS or short_name.startswith("__"):
            return list(dict.fromkeys(candidates))
        if len(repo_candidates) == 1:
            candidates.extend(repo_candidates)
        return list(dict.fromkeys(candidates))

    def _extract_call_name(self, node: ast.Call) -> Optional[str]:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = self._extract_attribute_base(func.value)
            if base in {"self", "cls"}:
                return f"{base}.{func.attr}"
            return None
        return None

    def _extract_attribute_base(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._extract_attribute_base(node.value)
            if parent:
                return f"{parent}.{node.attr}"
        return None

    def _extract_annotation_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._extract_attribute_base(node)
        if isinstance(node, ast.Subscript):
            return self._extract_annotation_name(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.split(".")[-1]
        return None

    def _annotation_names(self, node: ast.AST) -> list[str]:
        names: list[str] = []
        for child in ast.walk(node):
            name = self._extract_annotation_name(child)
            if name:
                names.append(name.split(".")[-1])
        return list(dict.fromkeys(names))

    def _source_segment(self, content: str, node: ast.AST) -> str:
        segment = ast.get_source_segment(content, node)
        if segment is not None:
            return segment
        lines = content.splitlines()
        start = max(getattr(node, "lineno", 1) - 1, 0)
        end = max(getattr(node, "end_lineno", getattr(node, "lineno", 1)), 1)
        return "\n".join(lines[start:end])
