"""
Helpers for mode-aware agentic-search prompts and lightweight evidence routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha1
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Any, Iterable
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from .core.config import AgenticSearchConfig, KnowledgeAccessMode

_PRIORITY_DOC_FILES = (
    "AGENTS.md",
    "README.md",
    "README.rst",
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "requirements.txt",
    "requirements-dev.txt",
)
_DOC_GLOBS = (
    "docs/**/*.md",
    "docs/**/*.rst",
    "doc/**/*.md",
    "doc/**/*.rst",
)
_COMMON_QUERY_STOPWORDS = {
    "about",
    "after",
    "agent",
    "already",
    "assert",
    "before",
    "benchmark",
    "blocker",
    "broader",
    "called",
    "change",
    "changes",
    "check",
    "clear",
    "clearing",
    "code",
    "command",
    "commit",
    "continue",
    "continueoncollectionerrors",
    "current",
    "error",
    "failed",
    "failure",
    "files",
    "first",
    "fix",
    "focus",
    "functionality",
    "implement",
    "immediate",
    "include",
    "issue",
    "library",
    "line",
    "lines",
    "module",
    "more",
    "next",
    "object",
    "output",
    "package",
    "pass",
    "passed",
    "patch",
    "plan",
    "prompt",
    "pytest",
    "python",
    "repo",
    "repository",
    "return",
    "review",
    "root",
    "run",
    "schema",
    "short",
    "should",
    "stage",
    "suite",
    "summary",
    "target",
    "task",
    "tests",
    "traceback",
    "validation",
    "value",
    "visible",
    "workspace",
}
_PATH_PATTERN = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.:-]+")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_TEST_NODE_PATTERN = re.compile(r"::([A-Za-z_][A-Za-z0-9_]+)")
_URL_PATTERN = re.compile(r"https?://[^\s)>\"]+")
_REFERENCE_URL_PATTERNS = (
    "readthedocs.io",
    "docs.",
    "documentation",
    "reference",
    "spec",
    "api",
)
_CACHE_LOCK = Lock()
_LOCAL_DOC_EVIDENCE_CACHE: dict[tuple[str, str, int, int], str] = {}
_LOCAL_DOC_EVIDENCE_ITEMS_CACHE: dict[
    tuple[str, str, int, int],
    list[dict[str, Any]],
] = {}
_EXTERNAL_EVIDENCE_ITEMS_CACHE: dict[
    tuple[str, str, int, int],
    list[dict[str, str]],
] = {}
_GUIDED_STAGE_DEFAULTS = ("localizer", "patcher")
_PROACTIVE_STAGE_DEFAULTS = ("patcher",)
_MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
_RST_HEADING_UNDERLINE_CHARS = {"=", "-", "~", "^", '"', "'", "`", ":", "+", "*", "#"}
_REFERENCE_DOC_HINTS = ("api", "reference", "usage", "guide", "tutorial", "spec")
_EXTERNAL_CONTRACT_HINTS = (
    "reference specification",
    "official documentation",
    "api reference",
    "public api",
    "documented behavior",
    "version-specific",
    "compatibility",
    "upstream",
    "read the docs",
    "readthedocs",
)
_TRUSTED_EXTERNAL_DOMAINS = (
    "readthedocs.io",
    "github.com",
    "stackoverflow.com",
    "stackexchange.com",
    "pypi.org",
    "docs.python.org",
    "pythonhosted.org",
)

# Domain *substrings* that are always blocked from external evidence even
# when INTERNET_AWARE is on. These cover known SWE-bench/Commit0 mirror
# locations whose contents are exactly the gold-patch material reviewers
# care about. Callers can extend the list at runtime with a per-task
# denylist via `external_search_denied_domains` on the agentic_search
# config — that is the right place for "the task's own upstream repo".
_BENCHMARK_GOLD_DOMAIN_DENYLIST = (
    "github.com/commit-0/",
    "github.com/aorwall/",  # Moatless / SWE-bench replays
    "github.com/princeton-nlp/SWE-bench",
    "github.com/SWE-bench/",
    "huggingface.co/datasets/princeton-nlp/SWE-bench",
    "huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
    "huggingface.co/datasets/wentingzhao/commit0",
)


def _url_matches_denylist(url: str, denied_substrings: tuple[str, ...]) -> bool:
    """Return True iff ``url`` contains any of ``denied_substrings``.

    Substring match (not regex) so that operators can specify denials
    like ``"github.com/<owner>/<repo>"`` without worrying about regex
    escaping. Comparisons are case-insensitive and matched against the
    lowercase URL string after normalising ``//`` collapsed to ``/``.
    """

    if not url or not denied_substrings:
        return False
    normalised = url.lower()
    for needle in denied_substrings:
        token = (needle or "").lower().strip()
        if token and token in normalised:
            return True
    return False


@dataclass
class EvidenceRoutingDecision:
    """Controller-visible routing decision for local-only versus escalated evidence use."""

    stage_name: str = ""
    access_mode: str = KnowledgeAccessMode.AIR_GAPPED.value
    route: str = "disabled"
    local_first: bool = True
    followup_memory_fired: bool = False
    gathered_information_fired: bool = False
    stall_detected: bool = False
    external_contract_uncertainty: bool = False
    allow_online_evidence: bool = False
    online_evidence_used: bool = False
    local_doc_evidence_used: bool = False
    local_evidence_count: int = 0
    external_evidence_count: int = 0
    search_focus: list[str] = field(default_factory=list)
    reference_urls: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "access_mode": self.access_mode,
            "route": self.route,
            "local_first": bool(self.local_first),
            "followup_memory_fired": bool(self.followup_memory_fired),
            "gathered_information_fired": bool(self.gathered_information_fired),
            "stall_detected": bool(self.stall_detected),
            "external_contract_uncertainty": bool(self.external_contract_uncertainty),
            "allow_online_evidence": bool(self.allow_online_evidence),
            "online_evidence_used": bool(self.online_evidence_used),
            "local_doc_evidence_used": bool(self.local_doc_evidence_used),
            "local_evidence_count": int(self.local_evidence_count),
            "external_evidence_count": int(self.external_evidence_count),
            "search_focus": list(self.search_focus),
            "reference_urls": list(self.reference_urls),
            "rationale": self.rationale,
        }


class _DuckDuckGoHTMLParser(HTMLParser):
    """Minimal parser for DuckDuckGo HTML result pages."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture_field: str = ""
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "") for key, value in attrs}
        classes = attributes.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._finalize_current()
            self._current = {"url": attributes.get("href", ""), "title": "", "snippet": ""}
            self._capture_field = "title"
            return
        if self._current is None:
            return
        if "result__snippet" in classes:
            self._capture_field = "snippet"
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._capture_field == "title" and tag == "a":
            self._capture_field = ""
            return
        if self._capture_field == "snippet" and tag in {"a", "div", "span"}:
            self._snippet_depth = max(0, self._snippet_depth - 1)
            if self._snippet_depth == 0:
                self._capture_field = ""

    def handle_data(self, data: str) -> None:
        if self._current is None or not self._capture_field:
            return
        self._current[self._capture_field] = self._current.get(self._capture_field, "") + data

    def close(self) -> None:
        super().close()
        self._finalize_current()

    def _finalize_current(self) -> None:
        if self._current is None:
            return
        title = _collapse_ws(self._current.get("title", ""))
        snippet = _collapse_ws(self._current.get("snippet", ""))
        url = self._normalize_url(self._current.get("url", ""))
        if title and url:
            self.results.append({"title": title, "url": url, "snippet": snippet})
        self._current = None
        self._capture_field = ""
        self._snippet_depth = 0

    def _normalize_url(self, raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            wrapped = parse_qs(parsed.query).get("uddg")
            if wrapped:
                return unescape(wrapped[0])
        return raw_url


class _HTMLTextSummaryParser(HTMLParser):
    """Very small HTML-to-text parser for extracting titles and visible text."""

    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.chunks: list[str] = []
        self._capture_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._capture_title = True
            return
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._capture_title = False
            return
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = _collapse_ws(data)
        if not text:
            return
        if self._capture_title:
            self.title.append(text)
        else:
            self.chunks.append(text)


def _coerce_agentic_search_config(config: Any) -> AgenticSearchConfig:
    if isinstance(config, AgenticSearchConfig):
        return config
    if isinstance(config, dict):
        payload = dict(config)
        if "access_mode" in payload and not isinstance(payload["access_mode"], KnowledgeAccessMode):
            payload["access_mode"] = KnowledgeAccessMode(
                str(payload["access_mode"] or KnowledgeAccessMode.AIR_GAPPED.value)
            )
        return AgenticSearchConfig(**payload)
    return AgenticSearchConfig()


def _coerce_access_mode(value: Any) -> KnowledgeAccessMode:
    if isinstance(value, KnowledgeAccessMode):
        return value
    try:
        return KnowledgeAccessMode(str(value or KnowledgeAccessMode.AIR_GAPPED.value))
    except ValueError:
        return KnowledgeAccessMode.AIR_GAPPED


def _normalize_stage_name(stage_name: Any) -> str:
    return str(stage_name or "").strip().lower()


def _stage_set_from_config(
    values: Any,
    *,
    default: Iterable[str],
) -> set[str]:
    if values is None:
        return {_normalize_stage_name(stage) for stage in default if _normalize_stage_name(stage)}
    stages: set[str] = set()
    for value in values if isinstance(values, (list, tuple, set, frozenset)) else [values]:
        normalized = _normalize_stage_name(value)
        if normalized:
            stages.add(normalized)
    return stages


def _knowledge_guidance_enabled_for_stage(config: Any, *, stage_name: str = "") -> bool:
    normalized_stage = _normalize_stage_name(stage_name)
    if not normalized_stage:
        return True
    policy = _coerce_agentic_search_config(config)
    guided_stages = _stage_set_from_config(
        getattr(policy, "guided_stage_names", None),
        default=_GUIDED_STAGE_DEFAULTS,
    )
    if not guided_stages:
        return False
    return normalized_stage in guided_stages


def _proactive_evidence_enabled_for_stage(config: Any, *, stage_name: str = "") -> bool:
    normalized_stage = _normalize_stage_name(stage_name)
    if not normalized_stage:
        return True
    policy = _coerce_agentic_search_config(config)
    proactive_stages = _stage_set_from_config(
        getattr(policy, "proactive_evidence_stage_names", None),
        default=_PROACTIVE_STAGE_DEFAULTS,
    )
    if not proactive_stages:
        return False
    return normalized_stage in proactive_stages


def _detect_external_contract_uncertainty(*values: Any) -> bool:
    source = "\n".join(str(value or "").strip() for value in values if str(value or "").strip())
    if not source:
        return False
    if any(_looks_like_reference_url(url) for url in _extract_urls(source)):
        return True
    lowered = source.lower()
    return any(marker in lowered for marker in _EXTERNAL_CONTRACT_HINTS)


def _build_evidence_routing_decision(
    config: Any,
    *,
    stage_name: str = "",
    query_text: str = "",
    stalled: bool = False,
    followup_mode: bool = False,
) -> EvidenceRoutingDecision:
    policy = _coerce_agentic_search_config(config)
    access_mode = _coerce_access_mode(policy.access_mode)
    guidance_enabled = _knowledge_guidance_enabled_for_stage(policy, stage_name=stage_name)
    reference_urls = [url for url in _extract_urls(query_text) if _looks_like_reference_url(url)]
    external_contract_uncertainty = _detect_external_contract_uncertainty(query_text)
    allow_online_evidence = (
        guidance_enabled
        and access_mode is KnowledgeAccessMode.INTERNET_AWARE
        and (stalled or external_contract_uncertainty)
    )
    followup_memory_fired = bool(
        followup_mode and query_text and policy.enable_followup_search_memory and guidance_enabled
    )
    gathered_information_fired = bool(
        followup_mode
        and query_text
        and getattr(policy, "enable_followup_gathered_information", False)
        and guidance_enabled
    )
    if not guidance_enabled:
        route = "disabled"
        rationale = "Stage is outside the configured guidance scope."
    elif gathered_information_fired:
        route = (
            "gathered_information_online" if allow_online_evidence else "gathered_information_local"
        )
        rationale = "Structured gathered information is active; online escalation is allowed only after stall detection or explicit external-contract uncertainty."
    elif followup_memory_fired:
        route = "followup_memory_local"
        rationale = "Follow-up memory is active, but the stage remains local-first."
    elif allow_online_evidence:
        route = "knowledge_access_online"
        rationale = "The task exposes external-contract uncertainty, so trusted online evidence may be used for this stage."
    else:
        route = "knowledge_access_local"
        rationale = "Default to local repository evidence first and defer online escalation until the controller observes stall or external-contract uncertainty."
    return EvidenceRoutingDecision(
        stage_name=_normalize_stage_name(stage_name),
        access_mode=access_mode.value,
        route=route,
        local_first=True,
        followup_memory_fired=followup_memory_fired,
        gathered_information_fired=gathered_information_fired,
        stall_detected=bool(stalled),
        external_contract_uncertainty=external_contract_uncertainty,
        allow_online_evidence=allow_online_evidence,
        reference_urls=reference_urls,
        rationale=rationale,
    )


def _collapse_ws(text: str, *, max_chars: int | None = None) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars is None or len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _iter_local_doc_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            if not path.is_file():
                return
            rel = str(path.resolve().relative_to(root.resolve()))
        except (OSError, ValueError):
            return
        if rel in seen:
            return
        seen.add(rel)
        candidates.append(path)

    for filename in _PRIORITY_DOC_FILES:
        add(root / filename)
    for pattern in _DOC_GLOBS:
        for path in sorted(root.glob(pattern)):
            add(path)
    return candidates


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in _URL_PATTERN.findall(str(text or "")):
        url = match.rstrip(".,;:)]}")
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _add_query_token(scores: dict[str, int], raw: str, weight: int) -> None:
    for piece in re.split(r"[/.:#_\-]+", str(raw or "").lower()):
        token = piece.strip("`'\"()[]{}<> ,;")
        if len(token) < 3 or token in _COMMON_QUERY_STOPWORDS:
            continue
        if token.isdigit() or not any(char.isalpha() for char in token):
            continue
        scores[token] = scores.get(token, 0) + max(1, weight)


def _extract_query_tokens(text: str, *, max_tokens: int = 10) -> list[str]:
    scores: dict[str, int] = {}
    source = str(text or "")

    for path in _PATH_PATTERN.findall(source):
        _add_query_token(scores, path, 3)
    for match in _IDENTIFIER_PATTERN.findall(source):
        weight = 2 if match.endswith("Error") or match[0].isupper() else 1
        _add_query_token(scores, match, weight)
    for test_name in _TEST_NODE_PATTERN.findall(source):
        _add_query_token(scores, test_name, 2)
    for url in _extract_urls(source):
        parsed = urlparse(url)
        _add_query_token(scores, parsed.netloc, 2)
        _add_query_token(scores, parsed.path, 2)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _ in ranked[: max(1, max_tokens)]]


def _query_cache_signature(text: str, *, max_tokens: int = 10) -> str:
    parts = _extract_query_tokens(text, max_tokens=max_tokens)
    for url in _extract_urls(text)[:2]:
        parsed = urlparse(url)
        normalized = _collapse_ws(f"{parsed.netloc}{parsed.path}", max_chars=120)
        if normalized:
            parts.append(normalized.lower())
    if not parts:
        return sha1(str(text or "").encode("utf-8", errors="replace")).hexdigest()
    return "|".join(parts[: max(1, max_tokens + 2)])


def _extract_doc_outline(lines: list[str], *, max_headings: int = 3) -> tuple[str, list[str]]:
    title = ""
    headings: list[str] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        heading_match = _MARKDOWN_HEADING_PATTERN.match(line)
        if heading_match:
            heading = _collapse_ws(heading_match.group(1), max_chars=80)
            if heading:
                if not title:
                    title = heading
                if heading not in headings:
                    headings.append(heading)
            continue
        if index + 1 < len(lines):
            underline = lines[index + 1].strip()
            if (
                underline
                and len(underline) >= max(3, len(stripped))
                and len(set(underline)) == 1
                and underline[0] in _RST_HEADING_UNDERLINE_CHARS
            ):
                heading = _collapse_ws(stripped, max_chars=80)
                if heading:
                    if not title:
                        title = heading
                    if heading not in headings:
                        headings.append(heading)
                continue
        if not title:
            title = _collapse_ws(stripped, max_chars=80)

    return title, headings[: max(0, max_headings)]


def _build_local_reference_index(
    repo_root: str | Path,
    *,
    max_files: int,
    max_sections_per_file: int = 2,
) -> list[str]:
    root = Path(repo_root)
    if not root.exists():
        return []

    entries: list[str] = []
    for path in _iter_local_doc_paths(root)[: max(1, max_files)]:
        try:
            rel_path = str(path.resolve().relative_to(root.resolve()))
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        title, headings = _extract_doc_outline(
            lines,
            max_headings=max_sections_per_file + 1,
        )
        visible_headings = [heading for heading in headings if heading != title]
        parts = [rel_path]
        if title:
            parts.append(title)
        if visible_headings:
            sections = ", ".join(visible_headings[: max(0, max_sections_per_file)])
            if sections:
                parts.append("sections: " + sections)
        entries.append(": ".join(part for part in parts if part))
    return entries


def _extract_repo_name(text: str) -> str:
    source = str(text or "")
    patterns = (
        re.compile(r"benchmark repo:\s*([^\s]+)", re.IGNORECASE),
        re.compile(r"benchmark instance:\s*commit-0/([A-Za-z0-9_.-]+)", re.IGNORECASE),
        re.compile(r"commit-0/([A-Za-z0-9_.-]+)"),
    )
    for pattern in patterns:
        match = pattern.search(source)
        if not match:
            continue
        value = str(match.group(1) or "").strip()
        if "/" in value:
            value = value.rsplit("/", 1)[-1]
        if value:
            return value
    for url in _extract_urls(source):
        parsed = urlparse(url)
        segments = [segment for segment in parsed.path.split("/") if segment]
        for segment in reversed(segments):
            lowered = segment.lower()
            if lowered in {"latest", "stable"}:
                continue
            if re.fullmatch(r"[0-9.]+", segment):
                continue
            return segment
    return ""


def _extract_error_signatures(text: str, *, max_items: int = 2) -> list[str]:
    source = str(text or "")
    candidates: list[str] = []
    patterns = (
        re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception): [^\n]+)"),
        re.compile(r"\b(AssertionError: [^\n]+)"),
        re.compile(r"\b(assert [^\n]+)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(source):
            candidate = _collapse_ws(match.group(1), max_chars=120)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
                if len(candidates) >= max(1, max_items):
                    return candidates
    return candidates


def _build_search_focus_lines(
    query_text: str,
    *,
    access_mode: KnowledgeAccessMode,
    max_items: int = 3,
) -> list[str]:
    repo_name = _extract_repo_name(query_text)
    tokens = _extract_query_tokens(query_text, max_tokens=6)
    preferred_domain = ""
    for url in _extract_urls(query_text):
        parsed = urlparse(url)
        if parsed.netloc:
            preferred_domain = parsed.netloc
            break

    suggestions: list[str] = []
    for error_signature in _extract_error_signatures(query_text):
        parts = [error_signature]
        if repo_name:
            parts.append(repo_name)
        suggestions.append(" ".join(parts))
    if tokens:
        suggestions.append(" ".join(([repo_name] if repo_name else []) + tokens[:4]).strip())
    if access_mode is KnowledgeAccessMode.INTERNET_AWARE:
        if preferred_domain and tokens:
            suggestions.append(f"{' '.join(tokens[:4])} site:{preferred_domain}")
        elif tokens:
            suggestions.append(f"{' '.join(tokens[:4])} official documentation")

    lines: list[str] = []
    seen: set[str] = set()
    for suggestion in suggestions:
        normalized = _collapse_ws(suggestion, max_chars=140)
        if not normalized:
            continue
        lowered = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
        if lowered in seen:
            continue
        seen.add(lowered)
        lines.append(f"- {normalized}")
        if len(lines) >= max(1, max_items):
            break
    return lines


def _search_local_doc_evidence(
    repo_root: str | Path,
    *,
    query_text: str,
    max_files: int,
    max_items: int,
) -> str:
    items = _retrieve_local_doc_evidence_items(
        repo_root,
        query_text=query_text,
        max_files=max_files,
        max_items=max_items,
    )
    return _render_local_doc_evidence_summary(items)


def _retrieve_local_doc_evidence_items(
    repo_root: str | Path,
    *,
    query_text: str,
    max_files: int,
    max_items: int,
) -> list[dict[str, Any]]:
    root = Path(repo_root)
    if not root.exists():
        return []

    tokens = _extract_query_tokens(query_text)
    if not tokens:
        return []

    cache_key = (
        str(root.resolve()),
        _query_cache_signature(query_text, max_tokens=10),
        max_files,
        max_items,
    )
    with _CACHE_LOCK:
        cached = _LOCAL_DOC_EVIDENCE_ITEMS_CACHE.get(cache_key)
    if cached is not None:
        return [dict(item) for item in cached]

    scored: list[tuple[int, str, list[tuple[int, int, str]]]] = []
    for path in _iter_local_doc_paths(root)[: max(1, max_files)]:
        try:
            rel_path = str(path.resolve().relative_to(root.resolve()))
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        rel_lower = rel_path.lower()
        path_score = sum(1 for token in tokens if token in rel_lower) * 3
        if any(hint in rel_lower for hint in _REFERENCE_DOC_HINTS):
            path_score += 2
        matches: list[tuple[int, int, str]] = []
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            hit_count = len({token for token in tokens if token in lowered})
            if hit_count <= 0:
                continue
            snippet = _collapse_ws(line, max_chars=220)
            if snippet:
                matches.append((hit_count, line_number, snippet))
        if not matches and path_score <= 0:
            continue
        if not matches and path_score < 4:
            continue
        matches.sort(key=lambda item: (-item[0], item[1], item[2]))
        score = path_score + sum(match[0] for match in matches[:3])
        scored.append((score, rel_path, matches))

    scored.sort(key=lambda item: (-item[0], item[1]))
    remaining = max(1, max_items)
    items: list[dict[str, Any]] = []
    for _, rel_path, matches in scored:
        if remaining <= 0:
            break
        top_matches = matches[: min(2, remaining)]
        if not top_matches:
            continue
        for _, line_number, snippet in top_matches:
            items.append(
                {
                    "source": "local_doc",
                    "title": rel_path,
                    "path": rel_path,
                    "line_number": int(line_number),
                    "snippet": snippet,
                    "trusted": True,
                }
            )
            remaining -= 1
            if remaining <= 0:
                break

    with _CACHE_LOCK:
        _LOCAL_DOC_EVIDENCE_ITEMS_CACHE[cache_key] = [dict(item) for item in items]
        _LOCAL_DOC_EVIDENCE_CACHE[cache_key] = _render_local_doc_evidence_summary(items)
    return [dict(item) for item in items]


def _render_local_doc_evidence_summary(
    items: Iterable[dict[str, Any]],
    *,
    heading: str = "Relevant local documentation evidence:",
) -> str:
    evidence = [dict(item) for item in items if isinstance(item, dict)]
    if not evidence:
        return ""
    grouped: dict[str, list[dict[str, Any]]] = {}
    ordered_paths: list[str] = []
    for item in evidence:
        rel_path = str(item.get("path") or item.get("title") or "").strip()
        if not rel_path:
            continue
        if rel_path not in grouped:
            grouped[rel_path] = []
            ordered_paths.append(rel_path)
        grouped[rel_path].append(item)

    lines = [heading]
    for rel_path in ordered_paths:
        lines.append(f"- {rel_path}")
        for item in grouped[rel_path]:
            line_number = item.get("line_number")
            snippet = _collapse_ws(item.get("snippet", ""), max_chars=220)
            if isinstance(line_number, int) and line_number > 0 and snippet:
                lines.append(f"  {line_number}: {snippet}")
            elif snippet:
                lines.append(f"  {snippet}")
    return "\n".join(lines)


def _looks_like_reference_url(url: str) -> bool:
    lowered = url.lower()
    return any(marker in lowered for marker in _REFERENCE_URL_PATTERNS)


def _fetch_url_text(
    url: str,
    *,
    timeout_seconds: int,
) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "apex-agentic-search/1.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def _summarize_reference_url(
    url: str,
    *,
    query_tokens: list[str],
    timeout_seconds: int,
) -> dict[str, str] | None:
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        title = parsed.netloc or "Reference document"
        return {
            "title": title,
            "url": url,
            "snippet": "Task supplied a PDF reference. Prefer it over generic web results if you need the canonical contract.",
        }

    try:
        html = _fetch_url_text(url, timeout_seconds=timeout_seconds)
    except Exception:
        return None

    if html.lstrip().startswith("%PDF-"):
        title = parsed.netloc or "Reference document"
        return {
            "title": title,
            "url": url,
            "snippet": "Task supplied a PDF reference. Prefer it over generic web results if you need the canonical contract.",
        }

    parser = _HTMLTextSummaryParser()
    parser.feed(html)
    parser.close()

    title = _collapse_ws(" ".join(parser.title), max_chars=120) or parsed.netloc or url
    scored_chunks: list[tuple[int, str]] = []
    for chunk in parser.chunks:
        snippet = _collapse_ws(chunk, max_chars=220)
        if len(snippet) < 40:
            continue
        hit_count = len({token for token in query_tokens if token in snippet.lower()})
        if hit_count <= 0:
            continue
        scored_chunks.append((hit_count, snippet))
    scored_chunks.sort(key=lambda item: (-item[0], item[1]))

    snippet = ""
    if scored_chunks:
        snippet = scored_chunks[0][1]
    elif parser.chunks:
        snippet = _collapse_ws(parser.chunks[0], max_chars=220)

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
    }


def _build_external_search_query(query_text: str, *, preferred_domain: str = "") -> str:
    repo_name = _extract_repo_name(query_text)
    error_signatures = _extract_error_signatures(query_text, max_items=1)
    tokens = _extract_query_tokens(query_text, max_tokens=4)
    parts: list[str] = []
    if error_signatures:
        parts.append(error_signatures[0])
    if repo_name:
        parts.append(repo_name)
    if tokens:
        parts.append(" ".join(tokens))
    query = _collapse_ws(" ".join(parts), max_chars=180)
    if not query:
        query = "official documentation"
    if preferred_domain:
        return f"{query} site:{preferred_domain}"
    return f"{query} official documentation"


def _external_result_allowed(
    url: str,
    *,
    preferred_domain: str = "",
    extra_denylist: tuple[str, ...] = (),
) -> bool:
    """Decide whether ``url`` may be surfaced as external evidence.

    Allowance gate (in order):
        1. Always reject if URL matches the hard-coded benchmark gold
           denylist or the caller-supplied ``extra_denylist``. This is
           the only safe defence against pulling SWE-bench / Commit0
           ground-truth repos as "evidence".
        2. Allow if the host matches ``preferred_domain``.
        3. Allow if the host belongs to ``_TRUSTED_EXTERNAL_DOMAINS``.
        4. Reject otherwise.
    """

    if not url:
        return False
    if _url_matches_denylist(url, _BENCHMARK_GOLD_DOMAIN_DENYLIST):
        return False
    if _url_matches_denylist(url, tuple(extra_denylist or ())):
        return False
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().strip()
    if not netloc:
        return False
    preferred = preferred_domain.lower().strip()
    if preferred and (netloc == preferred or netloc.endswith("." + preferred)):
        return True
    return any(
        netloc == domain or netloc.endswith("." + domain) for domain in _TRUSTED_EXTERNAL_DOMAINS
    )


def _search_external_results(
    query: str,
    *,
    max_results: int,
    timeout_seconds: int,
    preferred_domain: str = "",
    extra_denylist: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    html = _fetch_url_text(
        "https://duckduckgo.com/html/?q=" + quote_plus(query),
        timeout_seconds=timeout_seconds,
    )
    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    parser.close()
    filtered = [
        result
        for result in parser.results
        if _external_result_allowed(
            str(result.get("url") or ""),
            preferred_domain=preferred_domain,
            extra_denylist=extra_denylist,
        )
    ]
    return filtered[: max(1, max_results)]


def _retrieve_external_evidence_items(
    config: Any,
    *,
    query_text: str,
    max_items_override: int | None = None,
) -> list[dict[str, str]]:
    policy = _coerce_agentic_search_config(config)
    if _coerce_access_mode(policy.access_mode) is not KnowledgeAccessMode.INTERNET_AWARE:
        return []

    budget = max(0, int(policy.external_search_budget or 0))
    if budget <= 0:
        return []

    configured_max_items = (
        max_items_override
        if isinstance(max_items_override, int) and max_items_override > 0
        else int(policy.proactive_evidence_max_items or 4)
    )
    max_items = max(1, configured_max_items)
    timeout_seconds = max(3, int(policy.external_search_timeout_seconds or 12))
    extra_denylist = tuple(
        item
        for item in (getattr(policy, "external_search_denied_domains", None) or [])
        if isinstance(item, str) and item.strip()
    )
    # Cache key INCLUDES the denylist signature so that two solves on the
    # same repo with different per-task denylists do not collide. The
    # previous cache key omitted ``preferred_domain`` and the denylist,
    # which silently leaked one task's evidence into a sibling task that
    # happened to phrase the same question.
    cache_key = (
        _query_cache_signature(query_text, max_tokens=8),
        _coerce_access_mode(policy.access_mode).value,
        budget,
        max_items,
        tuple(sorted(item.lower() for item in extra_denylist)),
    )
    with _CACHE_LOCK:
        cached = _EXTERNAL_EVIDENCE_ITEMS_CACHE.get(cache_key)
    if cached is not None:
        return [dict(item) for item in cached]
    query_tokens = _extract_query_tokens(query_text, max_tokens=8)
    reference_urls = [url for url in _extract_urls(query_text) if _looks_like_reference_url(url)]
    evidence: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    preferred_domain = ""

    for url in reference_urls:
        if len(evidence) >= max_items:
            break
        if _url_matches_denylist(url, _BENCHMARK_GOLD_DOMAIN_DENYLIST) or _url_matches_denylist(
            url, extra_denylist
        ):
            continue
        parsed = urlparse(url)
        if parsed.netloc and not preferred_domain:
            preferred_domain = parsed.netloc
        summary = _summarize_reference_url(
            url,
            query_tokens=query_tokens,
            timeout_seconds=timeout_seconds,
        )
        if summary is None or summary["url"] in seen_urls:
            continue
        seen_urls.add(summary["url"])
        evidence.append(summary)

    if len(evidence) < max_items and budget > 0:
        search_query = _build_external_search_query(
            query_text,
            preferred_domain=preferred_domain,
        )
        try:
            search_results = _search_external_results(
                search_query,
                max_results=max_items - len(evidence),
                timeout_seconds=timeout_seconds,
                preferred_domain=preferred_domain,
                extra_denylist=extra_denylist,
            )
        except Exception:
            search_results = []
        for result in search_results:
            if result["url"] in seen_urls:
                continue
            seen_urls.add(result["url"])
            evidence.append(
                {
                    "title": _collapse_ws(result["title"], max_chars=120),
                    "url": result["url"],
                    "snippet": _collapse_ws(result.get("snippet", ""), max_chars=220),
                }
            )
            if len(evidence) >= max_items:
                break

    with _CACHE_LOCK:
        _EXTERNAL_EVIDENCE_ITEMS_CACHE[cache_key] = [dict(item) for item in evidence]
    return [dict(item) for item in evidence]


def _render_external_evidence_summary(
    evidence: Iterable[dict[str, str]],
    *,
    heading: str,
    intro_line: str,
) -> str:
    items = [dict(item) for item in evidence if isinstance(item, dict)]
    if not items:
        return ""

    lines = [heading]
    if intro_line:
        lines.append(intro_line)
    for item in items:
        lines.append(f"- {item['title']}")
        lines.append(f"  URL: {item['url']}")
        if item.get("snippet"):
            lines.append(f"  Snippet: {item['snippet']}")
    return "\n".join(lines)


def _build_proactive_external_evidence(
    config: Any,
    *,
    query_text: str,
) -> str:
    policy = _coerce_agentic_search_config(config)
    if (
        not bool(policy.enable_proactive_evidence)
        or _coerce_access_mode(policy.access_mode) is not KnowledgeAccessMode.INTERNET_AWARE
    ):
        return ""

    budget = max(0, int(policy.external_search_budget or 0))
    max_items = max(1, int(policy.proactive_evidence_max_items or 4))
    evidence = _retrieve_external_evidence_items(
        policy,
        query_text=query_text,
        max_items_override=max_items,
    )
    if not evidence:
        return ""
    return _render_external_evidence_summary(
        evidence[:max_items],
        heading="Proactive external evidence summary:",
        intro_line=(
            f"APEX pre-retrieved the references below. Interactive `search_web_evidence` "
            f"calls remain capped at {budget} unique lookups for this stage."
        ),
    )


def _build_followup_external_evidence(
    config: Any,
    *,
    query_text: str,
    max_items: int,
) -> str:
    policy = _coerce_agentic_search_config(config)
    if _coerce_access_mode(policy.access_mode) is not KnowledgeAccessMode.INTERNET_AWARE:
        return ""
    evidence = _retrieve_external_evidence_items(
        policy,
        query_text=query_text,
        max_items_override=max_items,
    )
    if not evidence:
        return ""
    budget = max(0, int(policy.external_search_budget or 0))
    return _render_external_evidence_summary(
        evidence[:max_items],
        heading="Relevant follow-up external evidence:",
        intro_line=(
            "Progress appears stalled on a similar blocker. APEX pre-retrieved the "
            "references below so you can validate the contract before another edit. "
            f"Interactive external lookups remain capped at {budget} unique queries "
            "for this stage."
        ),
    )


def agentic_search_internet_enabled(
    config: Any,
    *,
    stage_name: str = "",
    query_text: str = "",
    stalled: bool = False,
    external_contract_uncertainty: bool | None = None,
) -> bool:
    """Decide whether the rollout may issue external (online) search calls.

    Local-first policy (documented in ``ARCHITECTURE.md``): even in
    ``INTERNET_AWARE`` mode, online tools are only enabled when the rollout
    has *evidence* of needing them — repeated stalled progress, or explicit
    contract-uncertainty markers in the issue text. Returning ``True`` for
    callers that pass no signal at all defeats the documented gating and
    silently re-enables network access for tool-availability probes.
    """
    policy = _coerce_agentic_search_config(config)
    if _coerce_access_mode(policy.access_mode) is not KnowledgeAccessMode.INTERNET_AWARE:
        return False
    if not _knowledge_guidance_enabled_for_stage(policy, stage_name=stage_name):
        return False
    if external_contract_uncertainty is None and query_text:
        external_contract_uncertainty = _detect_external_contract_uncertainty(query_text)
    # Require *positive* evidence (stall or contract uncertainty). Absence
    # of signal is not a license to enable external network calls.
    return bool(stalled or external_contract_uncertainty)


def agentic_search_guidance_enabled(config: Any, *, stage_name: str = "") -> bool:
    return _knowledge_guidance_enabled_for_stage(config, stage_name=stage_name)


def collect_local_reference_files(
    repo_root: str | Path,
    *,
    max_files: int = 6,
) -> list[str]:
    root = Path(repo_root)
    if not root.exists():
        return []
    references: list[str] = []
    for path in _iter_local_doc_paths(root)[: max(1, max_files)]:
        try:
            rel_path = str(path.resolve().relative_to(root.resolve()))
        except (OSError, ValueError):
            continue
        references.append(rel_path)
    return references


def build_knowledge_access_appendix(
    config: Any,
    *,
    repo_root: str | Path,
    query_text: str = "",
    stage_name: str = "",
) -> str:
    policy = _coerce_agentic_search_config(config)
    if not _knowledge_guidance_enabled_for_stage(policy, stage_name=stage_name):
        return ""
    access_mode = _coerce_access_mode(policy.access_mode)
    if access_mode is KnowledgeAccessMode.AIR_GAPPED and not bool(policy.enable_local_doc_guidance):
        return ""
    routing_decision = _build_evidence_routing_decision(
        policy,
        stage_name=stage_name,
        query_text=query_text,
        followup_mode=False,
    )
    allow_online_evidence = bool(routing_decision.allow_online_evidence)

    local_refs = (
        _build_local_reference_index(
            repo_root,
            max_files=max(1, int(policy.local_doc_max_files or 6)),
        )
        if bool(policy.enable_local_doc_guidance)
        else []
    )

    lines = ["", "# Knowledge Access"]
    if access_mode is KnowledgeAccessMode.INTERNET_AWARE:
        budget = max(0, int(policy.external_search_budget or 0))
        lines.extend(
            [
                "Mode: internet-aware",
                "Policy: local evidence first.",
                (
                    f"Interactive external evidence remains capped at {budget} lookups for this stage."
                ),
                (
                    "Online escalation is enabled for this stage because the prompt already exposes external-contract uncertainty."
                    if allow_online_evidence
                    else "Do not browse by default. Escalate online only after repeated stalled failures or when an external contract remains genuinely uncertain."
                ),
                "Prefer official documentation, upstream source, or accepted community answers before generic blogs or summaries.",
            ]
        )
    else:
        lines.extend(
            [
                "Mode: air-gapped",
                "External internet lookups are unavailable in this run. Do not spend iterations trying to browse or search the web.",
                "Resolve the task from local repository evidence, nearby tests, runtime traces, and installed-package inspection inside the workspace.",
            ]
        )

    if local_refs:
        lines.extend(
            [
                "",
                "Local reference index:",
                *[f"- {path}" for path in local_refs],
            ]
        )

    search_focus = _build_search_focus_lines(
        query_text,
        access_mode=(
            KnowledgeAccessMode.INTERNET_AWARE
            if allow_online_evidence
            else KnowledgeAccessMode.AIR_GAPPED
        ),
        max_items=3,
    )
    if search_focus:
        heading = (
            "Suggested external query anchors:"
            if allow_online_evidence
            else "Suggested local search anchors:"
        )
        lines.extend(["", heading, *search_focus])

    lines.extend(
        [
            "",
            "Search workflow:",
            "- Start from the failing test, traceback, or focus files already in the brief.",
            "- Follow imports and call paths before broadening to unrelated files.",
            "- Stop searching once the contract is clear and switch back to editing and validation.",
        ]
    )
    routing_decision.search_focus = [
        item[2:].strip() if item.startswith("- ") else item.strip() for item in search_focus
    ]

    if bool(policy.enable_proactive_evidence) and _proactive_evidence_enabled_for_stage(
        policy,
        stage_name=stage_name,
    ):
        local_evidence_items = _retrieve_local_doc_evidence_items(
            repo_root,
            query_text=query_text,
            max_files=max(1, int(policy.local_doc_max_files or 6)),
            max_items=max(1, int(policy.proactive_evidence_max_items or 4)),
        )
        local_evidence = _render_local_doc_evidence_summary(local_evidence_items)
        if local_evidence:
            routing_decision.local_doc_evidence_used = True
            routing_decision.local_evidence_count = len(local_evidence_items)
            lines.extend(["", local_evidence])
        if allow_online_evidence:
            external_items = _retrieve_external_evidence_items(
                policy,
                query_text=query_text,
                max_items_override=max(1, int(policy.proactive_evidence_max_items or 4)),
            )
            external_evidence = _render_external_evidence_summary(
                external_items,
                heading="Proactive external evidence summary:",
                intro_line=(
                    f"APEX pre-retrieved the references below. Interactive `search_web_evidence` "
                    f"calls remain capped at {max(0, int(policy.external_search_budget or 0))} unique lookups for this stage."
                ),
            )
            if external_evidence:
                routing_decision.online_evidence_used = True
                routing_decision.external_evidence_count = len(external_items)
                lines.extend(["", external_evidence])

    return "\n".join(lines).strip()


def build_semiformal_editing_appendix(config: Any) -> str:
    policy = _coerce_agentic_search_config(config)
    if not bool(policy.enable_semiformal_review):
        return ""
    lines = [
        "",
        "# Semi-Formal Review",
        "Before finalizing a fix, reason through these steps internally:",
        "1. Premises: identify the concrete files, symbols, and test evidence that define the current contract.",
        "2. Trace: follow the exact import, call, or test path that reaches the failing behavior.",
        "3. Divergence: state the first behavioral mismatch between the expected contract and the current implementation.",
        "4. Edit plan: make the smallest change that repairs that divergence without regressing already-cleared behavior.",
    ]
    return "\n".join(lines).strip()


def build_semiformal_followup_appendix(
    config: Any,
    *,
    changed_files: Iterable[str] = (),
    blocker: str = "",
    failure_files: Iterable[str] = (),
) -> str:
    policy = _coerce_agentic_search_config(config)
    if not bool(policy.enable_semiformal_review):
        return ""
    changed = [str(path).strip() for path in changed_files if str(path).strip()]
    failed = [str(path).strip() for path in failure_files if str(path).strip()]
    lines = [
        "",
        "# Semi-Formal Review",
        "Before the next edit, explicitly tighten the plan around the current evidence:",
    ]
    if changed:
        lines.append("Current patch surface: " + ", ".join(changed[:6]))
    if blocker:
        lines.append("Current blocker: " + blocker)
    if failed:
        lines.append("Failure path files: " + ", ".join(failed[:6]))
    lines.extend(
        [
            "1. Premises: what behavior is already fixed, and what evidence still fails?",
            "2. Trace: what exact import, call, or test path reaches the remaining blocker?",
            "3. Divergence: where does the current code first depart from the intended contract on that path?",
            "4. Next edit: what is the smallest change that resolves that divergence while preserving the already-cleared behavior?",
        ]
    )
    return "\n".join(lines).strip()


def _compose_followup_query_text(
    *,
    blocker: str = "",
    failure_files: Iterable[str] = (),
    failed_tests: Iterable[str] = (),
    changed_files: Iterable[str] = (),
    tests_run: Iterable[str] = (),
) -> str:
    parts: list[str] = []
    blocker_text = _collapse_ws(blocker, max_chars=220)
    if blocker_text:
        parts.append(blocker_text)
    failed = [str(path).strip() for path in failure_files if str(path).strip()]
    if failed:
        parts.append("Failure files: " + ", ".join(failed[:4]))
    failing_tests = [str(test_id).strip() for test_id in failed_tests if str(test_id).strip()]
    if failing_tests:
        parts.append("Failing tests: " + ", ".join(failing_tests[:4]))
    changed = [str(path).strip() for path in changed_files if str(path).strip()]
    if changed:
        parts.append("Changed files: " + ", ".join(changed[:4]))
    validations = [str(entry).strip() for entry in tests_run if str(entry).strip()]
    if validations:
        parts.append("Validation commands: " + " | ".join(validations[:2]))
    return "\n".join(parts).strip()


def _compose_followup_reference_context(
    prompt_text: str,
    *,
    max_urls: int = 2,
) -> str:
    source = str(prompt_text or "").strip()
    if not source:
        return ""

    parts: list[str] = []
    repo_name = _extract_repo_name(source)
    if repo_name:
        parts.append("Benchmark repo: " + repo_name)
    urls = _extract_urls(source)
    if urls:
        parts.append("Reference URLs: " + ", ".join(urls[: max(1, max_urls)]))
    tokens = _extract_query_tokens(source, max_tokens=4)
    if tokens:
        parts.append("Original task anchors: " + ", ".join(tokens[:4]))
    return "\n".join(parts).strip()


def build_followup_evidence_routing_appendix(
    config: Any,
    *,
    repo_root: str | Path,
    base_prompt: str = "",
    blocker: str = "",
    previous_blocker: str = "",
    progress_assessment: str = "",
    stalled: bool = False,
    failure_files: Iterable[str] = (),
    failed_tests: Iterable[str] = (),
    changed_files: Iterable[str] = (),
    tests_run: Iterable[str] = (),
    stage_name: str = "patcher",
) -> tuple[str, EvidenceRoutingDecision]:
    policy = _coerce_agentic_search_config(config)
    followup_enabled = bool(
        policy.enable_followup_search_memory
        or getattr(policy, "enable_followup_gathered_information", False)
    )
    if not _knowledge_guidance_enabled_for_stage(policy, stage_name=stage_name):
        return "", EvidenceRoutingDecision(
            stage_name=_normalize_stage_name(stage_name),
            access_mode=_coerce_access_mode(policy.access_mode).value,
            route="disabled",
            rationale="Stage is outside the configured follow-up guidance scope.",
        )

    changed = [str(path).strip() for path in changed_files if str(path).strip()]
    failed = [str(path).strip() for path in failure_files if str(path).strip()]
    failing_tests = [str(test_id).strip() for test_id in failed_tests if str(test_id).strip()]
    blocker_text = _collapse_ws(blocker, max_chars=220)
    previous_blocker_text = _collapse_ws(previous_blocker, max_chars=220)
    progress_text = _collapse_ws(progress_assessment, max_chars=220)
    reference_context = _compose_followup_reference_context(base_prompt)
    query_text = _compose_followup_query_text(
        blocker=blocker_text,
        failure_files=failed,
        failed_tests=failing_tests,
        changed_files=changed,
        tests_run=tests_run,
    )
    if not query_text:
        return "", EvidenceRoutingDecision(
            stage_name=_normalize_stage_name(stage_name),
            access_mode=_coerce_access_mode(policy.access_mode).value,
            route="disabled",
            rationale="No structured follow-up evidence was available to route.",
        )
    evidence_query_text = "\n".join(
        part
        for part in (
            query_text,
            reference_context,
            ("Previous blocker: " + previous_blocker_text) if previous_blocker_text else "",
            ("Progress assessment: " + progress_text) if progress_text else "",
        )
        if part
    ).strip()

    max_items = max(1, int(policy.followup_search_memory_max_items or 3))
    routing_decision = _build_evidence_routing_decision(
        policy,
        stage_name=stage_name,
        query_text=evidence_query_text,
        stalled=stalled,
        followup_mode=followup_enabled,
    )
    if not followup_enabled:
        return "", routing_decision
    structured = bool(routing_decision.gathered_information_fired)
    search_access_mode = (
        KnowledgeAccessMode.INTERNET_AWARE
        if routing_decision.allow_online_evidence and structured
        else KnowledgeAccessMode.AIR_GAPPED
    )
    lines = [
        "# Gathered Information" if structured else "# Search Memory",
        (
            "Carry forward the confirmed evidence below before issuing new searches or widening the edit."
            if structured
            else "Reuse the confirmed evidence below before broadening search."
        ),
    ]
    if structured and (previous_blocker_text or blocker_text or progress_text):
        lines.extend(["", "Verification trajectory:"])
        if previous_blocker_text:
            lines.append(f"- Previous blocker: {previous_blocker_text}")
        if blocker_text:
            lines.append(f"- Current blocker: {blocker_text}")
        if progress_text:
            lines.append(f"- Progress assessment: {progress_text}")
    if changed:
        lines.extend(
            [
                "",
                "Confirmed patch surface:",
                *[f"- {path}" for path in changed[:max_items]],
            ]
        )
    if blocker_text or failed or failing_tests:
        lines.extend(["", "Confirmed failure surface:"])
        if blocker_text:
            lines.append(f"- Blocker: {blocker_text}")
        if failed:
            lines.extend(f"- File: {path}" for path in failed[:max_items])
        if failing_tests:
            lines.extend(f"- Test: {test_id}" for test_id in failing_tests[:max_items])
    if structured and progress_text:
        lines.extend(["", "Search posture:"])
        if stalled:
            lines.append(
                "- The last two verification traces look materially similar. Pivot through contract evidence or a new code path before repeating the same edit/search loop."
            )
        else:
            lines.append(
                "- The failure surface moved or improved. Keep search local to the current blocker before broadening."
            )

    search_focus = _build_search_focus_lines(
        evidence_query_text,
        access_mode=search_access_mode,
        max_items=max_items,
    )
    routing_decision.search_focus = [
        item[2:].strip() if item.startswith("- ") else item.strip() for item in search_focus
    ]
    if search_focus:
        heading = (
            "Refined external query anchors:"
            if search_access_mode is KnowledgeAccessMode.INTERNET_AWARE
            else "Refined local search anchors:"
        )
        lines.extend(["", heading, *search_focus])

    if bool(policy.enable_local_doc_guidance):
        local_evidence_items = _retrieve_local_doc_evidence_items(
            repo_root,
            query_text=evidence_query_text,
            max_files=max(1, int(policy.local_doc_max_files or 6)),
            max_items=max_items,
        )
        local_evidence = _render_local_doc_evidence_summary(
            local_evidence_items,
            heading="Relevant follow-up documentation evidence:",
        )
        if local_evidence:
            routing_decision.local_doc_evidence_used = True
            routing_decision.local_evidence_count = len(local_evidence_items)
            lines.extend(["", local_evidence])
    if structured and routing_decision.allow_online_evidence:
        external_items = _retrieve_external_evidence_items(
            policy,
            query_text=evidence_query_text,
            max_items_override=max_items,
        )
        external_evidence = _render_external_evidence_summary(
            external_items,
            heading="Relevant follow-up external evidence:",
            intro_line=(
                "Online escalation is active for this follow-up because the controller saw repeated stall or unresolved external-contract uncertainty. "
                f"Interactive external lookups remain capped at {max(0, int(policy.external_search_budget or 0))} unique queries for this stage."
            ),
        )
        if external_evidence:
            routing_decision.online_evidence_used = True
            routing_decision.external_evidence_count = len(external_items)
            lines.extend(["", external_evidence])

    return "\n".join(lines).strip(), routing_decision


def build_followup_search_memory_appendix(
    config: Any,
    *,
    repo_root: str | Path,
    base_prompt: str = "",
    blocker: str = "",
    previous_blocker: str = "",
    progress_assessment: str = "",
    stalled: bool = False,
    failure_files: Iterable[str] = (),
    failed_tests: Iterable[str] = (),
    changed_files: Iterable[str] = (),
    tests_run: Iterable[str] = (),
    stage_name: str = "patcher",
) -> str:
    appendix, _ = build_followup_evidence_routing_appendix(
        config,
        repo_root=repo_root,
        base_prompt=base_prompt,
        blocker=blocker,
        previous_blocker=previous_blocker,
        progress_assessment=progress_assessment,
        stalled=stalled,
        failure_files=failure_files,
        failed_tests=failed_tests,
        changed_files=changed_files,
        tests_run=tests_run,
        stage_name=stage_name,
    )
    return appendix


def augment_prompt_with_agentic_search_guidance(
    prompt: str,
    config: Any,
    *,
    repo_root: str | Path,
    stage_name: str = "",
    include_semiformal_editing: bool = False,
) -> str:
    extras: list[str] = []
    knowledge = build_knowledge_access_appendix(
        config,
        repo_root=repo_root,
        query_text=prompt,
        stage_name=stage_name,
    )
    if knowledge:
        extras.append(knowledge)
    if include_semiformal_editing:
        review = build_semiformal_editing_appendix(config)
        if review:
            extras.append(review)
    if not extras:
        return prompt
    return prompt.rstrip() + "\n\n" + "\n\n".join(extras)
