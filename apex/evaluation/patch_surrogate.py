"""V5 PatchSurrogate: candidate fix patches as oracle proxies.

The dual-version verifier (``apex.evaluation.dual_version_verifier``)
needs reference fix patches to score candidate tests against. SWT-Bench
forbids agents from seeing the ground-truth fix during generation, so we
synthesize candidate patches ourselves — a committee of imperfect
proxies that, in aggregate, supplies the F→P signal.

Echo / e-Otter++ / TEX-T all use external patch generators (Agentless,
Prometheus, etc.). APEX V5 uses APEX's OWN code-generation pipeline:

  - Co-evolution: better APEX-codegen → better surrogate → better
    selected tests (CURE-style joint training opportunity later).
  - Built-in pairing eval: every task gives us
    (test_quality, patch_quality, did_they_compose) data.
  - Architectural symmetry: same agentic CLI infrastructure.
  - No supply-chain risk on a separately-maintained patch agent.

This module exposes ``generate_candidate_patches(task, agent_models)``
that fans out one patch-generation request per agent in parallel, mirrors
the multi-agent ensemble pattern from testgen, and returns the unified
diffs (rejecting empty / non-applying outputs at the surface).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidatePatch:
    """One agent's candidate fix attempt for the bug under test."""

    agent: str
    diff: str
    raw_output: str = ""
    applied_cleanly: Optional[bool] = None
    error: str = ""

    @property
    def is_usable(self) -> bool:
        """A patch is usable as an oracle proxy when the diff is non-empty
        and we have no firm evidence it doesn't apply."""
        return bool(self.diff and self.diff.strip()) and self.applied_cleanly is not False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchSurrogateResult:
    candidates: list[CandidatePatch] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> list[CandidatePatch]:
        return [c for c in self.candidates if c.is_usable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "usable_count": len(self.usable),
            "diagnostics": dict(self.diagnostics),
        }


def generate_candidate_patches(
    *,
    task: Any,
    agent_models: list[str],
    output_dir: Path,
    generation_timeout_seconds: float = 300.0,
    request_parallelism: int = 4,
    bug_description: str = "",
) -> PatchSurrogateResult:
    """Fan out patch-generation requests across the agent ensemble.

    Each agent independently attempts to fix the bug under test in the
    focal module. Returns a list of candidate patches; the dual-version
    verifier consumes them as oracle proxies.

    The patches are not expected to all be correct — Echo/e-Otter++ rely
    on the same noisy-oracle dynamic. Per-agent fix accuracy of ~30-65%
    (matching SWE-Bench scores) is enough signal for the F→P matrix to
    discriminate good vs. bad tests in aggregate.
    """

    if not agent_models:
        return PatchSurrogateResult(diagnostics={"status": "no_agents"})

    output_dir.mkdir(parents=True, exist_ok=True)
    parallelism = max(1, min(int(request_parallelism or 1), len(agent_models)))
    metadata = dict(getattr(task, "metadata", {}) or {})
    source_truth_workdir = Path(
        getattr(task, "repo_path", None) or metadata.get("source_truth_workdir") or output_dir
    )

    def _request(agent: str) -> CandidatePatch:
        logger.warning("patch_surrogate: agent=%s starting", agent)
        agent_dir = output_dir / f"patch_{agent}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = agent_dir / "patch_prompt.md"
        raw_path = agent_dir / "patch_raw_output.txt"
        prompt = _render_patch_prompt(task=task, bug_description=bug_description)
        prompt_path.write_text(prompt, encoding="utf-8")
        agent_started = time.time()
        try:
            raw = _request_patch_via_agent(
                prompt=prompt,
                agent_name=agent,
                workdir=source_truth_workdir,
                output_dir=agent_dir,
                generation_timeout_seconds=generation_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - LLM/CLI errors are diagnostic
            logger.warning(
                "patch_surrogate: agent=%s raised %s: %s (after %.1fs)",
                agent,
                type(exc).__name__,
                exc,
                time.time() - agent_started,
            )
            return CandidatePatch(
                agent=agent,
                diff="",
                raw_output="",
                error=f"{type(exc).__name__}: {exc}",
            )
        raw_path.write_text(raw or "", encoding="utf-8")
        diff = _extract_unified_diff(raw or "")
        logger.warning(
            "patch_surrogate: agent=%s done in %.1fs (raw=%d bytes, diff=%d bytes)",
            agent,
            time.time() - agent_started,
            len(raw or ""),
            len(diff),
        )
        return CandidatePatch(
            agent=agent,
            diff=diff,
            raw_output=raw or "",
            applied_cleanly=None,  # populated by dual-version verifier
        )

    started = time.time()
    candidates: list[CandidatePatch] = []
    if parallelism == 1:
        for agent in agent_models:
            candidates.append(_request(agent))
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            candidates = list(pool.map(_request, agent_models))
    elapsed = time.time() - started
    diagnostics = {
        "status": "ok",
        "agent_count": len(agent_models),
        "usable_count": sum(1 for c in candidates if c.is_usable),
        "elapsed_seconds": round(elapsed, 2),
    }
    return PatchSurrogateResult(candidates=candidates, diagnostics=diagnostics)


def _render_patch_prompt(*, task: Any, bug_description: str) -> str:
    """Build the patch-generation prompt. Mirrors the testgen prompt
    structure so the agents see the same focal context for both roles."""

    focal_path = getattr(task, "focal_method_path", "") or ""
    focal_source = getattr(task, "focal_method_source", "") or ""
    test_source = getattr(task, "existing_test_source", "") or ""
    parts = [
        "Fix the bug in the focal module so that the existing test suite continues to pass.",
        "Output ONLY a unified diff (the kind `git apply` accepts) targeting the focal file. No prose, no fences.",
        "Do not modify the test file. Do not add new files. Keep the diff minimal and targeted at the bug.",
        "",
        f"Focal file: {focal_path}",
        "Focal source:",
        "```python",
        focal_source,
        "```",
        "",
        "Existing test file (do not modify):",
        "```python",
        test_source,
        "```",
        "",
    ]
    if bug_description:
        parts.extend(
            [
                "Bug description:",
                bug_description,
                "",
            ]
        )
    parts.extend(
        [
            "Output (unified diff only):",
        ]
    )
    return "\n".join(parts)


_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.DOTALL)


def _extract_unified_diff(raw_text: str) -> str:
    """Pull the first unified diff out of an agent's raw output. Tries
    fenced blocks first (in case the agent ignored the no-fences
    instruction), then falls back to scanning for ``diff --git`` /
    ``--- a/`` markers."""

    text = (raw_text or "").strip()
    if not text:
        return ""
    fence_match = _DIFF_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    if "diff --git " in text or text.lstrip().startswith("--- "):
        # Strip any leading prose before the first diff marker.
        idx = min(
            (i for i in (text.find("diff --git "), text.find("--- ")) if i >= 0),
            default=0,
        )
        return text[idx:].strip() + "\n"
    return ""


def _request_patch_via_agent(
    *,
    prompt: str,
    agent_name: str,
    workdir: Path,
    output_dir: Path,
    generation_timeout_seconds: float,
) -> str:
    """Run one patch-generation request through the named agent."""

    from apex._default_generators import build_agent_llm_config
    from apex.core.cli_backend import CLIModelClient

    llm_config = build_agent_llm_config(agent_name)
    timeout = max(1, int(generation_timeout_seconds))
    llm_config.cli_timeout = timeout
    llm_config.cli_hard_timeout_seconds = timeout
    llm_config.cli_strict_hard_timeout = True
    target_tool_env: dict[str, str] = {}
    try:
        from apex.evaluation.testgeneval_benchmark import _target_authoring_tool_env_overrides

        target_tool_env, _ = _target_authoring_tool_env_overrides(
            workdir=workdir,
            output_dir=output_dir,
            timeout_seconds=generation_timeout_seconds,
        )
    except Exception:
        target_tool_env = {}
    result = CLIModelClient(llm_config).run_structured_prompt(
        prompt=prompt,
        working_dir=str(workdir),
        schema=None,
        system_prompt=(
            "You are an expert software engineer. Produce minimal, correct "
            "unified diffs that fix bugs without modifying tests."
        ),
        allow_edits=False,
        internet_enabled=False,
        hard_timeout_seconds=timeout,
        env_overrides=target_tool_env or None,
    )
    return (
        getattr(result, "text", None)
        or (
            json.dumps(getattr(result, "parsed_json", None))
            if getattr(result, "parsed_json", None) is not None
            else ""
        )
    ) or ""
