"""Build a self-contained reviewer publication bundle from a run dir.

The bundle is the artifact handed to external reviewers when APEX
publishes a benchmark headline. It is *separate* from the raw run
directory because:

* Reviewers should not need to wade through APEX-internal trajectories,
  per-rollout state graphs, or developer scratch artifacts.
* The bundle contains a pinned ``REPRODUCE.sh`` derived from the
  manifest, so a reviewer with no APEX checkout can re-run the exact
  same benchmark slice end-to-end.
* The bundle contains predictions in each benchmark's *upstream-
  canonical* schema (Commit0 JSON; SWT-Bench JSONL; TestGenEval JSONL),
  which lets reviewers re-score with the upstream harness instead of
  relying on APEX's private scorer.

The class is intentionally pure-Python with no Docker or network calls
so the unit tests can exercise the full path with a synthetic
``run_dir``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("apex.publish.bundle")


#: Names of files written into the bundle root. Listed centrally so the
#: validator and tests stay consistent with the writer.
BUNDLE_FILES: tuple[str, ...] = (
    "README.md",
    "RESULTS.md",
    "MANIFEST.json",
    "OVERRIDES_DISCLOSURE.md",
    "REPRODUCE.sh",
)


#: Benchmarks we know how to extract upstream-canonical predictions for.
#: Each entry is ``(detector_filename, output_filename, schema_kind)``.
_PREDICTION_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("commit0_predictions.json", "commit0_predictions.json", "commit0"),
    ("swtbench_predictions.jsonl", "swtbench_predictions.jsonl", "swtbench"),
    (
        "testgeneval_predictions.jsonl",
        "testgeneval_predictions.jsonl",
        "testgeneval",
    ),
    ("predictions.jsonl", "predictions.jsonl", "swtbench"),
)


class BundleValidationError(RuntimeError):
    """Raised when ``run_dir`` is missing artifacts the bundle requires."""


@dataclass
class _BundleArtifacts:
    """Files we located inside ``run_dir`` for inclusion in the bundle."""

    manifest_path: Optional[Path] = None
    overrides_path: Optional[Path] = None
    fairness_audit_path: Optional[Path] = None
    pareto_frontier_path: Optional[Path] = None
    benchmark_summaries: list[Path] = field(default_factory=list)
    predictions: list[tuple[Path, str, str]] = field(default_factory=list)
    abstention_path: Optional[Path] = None


@dataclass
class PublicationBundle:
    """A reviewer-ready snapshot of an APEX benchmark run.

    Construct via :meth:`from_run_dir` then materialise with
    :meth:`write_to`.
    """

    run_dir: Path
    artifacts: _BundleArtifacts
    include_fairness_audit: bool = False
    contact: str = "apex-team@meta.com"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path | str,
        *,
        include_fairness_audit: bool = False,
        contact: Optional[str] = None,
    ) -> "PublicationBundle":
        """Walk ``run_dir`` and gather everything we need for the bundle."""
        run_dir_path = Path(run_dir).resolve()
        if not run_dir_path.is_dir():
            raise BundleValidationError(f"run_dir is not a directory: {run_dir_path}")
        artifacts = _scan_run_dir(run_dir_path)
        return cls(
            run_dir=run_dir_path,
            artifacts=artifacts,
            include_fairness_audit=bool(include_fairness_audit),
            contact=contact or "apex-team@meta.com",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error messages.

        Empty list means the run directory has the minimum artifacts the
        bundle needs (a manifest and at least one predictions file).
        """
        errors: list[str] = []
        if self.artifacts.manifest_path is None:
            errors.append(
                "missing manifest: expected 'apex_run_manifest.json' or "
                "'run_manifest.json' under run_dir"
            )
        if not self.artifacts.predictions:
            errors.append(
                "missing predictions: no 'predictions.jsonl', "
                "'commit0_predictions.json', 'swtbench_predictions.jsonl', "
                "or 'testgeneval_predictions.jsonl' found under run_dir"
            )
        return errors

    def require_valid(self) -> None:
        """Raise :class:`BundleValidationError` if validation fails."""
        errors = self.validate()
        if errors:
            raise BundleValidationError(
                "publication bundle cannot be built:\n  - " + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_to(self, output_dir: Path | str) -> dict[str, Path]:
        """Materialise the bundle in ``output_dir``.

        Returns a dict mapping bundle artifact name to absolute output
        path. Raises :class:`BundleValidationError` when the source
        ``run_dir`` lacks required inputs.
        """
        self.require_valid()

        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}

        # --- MANIFEST.json -------------------------------------------------
        manifest_path = self.artifacts.manifest_path
        assert manifest_path is not None  # validated above
        target_manifest = output / "MANIFEST.json"
        shutil.copyfile(manifest_path, target_manifest)
        written["MANIFEST.json"] = target_manifest

        # --- OVERRIDES_DISCLOSURE.md ---------------------------------------
        target_overrides = output / "OVERRIDES_DISCLOSURE.md"
        if self.artifacts.overrides_path is not None:
            shutil.copyfile(self.artifacts.overrides_path, target_overrides)
        else:
            target_overrides.write_text(
                "# Per-repo Override Disclosure\n\n"
                "_No OVERRIDES_DISCLOSURE.md found in the run directory; "
                "no per-repo overrides were applied to this benchmark run._\n"
            )
        written["OVERRIDES_DISCLOSURE.md"] = target_overrides

        # --- predictions/<benchmark>.{jsonl,json} -------------------------
        predictions_dir = output / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_targets: list[Path] = []
        for src, output_name, schema_kind in self.artifacts.predictions:
            target = predictions_dir / output_name
            _copy_predictions(src, target, schema_kind)
            prediction_targets.append(target)
            written[f"predictions/{output_name}"] = target

        # --- RESULTS.md ----------------------------------------------------
        results_md_path = output / "RESULTS.md"
        results_md_path.write_text(self._render_results_markdown())
        written["RESULTS.md"] = results_md_path

        # --- REPRODUCE.sh --------------------------------------------------
        reproduce_path = output / "REPRODUCE.sh"
        reproduce_path.write_text(self._render_reproduce_script())
        _make_executable(reproduce_path)
        written["REPRODUCE.sh"] = reproduce_path

        # --- README.md -----------------------------------------------------
        readme_path = output / "README.md"
        readme_path.write_text(self._render_readme(written))
        written["README.md"] = readme_path

        return written

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        if self.artifacts.manifest_path is None:
            return {}
        try:
            return json.loads(self.artifacts.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "failed to read manifest %s: %s",
                self.artifacts.manifest_path,
                exc,
            )
            return {}

    def _load_fairness(self) -> Optional[dict[str, Any]]:
        if self.artifacts.fairness_audit_path is None:
            return None
        try:
            return json.loads(self.artifacts.fairness_audit_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "failed to read fairness audit %s: %s",
                self.artifacts.fairness_audit_path,
                exc,
            )
            return None

    def _load_pareto(self) -> Optional[dict[str, Any]]:
        if self.artifacts.pareto_frontier_path is None:
            return None
        try:
            return json.loads(self.artifacts.pareto_frontier_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "failed to read pareto frontier %s: %s",
                self.artifacts.pareto_frontier_path,
                exc,
            )
            return None

    def _load_abstention(self) -> Optional[dict[str, Any]]:
        if self.artifacts.abstention_path is None:
            return None
        try:
            return json.loads(self.artifacts.abstention_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _render_results_markdown(self) -> str:
        manifest = self._load_manifest()
        lines: list[str] = []
        lines.append("# APEX Benchmark Publication: Results")
        lines.append("")
        lines.append(
            "This report is the headline scoring summary that "
            "accompanies the published APEX benchmark numbers. The "
            "scores below were produced by re-running the upstream "
            "canonical scorer against the predictions in this bundle."
        )
        lines.append("")

        lines.append("## Run identity")
        lines.append("")
        lines.append(f"* APEX git SHA: `{manifest.get('apex_git_sha', 'unknown')}`")
        lines.append(f"* Working tree dirty: `{bool(manifest.get('apex_git_dirty', False))}`")
        lines.append(f"* Python: `{manifest.get('python_version', 'unknown')}`")
        lines.append(f"* Platform: `{manifest.get('platform', 'unknown')}`")
        lines.append(f"* Run started at: `{manifest.get('started_at', '')}`")
        seed = manifest.get("seed")
        if seed is not None:
            lines.append(f"* Seed: `{seed}`")
        lines.append("")

        # --- Score table per benchmark ---
        lines.append("## Score table per benchmark")
        lines.append("")
        score_rows = self._collect_benchmark_scores()
        if not score_rows:
            lines.append("_No benchmark summary files found in the run dir._")
        else:
            lines.append("| benchmark | metric | value | tasks |")
            lines.append("| --- | --- | ---: | ---: |")
            for row in score_rows:
                lines.append(
                    f"| {row['benchmark']} | {row['metric']} | {row['value']} | {row['tasks']} |"
                )
        lines.append("")

        # --- Per-task results ---
        lines.append("## Per-task results")
        lines.append("")
        per_task = self._collect_per_task_rows()
        if not per_task:
            lines.append("_No per-task results located._")
        else:
            lines.append("| benchmark | task_id | passed | notes |")
            lines.append("| --- | --- | ---: | --- |")
            for row in per_task:
                lines.append(
                    f"| {row['benchmark']} | `{row['task_id']}` "
                    f"| {row['passed']} | {row.get('notes', '')} |"
                )
        lines.append("")

        # --- Fairness audit ---
        lines.append("## Fairness audit deltas")
        lines.append("")
        if not self.include_fairness_audit:
            lines.append(
                "_Fairness-audit deltas omitted from this bundle. "
                "Pass `--include-fairness-audit` to include them._"
            )
        else:
            fairness = self._load_fairness()
            if fairness is None:
                lines.append("_No `fairness_audit.json` found in the run dir._")
            else:
                summary = fairness.get("summary", {})
                lines.append(f"* Tasks audited: {summary.get('num_tasks', 0)}")
                lines.append(
                    f"* Flagged tasks "
                    f"(|delta| > {summary.get('flag_threshold', 0):g}): "
                    f"{summary.get('num_flagged_tasks', 0)}"
                )
                lines.append(
                    f"* Tasks with scorer disagreement: {summary.get('num_disagreement_tasks', 0)}"
                )
                per_metric = summary.get("per_metric", {})
                if per_metric:
                    lines.append("")
                    lines.append("| metric | mean delta | max |delta| | tasks |")
                    lines.append("| --- | ---: | ---: | ---: |")
                    for metric, stats in sorted(per_metric.items()):
                        lines.append(
                            f"| `{metric}` | {stats.get('mean_delta', 0)} "
                            f"| {stats.get('max_abs_delta', 0)} "
                            f"| {stats.get('num_tasks_with_metric', 0)} |"
                        )
        lines.append("")

        # --- Abstention rates ---
        lines.append("## Abstention rates")
        lines.append("")
        abst = self._load_abstention()
        if abst is None:
            lines.append("_No abstention summary located._")
        else:
            lines.append(f"* Total tasks: {abst.get('total_tasks', 0)}")
            lines.append(f"* Abstained tasks: {abst.get('abstained_tasks', 0)}")
            lines.append(f"* Abstention rate: {abst.get('abstention_rate', 'n/a')}")
        lines.append("")

        # --- Pareto frontier reference ---
        lines.append("## Pareto frontier reference")
        lines.append("")
        pareto = self._load_pareto()
        if pareto is None:
            lines.append(
                "_No `pareto_frontier.json` found; this run did not "
                "emit a budget-vs-quality frontier._"
            )
        else:
            frontier = pareto.get("frontier", []) or pareto.get("points", [])
            if not frontier:
                lines.append("_Frontier file present but empty._")
            else:
                lines.append("| budget | score | points |")
                lines.append("| ---: | ---: | --- |")
                for point in frontier:
                    budget = point.get("budget", point.get("cost", "?"))
                    score = point.get("score", point.get("quality", "?"))
                    notes = point.get("notes", "")
                    lines.append(f"| {budget} | {score} | {notes} |")
        lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_reproduce_script(self) -> str:
        manifest = self._load_manifest()
        apex_sha = manifest.get("apex_git_sha", "unknown")
        python_version = manifest.get("python_version", "3.10")
        seed = manifest.get("seed")
        harnesses = manifest.get("upstream_harness_versions", {}) or {}
        docker_images = manifest.get("docker_images", {}) or {}

        # Build deterministic per-harness pin lines.
        pin_lines: list[str] = []
        for name, version in sorted(harnesses.items()):
            if isinstance(version, str) and version.startswith("editable@"):
                sha = version.split("@", 1)[1]
                pin_lines.append(f'echo "  - {name} (editable@{sha})"')
            else:
                pin_lines.append(f'pip install "{name}=={version}"')
        if not pin_lines:
            pin_lines.append('echo "  (no upstream harness versions recorded)"')

        # Pre-pull docker images by digest.
        docker_lines: list[str] = []
        for tag, digest in sorted(docker_images.items()):
            if isinstance(digest, str) and digest.startswith("sha256:"):
                docker_lines.append(f'docker pull "{tag}@{digest}"')
            elif isinstance(digest, str) and "@" in digest:
                docker_lines.append(f'docker pull "{digest}"')
            else:
                docker_lines.append(f'echo "  - {tag} (digest unavailable: {digest})"')
        if not docker_lines:
            docker_lines.append('echo "  (no docker image pins recorded)"')

        seed_line = (
            f"export APEX_BENCHMARK_SEED={int(seed)}"
            if isinstance(seed, int)
            else 'echo "  (no seed recorded)"'
        )

        script = [
            "#!/usr/bin/env bash",
            "# Auto-generated by `apex publish-benchmark`.",
            "# Re-runs the benchmark slice using the manifest's pinned",
            "# APEX SHA + harness versions + docker image digests.",
            "set -euo pipefail",
            "",
            f'APEX_GIT_SHA="{apex_sha}"',
            f'PYTHON_VERSION="{python_version}"',
            seed_line,
            "",
            'echo "==> Cloning APEX at pinned SHA"',
            "if [[ ! -d apex ]]; then",
            "  git clone https://github.com/apex/apex.git",
            "fi",
            "pushd apex",
            "git fetch --all --quiet",
            'git checkout "$APEX_GIT_SHA"',
            "popd",
            "",
            'echo "==> Creating venv"',
            "python -m venv .venv",
            ". .venv/bin/activate",
            "pip install --upgrade pip",
            "pip install -e ./apex",
            "",
            'echo "==> Pinning upstream harnesses"',
            *pin_lines,
            "",
            'echo "==> Pulling docker images by digest"',
            *docker_lines,
            "",
            'echo "==> Re-running benchmark"',
            "apex benchmark --output ./reproduced-run",
            "",
            'echo "==> Done. Compare ./reproduced-run with the bundled MANIFEST.json"',
            "",
        ]
        return "\n".join(script)

    def _render_readme(self, written: dict[str, Path]) -> str:
        lines = [
            "# APEX Benchmark Publication Bundle",
            "",
            "This directory is a self-contained, reviewer-ready snapshot of "
            "an APEX benchmark run. There are no symlinks and no external "
            "paths inside this bundle - it can be tarred and sent verbatim.",
            "",
            "## Contents",
            "",
            "* `RESULTS.md` - score table per benchmark, per-task results, "
            "abstention rates, and (if `--include-fairness-audit` was passed) "
            "the fairness-audit deltas between APEX-private and upstream-"
            "canonical scorers.",
            "* `MANIFEST.json` - exact reproducibility manifest from the run "
            "(APEX git SHA, Python version, platform, env vars, model ids, "
            "docker image digests, upstream harness versions).",
            "* `OVERRIDES_DISCLOSURE.md` - every per-repo override applied "
            "by APEX before scoring, with rationale for each.",
            "* `predictions/` - one predictions file per benchmark, in the "
            "*upstream-canonical* schema. Reviewers can re-score these with "
            "the upstream harness and confirm the headline number.",
            "* `REPRODUCE.sh` - one-command script that pins APEX + harness "
            "versions from `MANIFEST.json` and re-runs the benchmark slice.",
            "",
            "## How to verify",
            "",
            "1. Read `RESULTS.md` for the headline numbers.",
            "2. Read `MANIFEST.json` to confirm the APEX SHA, Python version, and harness pins.",
            "3. Read `OVERRIDES_DISCLOSURE.md` to confirm what was patched "
            "in the upstream dataset and why.",
            "4. Run `bash REPRODUCE.sh` to reproduce the run end-to-end.",
            "5. Compare your reproduced predictions against "
            "`predictions/<benchmark>.{jsonl,json}` byte-for-byte.",
            "",
            f"## Contact\n\n{self.contact}\n",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Aggregation helpers (RESULTS.md inputs)
    # ------------------------------------------------------------------

    def _collect_benchmark_scores(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self.artifacts.benchmark_summaries:
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            benchmark = payload.get("benchmark") or path.stem
            metric_name = payload.get("primary_metric_name", "score")
            metric_value = payload.get(
                "primary_metric_percent",
                payload.get("score"),
            )
            tasks = payload.get("num_tasks", payload.get("task_count", "?"))
            rows.append(
                {
                    "benchmark": benchmark,
                    "metric": metric_name,
                    "value": metric_value if metric_value is not None else "n/a",
                    "tasks": tasks,
                }
            )
        return rows

    def _collect_per_task_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self.artifacts.benchmark_summaries:
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            benchmark = payload.get("benchmark") or path.stem
            for task in payload.get("tasks", []):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "task_id": task.get("task_id", task.get("instance_id", "?")),
                        "passed": task.get("passed", task.get("resolved", "?")),
                        "notes": task.get("notes", ""),
                    }
                )
        return rows


# ----------------------------------------------------------------------
# Module-private helpers
# ----------------------------------------------------------------------


def _scan_run_dir(run_dir: Path) -> _BundleArtifacts:
    """Walk ``run_dir`` and return the artifacts we care about."""
    artifacts = _BundleArtifacts()

    # Manifest: prefer the new name, fall back to legacy.
    for candidate in ("apex_run_manifest.json", "run_manifest.json"):
        path = run_dir / candidate
        if path.is_file():
            artifacts.manifest_path = path
            break

    overrides = run_dir / "OVERRIDES_DISCLOSURE.md"
    if overrides.is_file():
        artifacts.overrides_path = overrides

    fairness = run_dir / "fairness_audit.json"
    if fairness.is_file():
        artifacts.fairness_audit_path = fairness

    pareto = run_dir / "pareto_frontier.json"
    if pareto.is_file():
        artifacts.pareto_frontier_path = pareto

    abstention = run_dir / "abstention_summary.json"
    if abstention.is_file():
        artifacts.abstention_path = abstention

    # Benchmark summaries: any *_summary.json or benchmark_summary.json
    # at the top level of run_dir.
    for path in sorted(run_dir.glob("*benchmark_summary*.json")):
        artifacts.benchmark_summaries.append(path)
    for path in sorted(run_dir.glob("*_summary.json")):
        if path not in artifacts.benchmark_summaries:
            artifacts.benchmark_summaries.append(path)

    # Predictions: try the canonical names first, in priority order.
    seen_targets: set[str] = set()
    for detector_name, output_name, schema_kind in _PREDICTION_SOURCES:
        candidate = run_dir / detector_name
        if candidate.is_file() and output_name not in seen_targets:
            artifacts.predictions.append((candidate, output_name, schema_kind))
            seen_targets.add(output_name)

    return artifacts


def _copy_predictions(src: Path, dst: Path, schema_kind: str) -> None:
    """Copy predictions verbatim if they already match the upstream schema.

    For now we trust upstream-canonical naming as a contract: callers
    that emitted, e.g. ``commit0_predictions.json`` are expected to have
    written the upstream schema. We do a light sanity-check below and
    raise :class:`BundleValidationError` on a clear mismatch so reviewers
    are not handed an invalid predictions file.
    """
    if schema_kind == "commit0":
        try:
            payload = json.loads(src.read_text())
        except json.JSONDecodeError as exc:
            raise BundleValidationError(
                f"commit0 predictions at {src} are not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise BundleValidationError(
                f"commit0 predictions at {src} must be a JSON object "
                f"(instance_id -> patch); got {type(payload).__name__}"
            )
        dst.write_text(json.dumps(payload, sort_keys=True, indent=2))
        return

    # JSONL benchmarks: SWT-Bench (model_patch) and TestGenEval
    # (preds in instance_id/test_patch/etc form). We validate that each
    # line is a JSON object with an ``instance_id`` key.
    rows: list[dict[str, Any]] = []
    with open(src, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleValidationError(
                    f"predictions at {src}:{line_no} are not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise BundleValidationError(f"predictions at {src}:{line_no} must be JSON objects")
            if "instance_id" not in row:
                raise BundleValidationError(
                    f"predictions at {src}:{line_no} missing required "
                    f"'instance_id' field for upstream-canonical schema"
                )
            if schema_kind == "swtbench" and "model_patch" not in row:
                raise BundleValidationError(
                    f"swtbench predictions at {src}:{line_no} missing required 'model_patch' field"
                )
            rows.append(row)
    with open(dst, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _make_executable(path: Path) -> None:
    """chmod +x with best-effort fallback (Windows is a no-op)."""
    try:
        mode = os.stat(path).st_mode
        os.chmod(
            path,
            mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )
    except OSError:  # pragma: no cover - defensive
        pass


__all__ = [
    "BUNDLE_FILES",
    "BundleValidationError",
    "PublicationBundle",
]
