"""Cross-solve persistence primitives (per-repo memory, calibration logs)."""

from .calibration import (
    CalibrationBin,
    CalibrationDataset,
    CalibrationReport,
    build_calibration_reports,
    collect_calibration_datasets,
    render_reliability_markdown,
)
from .episodic_store import (
    Episode,
    EpisodicStore,
    task_signature_for,
)
from .repo_episodic_store import (
    RepoEpisode,
    RepoEpisodicStore,
    render_repo_episodes_prompt_block,
)
from .repo_memory import (
    PersistedInsight,
    RepoMemoryStore,
    is_repo_memory_disabled_via_env,
    repo_signature_for_path,
    repo_signature_legacy_for_path,
)
from .testgen_memory import (
    ALL_TESTGEN_INSIGHT_TYPES,
    INSIGHT_TYPE_TESTGEN_AXIS_COVERAGE_HOTSPOT,
    INSIGHT_TYPE_TESTGEN_F2P_BUG_PATTERN,
    INSIGHT_TYPE_TESTGEN_FOCUS_FILE_HOTSPOT,
    INSIGHT_TYPE_TESTGEN_KILLED_MUTATION_CLASS,
    INSIGHT_TYPE_TESTGEN_LOW_COVERAGE_HOTSPOT,
    INSIGHT_TYPE_TESTGEN_RESISTANT_MUTATION_CLASS,
    extract_testgen_insights_from_run_summary,
    persist_testgen_insights_for_repo,
    query_prior_testgen_insights_for_focus_files,
    render_prior_testgen_insights_prompt_block,
)

__all__ = [
    "ALL_TESTGEN_INSIGHT_TYPES",
    "CalibrationBin",
    "CalibrationDataset",
    "CalibrationReport",
    "Episode",
    "EpisodicStore",
    "INSIGHT_TYPE_TESTGEN_F2P_BUG_PATTERN",
    "INSIGHT_TYPE_TESTGEN_AXIS_COVERAGE_HOTSPOT",
    "INSIGHT_TYPE_TESTGEN_FOCUS_FILE_HOTSPOT",
    "INSIGHT_TYPE_TESTGEN_KILLED_MUTATION_CLASS",
    "INSIGHT_TYPE_TESTGEN_LOW_COVERAGE_HOTSPOT",
    "INSIGHT_TYPE_TESTGEN_RESISTANT_MUTATION_CLASS",
    "PersistedInsight",
    "RepoEpisode",
    "RepoEpisodicStore",
    "RepoMemoryStore",
    "build_calibration_reports",
    "collect_calibration_datasets",
    "extract_testgen_insights_from_run_summary",
    "is_repo_memory_disabled_via_env",
    "persist_testgen_insights_for_repo",
    "query_prior_testgen_insights_for_focus_files",
    "render_prior_testgen_insights_prompt_block",
    "render_reliability_markdown",
    "render_repo_episodes_prompt_block",
    "repo_signature_for_path",
    "repo_signature_legacy_for_path",
    "task_signature_for",
]
