"""
Commit0 benchmark runner used by the CAID paper.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import logging
import os
import re
import select
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import traceback

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from ..acceptance import (
    quick_verification_has_local_full_scope_pass,
    quick_verification_requires_authoritative_scoring,
)
from ..core.cli_backend import (
    _env_flag_enabled,
    clear_cli_cleanup_signal_requested,
    cli_cleanup_signal_requested,
    ensure_cli_process_cleanup_hooks,
)
from ..core.config import ApexConfig, BenchmarkEvaluationBackend
from ..core.docker_pinning import resolve_image as _resolve_docker_image
from ..core.failure_classifier import (
    FailureClass as _CoreFailureClass,
)
from ..core.failure_classifier import (
    classify_failure as _core_classify_failure,
)
from ..core.fairness_audit import (
    FairnessAuditAggregator,
    FairnessAuditMode,
    run_fairness_audit,
)
from ..core.filesystem import copy_tree
from ..core.git_utils import (
    is_ignored_change_path,
    normalize_changed_path,
    parse_porcelain_path,
    sync_git_submodules,
)
from ..core.git_utils import (
    list_changed_files as list_git_changed_files,
)
from ..core.pytest_report_utils import (
    VisibleTestEditDisposition,
    analyze_visible_test_edit,
    extract_pytest_report_outcomes,
    extract_pytest_report_tests,
    incomplete_test_files_from_context,
    load_pytest_json_report,
    normalize_pytest_outcome,
    parameterized_node_id_base,
    parse_pytest_terminal_summary_counts,
    protected_test_files_from_context,
    pytest_report_outcome,
    summarize_expected_pytest_coverage,
)
from ..core.pytest_utils import (
    infer_additional_pytest_packages,
    parse_pytest_command,
    render_pytest_command,
)
from ..core.run_manifest import RunManifest, detect_upstream_harness_versions
from ..core.run_supervisor import (
    APEX_BENCHMARK_LABEL,
    APEX_OWNER_PID_LABEL,
    apex_docker_labels,
    docker_label_args,
    parse_docker_label_string,
)
from ..core.stub_scanner import scan_files_for_stubs, summarize_findings
from ..core.subprocess_utils import PROCESS_REGISTRY, run_process_command, run_shell_command
from ..core.terminal_output import normalize_terminal_output
from ..orchestrator import ApexOrchestrator
from ..persistence.escrow import EscrowRecord, EscrowStore
from ..planning.manager import IssuePlan
from ..rollout.candidate_identity import CandidateIdentity, worktree_patch_hash
from ..rollout.patch_sanitizer import sanitize_candidate_worktree
from .benchmark import (
    append_benchmark_task_outcome_trace,
    build_apex_ablation_config,
    extract_apex_execution_metadata,
)
from .checkpointing import (
    RUN_STATE_FILENAME,
    atomic_write_json,
    atomic_write_text,
    build_run_state,
    ensure_clean_directory_for_task,
    load_json_if_exists,
    task_result_path,
    write_task_checkpoint,
)
from .contracts import (
    EvaluationContract,
    EvaluationDecision,
    RunnerHealth,
    ScoredCounts,
    decide_evaluation,
)
from .flake_firewall import (
    classify_oracle_failure,
    output_has_teardown_leak_signature,
)
from .run_artifacts import (
    build_allowed_backend_snapshots,
    build_benchmark_policy,
    build_prompt_template_fingerprints,
    build_run_manifest,
    capture_environment_snapshot,
    cluster_failures,
    ensure_run_manifest,
    load_run_manifest,
    manifest_summary,
    update_run_manifest,
    write_task_live_state,
    write_task_live_state_terminal,
)
from .target_runtime import (
    apply_target_tool_env_to_apex_config,
    docker_exec_runtime,
    docker_image_runtime,
    host_env_runtime,
    target_runtime_path_for_file,
    target_tool_env_overrides,
)

COMMIT0_LITE_REPOS = [
    "tinydb",
    "simpy",
    "deprecated",
    "wcwidth",
    "voluptuous",
    "cachetools",
    "imapclient",
    "marshmallow",
    "jinja",
    "cookiecutter",
    "portalocker",
    "parsel",
    "pyjwt",
    "chardet",
    "babel",
    "minitorch",
]

COMMIT0_BENCHMARK_HARNESS_NAME = "commit0_shared_harness"
COMMIT0_BENCHMARK_HARNESS_VERSION = "2026-04-20.1"
COMMIT0_BENCHMARK_REPORT_KIND_APEX = "apex_commit0"
COMMIT0_BENCHMARK_REPORT_KIND_RAW = "raw_commit0"
COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER = "commit0_official_local_docker"
COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST = "local_pytest_json_report"
COMMIT0_OFFICIAL_EVALUATION_TIMEOUT_SECONDS = 1800
COMMIT0_OFFICIAL_EVALUATION_NUM_CPUS = 1
COMMIT0_OFFICIAL_EVALUATION_SUBPROCESS_GRACE_SECONDS = 300
COMMIT0_OFFICIAL_GIT_LOCK_RETRY_DELAYS_SECONDS = (0.25, 0.75, 1.5)
COMMIT0_OFFICIAL_GIT_INDEX_LOCK_STALE_SECONDS = 2.0
_COMMIT0_AGENT_TARGET_OUTPUT_CAPTURE_MAX_CHARS = 65_536

# Task-framing block — the BINDING EVAL RULES, stated to every arm via the single shared
# worker-prompt gate (build_issue_description). It states the RULES of the eval (a from-scratch
# reimplementation; the upstream package is out of scope/unavailable; pass the visible suite by
# editing the source, not the tests) — it names NO package, version, enum member, locale,
# provider, count, or test-id, so it is leak-safe and is the model's/orchestrator's job to act
# on (it is NOT the banned design-contract, which pre-derived the answer shape). Surfaced to the
# orchestrator/scout via repo_map["task_framing"] too (single source of truth).
TASK_FRAMING_BLOCK = (
    "This is a from-scratch reimplementation task. Implement the library's behavior solely from "
    "the visible test suite and the existing source already present in this repository. The "
    "original upstream/published package is OUT OF SCOPE and is treated as UNAVAILABLE: do not "
    "fetch, download, install, vendor, copy, or otherwise import its source or any released "
    "version — solutions that do so are invalid and cannot be scored. Everything you need is in "
    "this repository and its tests. Scoring is REQUIRED to be by exact match against the suite's "
    "expected (gold) test ids — the visible tests ARE the gold scoring set, and only those expected "
    "test ids count toward the score. A solution is accepted ONLY when the in-repository "
    "implementation makes those expected tests pass, by editing the source files and not by "
    "altering, skipping, or removing the tests. Passing other checks, or making the suite green by "
    "changing the tests, does not count."
)

# Canonical commit0 GOLD evaluation contract. Pinning this onto a config's
# ``benchmark.evaluation_contract`` makes ``_commit0_expected_id_scoring_required``
# return True, so an empty/failed expected-id inventory becomes a HARNESS FAILURE
# (indeterminate, re-run) instead of silently falling through to the ``pytest_summary``
# visible-suite acceptance path (which would accept a green raw suite without verifying
# any gold expected-id match). Mirrors the gold default in ApexConfig
# (apex/core/config.py resolved_evaluation_contract_config), but pinned EXPLICITLY so
# every arm is guaranteed gold-scored and never relies on the empty-default fall-through.
COMMIT0_GOLD_EVALUATION_CONTRACT = {
    "mode": "gold_suite_visible",
    "scoring_universe": "expected_test_ids",
    "diagnostic_universes": ["extra_non_scored_tests", "raw_pytest_returncode"],
    "raw_returncode_policy": "diagnostic_only_when_scoring_filtered",
    "extra_result_policy": "diagnostic_only",
    "baseline_timeout_policy": "attempt_anyway",
    "environment_failure_policy": "fallback_runtime",
}
_GIT_INDEX_LOCK_PATH_RE = re.compile(
    r"(?:Unable to create|could not create) ['\"]?(?P<path>[^'\"\n]*index\.lock)['\"]?",
    re.IGNORECASE,
)
_COMMIT0_TASK_ROLLOUT_LIVE_STATE_KEYS = [
    "active_rollout_ids",
    "active_rollout_count",
    "current_rollout_id",
    "current_stage",
    "current_phase",
]


def _commit0_official_image_python_env_repair_shell(python_bin: str) -> str:
    quoted_python = shlex.quote(str(python_bin or "/testbed/.venv/bin/python"))
    return f"""
# APEX Commit0/Python harness repair: Commit0 cookiecutter official image ships
# NUL-filled installed dependency sources, so rewrite only those corrupted
# files offline and never fetch packages from the solve/eval container.
_apex_python={quoted_python}
if [ -x "$_apex_python" ]; then
  _apex_markupsafe_status=0
  "$_apex_python" - <<'PY' || _apex_markupsafe_status=$?
from importlib import metadata
import pathlib
import sys

try:
    metadata.distribution("MarkupSafe")
except metadata.PackageNotFoundError:
    sys.exit(0)

try:
    import markupsafe  # noqa: F401
except Exception as exc:
    site_roots = pathlib.Path(sys.prefix).glob("lib/python*/site-packages")
    candidates = []
    for site_root in site_roots:
        candidates.extend((site_root / "markupsafe").glob("*.py"))
    nul_paths = []
    for path in candidates:
        try:
            if path.read_bytes().count(b"\\x00"):
                nul_paths.append(str(path))
        except OSError:
            pass
    if nul_paths or "null bytes" in str(exc).lower():
        print("APEX_MARKUPSAFE_REPAIR_NEEDED: " + str(exc), file=sys.stderr)
        for path in nul_paths:
            print("APEX_MARKUPSAFE_NUL_FILE: " + path, file=sys.stderr)
        sys.exit(42)
    print("APEX_MARKUPSAFE_IMPORT_UNEXPECTED_FAILURE: " + repr(exc), file=sys.stderr)
    sys.exit(0)
sys.exit(0)
PY
  if [ "$_apex_markupsafe_status" -eq 42 ]; then
    "$_apex_python" - <<'PY' || exit $?
from pathlib import Path
import sys

site_roots = list(Path(sys.prefix).glob("lib/python*/site-packages"))
if not site_roots:
    raise SystemExit("APEX_MARKUPSAFE_REPAIR_FAILED: site-packages not found")
pkg_dir = site_roots[0] / "markupsafe"
pkg_dir.mkdir(parents=True, exist_ok=True)
(pkg_dir / "_native.py").write_text(
    '''
def _to_str(value):
    if hasattr(value, "__html__"):
        return str(value.__html__())
    return str(value)


def escape(s):
    from . import Markup

    return Markup(
        _to_str(s)
        .replace("&", "&amp;")
        .replace(">", "&gt;")
        .replace("<", "&lt;")
        .replace("'", "&#39;")
        .replace('"', "&#34;")
    )


def escape_silent(s):
    if s is None:
        from . import Markup

        return Markup("")
    return escape(s)


def soft_str(s):
    if not isinstance(s, str):
        return str(s)
    return s
'''.lstrip(),
    encoding="utf-8",
)
(pkg_dir / "__init__.py").write_text(
    '''
__version__ = "2.1.5"


class Markup(str):
    __slots__ = ()

    def __new__(cls, base="", encoding=None, errors="strict"):
        if hasattr(base, "__html__"):
            base = base.__html__()
        if encoding is None:
            return str.__new__(cls, base)
        return str.__new__(cls, base, encoding, errors)

    def __html__(self):
        return self

    @classmethod
    def escape(cls, s):
        rv = escape(s)
        if rv.__class__ is not cls:
            return cls(rv)
        return rv

    def __repr__(self):
        return self.__class__.__name__ + "(" + str.__repr__(self) + ")"

    def __add__(self, other):
        if isinstance(other, str) or hasattr(other, "__html__"):
            return self.__class__(str.__add__(self, self.escape(other)))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str) or hasattr(other, "__html__"):
            return self.escape(other).__add__(self)
        return NotImplemented

    def __mul__(self, num):
        if isinstance(num, int):
            return self.__class__(str.__mul__(self, num))
        return NotImplemented

    __rmul__ = __mul__

    def __mod__(self, arg):
        if isinstance(arg, tuple):
            arg = tuple(_MarkupEscapeHelper(x) for x in arg)
        elif hasattr(type(arg), "__getitem__") and not isinstance(arg, str):
            arg = _MarkupEscapeHelper(arg)
        else:
            arg = (_MarkupEscapeHelper(arg),)
        return self.__class__(str.__mod__(self, arg))

    def join(self, seq):
        return self.__class__(str.join(self, map(self.escape, seq)))

    def split(self, sep=None, maxsplit=-1):
        return [self.__class__(x) for x in str.split(self, sep, maxsplit)]

    def rsplit(self, sep=None, maxsplit=-1):
        return [self.__class__(x) for x in str.rsplit(self, sep, maxsplit)]

    def splitlines(self, keepends=False):
        return [self.__class__(x) for x in str.splitlines(self, keepends)]

    def unescape(self):
        import html

        return html.unescape(str(self))

    def striptags(self):
        import re

        return " ".join(re.sub(r"<[^>]*>", "", self.unescape()).split())

    def format(self, *args, **kwargs):
        return self.__class__(str.format(self, *args, **kwargs))

    def format_map(self, mapping):
        return self.__class__(str.format_map(self, mapping))


class _MarkupEscapeHelper:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return _MarkupEscapeHelper(self.obj[item])

    def __str__(self):
        return str(escape(self.obj))

    def __repr__(self):
        return str(escape(repr(self.obj)))

    def __int__(self):
        return int(self.obj)

    def __float__(self):
        return float(self.obj)


try:
    from ._speedups import escape, escape_silent, soft_str
except Exception:
    from ._native import escape, escape_silent, soft_str

soft_unicode = soft_str
__all__ = ["Markup", "escape", "escape_silent", "soft_str", "soft_unicode"]
'''.lstrip(),
    encoding="utf-8",
)
PY
    "$_apex_python" - <<'PY' || exit $?
import markupsafe  # noqa: F401
PY
  elif [ "$_apex_markupsafe_status" -ne 0 ]; then
    exit "$_apex_markupsafe_status"
  fi
  "$_apex_python" - <<'PY' || exit $?
from pathlib import Path
import importlib
import shutil
import sys

site_roots = list(Path(sys.prefix).glob("lib/python*/site-packages"))
if not site_roots:
    raise SystemExit("APEX_COOKIECUTTER_DEP_REPAIR_FAILED: site-packages not found")
site_root = site_roots[0]


def _has_nul(path):
    try:
        data = path.read_bytes()
    except OSError:
        return False
    return bool(data and b"\\x00" in data)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")


repaired = []
arrow_dir = site_root / "arrow"
arrow_locale = arrow_dir / "locales.py"
if _has_nul(arrow_locale):
    print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(arrow_locale), file=sys.stderr)
    _write(
        arrow_dir / "__init__.py",
        '''
__version__ = "1.3.0"


class Arrow:
    def __init__(self, value):
        self._value = value

    def shift(self, **kwargs):
        import datetime as _datetime

        return Arrow(self._value + _datetime.timedelta(**kwargs))

    def format(self, fmt="YYYY-MM-DD HH:mm:ssZZ", locale="en-us"):
        if fmt is None:
            fmt = "YYYY-MM-DD HH:mm:ssZZ"
        text = str(fmt)
        if "%" in text:
            return self._value.strftime(text)
        replacements = (
            ("YYYY", "%Y"),
            ("YY", "%y"),
            ("MMMM", "%B"),
            ("MMM", "%b"),
            ("MM", "%m"),
            ("DD", "%d"),
            ("D", "%-d"),
            ("HH", "%H"),
            ("H", "%-H"),
            ("hh", "%I"),
            ("h", "%-I"),
            ("mm", "%M"),
            ("ss", "%S"),
            ("ZZ", "%z"),
            ("Z", "%z"),
        )
        for token, replacement in replacements:
            text = text.replace(token, replacement)
        return self._value.strftime(text)

    @property
    def datetime(self):
        return self._value

    @property
    def naive(self):
        return self._value.replace(tzinfo=None)


def _tzinfo(timezone):
    if timezone in (None, "local"):
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(str(timezone))
    except Exception:
        import datetime as _datetime

        return _datetime.timezone.utc


def now(timezone=None):
    import datetime as _datetime

    return Arrow(_datetime.datetime.now(_tzinfo(timezone)))


def utcnow():
    import datetime as _datetime

    return Arrow(_datetime.datetime.now(_datetime.timezone.utc))


def get(value=None, *args, **kwargs):
    import datetime as _datetime

    if value is None:
        return now(kwargs.get("tzinfo") or kwargs.get("timezone"))
    if isinstance(value, Arrow):
        return value
    if isinstance(value, _datetime.datetime):
        return Arrow(value)
    if isinstance(value, _datetime.date):
        return Arrow(_datetime.datetime.combine(value, _datetime.time()))
    if isinstance(value, (int, float)):
        return Arrow(_datetime.datetime.fromtimestamp(value, _datetime.timezone.utc))
    if isinstance(value, str):
        try:
            return Arrow(_datetime.datetime.fromisoformat(value))
        except ValueError:
            return Arrow(_datetime.datetime.strptime(value, "%Y-%m-%d"))
    raise TypeError("Unsupported arrow.get() value: " + repr(value))


__all__ = ["Arrow", "get", "now", "utcnow"]
''',
    )
    _write(
        arrow_locale,
        '''
class Locale:
    names = ["en", "en-us"]

    def __init__(self, name="en-us"):
        self.name = name

    def describe(self, *args, **kwargs):
        return ""

    def ordinal_number(self, n):
        return str(n)

    def meridian(self, hour, token):
        return "am" if hour < 12 else "pm"


class EnglishLocale(Locale):
    pass


def get_locale(name):
    return EnglishLocale(str(name or "en-us"))
''',
    )
    shutil.rmtree(arrow_dir / "__pycache__", ignore_errors=True)
    repaired.append("arrow")

text_unidecode_dir = site_root / "text_unidecode"
text_unidecode_init = text_unidecode_dir / "__init__.py"
if _has_nul(text_unidecode_init):
    print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(text_unidecode_init), file=sys.stderr)
    _write(
        text_unidecode_init,
        '''
def unidecode(value):
    import unicodedata

    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


__all__ = ["unidecode"]
''',
    )
    shutil.rmtree(text_unidecode_dir / "__pycache__", ignore_errors=True)
    repaired.append("text_unidecode")

slugify_dir = site_root / "slugify"
slugify_files = [
    slugify_dir / "__init__.py",
    slugify_dir / "slugify.py",
    slugify_dir / "special.py",
    slugify_dir / "__version__.py",
    slugify_dir / "__main__.py",
]
if any(_has_nul(path) for path in slugify_files):
    for path in slugify_files:
        if _has_nul(path):
            print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(path), file=sys.stderr)
    _write(
        slugify_dir / "slugify.py",
        '''
import re
import unicodedata


def _ascii(value):
    return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")


def smart_truncate(string, max_length=0, word_boundary=False, separator=" ", save_order=False):
    text = str(string)
    if not max_length or len(text) <= max_length:
        return text
    if word_boundary:
        truncated = text[:max_length]
        boundary = truncated.rfind(separator)
        if boundary > 0:
            return truncated[:boundary]
    return text[:max_length]


def slugify(
    text,
    entities=True,
    decimal=True,
    hexadecimal=True,
    max_length=0,
    word_boundary=False,
    separator="-",
    save_order=False,
    stopwords=(),
    regex_pattern=None,
    lowercase=True,
    replacements=(),
    allow_unicode=False,
):
    value = str(text)
    for old, new in replacements or ():
        value = value.replace(str(old), str(new))
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = _ascii(value)
    if lowercase:
        value = value.lower()
    pattern = regex_pattern or r"[^\\w\\s-]"
    value = re.sub(pattern, separator, value)
    value = re.sub(r"[-\\s_]+", separator, value).strip(separator)
    if stopwords:
        blocked = set(str(word).lower() if lowercase else str(word) for word in stopwords)
        value = separator.join(part for part in value.split(separator) if part not in blocked)
    if max_length:
        value = smart_truncate(value, max_length, word_boundary, separator, save_order)
        value = value.strip(separator)
    return value
''',
    )
    _write(
        slugify_dir / "__init__.py",
        '''
from .slugify import slugify, smart_truncate

__all__ = ["slugify", "smart_truncate"]
''',
    )
    _write(slugify_dir / "special.py", "CHARACTER_REPLACEMENTS = []\\n")
    _write(slugify_dir / "__version__.py", "__version__ = '8.0.4'\\n")
    _write(
        slugify_dir / "__main__.py",
        '''
from .slugify import slugify


def main():
    import sys

    print(slugify(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
''',
    )
    shutil.rmtree(slugify_dir / "__pycache__", ignore_errors=True)
    repaired.append("slugify")

six_file = site_root / "six.py"
if _has_nul(six_file):
    print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(six_file), file=sys.stderr)
    _write(
        six_file,
        '''
import sys
import types

PY2 = False
PY3 = True
string_types = (str,)
text_type = str
binary_type = bytes
integer_types = (int,)
class_types = (type,)
MAXSIZE = sys.maxsize


def iteritems(mapping, **kwargs):
    return iter(mapping.items(**kwargs))


def iterkeys(mapping, **kwargs):
    return iter(mapping.keys(**kwargs))


def itervalues(mapping, **kwargs):
    return iter(mapping.values(**kwargs))


def advance_iterator(iterator):
    return next(iterator)


def raise_from(value, from_value):
    raise value from from_value


def add_metaclass(metaclass):
    def wrapper(cls):
        attrs = dict(cls.__dict__)
        attrs.pop("__dict__", None)
        attrs.pop("__weakref__", None)
        return metaclass(cls.__name__, cls.__bases__, attrs)

    return wrapper


def with_metaclass(metaclass, *bases):
    class TemporaryClass(*bases):
        pass

    attrs = dict(TemporaryClass.__dict__)
    attrs.pop("__dict__", None)
    attrs.pop("__weakref__", None)
    return metaclass("TemporaryClass", bases or (object,), attrs)


def b(value):
    return value.encode("latin-1") if isinstance(value, str) else value


def u(value):
    return value


class _Moves(types.ModuleType):
    pass


moves = _Moves("six.moves")
try:
    import _thread as _thread_module
except Exception:
    _thread_module = None
moves._thread = _thread_module
moves.range = range
try:
    import winreg as _winreg_module
except Exception:
    pass
else:
    moves.winreg = _winreg_module
sys.modules[__name__ + ".moves"] = moves
''',
    )
    try:
        (site_root / "__pycache__").mkdir(exist_ok=True)
    except OSError:
        pass
    repaired.append("six")

pytest_mock_dir = site_root / "pytest_mock"
pytest_mock_files = [
    pytest_mock_dir / "__init__.py",
    pytest_mock_dir / "plugin.py",
    pytest_mock_dir / "_util.py",
    pytest_mock_dir / "_version.py",
]
if any(_has_nul(path) for path in pytest_mock_files):
    for path in pytest_mock_files:
        if _has_nul(path):
            print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(path), file=sys.stderr)
    _write(pytest_mock_dir / "__init__.py", "__version__ = '3.14.0'\\n")
    _write(pytest_mock_dir / "_version.py", "version = '3.14.0'\\n")
    _write(pytest_mock_dir / "_util.py", "")
    _write(
        pytest_mock_dir / "plugin.py",
        '''
import pytest
from unittest import mock


class _Patcher:
    def __init__(self, fixture):
        self._fixture = fixture

    def __call__(self, *args, **kwargs):
        return self._fixture._start(mock.patch(*args, **kwargs))

    def object(self, *args, **kwargs):
        return self._fixture._start(mock.patch.object(*args, **kwargs))

    def dict(self, *args, **kwargs):
        return self._fixture._start(mock.patch.dict(*args, **kwargs))

    def multiple(self, *args, **kwargs):
        return self._fixture._start(mock.patch.multiple(*args, **kwargs))


class MockerFixture:
    Mock = mock.Mock
    MagicMock = mock.MagicMock
    PropertyMock = mock.PropertyMock
    call = mock.call
    ANY = mock.ANY

    def __init__(self):
        self._patches = []
        self._mocks = []
        self.patch = _Patcher(self)

    def _start(self, patcher):
        value = patcher.start()
        self._patches.append(patcher)
        if hasattr(value, "reset_mock"):
            self._mocks.append(value)
        return value

    def stopall(self):
        while self._patches:
            self._patches.pop().stop()

    def resetall(self, *args, **kwargs):
        for value in list(self._mocks):
            value.reset_mock(*args, **kwargs)

    def spy(self, obj, name):
        original = getattr(obj, name)
        spy_obj = mock.MagicMock(wraps=original)
        self.patch.object(obj, name, spy_obj)
        return spy_obj

    def stub(self, name=None):
        return mock.MagicMock(name=name)


@pytest.fixture
def mocker():
    fixture = MockerFixture()
    try:
        yield fixture
    finally:
        fixture.stopall()
''',
    )
    shutil.rmtree(pytest_mock_dir / "__pycache__", ignore_errors=True)
    repaired.append("pytest_mock")

pytest_jsonreport_dir = site_root / "pytest_jsonreport"
pytest_jsonreport_plugin = pytest_jsonreport_dir / "plugin.py"
pytest_jsonreport_serialize = pytest_jsonreport_dir / "serialize.py"
if _has_nul(pytest_jsonreport_plugin) or _has_nul(pytest_jsonreport_serialize):
    for path in (pytest_jsonreport_plugin, pytest_jsonreport_serialize):
        if _has_nul(path):
            print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(path), file=sys.stderr)
    _write(pytest_jsonreport_dir / "__init__.py", "__version__ = '1.5.0'\\n")
    _write(
        pytest_jsonreport_serialize,
        '''
def pytest_json_modifyreport(json_report):
    return None
''',
    )
    _write(
        pytest_jsonreport_plugin,
        '''
import json
import time
from pathlib import Path

_enabled = False
_started_at = 0.0
_tests = {{}}
_collection_errors = []


def _safe_addoption(group, *args, **kwargs):
    try:
        group.addoption(*args, **kwargs)
    except ValueError:
        pass


def pytest_addoption(parser):
    group = parser.getgroup("json-report")
    _safe_addoption(group, "--json-report", action="store_true", dest="json_report", default=False)
    _safe_addoption(group, "--json-report-file", action="store", dest="json_report_file", default="report.json")
    _safe_addoption(group, "--json-report-indent", action="store", dest="json_report_indent", default=None)
    _safe_addoption(group, "--json-report-verbosity", action="store", dest="json_report_verbosity", default=None)
    _safe_addoption(group, "--json-report-summary", action="store_true", dest="json_report_summary", default=True)
    _safe_addoption(group, "--json-report-omit", action="append", dest="json_report_omit", default=[])


def pytest_configure(config):
    global _enabled, _started_at, _tests, _collection_errors
    try:
        _enabled = bool(config.getoption("json_report"))
    except Exception:
        _enabled = False
    _started_at = time.time()
    _tests = {{}}
    _collection_errors = []


def _phase_payload(report):
    payload = {{"outcome": report.outcome}}
    duration = getattr(report, "duration", None)
    if duration is not None:
        payload["duration"] = duration
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail:
        payload["wasxfail"] = wasxfail
    longrepr = getattr(report, "longreprtext", "")
    if longrepr:
        payload["longrepr"] = longrepr
    return payload


def _final_outcome(test):
    setup = test.get("setup") or {{}}
    call = test.get("call") or {{}}
    teardown = test.get("teardown") or {{}}
    for phase in (setup, teardown):
        if phase.get("outcome") == "failed":
            return "error"
    if call:
        if call.get("wasxfail") and call.get("outcome") == "skipped":
            return "xfailed"
        if call.get("wasxfail") and call.get("outcome") == "passed":
            return "xpassed"
        return call.get("outcome") or "passed"
    if setup.get("outcome") == "skipped":
        return "skipped"
    if setup.get("outcome") == "failed":
        return "error"
    return test.get("outcome") or "passed"


def pytest_runtest_logreport(report):
    if not _enabled:
        return
    entry = _tests.setdefault(
        report.nodeid,
        {{"nodeid": report.nodeid, "keywords": [], "outcome": "passed"}},
    )
    entry[report.when] = _phase_payload(report)
    entry["outcome"] = _final_outcome(entry)


def pytest_collectreport(report):
    if not _enabled or getattr(report, "outcome", "") != "failed":
        return
    nodeid = getattr(report, "nodeid", "") or getattr(report, "fspath", "")
    text = getattr(report, "longreprtext", "")
    _collection_errors.append(
        {{
            "nodeid": str(nodeid),
            "outcome": "error",
            "setup": {{"outcome": "failed", "longrepr": str(text)}},
        }}
    )


def pytest_sessionfinish(session, exitstatus):
    if not _enabled:
        return
    tests = list(_tests.values()) + list(_collection_errors)
    for test in tests:
        test["outcome"] = _final_outcome(test)
    summary = {{
        "passed": 0,
        "failed": 0,
        "error": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }}
    for test in tests:
        outcome = test.get("outcome")
        if outcome in summary:
            summary[outcome] += 1
        elif outcome == "error":
            summary["error"] += 1
    summary["total"] = len(tests)
    summary["collected"] = len(tests)
    payload = {{
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(_started_at)),
        "duration": max(0.0, time.time() - _started_at),
        "exitcode": int(exitstatus),
        "root": str(getattr(session.config, "rootpath", session.config.rootdir)),
        "summary": summary,
        "tests": tests,
    }}
    report_file = session.config.getoption("json_report_file") or "report.json"
    indent = session.config.getoption("json_report_indent")
    try:
        indent_value = None if indent in (None, "", "none", "None") else int(indent)
    except Exception:
        indent_value = None
    path = Path(str(report_file))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent_value), encoding="utf-8")
''',
    )
    shutil.rmtree(pytest_jsonreport_dir / "__pycache__", ignore_errors=True)
    repaired.append("pytest_jsonreport")

pytest_cov_dir = site_root / "pytest_cov"
pytest_cov_files = [
    pytest_cov_dir / "__init__.py",
    pytest_cov_dir / "plugin.py",
    pytest_cov_dir / "compat.py",
    pytest_cov_dir / "embed.py",
    pytest_cov_dir / "engine.py",
]
if any(_has_nul(path) for path in pytest_cov_files):
    for path in pytest_cov_files:
        if _has_nul(path):
            print("APEX_COOKIECUTTER_DEP_NUL_FILE: " + str(path), file=sys.stderr)
    pytest_cov_plugin = '''
def _safe_addoption(group, *args, **kwargs):
    try:
        group.addoption(*args, **kwargs)
    except ValueError:
        pass


def pytest_addoption(parser):
    group = parser.getgroup("cov")
    _safe_addoption(group, "--cov", action="append", default=[], dest="cov_source")
    _safe_addoption(group, "--cov-reset", action="store_true", default=False, dest="cov_reset")
    _safe_addoption(group, "--cov-report", action="append", default=[], dest="cov_report")
    _safe_addoption(group, "--cov-config", action="store", default=".coveragerc", dest="cov_config")
    _safe_addoption(group, "--cov-branch", action="store_true", default=False, dest="cov_branch")
    _safe_addoption(group, "--cov-append", action="store_true", default=False, dest="cov_append")
    _safe_addoption(group, "--cov-fail-under", action="store", default=None, dest="cov_fail_under")
    _safe_addoption(group, "--cov-context", action="store", default=None, dest="cov_context")
    _safe_addoption(group, "--no-cov", action="store_true", default=False, dest="no_cov")
    _safe_addoption(group, "--no-cov-on-fail", action="store_true", default=False, dest="no_cov_on_fail")


def pytest_configure(config):
    return None
'''
    _write(
        pytest_cov_dir / "__init__.py",
        pytest_cov_plugin + "\\n__version__ = '4.1.0'\\n",
    )
    _write(pytest_cov_dir / "plugin.py", pytest_cov_plugin)
    _write(pytest_cov_dir / "compat.py", "")
    _write(pytest_cov_dir / "embed.py", "")
    _write(pytest_cov_dir / "engine.py", "")
    shutil.rmtree(pytest_cov_dir / "__pycache__", ignore_errors=True)
    repaired.append("pytest_cov")

if repaired:
    importlib.invalidate_caches()
    if "arrow" in repaired:
        sys.modules.pop("arrow", None)
        import arrow

        arrow.now().shift(days=1).format("%Y-%m-%d")
    if "slugify" in repaired or "text_unidecode" in repaired:
        sys.modules.pop("slugify", None)
        from slugify import slugify

        assert slugify("It's slugified Foobar") == "it-s-slugified-foobar"
    if "six" in repaired:
        sys.modules.pop("six", None)
        sys.modules.pop("six.moves", None)
        import six
        from six.moves import _thread, range

        assert six.PY3 and list(range(1)) == [0] and _thread is not None
    if "pytest_mock" in repaired:
        sys.modules.pop("pytest_mock", None)
        sys.modules.pop("pytest_mock.plugin", None)
        import pytest_mock.plugin  # noqa: F401
    if "pytest_jsonreport" in repaired:
        sys.modules.pop("pytest_jsonreport.plugin", None)
        import pytest_jsonreport.plugin  # noqa: F401
    if "pytest_cov" in repaired:
        sys.modules.pop("pytest_cov", None)
        import pytest_cov  # noqa: F401
    print("APEX_COOKIECUTTER_DEP_REPAIRED: " + ",".join(repaired), file=sys.stderr)
PY
fi
""".strip()


def _commit0_official_image_python_repair_eval_step() -> str:
    return _commit0_official_image_python_env_repair_shell("/testbed/.venv/bin/python")


def _commit0_official_image_python_repair_command(container_venv: str) -> str:
    container_venv = str(container_venv or _COMMIT0_OFFICIAL_TESTBED_VENV).rstrip("/")
    return _commit0_official_image_python_env_repair_shell(f"{container_venv}/bin/python")


def _commit0_official_image_python_repair_required(repo_name: Any) -> bool:
    # Commit0 cookiecutter image fact: that image has NUL-filled installed
    # dependency/test-plugin files; other official images should not carry the
    # large repair script in per-rollout Docker argv/env.
    return str(repo_name or "").strip().rstrip("/").split("/")[-1] == "cookiecutter"


def _commit0_official_runner_script() -> str:
    # Commit0 run_pytest_ids emits non-binary git diffs; binary .dat files need
    # --binary/--full-index so the official Docker image can apply candidate state.
    script = """
import json
import os
import sys
import traceback

from commit0.harness import run_pytest_ids as _apex_commit0_run_pytest_ids


_APEX_OFFICIAL_IMAGE_PYTHON_ENV_REPAIR = __APEX_OFFICIAL_IMAGE_PYTHON_ENV_REPAIR__


_APEX_PATCH_ADD_CLEANUP = r'''
python - <<'PY'
from pathlib import Path
import shutil
import shlex
import subprocess


def _is_tracked(path: str) -> bool:
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


removed = set()
current_new_path = None
old_path = None


def _remove_untracked(path: str) -> None:
    if not path or path == "/dev/null" or path in removed or _is_tracked(path):
        return
    target = Path(path)
    if target.is_symlink() or target.is_file():
        target.unlink()
        removed.add(path)
    elif target.is_dir():
        shutil.rmtree(target)
        removed.add(path)


for line in Path("/patch.diff").read_text(errors="replace").splitlines():
    if line.startswith("diff --git "):
        old_path = None
        current_new_path = None
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        if len(parts) >= 4:
            current_new_path = parts[3]
            if current_new_path.startswith("b/"):
                current_new_path = current_new_path[2:]
        continue
    if line.startswith("new file mode ") and current_new_path:
        _remove_untracked(current_new_path)
        continue
    if line.startswith("--- "):
        old_path = line[4:].strip()
        continue
    if not line.startswith("+++ ") or old_path != "/dev/null":
        continue
    new_path = line[4:].strip()
    if new_path.startswith("b/"):
        new_path = new_path[2:]
    _remove_untracked(new_path)
PY
'''.strip()


def _apex_generate_binary_patch_between_commits(repo, old_commit, new_commit):
    def _commit_exists(rev):
        try:
            repo.git.rev_parse("--verify", "--quiet", str(rev) + "^{commit}")
            return True
        except Exception:
            return False

    base = old_commit
    # The V1 anti-cheat flatten (rm -rf .git + git init -> a fresh rootless
    # ``apex-base`` commit) physically removes the original commit0 base commit,
    # so the dataset base SHA is a "bad object" in the audited checkout and the
    # base->head diff aborts with exit 128 (audit reported pass_rate=0.0 /
    # parser_error although the candidate was a real 1.0). Fall back to the
    # checkout's OWN root commit (the flattened base, identical in tracked
    # content to the original base) so the candidate patch stays base->head and
    # applies cleanly in the official Docker image. No-op when the base exists.
    if not _commit_exists(base):
        roots = []
        try:
            roots = repo.git.rev_list("--max-parents=0", new_commit).split()
        except Exception:
            roots = []
        if not roots:
            raise Exception(
                "Error generating patch: base commit "
                + str(old_commit)
                + " not found and no root commit resolvable from "
                + str(new_commit)
            )
        base = roots[-1]
    try:
        patch = repo.git.diff(
            base,
            new_commit,
            "--binary",
            "--full-index",
            "--",
            ".",
            ":(exclude)spec.pdf.bz2",
        )
        return patch + "\\n\\n"
    except Exception as exc:
        raise Exception(f"Error generating patch: {exc}") from exc


_apex_original_make_spec = _apex_commit0_run_pytest_ids.make_spec


def _apex_make_spec_with_patch_cleanup(example):
    spec = _apex_original_make_spec(example)
    spec.eval_script_list = list(spec.eval_script_list)
    _apex_repo_name = str((example or {}).get("repo", "")).rstrip("/").split("/")[-1]
    if (
        _apex_repo_name == "cookiecutter"
        and _APEX_OFFICIAL_IMAGE_PYTHON_ENV_REPAIR not in spec.eval_script_list
    ):
        spec.eval_script_list.insert(0, _APEX_OFFICIAL_IMAGE_PYTHON_ENV_REPAIR)
    marker = "git apply --allow-empty -v /patch.diff"
    if marker in spec.eval_script_list:
        # Commit0 repo images can carry untracked generated artifacts; remove only
        # untracked paths the candidate patch is about to add before git apply.
        patched_steps = []
        for step in spec.eval_script_list:
            if step == marker:
                patched_steps.append(_APEX_PATCH_ADD_CLEANUP)
            patched_steps.append(step)
        spec.eval_script_list = patched_steps
    return spec


def _apex_build_merged_dataset_loader(_real_load_dataset, _primary_rev, _fallback_revs):
    # commit0.run_pytest_ids.main loads the dataset at the latest mutable revision
    # and matches the audited repo's spec by name. Gold-suite rows dropped from that
    # revision (pytest, dropped 2024-09-22) abort with "No spec available", scoring a
    # real candidate as audit_inconclusive. Return a loader that merges the
    # fallback-revision rows for any repo missing from the primary load so the
    # audited repo's spec is always resolvable -- mirrors the benchmark's own
    # task-resolution. Primary rows win on repo-name collision; fallbacks are
    # appended only for repos the primary revision no longer carries.
    def _apex_merged_load_dataset(_name, *_a, **_kw):
        if _primary_rev and "revision" not in _kw:
            _kw = dict(_kw)
            _kw["revision"] = _primary_rev
        try:
            _rows = list(_real_load_dataset(_name, *_a, **_kw))
        except Exception:
            _rows = []
        _seen = set()
        for _row in _rows:
            try:
                _seen.add(_row["repo"].split("/")[-1])
            except Exception:
                pass
        for _rev in _fallback_revs:
            try:
                _kw2 = dict(_kw)
                _kw2["revision"] = _rev
                for _row in _real_load_dataset(_name, *_a, **_kw2):
                    try:
                        _rn = _row["repo"].split("/")[-1]
                    except Exception:
                        continue
                    if _rn not in _seen:
                        _rows.append(_row)
                        _seen.add(_rn)
            except Exception:
                continue
        return _rows

    return _apex_merged_load_dataset


_apex_commit0_run_pytest_ids.generate_patch_between_commits = (
    _apex_generate_binary_patch_between_commits
)
_apex_commit0_run_pytest_ids.make_spec = _apex_make_spec_with_patch_cleanup


def _apex_exit_code_from_system_exit(_exc):
    _code = _exc.code
    if _code is None:
        return 0
    if isinstance(_code, int):
        return int(_code)
    print(str(_code), file=sys.stderr)
    return 1


def _apex_exit_immediately(_code):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # Commit0's official local runner imports GitPython; after pytest has
        # completed and written logs/report, GitPython cleanup can leave a
        # ``git cat-file --batch-check`` helper blocking interpreter teardown.
        # This wrapper runs in a dedicated subprocess, so force-exit with the
        # upstream pytest code once scoring is complete.
        if os.environ.get("APEX_COMMIT0_OFFICIAL_RUNNER_IN_PROCESS_TEST") == "1":
            raise SystemExit(_code)
        os._exit(int(_code))

# These keys are APEX-internal (not part of commit0.main's signature); pop them
# before the call and use them to revision-pin the spec lookup.
_apex_kwargs = json.loads(sys.argv[1])
_apex_primary_rev = _apex_kwargs.pop("_apex_dataset_primary_revision", None)
_apex_fallback_revs = _apex_kwargs.pop("_apex_dataset_fallback_revisions", None) or []
if _apex_primary_rev or _apex_fallback_revs:
    _apex_commit0_run_pytest_ids.load_dataset = _apex_build_merged_dataset_loader(
        _apex_commit0_run_pytest_ids.load_dataset,
        _apex_primary_rev,
        _apex_fallback_revs,
    )
try:
    _apex_commit0_run_pytest_ids.main(**_apex_kwargs)
except SystemExit as _apex_exc:
    _apex_exit_immediately(_apex_exit_code_from_system_exit(_apex_exc))
except Exception:
    traceback.print_exc()
    _apex_exit_immediately(1)
else:
    _apex_exit_immediately(0)
""".strip()
    return script.replace(
        "__APEX_OFFICIAL_IMAGE_PYTHON_ENV_REPAIR__",
        repr(_commit0_official_image_python_repair_eval_step()),
    )


COMMIT0_DEFAULT_DATASET_NAME = "wentingzhao/commit0_combined"
# Commit0 dataset main dropped commit-0/pytest after 2024-09-22; keep the newest
# historical revision as a fallback so explicitly requested legacy rows resolve.
COMMIT0_DEFAULT_DATASET_FALLBACK_REVISIONS = [
    "afc4d5f9085597e14e2b2a5bdbae28577ecd7ecb",
]
_COMMIT0_PREPARED_RUNTIME_POLICY_DIRNAME = "commit0_prepared_runtime_python_policy"
_COMMIT0_EGRESS_ALLOW_HOSTS_ENV = "APEX_COMMIT0_EGRESS_ALLOW_HOSTS"
_COMMIT0_AGENT_COMMAND_TIMEOUT_ENV = "APEX_COMMIT0_AGENT_COMMAND_TIMEOUT_SECONDS"
_ACTIVE_OUTER_TASK_COUNT_ENV = "APEX_ACTIVE_OUTER_TASK_COUNT"
_WAITING_OUTER_TASK_COUNT_ENV = "APEX_WAITING_OUTER_TASK_COUNT"

_COMMIT0_PREPARED_RUNTIME_SITECUSTOMIZE = """\
# Intentionally empty. Commit0 solve-phase isolation is enforced by Docker
# mounts, git-history flattening, and the container/proxy egress boundary.
"""


def _commit0_prepared_runtime_container_guard_script(
    *,
    container_venv: str = "",
) -> str:
    target_python = ""
    if str(container_venv or "").strip():
        target_python = f"{str(container_venv).rstrip('/')}/bin/python"
    quoted_target_python = shlex.quote(target_python)
    return f"""set -eu
TARGET_PYTHON={quoted_target_python}
APEX_GUARD_SHELL="$(readlink -f /bin/sh 2>/dev/null || command -v sh 2>/dev/null || printf '%s' /bin/sh)"
install_python_guard_at_path() {{
  path="$1"
  route_to_target="${{2:-0}}"
  real="$path.apex-real"
  [ -e "$path" ] || return 0
  if [ ! -e "$real" ]; then
    resolved="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
    if [ -e "$resolved.apex-real" ]; then
      ln -s "$resolved.apex-real" "$real"
      rm -f "$path"
    elif [ "$resolved" = "$path" ]; then
      mv "$path" "$real"
    else
      ln -s "$resolved" "$real"
      rm -f "$path"
    fi
  fi
  cat > "$path" <<APEX_PYTHON_GUARD
#!$APEX_GUARD_SHELL
apex_commit0_python_guard_timeout_bin() {{
  for candidate in /usr/bin/timeout.apex-real /bin/timeout.apex-real /usr/local/bin/timeout.apex-real /usr/bin/timeout /bin/timeout /usr/local/bin/timeout; do
    [ -x "\\$candidate" ] || continue
    printf '%s\\n' "\\$candidate"
    return 0
  done
  return 1
}}
apex_commit0_python_guard_exec() {{
  executable="\\$1"
  shift
  timeout_seconds="\\${{{_COMMIT0_AGENT_COMMAND_TIMEOUT_ENV}:-}}"
  if [ -n "\\$timeout_seconds" ] && [ "\\$1" = "-m" ] && [ "\\$2" = "pytest" ]; then
    case "\\$timeout_seconds" in
      *[!0-9]*|0) ;;
      *)
        if timeout_bin="$(apex_commit0_python_guard_timeout_bin)"; then
          exec "\\$timeout_bin" -k 5 "\\$timeout_seconds" "\\$executable" "\\$@"
        fi
        ;;
    esac
  fi
  exec "\\$executable" "\\$@"
}}
if [ "$route_to_target" = "1" ] && [ -n "$TARGET_PYTHON" ] && [ -x "$TARGET_PYTHON.apex-real" ]; then
  apex_commit0_python_guard_exec "$TARGET_PYTHON.apex-real" "\\$@"
fi
if [ "$route_to_target" = "1" ] && [ -n "$TARGET_PYTHON" ] && [ -x "$TARGET_PYTHON" ]; then
  apex_commit0_python_guard_exec "$TARGET_PYTHON" "\\$@"
fi
apex_commit0_python_guard_exec "$real" "\\$@"
APEX_PYTHON_GUARD
  chmod 0755 "$path"
}}

install_pip_guard_at_path() {{
  path="$1"
  route_to_target="${{2:-0}}"
  real="$path.apex-real"
  [ -e "$path" ] || return 0
  if [ ! -e "$real" ]; then
    resolved="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
    if [ -e "$resolved.apex-real" ]; then
      ln -s "$resolved.apex-real" "$real"
      rm -f "$path"
    elif [ "$resolved" = "$path" ]; then
      mv "$path" "$real"
    else
      ln -s "$resolved" "$real"
      rm -f "$path"
    fi
  fi
  cat > "$path" <<APEX_PIP_GUARD
#!$APEX_GUARD_SHELL
if [ -n "$TARGET_PYTHON" ] && [ -x "$TARGET_PYTHON.apex-real" ]; then
  exec "$TARGET_PYTHON.apex-real" -m pip "\\$@"
fi
if [ "$route_to_target" = "1" ] && [ -n "$TARGET_PYTHON" ] && [ -x "$TARGET_PYTHON" ]; then
  exec "$TARGET_PYTHON" -m pip "\\$@"
fi
exec "$real" "\\$@"
APEX_PIP_GUARD
  chmod 0755 "$path"
}}

# Commit0 official uv-built venvs can symlink /testbed/.venv/bin/python
# into /root/.local; non-root CLI agents need execute traversal to run
# the scoring interpreter.
ensure_agent_can_execute_target_python() {{
  [ -n "$TARGET_PYTHON" ] || return 0
  real="$TARGET_PYTHON.apex-real"
  [ -e "$real" ] || return 0
  resolved="$(readlink -f "$real" 2>/dev/null || printf '%s' "$real")"
  [ -e "$resolved" ] || return 0
  chmod a+rx "$resolved" 2>/dev/null || true
  dir="$(dirname "$resolved")"
  while [ -n "$dir" ] && [ "$dir" != "/" ]; do
    chmod a+x "$dir" 2>/dev/null || true
    parent="$(dirname "$dir")"
    [ "$parent" = "$dir" ] && break
    dir="$parent"
  done
}}

for dir in /usr/local/bin /usr/bin; do
  [ -d "$dir" ] || continue
  for tool in python python3 python3.10 python3.11 python3.12 python3.13; do
    route=0
    [ -n "$TARGET_PYTHON" ] && route=1
    install_python_guard_at_path "$dir/$tool" "$route"
  done
  for tool in pip pip3; do
    route=0
    [ "$dir" = "/usr/local/bin" ] && route=1
    install_pip_guard_at_path "$dir/$tool" "$route"
  done
done

install_venv_guards() {{
  [ -n "$TARGET_PYTHON" ] || return 0
  venv_bin="$(dirname "$TARGET_PYTHON")"
  [ -d "$venv_bin" ] || return 0
  for candidate in "$venv_bin"/python "$venv_bin"/python3 "$venv_bin"/python3.*; do
    [ -e "$candidate" ] || continue
    install_python_guard_at_path "$candidate" 0
  done
  for candidate in "$venv_bin"/pip "$venv_bin"/pip3; do
    [ -e "$candidate" ] || continue
    install_pip_guard_at_path "$candidate" 1
  done
}}

install_venv_guards
ensure_agent_can_execute_target_python
"""


# Helpers for Commit0 expected-test-id scoring. The ids file is staged into
# each evaluation worktree so APEX can score the public Commit0 universe from
# pytest-json-report while still executing the benchmark's full test directory.
# The runner/plugin remain available for narrow diagnostics, but the primary
# gold-suite path does not deselect extra tests.
_APEX_EXPECTED_IDS_PLUGIN_NAME = "_apex_expected_ids_filter"
_APEX_EXPECTED_IDS_PLUGIN_FILENAME = f"{_APEX_EXPECTED_IDS_PLUGIN_NAME}.py"
_APEX_EXPECTED_IDS_RUNNER_NAME = "_apex_run_expected_ids"
_APEX_EXPECTED_IDS_RUNNER_FILENAME = f"{_APEX_EXPECTED_IDS_RUNNER_NAME}.py"
_APEX_EXPECTED_IDS_FILENAME = ".apex_expected_test_ids.txt"
_APEX_EXPECTED_IDS_MIRROR_FILENAME = "_apex_expected_test_ids.txt"
_APEX_LOCAL_EVAL_REPORT_FILENAME = "_apex_eval_report.json"
# Commit0/Python solve fact: rollout pytest JSON is harness evidence, so keep
# agent-run reports in APEX's ignored verification directory instead of repo root.
_APEX_ROLLOUT_REPORT_FILENAME = ".apex_verification_reports/rollout_report.json"
_APEX_PREPARE_DIAGNOSTICS_FILENAME = ".apex_prepare_diagnostics.json"
_APEX_EXPECTED_IDS_ENV_VAR = "APEX_EXPECTED_IDS_FILE"
_COMMIT0_PYTEST_XDIST_VENDOR_CONTAINER_ROOT = "/opt/apex-pytest-xdist-vendor"
_APEX_CORE_DIR = Path(__file__).resolve().parent.parent / "core"
_APEX_EXPECTED_IDS_PLUGIN_SOURCE = _APEX_CORE_DIR / _APEX_EXPECTED_IDS_PLUGIN_FILENAME
_APEX_EXPECTED_IDS_RUNNER_SOURCE = _APEX_CORE_DIR / _APEX_EXPECTED_IDS_RUNNER_FILENAME
# APEX-staged pytest harness helpers (the expected-id filter plugin + runner +
# id-list mirror). They sit at the repo root during a solve but are NEVER repo
# source modules, so they must not be treated as local module roots (which would
# pollute the baseline import-gap signal and trigger a spurious APEX_MISS /
# Docker retry). Stems are already lowercase.
_APEX_HARNESS_HELPER_STEMS = frozenset(
    {
        _APEX_EXPECTED_IDS_PLUGIN_NAME,
        _APEX_EXPECTED_IDS_RUNNER_NAME,
        Path(_APEX_EXPECTED_IDS_MIRROR_FILENAME).stem,
    }
)

# Commit0 expected-id scoring disables pytest plugin autoload; explicitly load
# common pytest plugin packages that provide fixtures/options used by tests.
_COMMIT0_PYTEST_PLUGIN_PACKAGE_MODULES = {
    "pytest-benchmark": "pytest_benchmark.plugin",
    # pytest-codspeed and pytest-benchmark expose distinct pytest11 entry points; Commit0 pydantic config declares both.
    "pytest-codspeed": "pytest_codspeed.plugin",
    "pytest-factoryboy": "pytest_factoryboy.plugin",
    "pytest-mock": "pytest_mock",
    "pytest-randomly": "pytest_randomly",
    "pytest-repeat": "pytest_repeat",
    # pytest-asyncio 0.21 exposes pytest hooks from pytest_asyncio.plugin; Commit0 disables autoload so load that module explicitly.
    "pytest-asyncio": "pytest_asyncio.plugin",
}
_COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES = {
    # Commit0 disables autoload; repos with --cov addopts need pytest-cov's option parser loaded explicitly.
    "pytest-cov": "pytest_cov",
    # P0.2: pydantic's addopts pass --memray; under PYTEST_DISABLE_PLUGIN_AUTOLOAD the
    # gate inferred+installed pytest-memray but never emitted `-p pytest_memray`, so
    # pytest exited rc=4 (unknown option) BEFORE collection -> correct code scored zero.
    "pytest-memray": "pytest_memray",
}

# O4/NEW-I7: repo addopts (read by pytest from pyproject/pytest.ini/setup.cfg/tox.ini
# at runtime) can carry plugin-provided options like ``--memray``. Commit0 scoring
# runs pytest with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, so a plugin option whose package
# is NOT importable in the scoring venv makes pytest exit rc=4 ("unrecognized
# arguments") BEFORE collection — scoring correct code as zero. This maps each
# plugin-provided pytest *option* (the literal flag, sans value) to the import module
# that must be loadable for the option to be recognized. Core pytest options are NOT
# listed here and are therefore never stripped. The package map above is folded in so
# both stay in sync.
_COMMIT0_PYTEST_OPTION_FLAG_PLUGIN_MODULES = {
    # pytest-memray
    "--memray": "pytest_memray",
    "--most-allocations": "pytest_memray",
    "--stacks": "pytest_memray",
    "--hide-memray-summary": "pytest_memray",
    "--memray-bin-path": "pytest_memray",
    "--memray-bin-prefix": "pytest_memray",
    "--native": "pytest_memray",
    "--trace-python-allocators": "pytest_memray",
    "--fail-on-increase": "pytest_memray",
    # pytest-cov
    "--cov": "pytest_cov",
    "--cov-append": "pytest_cov",
    "--cov-branch": "pytest_cov",
    "--cov-config": "pytest_cov",
    "--cov-context": "pytest_cov",
    "--cov-fail-under": "pytest_cov",
    "--cov-report": "pytest_cov",
    "--cov-reset": "pytest_cov",
    "--no-cov": "pytest_cov",
    "--no-cov-on-fail": "pytest_cov",
    # pytest-timeout
    "--timeout": "pytest_timeout",
    "--timeout-method": "pytest_timeout",
    "--timeout-disable-debugger-detection": "pytest_timeout",
    "--session-timeout": "pytest_timeout",
    # pytest-rerunfailures
    "--reruns": "pytest_rerunfailures",
    "--reruns-delay": "pytest_rerunfailures",
    "--only-rerun": "pytest_rerunfailures",
    # pytest-xdist
    "-n": "xdist",
    "--numprocesses": "xdist",
    "--dist": "xdist",
    "--tx": "xdist",
    "--maxschedchunk": "xdist",
    "--maxprocesses": "xdist",
    # pytest-benchmark
    "--benchmark-only": "pytest_benchmark",
    "--benchmark-disable": "pytest_benchmark",
    "--benchmark-enable": "pytest_benchmark",
    "--benchmark-skip": "pytest_benchmark",
    "--benchmark-autosave": "pytest_benchmark",
    "--benchmark-save": "pytest_benchmark",
    "--benchmark-columns": "pytest_benchmark",
    "--benchmark-group-by": "pytest_benchmark",
    "--benchmark-sort": "pytest_benchmark",
    "--benchmark-storage": "pytest_benchmark",
    "--benchmark-warmup": "pytest_benchmark",
    "--benchmark-warmup-iterations": "pytest_benchmark",
    # pytest-json-report
    "--json-report": "pytest_jsonreport.plugin",
    "--json-report-file": "pytest_jsonreport.plugin",
    # pytest-asyncio
    "--asyncio-mode": "pytest_asyncio.plugin",
}
# Pytest options that consume the following token as their value (so stripping the
# option must also strip its detached value). Only plugin-provided value options are
# listed; core options never reach the strip path.
_COMMIT0_PYTEST_PLUGIN_OPTION_FLAGS_WITH_VALUES = frozenset(
    {
        "-n",
        "--numprocesses",
        "--dist",
        "--tx",
        "--maxschedchunk",
        "--maxprocesses",
        "--cov",
        "--cov-config",
        "--cov-context",
        "--cov-fail-under",
        "--cov-report",
        "--timeout",
        "--timeout-method",
        "--session-timeout",
        "--reruns",
        "--reruns-delay",
        "--only-rerun",
        "--memray-bin-path",
        "--memray-bin-prefix",
        "--most-allocations",
        "--stacks",
        "--benchmark-save",
        "--benchmark-columns",
        "--benchmark-group-by",
        "--benchmark-sort",
        "--benchmark-storage",
        "--benchmark-warmup",
        "--benchmark-warmup-iterations",
        "--json-report-file",
        "--asyncio-mode",
    }
)
# Commit0/Python dependency shadows: when solve agents materialize installed
# packages at repo root, those paths are environment/harness artifacts, not
# source repairs. The aliases cover common distribution-name/import-root splits.
_COMMIT0_REQUIREMENT_IMPORT_ROOT_ALIASES = {
    "arrow": ("arrow", "dateutil", "six"),
    "beautifulsoup4": ("bs4",),
    "freezegun": ("freezegun",),
    "jinja2": ("jinja2", "markupsafe"),
    "markupsafe": ("markupsafe",),
    "pillow": ("PIL",),
    "pyyaml": ("yaml",),
    "python-dateutil": ("dateutil",),
    "python-slugify": ("slugify", "text_unidecode"),
    "requests": ("requests", "certifi", "charset_normalizer", "idna", "urllib3"),
    "text-unidecode": ("text_unidecode",),
}
_COMMIT0_PYTEST_PLUGIN_METADATA_FILES = (
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "tox.ini",
)


def _stage_expected_ids_filter(
    repo_dir: Path,
    expected_test_ids: list[str],
) -> Path:
    """Materialize the pytest plugin, runner, and ids file inside ``repo_dir``.

    Returns the path to the ids file. The primary Commit0 gold-suite command
    executes the full pytest directory and scores expected IDs from the JSON
    report; the runner/plugin remain staged for narrow diagnostics only.
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    _exclude_commit0_harness_helpers_from_git_status(repo_dir)
    ids_path = repo_dir / _APEX_EXPECTED_IDS_FILENAME
    ids_text = "\n".join(expected_test_ids) + "\n"
    ids_path.write_text(ids_text, encoding="utf-8")
    # Some Commit0 sanitized/copy evaluation paths preserve helper scripts but drop hidden dotfiles, so stage a non-hidden mirror.
    (repo_dir / _APEX_EXPECTED_IDS_MIRROR_FILENAME).write_text(
        ids_text,
        encoding="utf-8",
    )
    plugin_path = repo_dir / _APEX_EXPECTED_IDS_PLUGIN_FILENAME
    plugin_path.write_text(
        _APEX_EXPECTED_IDS_PLUGIN_SOURCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner_path = repo_dir / _APEX_EXPECTED_IDS_RUNNER_FILENAME
    runner_path.write_text(
        _APEX_EXPECTED_IDS_RUNNER_SOURCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return ids_path


# ---------------------------------------------------------------------------
# TIER 2 (T2.4): per-module expected-test-id subset mapping (Layer B).
#
# ECOSYSTEM FACT (Python/pytest, with inline rationale): a pytest node-id is
# ``<path/to/test_file.py>::<test>``. The test file's package/import path is
# the strongest path-evident signal for which source module a test exercises:
# a test at ``statsmodels/tsa/tests/test_arima.py`` belongs to the ``tsa``
# subpackage, whose source lives under ``statsmodels/tsa/``. We therefore map
# each expected node-id to a module group by matching the *package directory*
# the test file sits under against each group's owned source files' package
# directories. This is the only place the node-id<->package-path ecosystem
# fact lives; it stays in the Layer-B commit0 adapter and never leaks a
# repo/language conditional into Layer-A orchestration.
# ---------------------------------------------------------------------------


def _node_id_path(node_id: str) -> str:
    """Return the file portion of a pytest node-id (before ``::``)."""
    text = str(node_id or "").strip()
    if not text:
        return ""
    return text.split("::", 1)[0].replace("\\", "/").lstrip("./")


def _package_dir_tokens(rel_path: str) -> set[str]:
    """Return the set of package directory prefixes for a repo-relative path.

    ``statsmodels/tsa/tests/test_arima.py`` ->
    {"statsmodels", "statsmodels/tsa", "statsmodels/tsa/tests"} (minus the
    filename). These prefixes let us match a test file to the source files of
    the same subpackage by longest shared package prefix.
    """
    text = str(rel_path or "").strip().replace("\\", "/").lstrip("./")
    if not text:
        return set()
    parts = [part for part in Path(text).parts if part not in (".", "")]
    if not parts:
        return set()
    # Drop the trailing filename component; keep directory prefixes.
    dir_parts = parts[:-1] if parts[-1].endswith(".py") else parts
    tokens: set[str] = set()
    for index in range(1, len(dir_parts) + 1):
        tokens.add("/".join(dir_parts[:index]))
    return tokens


def _test_package_dir(node_id: str) -> str:
    """Return the package directory a test node-id sits in, with ``/tests`` and
    ``/test`` segments stripped (so the test maps to the source subpackage)."""
    path = _node_id_path(node_id)
    if not path:
        return ""
    parts = [part for part in Path(path).parts if part not in (".", "")]
    if parts and parts[-1].endswith(".py"):
        parts = parts[:-1]
    parts = [part for part in parts if part not in {"tests", "test", "__tests__"}]
    return "/".join(parts)


def _shared_prefix_depth(left: str, right: str) -> int:
    """Return the number of leading path segments ``left`` and ``right`` share."""
    left_parts = [p for p in str(left or "").split("/") if p]
    right_parts = [p for p in str(right or "").split("/") if p]
    depth = 0
    for a, b in zip(left_parts, right_parts):
        if a != b:
            break
        depth += 1
    return depth


def _module_match_strength(owned_tokens: set[str], test_pkg: str) -> int:
    """Strength (shared-prefix depth) of a test's link to an owned package set.

    Returns the deepest shared package prefix between the test's package
    directory and any owned source package token. A bare top-level root match
    (depth 1, e.g. both under ``statsmodels``) is intentionally weak: callers
    require depth >= 2 (subpackage level) so a test only maps to a group that
    owns source in the SAME subpackage, not merely the same top-level package.
    """
    best = 0
    for token in owned_tokens:
        best = max(best, _shared_prefix_depth(token, test_pkg))
    return best


def _expected_test_ids_for_module_group(
    repo_name: str,
    owned_files: list[str],
    expected_test_ids: list[str],
    *,
    group_index: int = 0,
    group_count: int = 1,
    is_largest_group: bool = False,
) -> list[str]:
    """Map the full expected node-id set onto one module group's owned files (T2.4).

    Each node-id is assigned to the group whose owned source files share the
    LONGEST package-directory prefix with the test file's package directory
    (subpackage level, depth >= 2 — a bare top-level root match does not count).
    Node-ids with no path-evident target (and bridge/ambiguous ones) fall back
    to the LARGEST group so the union of all per-group subsets is EXACTLY the
    full expected set with NO orphan ids (assert-checked by the caller).

    Because this maps the *full* set across *all* groups, calling it once per
    group with the same arguments reproduces a disjoint partition; the
    ``is_largest_group`` flag tells exactly one group to absorb the unassigned
    bucket.
    """
    owned_tokens: set[str] = set()
    for path in owned_files or []:
        owned_tokens |= _package_dir_tokens(path)
    subset: list[str] = []
    for node_id in expected_test_ids or []:
        test_pkg = _test_package_dir(node_id)
        if not test_pkg:
            # No path-evident target -> unassigned bucket (largest group).
            if is_largest_group:
                subset.append(node_id)
            continue
        if _module_match_strength(owned_tokens, test_pkg) >= 2:
            subset.append(node_id)
        elif is_largest_group:
            # Bridge / ambiguous / unmatched -> largest group, no orphans.
            # (Only added here when no group matched; see partition helper.)
            pass
    return subset


def _partition_expected_test_ids_by_module_group(
    repo_name: str,
    groups_owned_files: list[list[str]],
    expected_test_ids: list[str],
) -> list[list[str]]:
    """Disjointly assign every expected node-id to exactly one module group (T2.4).

    Returns a list parallel to ``groups_owned_files``: subset[i] are the
    node-ids assigned to group i. Each id is assigned to the group with the
    longest shared package-directory prefix; unassigned/ambiguous ids go to the
    largest group. ``union(subsets) == set(expected_test_ids)`` with no orphan
    and no duplicate (assert-checkable).
    """
    group_count = len(groups_owned_files)
    subsets: list[list[str]] = [[] for _ in range(group_count)]
    if group_count == 0:
        return subsets
    group_tokens: list[set[str]] = []
    for owned in groups_owned_files:
        tokens: set[str] = set()
        for path in owned or []:
            tokens |= _package_dir_tokens(path)
        group_tokens.append(tokens)
    buckets: dict[str, list[str]] = {}
    for node_id in expected_test_ids or []:
        node = str(node_id or "").strip()
        if not node:
            continue
        buckets.setdefault(node.split("::", 1)[0], []).append(node)
    loads = [0 for _ in range(group_count)]
    file_counts = [0 for _ in range(group_count)]
    # Pytest node IDs often name a broad test package rather than one source
    # owner; keep whole test-file buckets together, but load-balance ties and
    # unmatched buckets instead of dumping them into one synthetic mega-group.
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for test_file, node_ids in ordered_buckets:
        test_pkg = _test_package_dir(test_file)
        candidate_indices: list[int] = []
        if test_pkg:
            depths = [_module_match_strength(tokens, test_pkg) for tokens in group_tokens]
            best_depth = max(depths or [0])
            if best_depth >= 2:
                candidate_indices = [
                    index for index, depth in enumerate(depths) if depth == best_depth
                ]
        if not candidate_indices:
            candidate_indices = list(range(group_count))
        best_index = min(
            candidate_indices,
            key=lambda index: (
                loads[index],
                file_counts[index],
                len(groups_owned_files[index] or []),
                index,
            ),
        )
        subsets[best_index].extend(node_ids)
        loads[best_index] += len(node_ids)
        file_counts[best_index] += 1
    return subsets


def assert_module_group_subsets_cover_expected(
    groups_owned_files: list[list[str]],
    expected_test_ids: list[str],
) -> dict[str, Any]:
    """Verify the disjoint per-group subsets union to EXACTLY the full set (T2.4).

    Returns a diagnostics dict and raises ``AssertionError`` if any expected id
    is orphaned (assigned to no group) or duplicated (assigned to >1 group).
    This is the union-of-subsets == full-expected-set, no-orphan invariant.
    """
    subsets = _partition_expected_test_ids_by_module_group(
        repo_name="",
        groups_owned_files=groups_owned_files,
        expected_test_ids=expected_test_ids,
    )
    full = [str(test_id) for test_id in (expected_test_ids or []) if str(test_id).strip()]
    full_set = set(full)
    union: set[str] = set()
    duplicates: set[str] = set()
    for subset in subsets:
        for node_id in subset:
            if node_id in union:
                duplicates.add(node_id)
            union.add(node_id)
    orphans = full_set - union
    assert not orphans, f"module-group partition orphaned {len(orphans)} expected ids"
    assert not duplicates, f"module-group partition duplicated {len(duplicates)} expected ids"
    assert union == full_set, "module-group subsets union != full expected set"
    return {
        "group_count": len(subsets),
        "expected_total": len(full_set),
        "assigned_total": len(union),
        "per_group_counts": [len(subset) for subset in subsets],
        "orphans": 0,
        "duplicates": 0,
    }


def _make_module_group_expected_id_mapper(
    repo_name: str,
    expected_test_ids: list[str],
) -> Any:
    """Return a ``(owned_files) -> [node_id]`` callback for the planner (T2.4).

    The planner (Layer A) holds the module groups but must not know the
    node-id<->package ecosystem fact; this Layer-B closure bridges the two. The
    callback maps the supplied group's owned files to its expected-id subset
    using the path-evident package rule above. To keep subsets disjoint across
    repeated single-group calls we route every call through the same
    full-partition computation keyed by the owned-file signature.
    """
    expected = [str(test_id) for test_id in (expected_test_ids or []) if str(test_id).strip()]

    def _mapper(owned_files: list[str]) -> list[str]:
        # Single-group view: assign ids whose test package matches this group's
        # owned packages at the subpackage level (depth >= 2). A bare top-level
        # root match does not count, so a tsa group does not absorb stats tests.
        return _expected_test_ids_for_module_group(
            repo_name,
            list(owned_files or []),
            expected,
        )

    return _mapper


def _make_module_group_expected_id_partitioner(
    repo_name: str,
    expected_test_ids: list[str],
) -> Any:
    """Return a ``(groups_owned_files) -> [[node_id], ...]`` callback (T2.4).

    Unlike :func:`_make_module_group_expected_id_mapper` (one independent call
    per group, which DOUBLE-assigns a test when several groups co-own files in
    the same subpackage), this partitions the full expected set across ALL
    groups in a single global argmax: every node-id goes to exactly one group
    (longest shared package prefix; unassigned -> largest group). The resulting
    per-group subsets are therefore disjoint and union to the full expected set
    -- verified here by the cover-assert so a wiring regression fails loudly.

    The planner (Layer A) hands the parallel owned-files lists and receives the
    parallel disjoint subsets; the node-id<->package ecosystem fact stays in
    this Layer-B closure.
    """
    expected = [str(test_id) for test_id in (expected_test_ids or []) if str(test_id).strip()]

    def _partitioner(groups_owned_files: list[list[str]]) -> list[list[str]]:
        owned = [list(group or []) for group in (groups_owned_files or [])]
        subsets = _partition_expected_test_ids_by_module_group(repo_name, owned, expected)
        # Disjoint + full-cover invariant (raises AssertionError on orphan/dup).
        assert_module_group_subsets_cover_expected(owned, expected)
        return subsets

    return _partitioner


def _expected_id_pytest_plugin_args(
    task: "Commit0Task",
    repo_dir: Path,
) -> list[str]:
    modules: list[str] = []
    modules.extend(_repo_declared_pytest_plugins(repo_dir))
    modules.extend(_repo_declared_pytest_plugin_dependency_modules(repo_dir))
    modules.extend(_task_pytest_plugin_package_modules(task))
    modules.extend(_repo_requested_pytest_option_plugin_modules(task, repo_dir))
    blocked = {
        "pytest_jsonreport.plugin",
        _APEX_EXPECTED_IDS_PLUGIN_NAME,
    }
    deduped: list[str] = []
    seen: set[str] = set()
    for module in modules:
        normalized = str(module or "").strip()
        if not normalized or normalized in blocked or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    args: list[str] = []
    for module in deduped:
        args.extend(["-p", module])
    # pytest-asyncio defaults to strict mode; Commit0 repos with async tests but no explicit asyncio_mode need auto mode under disabled autoload.
    if _loads_pytest_asyncio_plugin(seen) and _repo_needs_pytest_asyncio_auto_mode(repo_dir):
        args.append("--asyncio-mode=auto")
    return args


def _repo_requested_pytest_option_plugin_modules(
    task: "Commit0Task",
    repo_dir: Path,
) -> list[str]:
    packages = _infer_additional_test_packages(task.test_cmd, repo_root=repo_dir)
    modules: list[str] = []
    for package in packages:
        module = _COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES.get(package)
        if module:
            modules.append(module)
    return modules


def _read_repo_addopts_tokens(repo_dir: Path) -> list[str]:
    """Return the ``addopts`` tokens pytest reads from the repo's config.

    O4/NEW-I7: the Commit0 scoring pytest invocation runs ``task.test_cmd`` inside
    ``repo_dir``, and pytest itself applies ``addopts`` from the first config file it
    finds (``pyproject.toml [tool.pytest.ini_options]``, ``pytest.ini [pytest]``,
    ``tox.ini [pytest]``, ``setup.cfg [tool:pytest]``). APEX never passes these
    tokens explicitly, but they ARE applied — so a plugin option like ``--memray``
    sitting in addopts can fail the run. This reads them (config-precedence order,
    first hit wins) so the scoring path can audit/strip unloadable plugin options.
    """
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        ini_options = (
            data.get("tool", {}).get("pytest", {}).get("ini_options", {})
            if isinstance(data, dict)
            else {}
        )
        if isinstance(ini_options, dict) and "addopts" in ini_options:
            return _coerce_addopts_to_tokens(ini_options.get("addopts"))

    for filename, section in (
        ("pytest.ini", "pytest"),
        ("tox.ini", "pytest"),
        ("setup.cfg", "tool:pytest"),
    ):
        path = repo_dir / filename
        if not path.is_file():
            continue
        import configparser

        parser = configparser.ConfigParser()
        try:
            parser.read(path, encoding="utf-8")
        except (OSError, configparser.Error):
            continue
        if parser.has_option(section, "addopts"):
            return _coerce_addopts_to_tokens(parser.get(section, "addopts"))
    return []


def _coerce_addopts_to_tokens(addopts: object) -> list[str]:
    """Normalize a config ``addopts`` value (str or list) to a token list."""
    if isinstance(addopts, str):
        try:
            return shlex.split(addopts)
        except ValueError:
            return addopts.split()
    if isinstance(addopts, (list, tuple)):
        tokens: list[str] = []
        for item in addopts:
            tokens.extend(_coerce_addopts_to_tokens(item))
        return tokens
    return []


def _addopts_option_plugin_module(token: str) -> Optional[str]:
    """Return the plugin import module a plugin-provided addopts option needs.

    Returns ``None`` for core pytest options (and unknown options), so those are
    never stripped. Handles both ``--cov`` and ``--cov=...``/``--cov report`` forms.
    """
    if not token.startswith("-"):
        return None
    flag = token.split("=", 1)[0]
    return _COMMIT0_PYTEST_OPTION_FLAG_PLUGIN_MODULES.get(flag)


def _strip_unimportable_plugin_addopts(
    addopts_tokens: list[str],
    *,
    is_module_importable: "Callable[[str], bool]",
) -> tuple[list[str], list[str]]:
    """Drop addopts options whose plugin is not importable in the scoring venv.

    Returns ``(kept_tokens, stripped_options)``. A plugin-provided option (per
    ``_COMMIT0_PYTEST_OPTION_FLAG_PLUGIN_MODULES``) is dropped — together with its
    detached value, when it takes one — iff its plugin module is NOT importable.
    Core pytest options and loadable-plugin options are always kept. The
    importability check is memoized per module so each is probed at most once
    immediately before invocation.
    """
    kept: list[str] = []
    stripped: list[str] = []
    importable_cache: dict[str, bool] = {}

    def _importable(module: str) -> bool:
        if module not in importable_cache:
            importable_cache[module] = bool(is_module_importable(module))
        return importable_cache[module]

    index = 0
    total = len(addopts_tokens)
    while index < total:
        token = addopts_tokens[index]
        module = _addopts_option_plugin_module(token)
        if module is None or _importable(module):
            kept.append(token)
            index += 1
            continue
        # Plugin not loadable -> strip this option (and its detached value).
        flag = token.split("=", 1)[0]
        stripped.append(token)
        index += 1
        if (
            "=" not in token
            and flag in _COMMIT0_PYTEST_PLUGIN_OPTION_FLAGS_WITH_VALUES
            and index < total
            and not addopts_tokens[index].startswith("-")
        ):
            stripped.append(addopts_tokens[index])
            index += 1
    return kept, stripped


def _loads_pytest_asyncio_plugin(modules: set[str]) -> bool:
    return "pytest_asyncio" in modules or "pytest_asyncio.plugin" in modules


def _repo_declared_pytest_plugins(repo_dir: Path) -> list[str]:
    """Return pytest11 plugin modules declared by repo metadata.

    Pytest plugin autoload stays disabled for Commit0 scoring, so project-local
    pytest11 entry points need explicit ``-p`` loading to keep repo fixtures
    available without candidate edits to test configuration.
    """

    plugins: list[str] = []
    plugins.extend(_repo_declared_pytest_plugins_from_pyproject(repo_dir))
    plugins.extend(_repo_declared_pytest_plugins_from_setup_py(repo_dir))
    return plugins


def _repo_declared_pytest_plugins_from_pyproject(repo_dir: Path) -> list[str]:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    plugins: list[str] = []
    project_entry_points = data.get("project", {}).get("entry-points", {}).get("pytest11", {})
    if isinstance(project_entry_points, dict):
        plugins.extend(_entry_point_modules(project_entry_points.values()))

    poetry_plugins = data.get("tool", {}).get("poetry", {}).get("plugins", {}).get("pytest11", {})
    if isinstance(poetry_plugins, dict):
        plugins.extend(_entry_point_modules(poetry_plugins.values()))
    return plugins


_COMMIT0_TEST_EXTRA_NAME_HINTS: tuple[str, ...] = (
    "dev",
    "devel",
    "develop",
    "development",
    "test",
    "tests",
    "testing",
    "test-suite",
    "checks",
    "ci",
    "lint",
)


def _repo_test_extra_names(repo_dir: Path) -> list[str]:
    """B4: discover declared test/dev extras (PEP 621 optional-dependencies,
    Poetry extras, and legacy setuptools extras_require) so collection-time test
    dependencies (pytest plugins, fixtures, runners) can be installed best-effort.

    Returns only extra group names that actually exist in the project's metadata,
    intersected with common test/dev hints, so a ``pip install -e .[<extra>]`` can
    never reference a non-existent extra.
    """
    declared: set[str] = set()
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        if isinstance(data, dict):
            project = data.get("project")
            if isinstance(project, dict):
                optional = project.get("optional-dependencies")
                if isinstance(optional, dict):
                    declared.update(str(name) for name in optional)
            tool = data.get("tool")
            if isinstance(tool, dict):
                poetry = tool.get("poetry")
                if isinstance(poetry, dict):
                    extras = poetry.get("extras")
                    if isinstance(extras, dict):
                        declared.update(str(name) for name in extras)
    setup_cfg = repo_dir / "setup.cfg"
    if setup_cfg.is_file():
        try:
            import configparser

            parser = configparser.ConfigParser()
            parser.read(setup_cfg, encoding="utf-8")
            if parser.has_section("options.extras_require"):
                declared.update(parser.options("options.extras_require"))
        except Exception:  # noqa: BLE001 - extras discovery is best-effort
            pass
    hints = {hint.lower() for hint in _COMMIT0_TEST_EXTRA_NAME_HINTS}
    return sorted(name for name in declared if name.lower() in hints)


def _repo_declared_pytest_plugins_from_setup_py(repo_dir: Path) -> list[str]:
    setup_py = repo_dir / "setup.py"
    if not setup_py.is_file():
        return []
    try:
        tree = ast.parse(setup_py.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []

    literal_assignments: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = _literal_ast_value(node.value)
            if value is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literal_assignments[target.id] = value

    plugins: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_setup_call(node.func):
            continue
        for keyword in node.keywords:
            if keyword.arg != "entry_points":
                continue
            value = _literal_ast_value(keyword.value)
            if value is None and isinstance(keyword.value, ast.Name):
                value = literal_assignments.get(keyword.value.id)
            plugins.extend(_pytest11_entry_point_modules(value))
    return plugins


def _is_setup_call(func: ast.AST) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "setup"
    if isinstance(func, ast.Attribute):
        return func.attr == "setup"
    return False


def _literal_ast_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _pytest11_entry_point_modules(entry_points: Any) -> list[str]:
    if isinstance(entry_points, dict):
        pytest11 = entry_points.get("pytest11")
        if isinstance(pytest11, dict):
            return _entry_point_modules(pytest11.values())
        if isinstance(pytest11, (list, tuple, set)):
            return _entry_point_modules(pytest11)
        if isinstance(pytest11, str):
            return _entry_point_modules([pytest11])
    if isinstance(entry_points, (list, tuple, set)):
        modules: list[str] = []
        in_pytest11_section = False
        for value in entry_points:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            section_match = re.match(r"\[([^\]]+)\]\s*$", stripped)
            if section_match:
                in_pytest11_section = section_match.group(1).strip() == "pytest11"
                continue
            if in_pytest11_section:
                modules.extend(_entry_point_modules([stripped]))
        return modules
    return []


def _repo_declared_pytest_plugin_dependency_modules(repo_dir: Path) -> list[str]:
    """Return known pytest plugin modules declared in repo dependency metadata."""

    metadata = []
    for filename in _COMMIT0_PYTEST_PLUGIN_METADATA_FILES:
        path = repo_dir / filename
        if not path.is_file():
            continue
        try:
            metadata.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    for path in sorted(repo_dir.glob("requirements*.txt")):
        if not path.is_file():
            continue
        try:
            metadata.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    if not metadata:
        return []

    combined = "\n".join(metadata).lower()
    modules: list[str] = []
    for package, module in _COMMIT0_PYTEST_PLUGIN_PACKAGE_MODULES.items():
        package_pattern = re.escape(package).replace("\\-", "[-_]")
        if re.search(rf"(?<![a-z0-9_.-]){package_pattern}(?![a-z0-9_.-])", combined):
            modules.append(module)
    return modules


def _repo_needs_pytest_asyncio_auto_mode(repo_dir: Path) -> bool:
    if _repo_pytest_config_declares_asyncio_mode(repo_dir):
        return False
    return _repo_has_async_pytest_tests(repo_dir)


def _repo_pytest_config_declares_asyncio_mode(repo_dir: Path) -> bool:
    for filename in _COMMIT0_PYTEST_PLUGIN_METADATA_FILES:
        path = repo_dir / filename
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"(?m)^\s*asyncio_mode\s*=", text):
            return True
    return False


def _repo_has_async_pytest_tests(repo_dir: Path) -> bool:
    roots = [repo_dir / "tests", repo_dir / "test"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(r"(?m)^\s*async\s+def\s+test_", text):
                return True
    return False


def _entry_point_modules(values: Any) -> list[str]:
    modules: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        module_spec = value.split("=", 1)[-1].strip()
        module = module_spec.split(":", 1)[0].strip()
        if module:
            modules.append(module)
    return modules


def _task_pytest_plugin_package_modules(task: "Commit0Task") -> list[str]:
    modules: list[str] = []
    for requirement in getattr(task, "pip_packages", []) or []:
        name = _requirement_project_name(str(requirement))
        module = _COMMIT0_PYTEST_PLUGIN_PACKAGE_MODULES.get(name)
        if module:
            modules.append(module)
    return modules


def _requirement_project_name(requirement: str) -> str:
    head = re.split(r"[<>=!~;\s\[]", requirement.strip(), maxsplit=1)[0]
    return head.replace("_", "-").lower()


def _commit0_requirement_import_roots(requirement: str) -> list[str]:
    project_name = _requirement_project_name(requirement)
    if not project_name:
        return []
    roots: list[str] = []
    plugin_module = _COMMIT0_PYTEST_PLUGIN_PACKAGE_MODULES.get(
        project_name,
    ) or _COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES.get(project_name)
    if plugin_module:
        roots.append(plugin_module.split(".", 1)[0])
    roots.extend(_COMMIT0_REQUIREMENT_IMPORT_ROOT_ALIASES.get(project_name, ()))
    normalized = project_name.replace("-", "_")
    if normalized:
        roots.append(normalized)
    return list(dict.fromkeys(root for root in roots if root))


def _repo_declared_requirement_import_roots(
    task: Optional["Commit0Task"],
    repo_dir: Path,
) -> list[str]:
    metadata_paths: list[Path] = []
    requirement_paths: list[Path] = []
    if task is not None:
        for raw_path in list(getattr(task, "packages", []) or []):
            package_path = repo_dir / str(raw_path).strip()
            if package_path.is_file():
                metadata_paths.append(package_path)
                if package_path.suffix == ".txt":
                    requirement_paths.append(package_path)
    for filename in _COMMIT0_PYTEST_PLUGIN_METADATA_FILES:
        path = repo_dir / filename
        if path.is_file():
            metadata_paths.append(path)
    requirements_paths = [
        path for path in sorted(repo_dir.glob("requirements*.txt")) if path.is_file()
    ]
    metadata_paths.extend(requirements_paths)
    requirement_paths.extend(requirements_paths)

    texts: list[str] = []
    requirement_texts: list[str] = []
    for path in dict.fromkeys(metadata_paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texts.append(text)
        if path in requirement_paths:
            requirement_texts.append(text)
    if not texts:
        return []

    roots: list[str] = []
    combined = "\n".join(texts).lower()
    known_projects = set(_COMMIT0_REQUIREMENT_IMPORT_ROOT_ALIASES)
    known_projects.update(_COMMIT0_PYTEST_PLUGIN_PACKAGE_MODULES)
    known_projects.update(_COMMIT0_PYTEST_OPTION_PLUGIN_PACKAGE_MODULES)
    for project in sorted(known_projects):
        package_pattern = re.escape(project).replace("\\-", "[-_]")
        if re.search(rf"(?<![a-z0-9_.-]){package_pattern}(?![a-z0-9_.-])", combined):
            roots.extend(_commit0_requirement_import_roots(project))

    for text in requirement_texts:
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip().strip("',\",")
            if not line or line.startswith(("-", ".", "/", "[", "{", "}")):
                continue
            project = _requirement_project_name(line)
            if re.match(r"^[a-z0-9][a-z0-9_.-]*$", project):
                roots.extend(_commit0_requirement_import_roots(project))
    return list(dict.fromkeys(root for root in roots if root))


def _exclude_commit0_harness_helpers_from_git_status(repo_dir: Path) -> None:
    """Keep staged Commit0 harness helpers out of git status/diffs."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if result.returncode != 0:
        return
    raw_path = (result.stdout or "").strip()
    if not raw_path:
        return
    exclude_path = Path(raw_path)
    if not exclude_path.is_absolute():
        exclude_path = repo_dir / exclude_path
    try:
        existing = exclude_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        existing = ""
    patterns = [
        _APEX_EXPECTED_IDS_FILENAME,
        _APEX_EXPECTED_IDS_MIRROR_FILENAME,
        _APEX_EXPECTED_IDS_PLUGIN_FILENAME,
        _APEX_EXPECTED_IDS_RUNNER_FILENAME,
        _APEX_LOCAL_EVAL_REPORT_FILENAME,
        ".apex_verification_reports/",
        # Commit0 harness diagnostic: operator-only prepare metadata, never a repo solution file.
        _APEX_PREPARE_DIAGNOSTICS_FILENAME,
    ]
    missing = [pattern for pattern in patterns if pattern not in existing.splitlines()]
    if not missing:
        return
    try:
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        with exclude_path.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            for pattern in missing:
                handle.write(f"{pattern}\n")
    except OSError:
        return


# Per-repo task overrides for upstream defects in the Commit0 dataset that
# would otherwise prevent APEX from ever running. Source of truth is the
# tracked JSON config at configs/commit0_task_overrides.json — see that
# file for per-entry justifications. Keys per entry: install_command,
# pre_install_drop_substrings (list[str]), pre_install_extra (list[str]),
# pip_packages_extra (list[str]), evaluation_timeout_seconds (int).
_COMMIT0_TASK_OVERRIDES_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "commit0_task_overrides.json"
)


def _load_commit0_task_overrides() -> dict[str, dict[str, Any]]:
    """Load per-repo Commit0 overrides from the tracked JSON config.

    The override list used to live as a Python dict literal in this file
    (~100 lines of inlined comments + key/value pairs). Promoting it to
    a JSON config separates "what we corrected" from "how we apply
    corrections", makes the override list reviewable in isolation, and
    gives benchmark reports a single artifact path to cite when
    reviewers ask which environmental fixes were applied.

    The JSON file may carry a top-level ``_doc`` field and per-entry
    ``_comment`` fields; both are informational and stripped here.
    """
    config_path = _COMMIT0_TASK_OVERRIDES_CONFIG_PATH
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        logger.warning(
            "Commit0 task overrides config missing at %s — running without overrides.",
            config_path,
        )
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Commit0 task overrides config at %s could not be loaded (%s) — "
            "running without overrides.",
            config_path,
            exc,
        )
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "Commit0 task overrides config at %s does not contain a JSON object — ignoring.",
            config_path,
        )
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for repo_name, entry in raw.items():
        if repo_name.startswith("_"):
            # Top-level metadata fields like `_doc` are informational.
            continue
        if not isinstance(entry, dict):
            continue
        cleaned = {key: value for key, value in entry.items() if not key.startswith("_")}
        overrides[str(repo_name)] = cleaned
    return overrides


_COMMIT0_TASK_OVERRIDES: dict[str, dict[str, Any]] = _load_commit0_task_overrides()


# --- perturbed-commit0 de-contaminated variants (ADDITIVE sidecar) -----------
# A perturbed variant (``<repo>_perturbed``) is a separately-built, byte-distinct
# benchmark target produced by ``apex_omega.eval.perturb`` (consistent symbol
# alpha-rename + docstring neutralization, semantics-preserved via a gold-test
# validation gate).  It is NOT in the HuggingFace dataset, so ``discover_tasks``
# synthesizes its ``Commit0Task`` from this sidecar JSON, and ``_git_clone_with_retry``
# sources it from the emitted LOCAL git mirror instead of GitHub.  When the sidecar
# is absent, ALL of this is inert and vanilla commit0 is byte-identical.
_COMMIT0_PERTURBED_TARGETS_SIDECAR = (
    Path(__file__).resolve().parents[2]
    / "apex_omega" / "eval" / "perturb" / "variants" / "perturbed_targets.json"
)


def _load_commit0_perturbed_targets() -> dict[str, dict[str, Any]]:
    """Load the perturbed-variant sidecar (``{name: {repo, base_commit, ...}}``).

    Missing/invalid file -> ``{}`` (vanilla commit0 unchanged).
    """
    path = _COMMIT0_PERTURBED_TARGETS_SIDECAR
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Perturbed-commit0 sidecar at %s unreadable (%s) — ignoring.", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    targets = raw.get("targets")
    return targets if isinstance(targets, dict) else {}


_COMMIT0_PERTURBED_TARGETS: dict[str, dict[str, Any]] = _load_commit0_perturbed_targets()


def _build_synthetic_perturbed_task(repo_name: str) -> Optional["Commit0Task"]:
    """Synthesize a :class:`Commit0Task` for a perturbed variant from the sidecar.

    Returns ``None`` if *repo_name* is not a known perturbed target.
    """
    entry = _COMMIT0_PERTURBED_TARGETS.get(repo_name)
    if not entry:
        return None
    setup_python = str(entry.get("python_version", "3.12"))
    task = Commit0Task(
        instance_id=str(entry.get("repo") or repo_name),
        repo=str(entry["repo"]),
        original_repo="",
        base_commit=str(entry["base_commit"]),
        reference_commit=str(entry.get("reference_commit", "")),
        python_version=setup_python,
        specification="",
        install_command=str(entry.get("install_command", "pip install -e .")),
        packages=list(entry.get("packages") or []),
        pip_packages=list(entry.get("pip_packages") or []),
        pre_install=list(entry.get("pre_install") or []),
        src_dir=str(entry.get("src_dir", "")),
        test_cmd=str(entry.get("test_cmd", "pytest")),
        test_dir=str(entry.get("test_dir", "")),
    )
    _apply_commit0_task_overrides(task)
    return task


def _perturbed_mirror_roots() -> list[str]:
    """Local-mirror parent dirs declared by perturbed sidecar entries."""
    roots: list[str] = []
    for entry in _COMMIT0_PERTURBED_TARGETS.values():
        root = entry.get("mirror_root")
        if root and root not in roots:
            roots.append(str(root))
    return roots


def _is_perturbed_task(task: "Commit0Task") -> bool:
    return task.repo_name in _COMMIT0_PERTURBED_TARGETS


def _is_repo_memory_disabled_via_env_safe() -> bool:
    """Defensive wrapper around the persistence helper.

    Reports must render even if persistence imports fail (rare but
    possible when reports are reconstructed from disk in tooling that
    doesn't import the full apex stack). Treats any failure as
    "not-disabled" so we don't falsely claim the override was active.
    """
    try:
        from ..persistence import is_repo_memory_disabled_via_env

        return is_repo_memory_disabled_via_env()
    except Exception:
        return False


# Repos that must run inside the Linux container even though their
# pre_install doesn't contain an apt-get hook. These have macOS-sandbox
# ceilings the auto-detection can't see (PTY/signal semantics, OpenSSL
# pkg-config bindings, fixture downloads, etc.).
_COMMIT0_FORCE_LINUX_CONTAINER_REPOS: set[str] = {
    "pexpect",
    "pypdf",
    "dnspython",
    "tlslite-ng",
    "filesystem_spec",
}


# Expected-id inventory size is a benchmark-provided public complexity
# signal. Use it to seed rollout budgets without encoding repo identities
# into the agentic selection/orchestration layer.
_COMMIT0_EXPECTED_TEST_COUNT_BUDGET_TIERS: tuple[tuple[int, dict[str, int]], ...] = (
    (10_000, {"min_rollouts": 12, "max_rollouts": 32}),
    (1_000, {"min_rollouts": 10, "max_rollouts": 24}),
    (100, {"min_rollouts": 6, "max_rollouts": 16}),
    (1, {"max_rollouts": 8}),
)


def _commit0_rollout_budget_for_expected_test_count(
    expected_test_count: int,
) -> dict[str, int]:
    if expected_test_count <= 0:
        return {}
    for floor, budget in _COMMIT0_EXPECTED_TEST_COUNT_BUDGET_TIERS:
        if expected_test_count >= floor:
            return dict(budget)
    return {}


def _apply_task_complexity_rollout_budget(
    config: "ApexConfig",
    *,
    expected_test_count: int,
) -> None:
    overrides = _commit0_rollout_budget_for_expected_test_count(expected_test_count)
    if not overrides:
        return
    rollout_cfg = config.rollout
    min_override = overrides.get("min_rollouts")
    max_override = overrides.get("max_rollouts")
    if isinstance(max_override, int) and max_override > 0:
        rollout_cfg.max_rollouts = max_override
        if rollout_cfg.num_rollouts > max_override:
            rollout_cfg.num_rollouts = max_override
    if isinstance(min_override, int) and min_override > 0:
        rollout_cfg.min_rollouts = min(min_override, rollout_cfg.max_rollouts)


def _apply_commit0_task_overrides(task: "Commit0Task") -> None:
    overrides = _COMMIT0_TASK_OVERRIDES.get(task.repo_name)
    if not overrides:
        return
    if "python_version" in overrides:
        # Commit0 official Docker images own the scoring interpreter; local
        # rollouts must not override it through APEX config.
        logger.warning(
            "Ignoring Commit0 python_version override for %s; official audit image owns runtime Python.",
            task.repo_name,
        )
    if "install_command" in overrides:
        task.install_command = str(overrides["install_command"])
    drop_substrings = overrides.get("pre_install_drop_substrings") or []
    if drop_substrings:
        task.pre_install = [
            cmd for cmd in task.pre_install if not any(sub in cmd for sub in drop_substrings)
        ]
    pre_install_extra = overrides.get("pre_install_extra") or []
    if pre_install_extra:
        task.pre_install = list(task.pre_install) + list(pre_install_extra)
    pip_packages_extra = overrides.get("pip_packages_extra") or []
    if pip_packages_extra:
        task.pip_packages = list(task.pip_packages) + list(pip_packages_extra)


_COMMIT0_UNTRACKED_SNAPSHOT_FILENAMES = {
    ".coveragerc",
    "conftest.py",
    "hatch.toml",
    "manifest.in",
    "noxfile.py",
    "pdm.lock",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "tox.ini",
}

logger = logging.getLogger("apex.evaluation.commit0")

_COMMIT0_TRANSIENT_PATH_PATTERNS = (
    re.compile(r"/(?:private/)?var/folders/[^\s\"'`]+"),
    re.compile(r"/tmp/[^\s\"'`]+"),
)
_COMMIT0_REFERENCE_HINT_PATTERNS = (
    re.compile(r"(?i)parallel reference checkout"),
    re.compile(r"(?i)reference upstream"),
)
_COMMIT0_SOLVE_PHASE_PYTEST_ADDOPTS = "--tb=short --disable-warnings --color=no"
_COMMIT0_DOCKER_RUNTIME_PASSTHROUGH_ENV_KEYS = (
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PIP_DISABLE_PIP_VERSION_CHECK",
    "PYTEST_ADDOPTS",
    # Commit0 expected-ID scoring and pytest-json-report loading are controlled
    # by env vars on the host-side shell command; Docker wrappers must forward them.
    _APEX_EXPECTED_IDS_ENV_VAR,
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "COVERAGE_FILE",
    # Colima/Commit0 Docker execs need the harness-rewritten proxy, not host loopback.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "X2P_AGENT_PROXY_ADDRESS",
    # Meta Claude Code in scrubbed Linux Docker reaches Plugboard V2 through
    # Commit0's benchmark-routed X2P proxy; these nonsecret provider facts must
    # survive docker-exec target-runtime wiring.
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_DISABLE_ADVISOR_TOOL",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
    # Host-auth model proxies are benchmark runtime plumbing; pass only endpoint
    # URLs so containerized agent CLIs keep model transport without host secrets.
    "APEX_AGENT_MODEL_PROXY_URL",
    "APEX_HOST_MODEL_PROXY_URL",
    "APEX_CODEX_CLI_MODEL_PROXY_URL",
    "APEX_CODEX_MODEL_PROXY_URL",
    "APEX_CLAUDE_CLI_MODEL_PROXY_URL",
    "APEX_CLAUDE_MODEL_PROXY_URL",
    "APEX_GEMINI_CLI_MODEL_PROXY_URL",
    "APEX_GEMINI_MODEL_PROXY_URL",
    "APEX_OPENCODE_CLI_MODEL_PROXY_URL",
    "APEX_OPENCODE_MODEL_PROXY_URL",
    "APEX_METACODE_CLI_MODEL_PROXY_URL",
    "APEX_METACODE_MODEL_PROXY_URL",
    _COMMIT0_EGRESS_ALLOW_HOSTS_ENV,
)
_COMMIT0_AGENT_CLI_NODE_VERSION = "20.19.5"
_COMMIT0_AGENT_CLI_NPM_PACKAGES: dict[str, str] = {
    "codex": "@openai/codex@0.132.0",
    "claude": "@anthropic-ai/claude-code@2.1.146",
    "gemini": "@google/gemini-cli@0.42.0",
    "opencode": "opencode-ai@1.15.6",
    # Commit0 Docker fact: MetaCode is not published as a reproducible npm
    # package, so explicit metacode_cli use must come from a preflight-proven
    # target-container binary rather than the default installer bundle.
}
_COMMIT0_DEFAULT_AGENT_CLI_BINARIES: tuple[str, ...] = ("codex", "claude", "gemini")
_COMMIT0_BACKEND_AGENT_CLI_BINARY: dict[str, str] = {
    "codex_cli": "codex",
    "claude_cli": "claude",
    "gemini_cli": "gemini",
    "opencode_cli": "opencode",
}
_COMMIT0_HOST_CLI_AUTH_MODES = frozenset(
    {
        "host_cli",
        "host_auth_cli",
        "host_cli_container_tools",
        "host_auth_container_tools",
    }
)
_COMMIT0_MODEL_PROXY_AUTH_MODES = frozenset(
    {
        "host_model_proxy",
        "model_proxy",
        "proxy",
        "credentialless_model_proxy",
    }
)
_COMMIT0_DOCKER_SANDBOX_AUTH_MODES = frozenset(
    {
        "docker_sandbox",
        "docker_sandboxes",
        "docker_sandbox_host_auth",
        "host_auth_docker_sandbox",
        "host_docker_sandbox",
    }
)
_COMMIT0_DOCKER_IMAGE_AUTH_MODES = frozenset(
    {
        "container_image",
        "docker_image",
        "target_image",
    }
)
_COMMIT0_DOCKER_WORKSPACE_ROOT = "/workspace"
_COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT = "/opt/apex-agent-cli-filtered"
_COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT = "/opt/apex-agent-cli"
_COMMIT0_OFFICIAL_IMAGE_NAMESPACE = "wentingzhao"
_COMMIT0_OFFICIAL_IMAGE_PLATFORM = "linux/amd64"
_COMMIT0_OFFICIAL_TESTBED_VENV = "/testbed/.venv"
_COMMIT0_DOCKER_HOST_WORKDIR_ROOT_ENV = "APEX_COMMIT0_DOCKER_HOST_WORKDIR_ROOT"
_COMMIT0_DOCKER_CONTAINER_WORKDIR_ROOT_ENV = "APEX_COMMIT0_DOCKER_CONTAINER_WORKDIR_ROOT"
_COMMIT0_DOCKER_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_COMMIT0_DOCKER_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")
_COMMIT0_LOOPBACK_PROXY_HOSTS = {"localhost", "127.0.0.1", "::1"}
_COMMIT0_EGRESS_PROXY_ALIAS = "apex-commit0-egress"
_COMMIT0_EGRESS_PROXY_BASE_PORT = 18080
_COMMIT0_EGRESS_PROXY_MAPPINGS_ENV = "APEX_COMMIT0_EGRESS_PROXY_MAPPINGS"
_COMMIT0_SOLVE_NETWORK_PREFLIGHT_ENV = "APEX_COMMIT0_SOLVE_NETWORK_PREFLIGHT"

# V2 anti-cheat diagnostics: all dependency installs complete in SETUP before
# the solve container is moved to an internal Docker network. These hosts are
# source/package surfaces the sidecar proxy must not forward to during SOLVE.
_COMMIT0_EGRESS_DENY_HOSTS: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "pythonhosted.org",
    "github.com",
    "codeload.github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "githubusercontent.com",
    "pypi.python.org",
    "pythonhosted.com",
    # Common third-party PyPI / package mirrors.
    "mirrors.aliyun.com",
    "pypi.tuna.tsinghua.edu.cn",
    "mirror.baidu.com",
    "pypi.douban.com",
    "mirrors.cloud.tencent.com",
)
# Hosts that must remain reachable during the solve phase: the LLM-API model
# proxy is wired through the sidecar alias, with loopback/host-gateway used only
# by harness-owned relays outside the isolated agent container.
_COMMIT0_EGRESS_ALLOW_HOSTS: tuple[str, ...] = (
    _COMMIT0_EGRESS_PROXY_ALIAS,
    "host.docker.internal",
    "localhost",
    "127.0.0.1",
    "::1",
)
# Non-routable sinkhole the deny proxy points at. RFC 5737 TEST-NET-1 +
# discard port so any direct CONNECT to a denied host fails fast instead of
# silently succeeding through an upstream proxy.
_COMMIT0_EGRESS_SINKHOLE_PROXY = "http://192.0.2.1:9"
# Env var name advertised to rollout agents/diagnostics so the solve-phase
# policy is discoverable and auditable.
_COMMIT0_EGRESS_DENY_HOSTS_ENV = "APEX_COMMIT0_EGRESS_DENY_HOSTS"
_COMMIT0_MIN_DOCKER_ROOT_FREE_BYTES_ENV = "APEX_COMMIT0_MIN_DOCKER_ROOT_FREE_BYTES"
_COMMIT0_DEFAULT_MIN_DOCKER_ROOT_FREE_BYTES = 8 * 1024 * 1024 * 1024
_COMMIT0_TARGET_CONTAINER_PREFLIGHT_ATTEMPTS_ENV = (
    "APEX_COMMIT0_TARGET_CONTAINER_PREFLIGHT_ATTEMPTS"
)
_COMMIT0_DEFAULT_TARGET_CONTAINER_PREFLIGHT_ATTEMPTS = 3
_COMMIT0_DOCKER_MODEL_PROXY_ENV_KEYS = (
    "APEX_AGENT_MODEL_PROXY_URL",
    "APEX_HOST_MODEL_PROXY_URL",
    "APEX_CODEX_CLI_MODEL_PROXY_URL",
    "APEX_CODEX_MODEL_PROXY_URL",
    "APEX_CLAUDE_CLI_MODEL_PROXY_URL",
    "APEX_CLAUDE_MODEL_PROXY_URL",
    "APEX_GEMINI_CLI_MODEL_PROXY_URL",
    "APEX_GEMINI_MODEL_PROXY_URL",
    "APEX_OPENCODE_CLI_MODEL_PROXY_URL",
    "APEX_OPENCODE_MODEL_PROXY_URL",
    "APEX_METACODE_CLI_MODEL_PROXY_URL",
    "APEX_METACODE_MODEL_PROXY_URL",
)
_COMMIT0_CLAUDE_PLUGBOARD_V2_BASE_URL = (
    "http://plugboardv2.x2p.facebook.net/claude_code/passthrough"
)
_COMMIT0_CLAUDE_PLUGBOARD_PLACEHOLDER_API_KEY = "NOT_IMPLEMENTED_YET"
_COMMIT0_CLAUDE_PLUGBOARD_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_DISABLE_ADVISOR_TOOL",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
)
_COMMIT0_CLAUDE_PROVIDER_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "APEX_CLAUDE_CLI_AUTH_STATE",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "CLAUDE_CODE_USE_VERTEX",
    "CLOUD_ML_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS",
)
_COMMIT0_CLAUDE_MODEL_PROXY_ENV_KEYS = (
    "APEX_AGENT_MODEL_PROXY_URL",
    "APEX_HOST_MODEL_PROXY_URL",
    "APEX_CLAUDE_CLI_MODEL_PROXY_URL",
    "APEX_CLAUDE_MODEL_PROXY_URL",
)
_COMMIT0_MODEL_TRANSPORT_URL_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "CODEX_BASE_URL",
    "CODE_ASSIST_ENDPOINT",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "OPENAI_BASE_URL",
    *_COMMIT0_DOCKER_MODEL_PROXY_ENV_KEYS,
)
_COMMIT0_MODEL_TRANSPORT_HOSTS = frozenset(
    {
        "api.anthropic.com",
        "api.openai.com",
        "aiplatform.googleapis.com",
        "chatgpt.com",
        "claude.ai",
        "generativelanguage.googleapis.com",
        "oauth2.googleapis.com",
        "plugboard.x2p.facebook.net",
        "plugboardv2.x2p.facebook.net",
        "sts.googleapis.com",
        "www.googleapis.com",
    }
)
_COMMIT0_MODEL_TRANSPORT_HOST_SUFFIXES = (
    ".anthropic.com",
    ".googleapis.com",
    ".openai.com",
    ".x2p.facebook.net",
)


def _commit0_min_docker_root_free_bytes() -> int:
    raw = str(os.environ.get(_COMMIT0_MIN_DOCKER_ROOT_FREE_BYTES_ENV) or "").strip()
    if raw:
        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            logger.debug(
                "Ignoring invalid %s=%r",
                _COMMIT0_MIN_DOCKER_ROOT_FREE_BYTES_ENV,
                raw,
            )
    return _COMMIT0_DEFAULT_MIN_DOCKER_ROOT_FREE_BYTES


def _commit0_target_container_preflight_attempts() -> int:
    raw = str(os.environ.get(_COMMIT0_TARGET_CONTAINER_PREFLIGHT_ATTEMPTS_ENV) or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.debug(
                "Ignoring invalid %s=%r",
                _COMMIT0_TARGET_CONTAINER_PREFLIGHT_ATTEMPTS_ENV,
                raw,
            )
    return _COMMIT0_DEFAULT_TARGET_CONTAINER_PREFLIGHT_ATTEMPTS


def _parse_posix_df_available_bytes(output: str) -> Optional[int]:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    parts = lines[-1].split()
    if len(parts) < 4:
        return None
    try:
        return int(parts[3]) * 1024
    except ValueError:
        return None


def _format_bytes_gib(value: int) -> str:
    return f"{max(value, 0) / (1024**3):.1f}GiB"


_DOCKER_SIZE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kmgtp]?i?b?)?\s*$",
    re.IGNORECASE,
)
_DOCKER_SIZE_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
    "p": 1024**5,
    "pb": 1024**5,
    "pib": 1024**5,
}


def _parse_docker_size_bytes(value: str) -> Optional[int]:
    match = _DOCKER_SIZE_RE.match(str(value or ""))
    if not match:
        return None
    multiplier = _DOCKER_SIZE_MULTIPLIERS.get(str(match.group("unit") or "").lower())
    if multiplier is None:
        return None
    try:
        return int(float(match.group("value")) * multiplier)
    except ValueError:
        return None


def _format_docker_size_bytes(value: int) -> str:
    gib = max(1, int(value) // (1024**3))
    return f"{gib}g"


def _commit0_normalize_host(value: str) -> str:
    host = str(value or "").strip().strip("[]").lower().rstrip(".")
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if ":" in host and host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host


def _commit0_url_host(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parse_value = text if "://" in text else f"http://{text}"
    try:
        return _commit0_normalize_host(urlsplit(parse_value).hostname or "")
    except ValueError:
        return ""


def _commit0_model_transport_hosts_from_env(env: Mapping[str, str] | None = None) -> set[str]:
    source = {str(k): str(v) for k, v in dict(os.environ).items()}
    source.update({str(k): str(v) for k, v in dict(env or {}).items()})
    hosts = set(_COMMIT0_MODEL_TRANSPORT_HOSTS)
    for key in _COMMIT0_MODEL_TRANSPORT_URL_ENV_KEYS:
        host = _commit0_url_host(source.get(key, ""))
        if host:
            hosts.add(host)
    return hosts


def _commit0_model_transport_host_allowed(host: str, allowed_hosts: set[str] | None = None) -> bool:
    normalized = _commit0_normalize_host(host)
    if not normalized:
        return False
    hosts = allowed_hosts or set(_COMMIT0_MODEL_TRANSPORT_HOSTS)
    if normalized in hosts:
        return True
    return any(normalized.endswith(suffix) for suffix in _COMMIT0_MODEL_TRANSPORT_HOST_SUFFIXES)


def _commit0_endpoint_host_port(value: str) -> tuple[str, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parse_value = text if "://" in text else f"//{text}"
    try:
        parsed = urlsplit(parse_value)
    except ValueError:
        return None
    host = _commit0_normalize_host(parsed.hostname or "")
    if not host:
        return None
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
        else:
            return None
    return (host, int(port))


def _commit0_solve_phase_allowed_egress_endpoints(
    env: Mapping[str, str] | None = None,
) -> list[tuple[str, int]]:
    source = {str(k): str(v) for k, v in dict(os.environ).items()}
    source.update({str(k): str(v) for k, v in dict(env or {}).items()})
    endpoints: list[tuple[str, int]] = []
    for key in (
        *_COMMIT0_DOCKER_PROXY_ENV_KEYS,
        "X2P_AGENT_PROXY_ADDRESS",
        *_COMMIT0_MODEL_TRANSPORT_URL_ENV_KEYS,
    ):
        parsed = _commit0_endpoint_host_port(source.get(key, ""))
        if parsed is None:
            continue
        host, _port = parsed
        if key in _COMMIT0_MODEL_TRANSPORT_URL_ENV_KEYS and not (
            host in _COMMIT0_EGRESS_ALLOW_HOSTS
            or _commit0_model_transport_host_allowed(
                host,
                _commit0_model_transport_hosts_from_env(source),
            )
        ):
            continue
        endpoints.append(parsed)
    return sorted(set(endpoints))


def _commit0_docker_network_pool_exhausted(output: str) -> bool:
    lowered = str(output or "").lower()
    return (
        "predefined address pools" in lowered
        or "could not find an available" in lowered
        or "no available subnet" in lowered
    )


def _commit0_proxy_request_host(payload: bytes) -> str:
    try:
        text = payload[:8192].decode("iso-8859-1", "ignore")
    except Exception:
        return ""
    if not text:
        return ""
    first_line = text.split("\r\n", 1)[0]
    parts = first_line.split()
    if len(parts) >= 2:
        method = parts[0].upper()
        target = parts[1]
        if method == "CONNECT":
            return _commit0_normalize_host(target)
        if target.startswith(("http://", "https://")):
            return _commit0_url_host(target)
    for line in text.split("\r\n")[1:]:
        if line.lower().startswith("host:"):
            return _commit0_normalize_host(line.split(":", 1)[1].strip())
    return ""


@dataclass(frozen=True)
class _Commit0EgressProxyMapping:
    listen_port: int
    upstream_host: str
    upstream_port: int
    http_proxy_acl: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "listen_port": self.listen_port,
            "upstream_host": self.upstream_host,
            "upstream_port": self.upstream_port,
            "http_proxy_acl": self.http_proxy_acl,
        }


def _commit0_rewrite_endpoint_host_port(value: str, host: str, port: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    has_scheme = "://" in text
    parse_value = text if has_scheme else f"//{text}"
    try:
        parsed = urlsplit(parse_value)
    except ValueError:
        return text
    credentials = ""
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials += f":{parsed.password}"
        credentials += "@"
    rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    rewritten = urlunsplit(
        (
            parsed.scheme,
            f"{credentials}{rendered_host}:{int(port)}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
    if not has_scheme and rewritten.startswith("//"):
        return rewritten[2:]
    return rewritten


def _commit0_restricted_no_proxy_value(existing: str = "") -> str:
    allowed = {_COMMIT0_EGRESS_PROXY_ALIAS, "localhost", "127.0.0.1", "::1"}
    for token in str(existing or "").split(","):
        host = token.strip()
        if not host:
            continue
        if _commit0_normalize_host(host) in {"localhost", "127.0.0.1", "::1"}:
            allowed.add(host)
    return ",".join(sorted(allowed))


def _commit0_build_egress_proxy_plan(
    proxy_env: Mapping[str, str] | None,
    model_proxy_env: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, str], tuple[_Commit0EgressProxyMapping, ...]]:
    """Rewrite solve-phase proxy endpoints through the internal-network sidecar.

    SETUP uses the original endpoints. SOLVE receives only sidecar addresses, and
    the sidecar is the only peer reachable from the agent's internal Docker net.
    """

    proxy_env = {str(k): str(v) for k, v in dict(proxy_env or {}).items()}
    model_proxy_env = {str(k): str(v) for k, v in dict(model_proxy_env or {}).items()}
    rewritten_proxy_env: dict[str, str] = {}
    rewritten_model_proxy_env: dict[str, str] = {}
    mappings_by_key: dict[tuple[str, int, bool], _Commit0EgressProxyMapping] = {}
    next_port = _COMMIT0_EGRESS_PROXY_BASE_PORT

    def mapping_for(value: str, *, http_proxy_acl: bool) -> _Commit0EgressProxyMapping | None:
        nonlocal next_port
        endpoint = _commit0_endpoint_host_port(value)
        if endpoint is None:
            return None
        host, upstream_port = endpoint
        key = (host, upstream_port, http_proxy_acl)
        mapping = mappings_by_key.get(key)
        if mapping is not None:
            return mapping
        mapping = _Commit0EgressProxyMapping(
            listen_port=next_port,
            upstream_host=host,
            upstream_port=upstream_port,
            http_proxy_acl=http_proxy_acl,
        )
        mappings_by_key[key] = mapping
        next_port += 1
        return mapping

    for key, value in proxy_env.items():
        if key not in _COMMIT0_DOCKER_PROXY_ENV_KEYS:
            continue
        mapping = mapping_for(value, http_proxy_acl=True)
        if mapping is None:
            rewritten_proxy_env[key] = value
            continue
        rewritten_proxy_env[key] = _commit0_rewrite_endpoint_host_port(
            value,
            _COMMIT0_EGRESS_PROXY_ALIAS,
            mapping.listen_port,
        )
    for key, value in model_proxy_env.items():
        if key not in _COMMIT0_DOCKER_MODEL_PROXY_ENV_KEYS:
            continue
        mapping = mapping_for(value, http_proxy_acl=False)
        if mapping is None:
            rewritten_model_proxy_env[key] = value
            continue
        rewritten_model_proxy_env[key] = _commit0_rewrite_endpoint_host_port(
            value,
            _COMMIT0_EGRESS_PROXY_ALIAS,
            mapping.listen_port,
        )
    if mappings_by_key:
        no_proxy = _commit0_restricted_no_proxy_value(
            proxy_env.get("NO_PROXY") or proxy_env.get("no_proxy") or ""
        )
        rewritten_proxy_env["NO_PROXY"] = no_proxy
        rewritten_proxy_env["no_proxy"] = no_proxy
    return (
        rewritten_proxy_env,
        rewritten_model_proxy_env,
        tuple(sorted(mappings_by_key.values(), key=lambda mapping: mapping.listen_port)),
    )


def _commit0_egress_proxy_mappings_json(
    mappings: tuple[_Commit0EgressProxyMapping, ...] | list[_Commit0EgressProxyMapping],
) -> str:
    return json.dumps([mapping.to_dict() for mapping in mappings], sort_keys=True)


def _commit0_egress_proxy_sidecar_script(
    mappings: tuple[_Commit0EgressProxyMapping, ...] | list[_Commit0EgressProxyMapping],
    *,
    allowed_hosts: set[str] | None = None,
) -> str:
    mapping_payload = json.dumps([mapping.to_dict() for mapping in mappings])
    allowed_payload = json.dumps(
        sorted(
            _commit0_normalize_host(host)
            for host in (allowed_hosts or set(_COMMIT0_MODEL_TRANSPORT_HOSTS))
            if host
        )
    )
    suffix_payload = json.dumps(list(_COMMIT0_MODEL_TRANSPORT_HOST_SUFFIXES))
    template = r"""
import json
import select
import socket
import socketserver
import sys
import threading
from urllib.parse import urlsplit

MAPPINGS = json.loads(__MAPPINGS_JSON__)
ALLOWED_HOSTS = set(json.loads(__ALLOWED_HOSTS_JSON__))
ALLOWED_SUFFIXES = tuple(json.loads(__ALLOWED_SUFFIXES_JSON__))


def normalize_host(value):
    host = str(value or "").strip().strip("[]").lower().rstrip(".")
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if ":" in host and host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host


def url_host(value):
    text = str(value or "").strip()
    if not text:
        return ""
    parse_value = text if "://" in text else "http://" + text
    try:
        return normalize_host(urlsplit(parse_value).hostname or "")
    except ValueError:
        return ""


def request_host(payload):
    try:
        text = payload[:8192].decode("iso-8859-1", "ignore")
    except Exception:
        return ""
    if not text:
        return ""
    first_line = text.split("\r\n", 1)[0]
    parts = first_line.split()
    if len(parts) >= 2:
        method = parts[0].upper()
        target = parts[1]
        if method == "CONNECT":
            return normalize_host(target)
        if target.startswith(("http://", "https://")):
            return url_host(target)
    for line in text.split("\r\n")[1:]:
        if line.lower().startswith("host:"):
            return normalize_host(line.split(":", 1)[1].strip())
    return ""


def host_allowed(host):
    normalized = normalize_host(host)
    if not normalized:
        return False
    if normalized in ALLOWED_HOSTS:
        return True
    return any(normalized.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


def relay(left, right):
    sockets = [left, right]
    try:
        for sock in sockets:
            sock.setblocking(False)
        while True:
            readable, _, _ = select.select(sockets, [], [], 60.0)
            if not readable:
                continue
            for source in readable:
                try:
                    payload = source.recv(65536)
                except OSError:
                    return
                if not payload:
                    return
                target = right if source is left else left
                try:
                    target.sendall(payload)
                except OSError:
                    return
    finally:
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        mapping = getattr(self.server, "apex_mapping", None)
        if not mapping:
            return
        initial = b""
        try:
            self.request.settimeout(15)
            initial = self.request.recv(65536)
        except OSError:
            return
        if not initial:
            return
        if mapping.get("http_proxy_acl"):
            requested = request_host(initial)
            if not host_allowed(requested):
                try:
                    self.request.sendall(
                        b"HTTP/1.1 403 Forbidden\r\n"
                        b"Connection: close\r\n"
                        b"Content-Length: 0\r\n\r\n"
                    )
                except OSError:
                    pass
                print(
                    "APEX Commit0 egress sidecar denied non-model destination: "
                    + (requested or "<unknown>"),
                    file=sys.stderr,
                    flush=True,
                )
                return
        try:
            upstream = socket.create_connection(
                (mapping["upstream_host"], int(mapping["upstream_port"])),
                timeout=15,
            )
        except OSError as exc:
            # Commit0 solve sidecar fact: model transport and denied egress share this relay, so connection failures need host-level diagnostics.
            requested = request_host(initial)
            print(
                "APEX Commit0 egress sidecar upstream connect failed: "
                + "upstream="
                + str(mapping.get("upstream_host"))
                + ":"
                + str(mapping.get("upstream_port"))
                + " requested="
                + (requested or "<unknown>")
                + " acl="
                + str(bool(mapping.get("http_proxy_acl")))
                + " error="
                + str(exc),
                file=sys.stderr,
                flush=True,
            )
            return
        with upstream:
            try:
                upstream.sendall(initial)
            except OSError:
                return
            relay(self.request, upstream)


servers = []
for mapping in MAPPINGS:
    server = Server(("0.0.0.0", int(mapping["listen_port"])), Handler)
    server.apex_mapping = mapping
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    servers.append((server, thread))
print("APEX Commit0 egress sidecar ready", file=sys.stderr, flush=True)
threading.Event().wait()
"""
    return (
        template.replace("__MAPPINGS_JSON__", repr(mapping_payload))
        .replace("__ALLOWED_HOSTS_JSON__", repr(allowed_payload))
        .replace("__ALLOWED_SUFFIXES_JSON__", repr(suffix_payload))
        .strip()
    )


class _Commit0ProxyTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _relay_tcp_streams(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    try:
        for sock in sockets:
            sock.setblocking(False)
        while True:
            readable, _, _ = select.select(sockets, [], [], 60.0)
            if not readable:
                continue
            for source in readable:
                try:
                    payload = source.recv(65536)
                except OSError:
                    return
                if not payload:
                    return
                target = right if source is left else left
                try:
                    target.sendall(payload)
                except OSError:
                    return
    finally:
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _Commit0ProxyRelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        target = getattr(self.server, "apex_target", None)
        if not target:
            return
        try:
            self.request.settimeout(15)
            initial_payload = self.request.recv(65536)
        except OSError:
            return
        if not initial_payload:
            return
        allowed_hosts = getattr(self.server, "apex_allowed_hosts", set())
        requested_host = _commit0_proxy_request_host(initial_payload)
        if not _commit0_model_transport_host_allowed(requested_host, set(allowed_hosts)):
            try:
                self.request.sendall(
                    b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            logger.info(
                "Commit0 solve-phase proxy denied non-model destination: %s",
                requested_host or "<unknown>",
            )
            return
        try:
            upstream = socket.create_connection(target, timeout=15)
        except OSError as exc:
            logger.debug("Commit0 Docker proxy relay connection failed: %s", exc)
            return
        with upstream:
            try:
                upstream.sendall(initial_payload)
            except OSError:
                return
            _relay_tcp_streams(self.request, upstream)


@dataclass
class _Commit0DockerProxyRelay:
    server: _Commit0ProxyTCPServer
    thread: threading.Thread
    target_host: str
    target_port: int

    @property
    def listen_port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@dataclass
class _Commit0ColimaProxyTunnel:
    process: subprocess.Popen[Any]
    listen_port: int
    target_host: str
    target_port: int

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _format_count_rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100.0 * _safe_ratio(numerator, denominator):.1f}%)"


def _format_optional_timeout_seconds(value: Any) -> str:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        return "n/a"
    if timeout <= 0:
        return "disabled"
    return f"{timeout}s"


def _commit0_gold_docker_requires_containerized_agent_cli(config: ApexConfig) -> bool:
    policy = dict(config.benchmark.runtime_policy or {})
    contract = config.benchmark.resolved_evaluation_contract_config("commit0")
    docker_mode = str(config.benchmark.commit0_docker_runtime_mode or "").strip().lower()
    return (
        str(contract.get("mode") or "").strip().lower() == "gold_suite_visible"
        and bool(policy.get("target_evaluation_runtime_required"))
        and docker_mode != "never"
    )


def _commit0_expected_id_scoring_required(config: ApexConfig) -> bool:
    contract = config.benchmark.resolved_evaluation_contract_config("commit0")
    mode = str(contract.get("mode") or "").strip().lower().replace("-", "_")
    scoring_universe = str(contract.get("scoring_universe") or "").strip().lower().replace("-", "_")
    return scoring_universe in {"expected_test_ids", "commit0_test_ids"} and (
        not mode or mode == "gold_suite_visible"
    )


def _commit0_target_cli_auth_mode(config: ApexConfig) -> str:
    mode = str((config.benchmark.runtime_policy or {}).get("target_cli_auth_mode") or "").strip()
    # Commit0 gold mode evaluates scrubbed Docker workspaces; host_cli would leave
    # the provider CLI process on the host, so require a containerized agent path.
    if (
        mode.lower() in _COMMIT0_HOST_CLI_AUTH_MODES
        and _commit0_gold_docker_requires_containerized_agent_cli(config)
    ):
        raise RuntimeError(
            "Commit0 gold_suite_visible Docker runs cannot use "
            "runtime_policy.target_cli_auth_mode=host_cli because the provider CLI "
            "would execute as a host process. Leave target_cli_auth_mode unset to "
            "run the agent CLI inside the Commit0 docker_exec runtime, or use a "
            "Docker Sandbox / model-proxy mode that keeps the agent CLI "
            "container-confined."
        )
    return mode


def _build_commit0_benchmark_policy(config: ApexConfig) -> dict[str, Any]:
    primary_backend = config.benchmark.commit0_primary_evaluation_backend.value
    return build_benchmark_policy(
        benchmark_name="commit0",
        benchmark_family="commit0",
        agent_input_contract={
            "repo_snapshot_visible": True,
            "issue_statement_visible": True,
            "install_command_visible": True,
            "test_command_visible": True,
            "visible_repo_tests_visible": True,
            "coverage_guardrail_visible_in_prompt": True,
            "expected_test_count_visible_in_prompt": True,
            "expected_test_ids_visible_in_prompt": True,
            "expected_test_ids_visible_via_workspace_file": _APEX_EXPECTED_IDS_FILENAME,
            "upstream_reference_visible_in_prompt": False,
        },
        orchestrator_input_contract={
            "benchmark_metadata_passed_to_orchestrator": True,
            "benchmark_metadata_fields": [
                "expected_test_count",
                "expected_test_ids",
                "protect_visible_test_files",
            ],
            "benchmark_metadata_visible_in_prompt": True,
            "benchmark_metadata_used_for_rollout_selection": True,
            "orchestrator_visible_hidden_evaluator_metadata": False,
            "hidden_evaluator_metadata_prompt_visible": False,
        },
        evaluation_protocol={
            "baseline_evaluation_backend": primary_backend,
            "final_evaluation_backend": primary_backend,
            "rollout_selection_policy": "public_commit0_expected_id_pytest_candidate_rescoring",
            "candidate_selection_scoring_source": "public_commit0_expected_test_ids_pytest_summary",
            "candidate_selection_uses_expected_test_ids": True,
            "official_audit_candidate_reranking_enabled": (
                config.benchmark.commit0_audit_candidate_selection
            ),
            "primary_metric": "average_expected_test_pass_rate",
            "primary_scoring_source": "commit0_test_ids_over_pytest_summary",
            "official_audit_selected": config.benchmark.commit0_official_audit_selected,
            "official_audit_only_if_primary_passes": (
                config.benchmark.commit0_official_audit_only_if_primary_passes
            ),
            "official_audit_backend": COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER,
            "sampling_protocol": "single_run_per_repo",
        },
        environment_policy={
            "agent_execution_isolation": "per_task_temp_sandbox",
            "workspace_isolation": "per_task_workspace_dir",
            "agent_network_access": "inherited_host",
            "primary_evaluator_network_access": "inherited_host",
            "official_audit_network_access": "docker_default",
            "persistent_outputs_outside_repo": True,
        },
        benchmark_specifics={
            "protect_visible_test_files": True,
            "evidence_mode": "gold_suite_visible",
            "expected_test_inventory_source": "commit0_public_test_inventory",
            "expected_test_ids_workspace_file": _APEX_EXPECTED_IDS_FILENAME,
            "issue_prompt_hash_logged": True,
        },
    )


def _artifact_safe_commit0_task_payload(task: "Commit0Task") -> dict[str, Any]:
    payload = task.to_dict()
    payload["original_repo"] = ""
    return payload


def _artifact_safe_issue_prompt_payload(
    *,
    issue_description: str,
    test_command: str,
) -> dict[str, Any]:
    return {
        "prompt_hash_sha256": hashlib.sha256(issue_description.encode("utf-8")).hexdigest(),
        "prompt_length_chars": len(issue_description),
        "prompt_length_lines": issue_description.count("\n") + 1,
        "test_command": test_command,
        "benchmark_metadata_redacted": True,
    }


def _artifact_safe_issue_plan_payload(issue_plan: Any) -> Any:
    if not isinstance(issue_plan, dict):
        return _scrub_commit0_hidden_artifact_payload(issue_plan)
    try:
        payload = IssuePlan.from_dict(issue_plan).to_dict()
    except Exception:
        payload = dict(issue_plan)
        payload["test_context"] = dict(payload.get("test_context") or {})
        payload["evaluation_constraints"] = dict(payload.get("evaluation_constraints") or {})
        payload["evaluation_constraints"]["test_inventory"] = dict(
            payload["evaluation_constraints"].get("test_inventory") or {}
        )
    return _scrub_commit0_hidden_artifact_payload(payload)


def _scrub_commit0_hidden_artifact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        scrubbed: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "issue_plan":
                scrubbed[key] = _artifact_safe_issue_plan_payload(value)
                continue
            if key == "original_repo":
                scrubbed[key] = ""
                continue
            if key in {
                "patch_path",
                "selected_worktree_path",
                "working_dir",
                "worktree_path",
            }:
                scrubbed[key] = ""
                continue
            scrubbed[key] = _scrub_commit0_hidden_artifact_payload(value)
        return scrubbed
    if isinstance(payload, list):
        return [_scrub_commit0_hidden_artifact_payload(item) for item in payload]
    if isinstance(payload, str):
        scrubbed = payload
        for pattern in _COMMIT0_TRANSIENT_PATH_PATTERNS:
            scrubbed = pattern.sub("[redacted-temp-path]", scrubbed)
        for pattern in _COMMIT0_REFERENCE_HINT_PATTERNS:
            scrubbed = pattern.sub("[redacted-reference]", scrubbed)
        return scrubbed
    return copy.deepcopy(payload)


def _rewrite_json_artifact_if_present(path: Path, transform: Callable[[Any], Any]) -> None:
    payload = load_json_if_exists(path)
    if payload is None:
        return
    atomic_write_json(path, transform(payload))


def _write_commit0_candidate_handoff(task_output_dir: Path) -> None:
    handoff_path = task_output_dir / "_internal" / "candidate_handoff.json"
    if handoff_path.exists():
        return
    payload = load_json_if_exists(task_output_dir / "apex_result.json")
    if not isinstance(payload, dict):
        return
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(handoff_path, payload)


def _scrub_commit0_run_artifacts(task_output_dir: Path) -> None:
    _write_commit0_candidate_handoff(task_output_dir)
    for relative_path in (
        "apex_result.json",
        "task_live_state.json",
        "task_state_graph.json",
    ):
        _rewrite_json_artifact_if_present(
            task_output_dir / relative_path,
            _scrub_commit0_hidden_artifact_payload,
        )
    for subdir_name in ("trajectories", "rollout_status"):
        directory = task_output_dir / subdir_name
        if not directory.is_dir():
            continue
        for artifact_path in sorted(directory.glob("*.json")):
            _rewrite_json_artifact_if_present(
                artifact_path,
                _scrub_commit0_hidden_artifact_payload,
            )


@contextmanager
def _interruptible_thread_pool(max_workers: int) -> Iterator[ThreadPoolExecutor]:
    """Thread pool that does not block on shutdown after interrupts/fatal errors."""

    executor = ThreadPoolExecutor(max_workers=max_workers)
    interrupted = False
    try:
        yield executor
    except BaseException:
        interrupted = True
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not interrupted and not bool(getattr(executor, "_apex_abandon_on_exit", False)):
            executor.shutdown(wait=True)


def _abandon_interruptible_thread_pool(executor: ThreadPoolExecutor) -> None:
    """Skip the clean-exit join for scorer jobs that were explicitly cancelled."""

    setattr(executor, "_apex_abandon_on_exit", True)
    executor.shutdown(wait=False, cancel_futures=True)


def _commit0_candidate_eval_task_id(task_instance_id: str, rollout_id: int) -> str:
    return f"{task_instance_id}:candidate_eval:{int(rollout_id)}"


class _TaskSolveSlot:
    """Idempotent release handle for a benchmark solve-lane permit."""

    def __init__(
        self,
        semaphore: threading.BoundedSemaphore,
        *,
        on_release: Optional[Callable[[], None]] = None,
    ):
        self._semaphore = semaphore
        self._on_release = on_release
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            self._released = True
        self._semaphore.release()
        if self._on_release is not None:
            self._on_release()
        return True


def _is_commit0_harness_failure(output: str) -> bool:
    message = output or ""
    lowered_message = message.lower()
    failure_markers = (
        "No spec available",
        "No example available",
        "No repo available",
        "are not git directories",
        "does not exist locally or remotely",
        "PermissionError: [Errno 1] Operation not permitted:",
        # Phase 4 10.Q-a: docker / network-layer infrastructure failures
        # the harness can't recover from — surface them as harness
        # failures so audit promotion doesn't overwrite a good local
        # result with a docker-layer error.
        "docker: Error response from daemon",
        "manifest unknown",
        "registry-1.docker.io",
        "connection refused",
        "no such host",
        "i/o timeout",
    )
    if any(marker in message for marker in failure_markers):
        return True
    unicode_harness_markers = (
        "surrogates not allowed",
        "codec can't encode",
        "codec can't decode",
        "unicodeencodeerror",
        "unicodedecodeerror",
    )
    if any(marker in lowered_message for marker in unicode_harness_markers):
        return True
    python_stdio_harness_markers = (
        "can't initialize sys standard streams",
        "bad file descriptor",
    )
    if any(marker in lowered_message for marker in python_stdio_harness_markers):
        return True
    if "runtime/.venv" in message and "PermissionError" in message:
        return True
    if "ModuleNotFoundError: No module named" in message and (
        "numba" in message or "hypothesis" in message
    ):
        return True
    return False


@contextmanager
def _stream_task_output_artifacts(
    sync_fn: Callable[[Path, Path], None],
    source_dir: Path,
    destination_dir: Path,
    *,
    interval_seconds: float = 1.0,
) -> Iterator[None]:
    """Mirror sandbox task artifacts into the persisted run dir during solve()."""

    if not source_dir.exists():
        yield
        return

    interval = max(float(interval_seconds or 0.0), 0.10)
    stop_event = threading.Event()

    def _sync_once() -> None:
        try:
            sync_fn(source_dir, destination_dir)
        except Exception:
            logger.debug(
                "Failed to mirror Commit0 task artifacts from %s to %s",
                source_dir,
                destination_dir,
                exc_info=True,
            )

    def _worker() -> None:
        while not stop_event.wait(interval):
            _sync_once()

    _sync_once()
    worker = threading.Thread(
        target=_worker,
        name=f"apex-commit0-sync-{destination_dir.name}",
        daemon=True,
    )
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=max(2.0, interval * 2.0))
        _sync_once()


@dataclass
class Commit0Task:
    """One Commit0 benchmark target."""

    instance_id: str
    repo: str
    original_repo: str
    base_commit: str
    reference_commit: str
    python_version: str
    specification: str
    install_command: str
    packages: list[str] = field(default_factory=list)
    pip_packages: list[str] = field(default_factory=list)
    pre_install: list[str] = field(default_factory=list)
    src_dir: str = ""
    test_cmd: str = "pytest"
    test_dir: str = "tests/"

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def src_root(self) -> str:
        src_dir = self.src_dir.strip().rstrip("/")
        if not src_dir:
            return ""
        parent = Path(src_dir).parent.as_posix()
        if parent not in {"", "."}:
            return parent
        # Layer B (Python src-layout ecosystem fact): a top-level ``src/``
        # directory is the conventional PEP 517 source root that *contains* the
        # importable package(s) (e.g. pytest's ``src/_pytest``), not a package
        # itself, so the import root is ``src`` rather than its parent. Returning
        # "" here would leave ``<repo>/src`` off PYTHONPATH and let a pre-installed
        # distribution shadow the candidate's edited source. A top-level package
        # directory (e.g. ``networkx``) keeps the repo root as the import root.
        if Path(src_dir).name == "src":
            return src_dir
        return ""

    def build_issue_description(
        self,
        test_command: str,
        *,
        include_upstream_reference: bool = False,
        expected_test_count: Optional[int] = None,
        expected_test_ids: Optional[list[str]] = None,
    ) -> str:
        lines = [
            "Implement the missing library functionality in this repository so the visible test suite passes.",
            "Treat this as a repository-completion task rather than a single bug fix.",
            "",
            TASK_FRAMING_BLOCK,
        ]
        specification = str(self.specification or "").strip()
        if specification:
            # P0.4 (de-seed): strip upstream URLs from the objective so neither arm is
            # handed a fetch target in the prompt. (Version-literal stripping is omitted
            # on purpose — it risks garbling the spec, e.g. "RFC 3986" / "Python 3".)
            import re as _re
            specification = _re.sub(r"https?://\S+", "", specification).strip()
        if specification:
            lines.append(f"Task objective: {specification}")
        lines.extend(
            [
                "",
                f"Target Python version: {self.python_version}",
                f"Primary source directory: {self.src_dir or '.'}",
                f"Install command already applied by the Commit0 harness: {self.install_command}",
                f"Repository test command: {test_command}",
            ]
        )
        if include_upstream_reference and self.original_repo:
            lines.append(f"Upstream reference repository: {self.original_repo}")
        lines.extend(
            [
                "",
                "Read the existing tests and source to infer the intended library behavior.",
                "Treat visible tests as read-only specification unless APEX identifies exact `incomplete_test_files` placeholder bodies; do not modify tests to make failures disappear or reduce coverage.",
            ]
        )
        if isinstance(expected_test_count, int) and expected_test_count > 0:
            lines.append(
                "Commit0 provides its unit tests to the agent; treat the expected "
                "pytest inventory as public development feedback, not a hidden oracle."
            )
            lines.append(f"Expected visible test count: {expected_test_count}")
            if expected_test_ids:
                lines.append(f"Expected visible test inventory file: {_APEX_EXPECTED_IDS_FILENAME}")
            lines.append(
                "Run the full repository test command and preserve test collection "
                "coverage while implementing the missing functionality."
            )
            lines.append(
                "Do not treat the task as solved if tests disappear from collection "
                "or parameterized coverage."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        try:
            self.contract_decision()
        except Exception:
            pass
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "original_repo": self.original_repo,
            "base_commit": self.base_commit,
            "reference_commit": self.reference_commit,
            "python_version": self.python_version,
            "specification": self.specification,
            "install_command": self.install_command,
            "packages": list(self.packages),
            "pip_packages": list(self.pip_packages),
            "pre_install": list(self.pre_install),
            "src_dir": self.src_dir,
            "test_cmd": self.test_cmd,
            "test_dir": self.test_dir,
        }


@dataclass
class Commit0Evaluation:
    """One benchmark test execution summary."""

    returncode: int
    output: str = ""
    raw_returncode: Optional[int] = None
    report_path: Optional[str] = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total_tests: int = 0
    scoring_source: str = "pytest_summary"
    evaluation_backend: str = "unknown"
    expected_test_coverage: dict[str, Any] = field(default_factory=dict)
    # Phase 1.2: declare which scoring source produced the headline rc
    # for this evaluation. One of:
    #   "upstream_audit"           — official commit0 docker harness
    #   "apex_private_pytest_json" — APEX-private exit-code rewrite from
    #                                pytest-json-report (gated by the
    #                                ``commit0_use_pytest_json_exitcode``
    #                                BenchmarkConfig flag)
    #   "commit0_benign_extra_normalized" — expected IDs passed and all
    #                                raw failures were classified as
    #                                benign non-scored Commit0 extras
    #   "shell_rc"                 — canonical shell returncode (default)
    score_source: str = "shell_rc"
    # Phase 1.2: structured diagnostics for downstream auditors. Notably
    # carries ``pytest_returncode_disagrees_with_report`` when the
    # private rewrite was suppressed but the pytest-json report's
    # exitcode diverges from the shell rc.
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Phase 1: failure classification produced by the new core
    # ``classify_failure``. ``failure_class`` is the FailureClass enum
    # value (e.g. "apex_miss", "env_network"); ``failure_classification``
    # is the full ClassificationResult.to_dict() payload for downstream
    # audit. Both default to None — populated by callers when a failure
    # is observed.
    failure_class: Optional[str] = None
    failure_classification: Optional[dict[str, Any]] = None
    evaluation_contract: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        # Commit0 expected-ID scoring is a visible gold contract: an expected
        # test that pytest skipped did not pass and must remain score-bearing.
        skipped = self.skipped if self.scoring_source == "commit0_test_ids" else 0
        runnable = self.passed + self.failed + self.errors + skipped
        if runnable <= 0:
            return 0.0
        return self.passed / runnable

    @property
    def scored_success(self) -> bool:
        return bool(_commit0_evaluation_success(self))

    @property
    def raw_pytest_success(self) -> bool:
        raw_returncode = self.raw_returncode
        if raw_returncode is None:
            raw_returncode = self.returncode
        return int(raw_returncode) == 0

    @property
    def scored_returncode(self) -> int:
        return 0 if self.scored_success else 1

    @property
    def evaluation_status(self) -> str:
        decision = self.decision or _commit0_evaluation_decision(self).to_dict()
        if decision.get("kind") == "harness_failure":
            return "audit_inconclusive"
        if self.scored_success:
            extra = self.diagnostics.get("extra_non_scored_tests")
            if (
                int(self.returncode) != 0
                and isinstance(extra, dict)
                and (int(extra.get("failed") or 0) or int(extra.get("errors") or 0))
            ):
                return "solved_with_extra_non_scored_failures"
            return "solved"
        return "unsolved"

    def to_dict(self) -> dict[str, Any]:
        raw_returncode = self.raw_returncode
        if raw_returncode is None:
            raw_returncode = self.returncode
        return {
            "returncode": self.returncode,
            "raw_returncode": raw_returncode,
            "scored_returncode": self.scored_returncode,
            "scored_success": self.scored_success,
            "raw_pytest_success": self.raw_pytest_success,
            "evaluation_status": self.evaluation_status,
            "output": self.output,
            "report_path": self.report_path,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total_tests": self.total_tests,
            "pass_rate": self.pass_rate,
            "scoring_source": self.scoring_source,
            "evaluation_backend": self.evaluation_backend,
            "expected_test_coverage": copy.deepcopy(self.expected_test_coverage),
            "score_source": self.score_source,
            "diagnostics": copy.deepcopy(self.diagnostics),
            "failure_class": self.failure_class,
            "failure_classification": copy.deepcopy(self.failure_classification)
            if self.failure_classification is not None
            else None,
            "evaluation_contract": copy.deepcopy(self.evaluation_contract),
            "decision": copy.deepcopy(self.decision),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Commit0Evaluation":
        return cls(
            returncode=int(payload.get("returncode", 1)),
            output=str(payload.get("output", "") or ""),
            raw_returncode=(
                int(payload["raw_returncode"])
                if payload.get("raw_returncode") is not None
                else None
            ),
            report_path=payload.get("report_path"),
            passed=int(payload.get("passed", 0) or 0),
            failed=int(payload.get("failed", 0) or 0),
            errors=int(payload.get("errors", 0) or 0),
            skipped=int(payload.get("skipped", 0) or 0),
            total_tests=int(payload.get("total_tests", 0) or 0),
            scoring_source=str(payload.get("scoring_source", "pytest_summary") or "pytest_summary"),
            evaluation_backend=str(payload.get("evaluation_backend", "unknown") or "unknown"),
            expected_test_coverage=dict(payload.get("expected_test_coverage") or {}),
            score_source=str(payload.get("score_source", "shell_rc") or "shell_rc"),
            diagnostics=dict(payload.get("diagnostics") or {}),
            failure_class=payload.get("failure_class"),
            failure_classification=dict(payload["failure_classification"])
            if isinstance(payload.get("failure_classification"), dict)
            else None,
            evaluation_contract=dict(payload.get("evaluation_contract") or {}),
            decision=dict(payload.get("decision") or {}),
        )

    def contract_decision(self) -> EvaluationDecision:
        return _commit0_evaluation_decision(self)

    def contract_success(self) -> bool:
        return bool(self.contract_decision().is_success)


def _commit0_contract_for_evaluation(
    evaluation: Commit0Evaluation,
) -> EvaluationContract:
    if evaluation.scoring_source == "commit0_test_ids":
        return EvaluationContract.commit0_expected_ids()
    return EvaluationContract.full_runner_summary()


def _commit0_scored_counts(evaluation: Commit0Evaluation) -> ScoredCounts:
    coverage = dict(evaluation.expected_test_coverage or {})
    missing = int(coverage.get("missing_expected_test_count") or 0)
    failed = int(evaluation.failed or 0)
    if evaluation.scoring_source == "commit0_test_ids":
        # Commit0 gold fact: visible expected IDs are score-bearing cases; a
        # pytest skip did not pass the expected test and must lower the score.
        failed += int(evaluation.skipped or 0)
    return ScoredCounts(
        passed=int(evaluation.passed or 0),
        failed=failed,
        errors=int(evaluation.errors or 0),
        skipped=int(evaluation.skipped or 0),
        total=int(evaluation.total_tests or 0),
        missing=missing,
    )


def _commit0_runner_health(evaluation: Commit0Evaluation) -> RunnerHealth:
    diagnostics = dict(evaluation.diagnostics or {})
    if diagnostics.get("timeout"):
        return RunnerHealth.TIMEOUT
    if diagnostics.get("parser_error"):
        return RunnerHealth.PARSER_ERROR
    if diagnostics.get("harness_failure"):
        return RunnerHealth.HARNESS_FAILURE
    signal_count = int(evaluation.passed) + int(evaluation.failed) + int(evaluation.errors)
    if signal_count <= 0 and _is_commit0_harness_failure(evaluation.output):
        return RunnerHealth.HARNESS_FAILURE
    return RunnerHealth.SUCCESS


def _commit0_evaluation_decision(
    evaluation: Commit0Evaluation,
) -> EvaluationDecision:
    contract = _commit0_contract_for_evaluation(evaluation)
    diagnostics = dict(evaluation.diagnostics or {})
    decision = decide_evaluation(
        contract=contract,
        scored=_commit0_scored_counts(evaluation),
        raw_returncode=int(evaluation.returncode),
        runner_health=_commit0_runner_health(evaluation),
        diagnostics=diagnostics,
    )
    evaluation.evaluation_contract = contract.to_dict()
    evaluation.decision = decision.to_dict()
    return decision


def _apply_expected_id_terminal_summary_fallback(
    evaluation: Commit0Evaluation,
    *,
    expected_test_count: int,
    reason: str,
) -> bool:
    if expected_test_count <= 0:
        return False
    summary_counts = parse_pytest_terminal_summary_counts(evaluation.output)
    passed = (
        int(summary_counts.get("passed") or 0)
        + int(summary_counts.get("xfailed") or 0)
        + int(summary_counts.get("xpassed") or 0)
    )
    failed = int(summary_counts.get("failed") or 0)
    errors = int(summary_counts.get("errors") or 0)
    skipped = int(summary_counts.get("skipped") or 0)
    observed_total = passed + failed + errors + skipped
    if observed_total <= 0:
        return False
    # GAP1 (gold_suite_visible contract integrity): aggregate terminal-summary
    # counts can prove a full expected-id pass in EXACTLY one case — an exact,
    # fully-clean pass: the observed count equals the expected count, nothing
    # failed or errored, and nothing was skipped, so every observed test is an
    # unambiguous pass and the counts cannot be hiding an uncollected expected
    # ID. Every other shape is an UNVERIFIABLE scoring universe and must NOT be
    # credited from aggregate counts:
    #   - a superset (observed > expected) may be a wrong-rootdir / broad-collection
    #     run whose extras coincidentally pass while the real expected IDs were
    #     never collected (the previous min(passed, expected) backfill silently
    #     treated uncollected expected IDs as passes — a false-SOLVED path);
    #   - a count collision with skips hides uncollected expected IDs behind a
    #     coincidentally-matching total.
    # Without per-test node IDs we cannot attribute passes to the specific
    # expected IDs, so we reject (the caller stamps parser_error -> HARNESS_FAILURE)
    # rather than risk inflating the published headline. The legitimate
    # json-report-missing recovery still works for the exact-clean-pass case.
    if not (
        observed_total == expected_test_count
        and failed == 0
        and errors == 0
        and skipped == 0
        and passed == expected_test_count
    ):
        evaluation.diagnostics["pytest_expected_id_terminal_summary_fallback_rejected"] = {
            "reason": reason,
            "rejected": "terminal_summary_unverifiable_universe",
            "summary_counts": dict(summary_counts),
            "expected_test_count": expected_test_count,
            "observed_total": observed_total,
        }
        return False
    scored_passed = passed
    scored_skipped = 0

    evaluation.scoring_source = "commit0_test_ids"
    evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
    evaluation.total_tests = expected_test_count
    evaluation.passed = scored_passed
    evaluation.failed = failed
    evaluation.errors = errors
    evaluation.skipped = scored_skipped
    evaluation.expected_test_coverage = {
        "expected_test_count": expected_test_count,
        "matched_expected_test_count": expected_test_count,
        "missing_expected_test_count": 0,
        "skipped_expected_test_count": scored_skipped,
        "coverage_preserved": True,
        "collected_test_count": observed_total,
        "terminal_summary_fallback": True,
    }
    # Pytest JSON reports can omit per-test entries in some Commit0
    # wrapper/plugin combinations; when pytest's terminal summary cleanly
    # covers the visible expected-id universe, preserve aggregate scorer
    # parity instead of reporting an unobserved private score.
    evaluation.diagnostics["pytest_expected_id_terminal_summary_fallback"] = {
        "reason": reason,
        "summary_counts": dict(summary_counts),
    }
    _commit0_evaluation_decision(evaluation)
    return True


def _commit0_evaluation_success(evaluation: Commit0Evaluation) -> bool:
    return bool(_commit0_evaluation_decision(evaluation).is_success)


def _commit0_evaluation_signal_count(evaluation: Commit0Evaluation) -> int:
    return int(evaluation.passed) + int(evaluation.failed) + int(evaluation.errors)


def _commit0_official_audit_usable(evaluation: Commit0Evaluation) -> bool:
    if evaluation.evaluation_backend != COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER:
        return False
    if _commit0_evaluation_signal_count(evaluation) <= 0:
        return False
    diagnostics = dict(evaluation.diagnostics or {})
    if diagnostics.get("parser_error") or diagnostics.get("harness_failure"):
        return False
    return True


def _commit0_audit_error_is_transient_teardown_flake(evaluation: Commit0Evaluation) -> bool:
    """B5/NDFF: True iff the official audit is green except for scored ERRORS that
    match a known non-deterministic teardown/finalizer signature.

    Strict by construction so it never re-runs (and never hides) a genuine
    scored failure: it requires the official-docker backend, a usable result,
    ``failed == 0`` with ``errors > 0`` and ``passed > 0``, preserved expected-id
    coverage, no missing expected IDs, and at least one teardown marker in the
    captured output (the canonical marker set lives in
    :mod:`apex.evaluation.flake_firewall`). An AssertionError / real ``failed``
    outcome therefore can never be re-run away.
    """
    if not _commit0_official_audit_usable(evaluation):
        return False
    if int(evaluation.failed) != 0:
        return False
    if int(evaluation.errors) <= 0:
        return False
    if int(evaluation.passed) <= 0:
        return False
    coverage = dict(evaluation.expected_test_coverage or {})
    if coverage:
        if coverage.get("coverage_preserved") is False:
            return False
        missing = coverage.get("missing_expected_test_count")
        if isinstance(missing, int) and missing > 0:
            return False
    return output_has_teardown_leak_signature(evaluation.output)


def _commit0_audit_error_is_transient_harness_failure(
    evaluation: Commit0Evaluation,
) -> bool:
    """True for official-audit failures that produced no scored signal.

    This retry class is narrower than generic harness failure handling: it only
    covers the official Docker audit backend, requires zero scored pass/fail/error
    signal, and requires parser/harness diagnostics or a known harness-output
    marker. Retrying this shape cannot erase a candidate regression because the
    audit has not observed any score-bearing expected test yet.
    """
    if evaluation.evaluation_backend != COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER:
        return False
    if _commit0_evaluation_signal_count(evaluation) > 0:
        return False
    diagnostics = dict(evaluation.diagnostics or {})
    return bool(
        diagnostics.get("parser_error")
        or diagnostics.get("harness_failure")
        or _is_commit0_harness_failure(evaluation.output)
    )


def _git_index_lock_paths_from_output(output: str, *, cwd: Path | str) -> list[Path]:
    if "index.lock" not in output:
        return []
    cwd_path = Path(cwd)
    lock_paths: list[Path] = []
    seen: set[str] = set()
    for match in _GIT_INDEX_LOCK_PATH_RE.finditer(output):
        raw_path = (match.group("path") or "").strip()
        if not raw_path:
            continue
        lock_path = Path(raw_path)
        if not lock_path.is_absolute():
            lock_path = cwd_path / lock_path
        if lock_path.name != "index.lock":
            continue
        # Only recover locks that look like Git administrative state. This keeps
        # stale-lock cleanup scoped to the Commit0/Git execution mechanism.
        if ".git" not in lock_path.parts and "worktrees" not in lock_path.parts:
            continue
        key = str(lock_path)
        if key in seen:
            continue
        seen.add(key)
        lock_paths.append(lock_path)
    return lock_paths


def _pytest_extra_non_scored_diagnostics(
    *,
    repo_name: str = "",
    outcomes: dict[str, str],
    expected_test_ids: list[str],
    sample_limit: int = 20,
) -> dict[str, Any]:
    expected = {test_id for test_id in expected_test_ids if test_id}
    extra: list[tuple[str, str]] = [
        (node_id, outcome)
        for node_id, outcome in sorted(outcomes.items())
        if node_id not in expected
    ]
    if not extra:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "sample_failures": [],
        }
    passed = sum(1 for _, outcome in extra if outcome in {"passed", "xfailed", "xpassed"})
    failed = sum(1 for _, outcome in extra if outcome == "failed")
    errors = sum(1 for _, outcome in extra if outcome == "error")
    skipped = sum(1 for _, outcome in extra if outcome == "skipped")
    benign_failures: list[dict[str, str]] = []
    non_benign_failures: list[dict[str, str]] = []
    for node_id, outcome in extra:
        if outcome not in {"failed", "error"}:
            continue
        entry = {"nodeid": node_id, "outcome": outcome}
        benign_rationale = _commit0_benign_extra_test_rationale(
            repo_name=repo_name,
            nodeid=node_id,
            outcome=outcome,
        )
        if benign_rationale:
            entry["rationale"] = benign_rationale
            benign_failures.append(entry)
        else:
            non_benign_failures.append(entry)
    sample_failures = [*benign_failures, *non_benign_failures][:sample_limit]
    return {
        "total": len(extra),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "sample_failures": sample_failures,
        "benign_failed": sum(1 for item in benign_failures if item["outcome"] == "failed"),
        "benign_errors": sum(1 for item in benign_failures if item["outcome"] == "error"),
        "non_benign_failed": sum(1 for item in non_benign_failures if item["outcome"] == "failed"),
        "non_benign_errors": sum(1 for item in non_benign_failures if item["outcome"] == "error"),
        "benign_sample_failures": benign_failures[:sample_limit],
        "non_benign_sample_failures": non_benign_failures[:sample_limit],
        "all_failures_benign": bool(failed or errors) and not non_benign_failures,
    }


def _commit0_benign_extra_test_rationale(
    *,
    repo_name: str,
    nodeid: str,
    outcome: str,
) -> str:
    if outcome not in {"failed", "error"}:
        return ""
    repo = str(repo_name or "").strip().lower()
    test_id = str(nodeid or "").strip()
    if repo == "networkx":
        # NetworkX drawing tests are outside Commit0 expected IDs and depend on Matplotlib rendering details in the audit runtime.
        if test_id.startswith(
            "networkx/drawing/tests/test_pylab.py::"
            "test_draw_networkx_edges_multiedge_connectionstyle"
        ) or test_id.startswith(
            "networkx/drawing/tests/test_pylab.py::"
            "test_draw_networkx_edge_labels_multiedge_connectionstyle"
        ):
            return (
                "networkx drawing extra outside Commit0 expected IDs; "
                "Matplotlib rendering-sensitive diagnostic"
            )
    if repo == "seaborn":
        # Seaborn extra KDE weights test is outside Commit0 expected IDs and fails on NumPy VisibleDeprecationWarning removal in the audit runtime.
        if test_id == "tests/test_distributions.py::TestKDEPlotBivariate::test_weights":
            return (
                "seaborn extra outside Commit0 expected IDs; "
                "NumPy VisibleDeprecationWarning runtime-compatibility diagnostic"
            )
    if repo == "web3.py":
        # web3.py beacon/go-ethereum tests are outside Commit0 expected IDs and require external service fixtures unavailable in the audit runtime.
        if test_id.startswith("tests/beacon/") or test_id.startswith(
            "tests/integration/go_ethereum/"
        ):
            return (
                "web3.py integration extra outside Commit0 expected IDs; "
                "external service fixture diagnostic"
            )
    return ""


def _commit0_can_normalize_benign_extra_returncode(
    evaluation: Commit0Evaluation,
    extra_diagnostics: dict[str, Any],
) -> bool:
    if int(evaluation.returncode) == 0:
        return False
    if evaluation.scoring_source != "commit0_test_ids":
        return False
    if int(evaluation.failed or 0) or int(evaluation.errors or 0):
        return False
    coverage = dict(evaluation.expected_test_coverage or {})
    if int(coverage.get("missing_expected_test_count") or 0):
        return False
    if not bool(coverage.get("coverage_preserved")):
        return False
    failed = int(extra_diagnostics.get("failed") or 0)
    errors = int(extra_diagnostics.get("errors") or 0)
    if failed + errors <= 0:
        return False
    return bool(extra_diagnostics.get("all_failures_benign"))


# P0 harness fix: a native interpreter crash (segfault / abort / OOM-kill / signal)
# is an ENVIRONMENT failure, never a real test outcome. On POSIX a signal-killed
# subprocess surfaces either as a negative returncode (``-signal``) or, when a shell
# wrapper reports it, as ``128 + signal``. The signals below cover SIGABRT(6→134),
# SIGKILL(9→137), SIGPIPE(13→138) and SIGSEGV(11→139) — the crashes that make pytest
# exit WITHOUT a JSON report. Classifying these as harness_failure (→ indeterminate)
# prevents a crashed interpreter from being scored as a genuine 0 (run-4 false-zero class).
_COMMIT0_NATIVE_CRASH_RETURNCODES = frozenset({134, 137, 138, 139})


def _commit0_returncode_is_native_crash(returncode: int) -> bool:
    try:
        rc = int(returncode)
    except (TypeError, ValueError):
        return False
    # rc<0 == killed by signal -rc (Python subprocess convention);
    # 134-139 == 128 + fatal-signal (shell convention).
    return rc < 0 or rc in _COMMIT0_NATIVE_CRASH_RETURNCODES


def _pytest_output_indicates_collection_failure_before_report(output: str) -> bool:
    normalized = normalize_terminal_output(output).lower()
    if "unrecognized arguments" in normalized and "--json-report" in normalized:
        return False
    if "error importing plugin" in normalized and "/workspace/" in normalized:
        # Pytest imports explicit -p plugins before pytest-json-report can flush;
        # Commit0 project plugin import failures through /workspace are source
        # collection failures, not runner parser failures.
        return True
    return any(
        marker in normalized
        for marker in (
            "importerror while loading conftest",
            "error collecting",
            "errors during collection",
            "collected 0 items /",
            "syntaxerror:",
        )
    )


_PYTHON_IMPORT_FROM_MODULE_RE = re.compile(
    r"\bImportError:\s+cannot import name\b[^\n]*?\bfrom\s+['\"](?P<module>[A-Za-z_][\w.]*)['\"]",
    re.IGNORECASE,
)
_PYTHON_MODULE_NOT_FOUND_RE = re.compile(
    r"\bModuleNotFoundError:\s+No module named\s+['\"](?P<module>[A-Za-z_][\w.]*)['\"]",
    re.IGNORECASE,
)


def _commit0_python_local_module_roots(repo_dir: Path) -> list[str]:
    roots: list[str] = []
    for parent in (repo_dir, repo_dir / "src"):
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            name = child.name
            if not name or name.startswith(".") or name in {"test", "tests", "docs"}:
                continue
            if child.is_dir() and (child / "__init__.py").is_file():
                roots.append(name.lower())
            elif (
                child.is_file()
                and child.suffix == ".py"
                and name
                not in {
                    "setup.py",
                    "conftest.py",
                }
                and child.stem.lower() not in _APEX_HARNESS_HELPER_STEMS
            ):
                roots.append(child.stem.lower())
    return sorted(set(roots))


def _commit0_output_mentions_local_python_import(
    output: str,
    *,
    local_module_roots: list[str],
) -> bool:
    roots = {str(root or "").strip().lower() for root in local_module_roots if root}
    if not roots:
        return False
    text = str(output or "")
    for regex in (_PYTHON_IMPORT_FROM_MODULE_RE, _PYTHON_MODULE_NOT_FOUND_RE):
        for match in regex.finditer(text):
            module = str(match.group("module") or "").strip().lower()
            if any(module == root or module.startswith(f"{root}.") for root in roots):
                return True
    return False


# Native/compiled-extension build failure signature (numpy/scipy/cython/fortran,
# undefined symbols, a failed gcc/gfortran build command). The regex form of
# "error: command '...' failed" also catches absolute toolchain paths
# (``/usr/bin/gcc``) that the old bare "error: command 'gcc' failed" substring
# missed.
_COMMIT0_NATIVE_BUILD_SIGNATURE_RE = re.compile(
    r"undefined symbol"
    r"|cannot open shared object"
    r"|undefined reference to"
    r"|\.so[\"']?:?\s*cannot"
    r"|error:\s*command\s*['\"][^'\"]*['\"]\s*failed"
    r"|\bgfortran\b|\bfortran\b"
    r"|\bcython\b"
    r"|numpy\.core\.(?:multiarray|_multiarray_umath)"
    r"|No module named\s*['\"](?:numpy|scipy|cython|pandas)\b",
    re.IGNORECASE,
)


def _commit0_output_has_native_build_signature(output: str) -> bool:
    """Whether baseline output shows a NATIVE / compiled-extension build failure.

    Such a failure (numpy/scipy/cython/fortran, undefined symbols, a failed
    gcc/gfortran build) is an ENV/build limitation a clean Linux container fixes,
    NOT an APEX source gap — so the baseline must keep its env classification
    (Docker retry) rather than being reclassified APEX_MISS just because the
    failing import happens to name a local module whose compiled extension never
    built. Runs ONLY on the baseline (agent never touched it), so there is no
    agent-manufacturable vector; it only re-routes a genuine env failure from a
    silent zero to an env retry (strictly safer)."""
    return bool(_COMMIT0_NATIVE_BUILD_SIGNATURE_RE.search(str(output or "")))


@dataclass
class Commit0TaskResult:
    """Execution result for one Commit0 benchmark repository."""

    task_name: str
    instance_id: str
    repo: str
    success: bool
    baseline_failed: bool
    final_tests_passed: bool
    baseline: Commit0Evaluation
    final: Commit0Evaluation
    orchestrator_success: bool = False
    candidate_found: bool = False
    orchestrator_selected_rollout_id: Optional[int] = None
    orchestrator_selected_worktree_path: Optional[str] = None
    selected_rollout_id: Optional[int] = None
    selected_worktree_path: Optional[str] = None
    orchestrator_nomination_candidate_id: Optional[str] = None
    orchestrator_nomination_rollout_id: Optional[int] = None
    benchmark_rescored_candidate_id: Optional[str] = None
    official_audit_candidate_id: Optional[str] = None
    final_candidate_id: Optional[str] = None
    final_patch_id: Optional[str] = None
    final_decision_source: Optional[str] = None
    candidate_identity: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    duration_seconds: float = 0.0
    result_path: Optional[str] = None
    failure_reason: Optional[str] = None
    skipped: bool = False
    skip_category: Optional[str] = None
    official_audit: Optional[Commit0Evaluation] = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate_delta(self) -> float:
        return self.final.pass_rate - self.baseline.pass_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "success": self.success,
            "baseline_failed": self.baseline_failed,
            "final_tests_passed": self.final_tests_passed,
            "baseline": self.baseline.to_dict(),
            "final": self.final.to_dict(),
            "baseline_pass_rate": self.baseline.pass_rate,
            "final_pass_rate": self.final.pass_rate,
            "orchestrator_success": self.orchestrator_success,
            "candidate_found": self.candidate_found,
            "orchestrator_selected_rollout_id": self.orchestrator_selected_rollout_id,
            "orchestrator_selected_worktree_path": self.orchestrator_selected_worktree_path,
            "pass_rate_delta": self.pass_rate_delta,
            "selected_rollout_id": self.selected_rollout_id,
            "selected_worktree_path": self.selected_worktree_path,
            "orchestrator_nomination_candidate_id": self.orchestrator_nomination_candidate_id,
            "orchestrator_nomination_rollout_id": self.orchestrator_nomination_rollout_id,
            "benchmark_rescored_candidate_id": self.benchmark_rescored_candidate_id,
            "official_audit_candidate_id": self.official_audit_candidate_id,
            "final_candidate_id": self.final_candidate_id,
            "final_patch_id": self.final_patch_id,
            "final_decision_source": self.final_decision_source,
            "candidate_identity": copy.deepcopy(self.candidate_identity),
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "result_path": self.result_path,
            "failure_reason": self.failure_reason,
            "skipped": self.skipped,
            "skip_category": self.skip_category,
            "official_audit": self.official_audit.to_dict() if self.official_audit else None,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Commit0TaskResult":
        uses_explicit_benchmark_success = any(
            key in payload
            for key in (
                "orchestrator_success",
                "candidate_found",
                "orchestrator_selected_rollout_id",
                "orchestrator_selected_worktree_path",
            )
        )
        final_tests_passed = bool(payload.get("final_tests_passed", False))
        return cls(
            task_name=str(payload["task_name"]),
            instance_id=str(payload["instance_id"]),
            repo=str(payload["repo"]),
            success=(
                bool(payload.get("success", False))
                if uses_explicit_benchmark_success
                else final_tests_passed
            ),
            baseline_failed=bool(payload.get("baseline_failed", False)),
            final_tests_passed=final_tests_passed,
            baseline=Commit0Evaluation.from_dict(dict(payload.get("baseline") or {})),
            final=Commit0Evaluation.from_dict(dict(payload.get("final") or {})),
            orchestrator_success=bool(
                payload.get("orchestrator_success", payload.get("success", False))
            ),
            candidate_found=bool(payload.get("candidate_found", False)),
            orchestrator_selected_rollout_id=payload.get("orchestrator_selected_rollout_id"),
            orchestrator_selected_worktree_path=payload.get("orchestrator_selected_worktree_path"),
            selected_rollout_id=payload.get("selected_rollout_id"),
            selected_worktree_path=payload.get("selected_worktree_path"),
            orchestrator_nomination_candidate_id=payload.get(
                "orchestrator_nomination_candidate_id"
            ),
            orchestrator_nomination_rollout_id=payload.get("orchestrator_nomination_rollout_id"),
            benchmark_rescored_candidate_id=payload.get("benchmark_rescored_candidate_id"),
            official_audit_candidate_id=payload.get("official_audit_candidate_id"),
            final_candidate_id=payload.get("final_candidate_id"),
            final_patch_id=payload.get("final_patch_id"),
            final_decision_source=payload.get("final_decision_source"),
            candidate_identity=dict(payload.get("candidate_identity") or {}),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            result_path=payload.get("result_path"),
            failure_reason=payload.get("failure_reason"),
            skipped=bool(payload.get("skipped", False)),
            skip_category=payload.get("skip_category"),
            official_audit=(
                Commit0Evaluation.from_dict(dict(payload.get("official_audit") or {}))
                if payload.get("official_audit") is not None
                else None
            ),
            execution_metadata=dict(payload.get("execution_metadata") or {}),
        )


def _task_is_nondeterministic_flake(task: "Commit0TaskResult") -> bool:
    """WS2B: True iff the task's final eval was stamped NON_DETERMINISTIC (a
    budget-exhausted gold-oracle teardown flake) — used to carve it out of the
    strict headline denominator."""
    final = getattr(task, "final", None)
    failure_class = getattr(final, "failure_class", None) if final is not None else None
    return str(failure_class or "") == _CoreFailureClass.NON_DETERMINISTIC.value


@dataclass
class Commit0BenchmarkReport:
    """Aggregate Commit0 benchmark report."""

    tasks: list[Commit0TaskResult] = field(default_factory=list)
    requested_task_ids: list[str] = field(default_factory=list)
    requested_repo_names: list[str] = field(default_factory=list)
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float = 0.0
    dataset_name: str = "wentingzhao/commit0_combined"
    dataset_split: str = "test"
    dataset_revision: Optional[str] = None
    dataset_fallback_revisions: list[str] = field(default_factory=list)
    split: str = "lite"
    report_kind: str = COMMIT0_BENCHMARK_REPORT_KIND_APEX
    harness_name: str = COMMIT0_BENCHMARK_HARNESS_NAME
    harness_version: str = COMMIT0_BENCHMARK_HARNESS_VERSION
    config_source: Optional[str] = None
    model_config: list[dict[str, Any]] = field(default_factory=list)
    ablation_config: dict[str, Any] = field(default_factory=dict)
    run_manifest: dict[str, Any] = field(default_factory=dict)
    # WS2B (NDFF): when True, tasks whose final.failure_class is NON_DETERMINISTIC
    # (a budget-exhausted gold-oracle teardown flake) are carved out of the
    # score_strict denominator — a flaky gold test must never charge an APEX miss.
    # Default False keeps bare-constructed reports (tests) byte-identical; the
    # runner sets it from BenchmarkConfig.commit0_ndff_exclude_nondeterministic.
    ndff_exclude_nondeterministic: bool = False

    @property
    def repo_names(self) -> list[str]:
        if self.requested_repo_names:
            return list(self.requested_repo_names)
        return [task.task_name for task in self.tasks]

    @property
    def total_tasks(self) -> int:
        if self.requested_task_ids:
            return len(self.requested_task_ids)
        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        return len(self.tasks)

    @property
    def completed(self) -> bool:
        return self.finished_at > 0.0

    @property
    def duration_seconds(self) -> float:
        if self.started_at <= 0.0:
            return 0.0
        end_time = self.finished_at or self.updated_at or self.started_at
        return max(0.0, end_time - self.started_at)

    @property
    def solved_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.final_tests_passed)

    @property
    def skipped_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.skipped)

    @property
    def runnable_tasks(self) -> int:
        return sum(1 for task in self.tasks if not task.skipped)

    @property
    def solved_runnable_tasks(self) -> int:
        return sum(1 for task in self.tasks if not task.skipped and task.final_tests_passed)

    @property
    def score(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(task.final.pass_rate for task in self.tasks) / len(self.tasks)

    @property
    def score_completed(self) -> float:
        """Average pass rate across completed tasks only.

        This is equivalent to the historical ``score`` field and is
        intentionally not the stopped-run headline.
        """
        return self.score

    @property
    def average_pass_rate(self) -> float:
        return self.score

    @property
    def baseline_score(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(task.baseline.pass_rate for task in self.tasks) / len(self.tasks)

    @property
    def baseline_score_strict(self) -> float:
        denominator = self.total_tasks
        if denominator <= 0:
            return 0.0
        return sum(task.baseline.pass_rate for task in self.tasks) / denominator

    @property
    def average_baseline_pass_rate(self) -> float:
        return self.baseline_score

    @property
    def score_improvement(self) -> float:
        return self.score - self.baseline_score

    @property
    def average_pass_rate_improvement(self) -> float:
        return self.score_improvement

    @property
    def runnable_score(self) -> float:
        # DEPRECATED (Phase 1.6): kept for backwards compatibility with
        # consumers that still read ``runnable_score``. Prefer
        # ``score_runnable`` (semantically identical) or, for the
        # publication-headline number, ``score_strict``.
        runnable = [task for task in self.tasks if not task.skipped]
        if not runnable:
            return 0.0
        return sum(task.final.pass_rate for task in runnable) / len(runnable)

    @property
    def runnable_average_pass_rate(self) -> float:
        return self.runnable_score

    # ------------------------------------------------------------------
    # Phase 1.6: three-number reporting + env-skip ledger.
    # ``score_strict``     — denominator = ALL tasks; skipped → 0.
    #                       This is the publishable headline.
    # ``score_runnable``   — denominator excludes env-skipped tasks.
    #                       Identical to legacy ``runnable_score`` and
    #                       reported for transparency.
    # ``score_attempted``  — denominator includes only tasks where the
    #                       orchestrator actually started (non-skipped
    #                       AND a baseline was produced AND we have a
    #                       non-zero candidate signal). Tightest
    #                       denominator for "what APEX tried to solve".
    # ``env_skip_ledger``  — per-skipped-task audit row with the new
    #                       core failure classification + retry hint.
    # ------------------------------------------------------------------

    @property
    def score_strict(self) -> float:
        """Strict denominator: ALL requested tasks; missing/skipped tasks score 0.

        This is the publishable headline number per the Phase 1
        remediation plan. It intentionally penalises APEX for env-skips
        that the runner couldn't recover from, and it also penalises
        interrupted or partial runs for tasks that never completed.
        """
        # WS2B (NDFF): optionally exclude non-deterministic gold-oracle flakes
        # from BOTH numerator and denominator so a flaky gold test cannot drag the
        # published headline down (it was never an APEX miss).
        scored_tasks = self.tasks
        excluded = 0
        if self.ndff_exclude_nondeterministic:
            scored_tasks = [t for t in self.tasks if not _task_is_nondeterministic_flake(t)]
            excluded = len(self.tasks) - len(scored_tasks)
        denominator = self.total_tasks - excluded
        if denominator <= 0:
            return 0.0
        return sum(task.final.pass_rate for task in scored_tasks) / denominator

    @property
    def score_runnable(self) -> float:
        """Same denominator as the legacy ``runnable_score`` property."""
        return self.runnable_score

    @property
    def score_attempted(self) -> float:
        """Denominator includes only tasks where the orchestrator started.

        A task counts as "attempted" when it is NOT skipped AND its
        baseline evaluation actually executed (returncode != 1 with no
        signal, or the task has a non-empty execution_metadata payload
        recording an orchestrator handoff).
        """
        attempted = [task for task in self.tasks if self._task_was_attempted(task)]
        if not attempted:
            return 0.0
        return sum(task.final.pass_rate for task in attempted) / len(attempted)

    @staticmethod
    def _task_was_attempted(task: "Commit0TaskResult") -> bool:
        if task.skipped:
            return False
        # The orchestrator was reached if there's either an
        # execution_metadata payload OR a real baseline signal (any
        # tests counted, regardless of pass/fail).
        if task.execution_metadata:
            return True
        baseline_total = (
            int(task.baseline.passed) + int(task.baseline.failed) + int(task.baseline.errors)
        )
        return baseline_total > 0

    @property
    def attempted_tasks(self) -> int:
        return sum(1 for task in self.tasks if self._task_was_attempted(task))

    @property
    def env_skip_ledger(self) -> list[dict[str, Any]]:
        """Per-skipped-task audit ledger.

        Each entry surfaces:

        * ``task_id`` — instance_id of the skipped task.
        * ``repo`` — task.repo for human-friendly grouping.
        * ``skip_category`` — the historical category string set by
          ``_classify_prepare_error`` (kept for backwards compat).
        * ``classification`` — the new core ``failure_class`` string
          (or ``"unclassified"``) produced by ``classify_failure``.
        * ``classification_detail`` — the full ClassificationResult
          dict (confidence, matched_pattern, reason).
        * ``reason`` — the failure_reason string the runner already
          recorded.
        * ``retry_recommended`` — True when the classification's
          ``is_environment`` predicate is True (caller policy hint).
        """
        ledger: list[dict[str, Any]] = []
        for task in self.tasks:
            if not task.skipped:
                continue
            classification = task.final.failure_classification or {}
            failure_class = task.final.failure_class or classification.get("failure_class")
            retry_recommended = bool(
                failure_class
                in {
                    _CoreFailureClass.ENV_NETWORK.value,
                    _CoreFailureClass.ENV_INSTALL.value,
                    _CoreFailureClass.ENV_TIMEOUT.value,
                    _CoreFailureClass.ENV_RESOURCE.value,
                }
            )
            ledger.append(
                {
                    "task_id": task.instance_id,
                    "repo": task.repo,
                    "skip_category": task.skip_category,
                    "classification": failure_class or "unclassified",
                    "classification_detail": classification or None,
                    "reason": task.failure_reason,
                    "retry_recommended": retry_recommended,
                }
            )
        return ledger

    @property
    def score_strict_percent(self) -> float:
        return 100.0 * self.score_strict

    @property
    def baseline_score_strict_percent(self) -> float:
        return 100.0 * self.baseline_score_strict

    @property
    def score_strict_improvement(self) -> float:
        return self.score_strict - self.baseline_score_strict

    @property
    def score_strict_improvement_percent(self) -> float:
        return 100.0 * self.score_strict_improvement

    @property
    def score_runnable_percent(self) -> float:
        return 100.0 * self.score_runnable

    @property
    def score_attempted_percent(self) -> float:
        return 100.0 * self.score_attempted

    @property
    def runnable_baseline_score(self) -> float:
        runnable = [task for task in self.tasks if not task.skipped]
        if not runnable:
            return 0.0
        return sum(task.baseline.pass_rate for task in runnable) / len(runnable)

    @property
    def runnable_average_baseline_pass_rate(self) -> float:
        return self.runnable_baseline_score

    @property
    def runnable_score_improvement(self) -> float:
        return self.runnable_score - self.runnable_baseline_score

    @property
    def runnable_average_pass_rate_improvement(self) -> float:
        return self.runnable_score_improvement

    @property
    def solved_rate(self) -> float:
        return _safe_ratio(self.solved_tasks, self.total_tasks)

    @property
    def runnable_solved_rate(self) -> float:
        return _safe_ratio(self.solved_runnable_tasks, self.runnable_tasks)

    @property
    def score_percent(self) -> float:
        return 100.0 * self.score

    @property
    def score_completed_percent(self) -> float:
        return 100.0 * self.score_completed

    @property
    def average_pass_rate_percent(self) -> float:
        return self.score_percent

    @property
    def baseline_score_percent(self) -> float:
        return 100.0 * self.baseline_score

    @property
    def average_baseline_pass_rate_percent(self) -> float:
        return self.baseline_score_percent

    @property
    def score_improvement_percent(self) -> float:
        return 100.0 * self.score_improvement

    @property
    def average_pass_rate_improvement_percent(self) -> float:
        return self.score_improvement_percent

    @property
    def runnable_score_percent(self) -> float:
        return 100.0 * self.runnable_score

    @property
    def runnable_average_pass_rate_percent(self) -> float:
        return self.runnable_score_percent

    @property
    def runnable_baseline_score_percent(self) -> float:
        return 100.0 * self.runnable_baseline_score

    @property
    def runnable_average_baseline_pass_rate_percent(self) -> float:
        return self.runnable_baseline_score_percent

    @property
    def runnable_score_improvement_percent(self) -> float:
        return 100.0 * self.runnable_score_improvement

    @property
    def runnable_average_pass_rate_improvement_percent(self) -> float:
        return self.runnable_score_improvement_percent

    @property
    def solved_rate_percent(self) -> float:
        return 100.0 * self.solved_rate

    @property
    def runnable_solved_rate_percent(self) -> float:
        return 100.0 * self.runnable_solved_rate

    @property
    def scoring_method(self) -> str:
        sources = set()
        for task in self.tasks:
            if task.baseline.scoring_source:
                sources.add(task.baseline.scoring_source)
            if task.final.scoring_source:
                sources.add(task.final.scoring_source)
        if not sources:
            return "unknown"
        if len(sources) == 1:
            return next(iter(sources))
        return "mixed"

    @property
    def scoring_source(self) -> str:
        return self.scoring_method

    @property
    def evaluation_backend(self) -> str:
        backends = set()
        for task in self.tasks:
            if task.baseline.evaluation_backend:
                backends.add(task.baseline.evaluation_backend)
            if task.final.evaluation_backend:
                backends.add(task.final.evaluation_backend)
        if not backends:
            return "unknown"
        if len(backends) == 1:
            return next(iter(backends))
        return "mixed"

    @property
    def failure_clusters(self) -> list[dict[str, Any]]:
        return cluster_failures(
            [task.to_dict() for task in self.tasks],
            benchmark_family="commit0",
        )

    @property
    def standalone_anchor_scorecard(self) -> dict[str, Any]:
        by_label: dict[str, dict[str, Any]] = {}
        entries: list[dict[str, Any]] = []
        for task in self.tasks:
            metadata = dict(task.execution_metadata or {})
            for raw_entry in list(metadata.get("standalone_anchor_results") or []):
                if not isinstance(raw_entry, dict):
                    continue
                entry = dict(raw_entry)
                label = str(
                    entry.get("standalone_anchor_label")
                    or entry.get("label")
                    or "standalone_anchor"
                )
                pass_rate = float(entry.get("pass_rate") or 0.0)
                scored_success = bool(entry.get("scored_success"))
                entries.append(entry)
                aggregate = by_label.setdefault(
                    label,
                    {
                        "label": label,
                        "backend": entry.get("standalone_anchor_backend"),
                        "model": entry.get("standalone_anchor_model"),
                        "candidate_count": 0,
                        "solved_count": 0,
                        "pass_rate_sum": 0.0,
                    },
                )
                aggregate["candidate_count"] += 1
                aggregate["solved_count"] += 1 if scored_success else 0
                aggregate["pass_rate_sum"] += pass_rate
        rows = []
        for aggregate in by_label.values():
            count = int(aggregate.pop("candidate_count"))
            pass_rate_sum = float(aggregate.pop("pass_rate_sum"))
            rows.append(
                {
                    **aggregate,
                    "candidate_count": count,
                    "solved_count": int(aggregate.get("solved_count") or 0),
                    "average_pass_rate": pass_rate_sum / count if count else 0.0,
                }
            )
        rows.sort(key=lambda item: str(item.get("label") or ""))
        return {
            "candidate_count": len(entries),
            "by_label": rows,
        }

    @property
    def diagnostic_score_only_summary(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for task in self.tasks:
            metadata = dict(task.execution_metadata or {})
            for raw_entry in list(metadata.get("diagnostic_score_only_candidates") or []):
                if not isinstance(raw_entry, dict):
                    continue
                evaluation = (
                    dict(raw_entry.get("evaluation") or {})
                    if isinstance(raw_entry.get("evaluation"), dict)
                    else {}
                )
                entry = {
                    "task_name": task.task_name,
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "final_decision_source": task.final_decision_source,
                    "rollout_id": raw_entry.get("rollout_id"),
                    "pass_rate": float(evaluation.get("pass_rate") or 0.0),
                    "scored_success": bool(evaluation.get("scored_success")),
                    "evaluation_status": evaluation.get("evaluation_status"),
                }
                entries.append(entry)
        return {
            "candidate_count": len(entries),
            "solved_count": sum(1 for entry in entries if entry.get("scored_success")),
            "tasks": entries,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_kind": self.report_kind,
            "harness_name": self.harness_name,
            "harness_version": self.harness_version,
            "config_source": self.config_source,
            "requested_task_ids": list(self.requested_task_ids),
            "repo_names": self.repo_names,
            "model_config": copy.deepcopy(self.model_config),
            "ablation_config": copy.deepcopy(self.ablation_config),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "dataset_revision": self.dataset_revision,
            "dataset_fallback_revisions": list(self.dataset_fallback_revisions),
            "split": self.split,
            "completed_tasks": self.completed_tasks,
            "solved_tasks": self.solved_tasks,
            "total_tasks": self.total_tasks,
            "skipped_tasks": self.skipped_tasks,
            "runnable_tasks": self.runnable_tasks,
            "solved_runnable_tasks": self.solved_runnable_tasks,
            "solved_rate": self.solved_rate,
            "solved_rate_percent": self.solved_rate_percent,
            "runnable_solved_rate": self.runnable_solved_rate,
            "runnable_solved_rate_percent": self.runnable_solved_rate_percent,
            "score": self.score,
            "score_percent": self.score_percent,
            "score_completed": self.score_completed,
            "score_completed_percent": self.score_completed_percent,
            "average_pass_rate": self.average_pass_rate,
            "average_pass_rate_percent": self.average_pass_rate_percent,
            "baseline_score": self.baseline_score,
            "baseline_score_percent": self.baseline_score_percent,
            "average_baseline_pass_rate": self.average_baseline_pass_rate,
            "average_baseline_pass_rate_percent": self.average_baseline_pass_rate_percent,
            "score_improvement": self.score_improvement,
            "score_improvement_percent": self.score_improvement_percent,
            "average_pass_rate_improvement": self.average_pass_rate_improvement,
            "average_pass_rate_improvement_percent": self.average_pass_rate_improvement_percent,
            "runnable_score": self.runnable_score,
            "runnable_score_percent": self.runnable_score_percent,
            "runnable_average_pass_rate": self.runnable_average_pass_rate,
            "runnable_average_pass_rate_percent": self.runnable_average_pass_rate_percent,
            "runnable_baseline_score": self.runnable_baseline_score,
            "runnable_baseline_score_percent": self.runnable_baseline_score_percent,
            "runnable_average_baseline_pass_rate": self.runnable_average_baseline_pass_rate,
            "runnable_average_baseline_pass_rate_percent": self.runnable_average_baseline_pass_rate_percent,
            "runnable_score_improvement": self.runnable_score_improvement,
            "runnable_score_improvement_percent": self.runnable_score_improvement_percent,
            "runnable_average_pass_rate_improvement": self.runnable_average_pass_rate_improvement,
            "runnable_average_pass_rate_improvement_percent": self.runnable_average_pass_rate_improvement_percent,
            "metric_semantics": {
                "score_percent": (
                    "Legacy completed-only average per-repository final pass rate. "
                    "Use score_strict_percent for stopped or full-run headline reporting."
                ),
                "score_completed_percent": (
                    "Average per-repository final pass rate across completed repos only."
                ),
                "solved_rate_percent": "Percent of repos with final_tests_passed == true.",
                "runnable_score_percent": "Average per-repository final pass rate across non-skipped repos.",
                "runnable_solved_rate_percent": "Percent of non-skipped repos with final_tests_passed == true.",
            },
            "task_field_semantics": {
                "success": "Whether final benchmark evaluation was fully green. This intentionally matches final_tests_passed.",
                "orchestrator_success": "Whether Apex internally accepted a rollout before benchmark-level rescoring.",
                "candidate_found": "Whether the benchmark runner found a rollout candidate to score from rollout worktrees.",
                "selected_rollout_id": "Rollout selected for final benchmark scoring.",
                "orchestrator_selected_rollout_id": "Rollout Apex internally selected before benchmark rescoring.",
                "orchestrator_nomination_candidate_id": "Stable candidate id for Apex's internal nomination.",
                "benchmark_rescored_candidate_id": "Stable candidate id for the candidate chosen by benchmark-level rescoring.",
                "official_audit_candidate_id": "Stable candidate id audited by the official scorer, when audit ran.",
                "final_candidate_id": "Stable candidate id used for the task's final decision.",
                "final_patch_id": "SHA-256 hash of the normalized selected patch when available.",
                "final_decision_source": "Decision source: orchestrator_nomination, benchmark_rescore, official_audit, or no_candidate.",
            },
            "scoring_source": self.scoring_source,
            "scoring_method": self.scoring_method,
            "evaluation_backend": self.evaluation_backend,
            "run_manifest": manifest_summary(self.run_manifest),
            "failure_clusters": self.failure_clusters,
            "standalone_anchor_scorecard": self.standalone_anchor_scorecard,
            "diagnostic_score_only": self.diagnostic_score_only_summary,
            "tasks": [task.to_dict() for task in self.tasks],
            # Phase 1.6: three-number reporting + env-skip ledger.
            # ``score_strict`` is the publishable headline; the other
            # two are reported for transparency. ``runnable_score``
            # remains in the payload above for backwards compat but is
            # marked deprecated in the markdown report.
            "score_strict": self.score_strict,
            "score_strict_percent": self.score_strict_percent,
            "headline_score": self.score_strict,
            "headline_score_percent": self.score_strict_percent,
            "headline_score_semantics": (
                "Strict all-requested denominator; incomplete and skipped tasks score 0."
            ),
            "baseline_score_strict": self.baseline_score_strict,
            "baseline_score_strict_percent": self.baseline_score_strict_percent,
            "score_strict_improvement": self.score_strict_improvement,
            "score_strict_improvement_percent": self.score_strict_improvement_percent,
            "score_runnable": self.score_runnable,
            "score_runnable_percent": self.score_runnable_percent,
            "score_attempted": self.score_attempted,
            "score_attempted_percent": self.score_attempted_percent,
            "attempted_tasks": self.attempted_tasks,
            "env_skip_ledger": self.env_skip_ledger,
            "score_semantics": {
                "score_strict": (
                    "Headline. Denominator = ALL requested tasks; incomplete and "
                    "env-skipped tasks score 0."
                ),
                "baseline_score_strict": (
                    "Baseline scored with the same all-requested denominator as score_strict."
                ),
                "score_completed": (
                    "Completed-only average. Equivalent to legacy score/average_pass_rate."
                ),
                "score_runnable": (
                    "Denominator excludes env-skipped tasks. Equivalent to legacy "
                    "runnable_score (deprecated)."
                ),
                "score_attempted": (
                    "Denominator includes only tasks where the orchestrator actually started."
                ),
            },
        }

    def to_markdown(self) -> str:
        rollout_buckets = (self.ablation_config.get("allocator") or {}).get("rollout_buckets") or []
        overlap_diversity_cap_enabled = (self.ablation_config.get("allocator") or {}).get(
            "overlap_diversity_cap_enabled",
            False,
        )
        min_overlap_diversity_parallel_workers = (self.ablation_config.get("allocator") or {}).get(
            "min_overlap_diversity_parallel_workers",
            "n/a",
        )
        planner_brief_family_cap = (self.ablation_config.get("scaffold") or {}).get(
            "planner_brief_family_cap",
            "n/a",
        )
        orchestrated_multi_agent_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "orchestrated_multi_agent_enabled",
            False,
        )
        task_state_graph_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "task_state_graph_enabled",
            False,
        )
        frontier_targeting_enabled = (self.ablation_config.get("scaffold") or {}).get(
            "frontier_targeting_enabled",
            False,
        )
        dynamic_transitions = (self.ablation_config.get("scaffold") or {}).get(
            "dynamic_transitions_enabled",
            False,
        )
        feedback_config = self.ablation_config.get("feedback") or {}
        search_mode = (self.ablation_config.get("search") or {}).get("mode", "off")
        selection_config = self.ablation_config.get("selection") or {}
        memory_config = self.ablation_config.get("memory") or {}
        status = "completed" if self.completed else "in_progress"
        lines = [
            "# APEX Commit0 Benchmark Report",
            "",
            f"- Harness: {self.harness_name} v{self.harness_version}",
            f"- Report kind: {self.report_kind}",
            f"- Status: {status}",
            f"- Config source: {self.config_source or 'default'}",
            f"- Model config: {_format_model_config_summary(self.model_config)}",
            f"- Rollout allocator: {(self.ablation_config.get('allocator') or {}).get('policy', 'unknown')}",
            f"- Rollout buckets: {', '.join(str(bucket) for bucket in rollout_buckets) or 'n/a'}",
            (
                "- Overlap diversity cap: "
                f"{'enabled' if overlap_diversity_cap_enabled else 'disabled'} "
                f"(outer_floor={min_overlap_diversity_parallel_workers})"
            ),
            f"- Scaffold mode: {(self.ablation_config.get('scaffold') or {}).get('policy', 'unknown')}",
            (
                "- Orchestrated multi-agent delegation: "
                f"{'enabled' if orchestrated_multi_agent_enabled else 'disabled'}"
            ),
            f"- Planner brief family cap: {planner_brief_family_cap}",
            f"- Task-state graph: {'enabled' if task_state_graph_enabled else 'disabled'}",
            f"- Frontier targeting: {'enabled' if frontier_targeting_enabled else 'disabled'}",
            f"- Explicit search: {search_mode}",
            f"- COP transitions: {'enabled' if dynamic_transitions else 'disabled'}",
            (
                "- Rollout quick verification: "
                f"{'enabled' if feedback_config.get('quick_verification_enabled') else 'disabled'} "
                f"(max_tests={feedback_config.get('quick_verification_max_tests', 'n/a')}, "
                f"timeout={_format_optional_timeout_seconds(feedback_config.get('quick_verification_timeout_seconds'))})"
            ),
            (
                "- Selection critic: "
                f"{'enabled' if selection_config.get('critic_reranking_enabled') else 'disabled'} "
                f"(weight={selection_config.get('critic_weight', 0)})"
            ),
            (
                "- Repo memory: "
                + (
                    "enabled (non-i.i.d.; disclose when comparing to fresh-run baselines)"
                    if memory_config.get("repo_memory_enabled")
                    else "disabled"
                )
                + (
                    " — forced off via APEX_DISABLE_REPO_MEMORY"
                    if _is_repo_memory_disabled_via_env_safe()
                    else ""
                )
            ),
            (
                "- Per-repo overrides: "
                f"{len(_COMMIT0_TASK_OVERRIDES)} "
                "(see configs/commit0_task_overrides.json for justifications)"
            ),
            f"- Repos: {', '.join(self.repo_names) or 'none'}",
            f"- Completed repos: {self.completed_tasks}/{self.total_tasks}",
            (
                "- Subset: "
                f"{self.completed_tasks} of {self.total_tasks} tasks completed "
                f"on split={self.split}"
            ),
            f"- Split: {self.split}",
            (
                # Phase 1.1 + 1.6: the headline is ``score_strict`` (all
                # tasks, env-skipped → 0). When the official audit ran
                # successfully the underlying per-task numbers are the
                # audit numbers — see the "Headline source" note below.
                f"- Headline pass rate (strict, all tasks): "
                f"{self.score_strict_percent:.1f}% "
                f"(baseline {self.baseline_score_strict_percent:.1f}%, "
                f"delta {self.score_strict_improvement_percent:+.1f}%)"
            ),
            (
                f"- Headline solve rate (strict, all tasks): "
                f"{_format_count_rate(self.solved_tasks, self.total_tasks)}"
            ),
            (
                f"- Completed-only pass rate: "
                f"{self.score_completed_percent:.1f}% "
                f"({self.completed_tasks}/{self.total_tasks} completed; "
                "legacy `score_percent`)"
            ),
            (
                f"- Runnable-only pass rate (denominator excludes "
                f"{self.skipped_tasks} env-skipped): "
                f"{self.score_runnable_percent:.1f}% "
                f"(baseline {self.runnable_average_baseline_pass_rate_percent:.1f}%, "
                f"delta {self.runnable_score_improvement_percent:+.1f}%) "
                "[deprecated; reported for transparency]"
            ),
            (
                f"- Attempted-only pass rate (denominator = orchestrator "
                f"reached): {self.score_attempted_percent:.1f}% "
                f"({self.attempted_tasks}/{self.total_tasks} attempted)"
            ),
            (
                f"- Runnable-only solve rate: "
                f"{_format_count_rate(self.solved_runnable_tasks, self.runnable_tasks)}"
            ),
            (
                "- Headline source: per-task `score_source` field (one of "
                "`upstream_audit` / `apex_private_pytest_json` / `shell_rc`). "
                "When `commit0_official_audit_selected` is True (default), "
                "successful audit runs publish the upstream docker-harness number "
                "as the headline; the APEX-private local pytest number is preserved "
                "in `final.diagnostics` for audit only."
            ),
            "- Metric note: `score_strict` is the headline. `score_runnable` is deprecated (kept for back-compat) and `score_attempted` is reported for transparency.",
            f"- Scoring source: {self.scoring_source}",
            f"- Evaluation backend: {self.evaluation_backend}",
            f"- Duration: {self.duration_seconds:.1f}s",
            "",
        ]
        # Runnable headline rolled into the canonical block above to avoid
        # ambiguity about which number is the published metric.
        lines.extend(
            [
                "| Repo | Baseline | Final | Delta | Solved | Status | Tokens | Duration (s) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for task in self.tasks:
            status = "skipped"
            if not task.skipped:
                status = "scored"
            if task.skip_category:
                status = f"{status} ({task.skip_category})"
            lines.append(
                "| {name} | {baseline:.1f}% | {final:.1f}% | {delta:+.1f}% | {solved} | {status} | {tokens} | {duration:.1f} |".format(
                    name=task.task_name,
                    baseline=100.0 * task.baseline.pass_rate,
                    final=100.0 * task.final.pass_rate,
                    delta=100.0 * task.pass_rate_delta,
                    solved="yes" if task.final_tests_passed else "no",
                    status=status,
                    tokens=task.total_tokens,
                    duration=task.duration_seconds,
                )
            )
        failure_clusters = self.failure_clusters
        if failure_clusters:
            lines.extend(
                [
                    "",
                    "## Failure Clusters",
                    "",
                    "| Root Cause | Count | Example Repos |",
                    "| --- | --- | --- |",
                ]
            )
            for cluster in failure_clusters:
                lines.append(
                    "| {bucket} | {count} | {tasks} |".format(
                        bucket=cluster.get("bucket"),
                        count=cluster.get("count"),
                        tasks=", ".join(cluster.get("tasks") or []) or "-",
                    )
                )
        # Phase 1.6: env-skip ledger. Surfaced in markdown so reviewers
        # can audit which tasks were excluded from the runnable
        # denominator and whether the orchestrator should retry them
        # with a fresh container.
        ledger = self.env_skip_ledger
        if ledger:
            lines.extend(
                [
                    "",
                    "## Env-skip Ledger",
                    "",
                    "Per-skipped-task audit. `classification` uses the new core "
                    "`apex.core.failure_classifier.FailureClass` taxonomy. "
                    "`retry_recommended` is True for env_* classes (network/install/"
                    "timeout/resource) — the orchestrator can safely retry these on "
                    "a fresh container.",
                    "",
                    "| Task ID | Repo | Skip Category | Classification | Retry? | Reason |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for entry in ledger:
                reason = (entry.get("reason") or "").splitlines()[0] if entry.get("reason") else ""
                if len(reason) > 120:
                    reason = reason[:117] + "..."
                lines.append(
                    "| {tid} | {repo} | {cat} | {cls} | {retry} | {reason} |".format(
                        tid=entry.get("task_id"),
                        repo=entry.get("repo"),
                        cat=entry.get("skip_category") or "-",
                        cls=entry.get("classification") or "unclassified",
                        retry="yes" if entry.get("retry_recommended") else "no",
                        reason=reason or "-",
                    )
                )
        diagnostic_score_only = self.diagnostic_score_only_summary
        if diagnostic_score_only["candidate_count"]:
            lines.extend(
                [
                    "",
                    "## Diagnostic-Only Candidate Scores",
                    "",
                    "Invalid-for-submission candidates scored for diagnostics only. "
                    "These rows never contribute to headline pass or solve rates.",
                    "",
                    "| Repo | Rollout | Pass Rate | Solved | Final Decision Source |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for entry in diagnostic_score_only["tasks"]:
                lines.append(
                    "| {repo} | {rollout} | {pass_rate:.1f}% | {solved} | {source} |".format(
                        repo=entry.get("task_name") or entry.get("instance_id"),
                        rollout=entry.get("rollout_id"),
                        pass_rate=100.0 * float(entry.get("pass_rate") or 0.0),
                        solved="yes" if entry.get("scored_success") else "no",
                        source=entry.get("final_decision_source") or "-",
                    )
                )
        return "\n".join(lines)


@dataclass
class _CandidateFinalResult:
    rollout_id: int
    worktree_path: Path
    evaluation: Commit0Evaluation
    changed_files: list[str] = field(default_factory=list)
    stub_findings: list[Any] = field(default_factory=list)
    quality_gate: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TaskExecutionLayout:
    sandbox_root: Path
    repo_dir: Path
    runtime_dir: Path
    task_output_dir: Path
    workspace_dir: Path


@dataclass(frozen=True)
class _EnvironmentSmokeTest:
    name: str
    probe: str
    remediation_package: Optional[str] = None


def _copy_tree_error_is_only_missing_files(exc: shutil.Error) -> bool:
    errors = exc.args[0] if exc.args else []
    if not isinstance(errors, list) or not errors:
        return False
    for entry in errors:
        message = ""
        if isinstance(entry, tuple) and len(entry) >= 3:
            message = str(entry[2] or "")
        else:
            message = str(entry or "")
        if "[Errno 2]" not in message and "No such file or directory" not in message:
            return False
    return True


def _copy_file_atomic(source_path: Path, destination_path: Path) -> None:
    temp_path = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        shutil.copy2(source_path, temp_path)
        os.replace(temp_path, destination_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _is_commit0_atomic_write_temp_artifact(relative_path: Path) -> bool:
    name = relative_path.name
    return name.startswith(".") and name.endswith(".tmp")


_COMMIT0_PERSISTED_WORKSPACE_SKIP_NAMES = frozenset(
    {
        ".apex_agent_runtime",
        ".locks",
        "_pool",
    }
)


def _commit0_persisted_workspace_ignore(directory: str, names: list[str]) -> list[str]:
    # Commit0 workspace evidence only needs materialized rollout worktrees; pool/runtime
    # internals can be large transient CLI state and are deleted before evaluation anyway.
    return [name for name in names if name in _COMMIT0_PERSISTED_WORKSPACE_SKIP_NAMES]


class Commit0BenchmarkRunner:
    """Run APEX against Commit0 or Commit0-Lite repositories."""

    def __init__(
        self,
        config: ApexConfig,
        output_dir: str,
        dataset_name: str = COMMIT0_DEFAULT_DATASET_NAME,
        dataset_split: str = "test",
        dataset_revision: Optional[str] = None,
        dataset_fallback_revisions: Optional[list[str]] = None,
        split: str = "lite",
    ):
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dataset_revision = dataset_revision
        self.dataset_fallback_revisions = list(
            dataset_fallback_revisions
            if dataset_fallback_revisions
            else (
                COMMIT0_DEFAULT_DATASET_FALLBACK_REVISIONS
                if dataset_name == COMMIT0_DEFAULT_DATASET_NAME
                else []
            )
        )
        self.split = split
        self.config_source: Optional[str] = None
        # Phase 1: per-run reproducibility manifest. Populated lazily in
        # ``run()`` (and on first docker resolution) so tests that
        # construct a runner without invoking ``run()`` don't pay the
        # cost of capturing platform/env state.
        self.run_manifest: Optional[RunManifest] = None
        # Phase 1: side-by-side fairness audit aggregator. Initialised in
        # ``run()`` only when fairness_audit_mode != OFF.
        self.fairness_audit_aggregator: Optional[FairnessAuditAggregator] = None
        self._live_requested_task_ids: list[str] = []
        self._live_task_output_rel_by_id: dict[str, str] = {}
        self._commit0_docker_proxy_relays: dict[
            tuple[str, int],
            _Commit0DockerProxyRelay,
        ] = {}
        self._commit0_colima_proxy_tunnels: dict[
            tuple[str, int],
            _Commit0ColimaProxyTunnel,
        ] = {}
        self._commit0_docker_proxy_relay_lock = threading.Lock()
        self._commit0_agent_cli_bundle_lock = threading.Lock()
        self._target_container_backend_preflight_lock = threading.Lock()
        self._target_container_backend_preflight_healthy: dict[
            tuple[str, str, str, str], dict[str, Any]
        ] = {}
        self._commit0_pytest_xdist_disabled_repos: set[str] = set()
        self._task_solve_semaphore: Optional[threading.BoundedSemaphore] = None
        self._task_audit_semaphore: Optional[threading.BoundedSemaphore] = None
        self._task_lane_state = threading.local()
        self._task_solve_slot_lock = threading.Lock()
        self._active_solve_task_count = 0
        self._waiting_solve_task_count = 0

    def _benchmark_timeout_seconds(self, field_name: str, default: int) -> int:
        value = getattr(self.config.benchmark, field_name, default)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(default))

    def _commit0_repo_clone_timeout_seconds(self) -> int:
        return self._benchmark_timeout_seconds(
            "commit0_repo_clone_timeout_seconds",
            1800,
        )

    def _commit0_runtime_setup_timeout_seconds(self) -> int:
        return self._benchmark_timeout_seconds(
            "commit0_runtime_setup_timeout_seconds",
            1800,
        )

    def _commit0_dependency_install_timeout_seconds(self) -> int:
        return self._benchmark_timeout_seconds(
            "commit0_dependency_install_timeout_seconds",
            3600,
        )

    def _commit0_evaluation_timeout_seconds(
        self,
        task: Optional["Commit0Task"] = None,
    ) -> int:
        global_timeout = self._benchmark_timeout_seconds(
            "commit0_evaluation_timeout_seconds",
            COMMIT0_OFFICIAL_EVALUATION_TIMEOUT_SECONDS,
        )
        if task is None:
            return global_timeout
        override = _COMMIT0_TASK_OVERRIDES.get(task.repo_name, {}).get("evaluation_timeout_seconds")
        if isinstance(override, int) and override > 0:
            return max(override, global_timeout)
        return global_timeout

    def _commit0_baseline_evaluation_timeout_seconds(
        self,
        task: Optional["Commit0Task"] = None,
    ) -> int:
        global_timeout = self._benchmark_timeout_seconds(
            "commit0_baseline_evaluation_timeout_seconds",
            1800,
        )
        if task is None:
            return global_timeout
        override = _COMMIT0_TASK_OVERRIDES.get(task.repo_name, {}).get(
            "baseline_evaluation_timeout_seconds"
        )
        if isinstance(override, int) and override > 0:
            # Mirror _commit0_evaluation_timeout_seconds: a per-repo override may
            # only RAISE the ceiling, never shrink it below the global default
            # (a too-low baseline timeout would spuriously inflate the delta).
            return max(override, global_timeout)
        return global_timeout

    def _commit0_agent_target_tool_timeout_seconds(
        self,
        task: Optional["Commit0Task"] = None,
        *,
        expected_test_count: Optional[int] = None,
    ) -> int:
        configured_timeout = self._benchmark_timeout_seconds(
            "commit0_agent_target_tool_timeout_seconds",
            300,
        )
        try:
            count = max(0, int(expected_test_count or 0))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            return configured_timeout
        scaled_timeout = configured_timeout
        if count >= 30_000:
            scaled_timeout = max(scaled_timeout, 3600)
        elif count >= 10_000:
            scaled_timeout = max(scaled_timeout, 2400)
        elif count >= 5_000:
            scaled_timeout = max(scaled_timeout, 1200)
        elif count >= 1_000:
            scaled_timeout = max(scaled_timeout, 600)
        # Commit0/Python fact: large expected pytest universes need longer
        # exploratory pytest runs, but never longer than official scoring.
        return min(scaled_timeout, self._commit0_evaluation_timeout_seconds(task))

    def discover_tasks(
        self,
        repos: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[Commit0Task]:
        from datasets import load_dataset

        allowed_repos = self._resolve_repo_filter(repos)
        requested_repo_order: list[str] = []
        if allowed_repos:
            if repos:
                requested_repo_order = [
                    repo.split("/")[-1] for repo in repos if repo.split("/")[-1]
                ]
            elif self.split == "lite":
                requested_repo_order = list(COMMIT0_LITE_REPOS)
            else:
                requested_repo_order = [self.split]
        tasks: list[Commit0Task] = []
        seen_repos: set[str] = set()

        def _dataset_rows(revision: Optional[str]) -> Iterator[dict[str, Any]]:
            dataset_kwargs: dict[str, Any] = {"split": self.dataset_split}
            if revision:
                # Commit0 datasets are mutable on HuggingFace; revision pinning keeps
                # the evaluated repo universe reproducible.
                dataset_kwargs["revision"] = revision
            return iter(load_dataset(self.dataset_name, **dataset_kwargs))

        def _append_matching_rows(
            dataset: Iterator[dict[str, Any]],
            *,
            missing_repos: Optional[set[str]] = None,
        ) -> None:
            for row in dataset:
                repo_name = row["repo"].split("/")[-1]
                if allowed_repos and repo_name not in allowed_repos:
                    continue
                if missing_repos is not None and repo_name not in missing_repos:
                    continue
                if repo_name in seen_repos:
                    continue

                setup = row.get("setup") or {}
                test = row.get("test") or {}
                # Older Commit0 dataset revisions predate the explicit instance_id field;
                # the repo slug is the benchmark task identifier in those rows.
                instance_id = str(row.get("instance_id") or row["repo"])
                task = Commit0Task(
                    instance_id=instance_id,
                    repo=row["repo"],
                    original_repo=row.get("original_repo", ""),
                    base_commit=row["base_commit"],
                    reference_commit=row.get("reference_commit", ""),
                    python_version=str(setup.get("python", "3.12")),
                    specification=setup.get("specification", ""),
                    install_command=setup.get("install", "pip install -e ."),
                    packages=list(setup.get("packages") or []),
                    pip_packages=list(setup.get("pip_packages") or []),
                    pre_install=list(setup.get("pre_install") or []),
                    src_dir=row.get("src_dir", ""),
                    test_cmd=test.get("test_cmd", "pytest"),
                    test_dir=test.get("test_dir", "tests/"),
                )
                _apply_commit0_task_overrides(task)
                tasks.append(task)
                seen_repos.add(repo_name)
                if limit is not None and len(tasks) >= limit:
                    break

        _append_matching_rows(_dataset_rows(self.dataset_revision))
        # ADDITIVE: synthesize perturbed-variant tasks from the sidecar for any
        # requested ``<repo>_perturbed`` not present in the dataset (it never is).
        # When no perturbed repo is requested / no sidecar exists, this is inert.
        if allowed_repos:
            for repo_name in sorted(set(allowed_repos) - seen_repos):
                synthetic = _build_synthetic_perturbed_task(repo_name)
                if synthetic is None:
                    continue
                tasks.append(synthetic)
                seen_repos.add(repo_name)
                if limit is not None and len(tasks) >= limit:
                    break
        if allowed_repos and (limit is None or len(tasks) < limit):
            missing_repos = set(allowed_repos) - seen_repos
            for fallback_revision in self.dataset_fallback_revisions:
                if not missing_repos or (limit is not None and len(tasks) >= limit):
                    break
                # Commit0 dataset revisions can have different row universes; fallback
                # revisions recover explicitly requested repos missing from the primary.
                _append_matching_rows(
                    _dataset_rows(fallback_revision),
                    missing_repos=missing_repos,
                )
                missing_repos = set(allowed_repos) - seen_repos
        if allowed_repos:
            missing = [repo for repo in requested_repo_order if repo not in seen_repos]
            if missing:
                fallback_text = (
                    ", ".join(self.dataset_fallback_revisions)
                    if self.dataset_fallback_revisions
                    else "none"
                )
                raise ValueError(
                    "Requested Commit0 repo(s) not found in dataset "
                    f"{self.dataset_name!r} split {self.dataset_split!r}: "
                    f"{', '.join(missing)}. Fallback revisions tried: {fallback_text}."
                )
        return tasks

    def run(
        self,
        repos: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> Commit0BenchmarkReport:
        ensure_cli_process_cleanup_hooks()
        clear_cli_cleanup_signal_requested()
        # Phase 1: capture the per-run reproducibility manifest BEFORE
        # discovering tasks so the manifest's apex_config snapshot
        # reflects the exact config object the run will use. Idempotent
        # under retries because ``run()`` is single-shot per runner.
        self._init_run_manifest()
        # Phase 1: initialise the fairness audit aggregator iff configured.
        # The mode lives on BenchmarkConfig as a string so the dataclass
        # stays import-cycle-free; coerce to the enum here.
        self._init_fairness_audit_aggregator()
        tasks = self.discover_tasks(repos=repos, limit=limit)
        execution = {
            "entrypoint": "commit0-benchmark",
            "args": {
                "split": self.split,
                "repos": list(repos or []),
                "limit": limit,
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
                "dataset_revision": self.dataset_revision,
                "dataset_fallback_revisions": list(self.dataset_fallback_revisions),
                "task_parallelism": self.config.benchmark.task_parallelism,
                "commit0_official_audit_parallelism": (
                    self.config.benchmark.commit0_official_audit_parallelism
                ),
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_commit0_disk_headroom(tasks)
        requested_task_ids = [task.instance_id for task in tasks]
        self._live_requested_task_ids = list(requested_task_ids)
        self._live_task_output_rel_by_id = {task.instance_id: task.repo_name for task in tasks}
        existing_state = load_json_if_exists(self.output_dir / RUN_STATE_FILENAME) or {}
        report = Commit0BenchmarkReport(
            requested_task_ids=requested_task_ids,
            requested_repo_names=[task.repo_name for task in tasks],
            started_at=float(existing_state.get("started_at") or time.time()),
            dataset_name=self.dataset_name,
            dataset_split=self.dataset_split,
            dataset_revision=self.dataset_revision,
            dataset_fallback_revisions=list(self.dataset_fallback_revisions),
            split=self.split,
            config_source=self.config_source,
            model_config=serialize_llm_configs(self.config),
            ablation_config=build_apex_ablation_config(self.config),
            ndff_exclude_nondeterministic=bool(
                getattr(self.config.benchmark, "commit0_ndff_exclude_nondeterministic", True)
            ),
        )
        report.run_manifest = ensure_run_manifest(
            self.output_dir,
            build_run_manifest(
                config=self.config,
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                benchmark_family="commit0",
                output_dir=self.output_dir,
                config_source=self.config_source,
                requested_task_ids=requested_task_ids,
                execution=execution,
                extra_settings={
                    "dataset_name": self.dataset_name,
                    "dataset_split": self.dataset_split,
                    "dataset_revision": self.dataset_revision,
                    "dataset_fallback_revisions": list(self.dataset_fallback_revisions),
                    "split": self.split,
                    "requested_repo_names": [task.repo_name for task in tasks],
                },
                benchmark_policy=_build_commit0_benchmark_policy(self.config),
            ),
        )
        ordered_instance_ids = [task.instance_id for task in tasks]
        completed_results: dict[str, Commit0TaskResult] = {}
        pending_tasks: list[Commit0Task] = []
        # WS2A (CCEDF): replay the confirmed-candidate escrow WAL once. A live
        # per-task checkpoint ALWAYS wins (it reflects a real run); escrow only
        # fills tasks that have no checkpoint yet — recovering a confirmed pass
        # that a restart would otherwise re-run from scratch (or lose).
        escrow_by_task = self._replay_escrow_results()
        for task in tasks:
            checkpointed = self._load_checkpointed_task_result(task)
            if checkpointed is not None:
                completed_results[task.instance_id] = checkpointed
                continue
            escrow_result = self._task_result_from_escrow(task, escrow_by_task)
            if escrow_result is not None:
                completed_results[task.instance_id] = escrow_result
                # Persist as a normal checkpoint so a later restart treats it as
                # an ordinary completed task (idempotent, exactly-once apply).
                try:
                    write_task_checkpoint(self._task_output_dir(task), escrow_result.to_dict())
                except Exception:  # noqa: BLE001 - checkpoint write is best-effort
                    logger.debug(
                        "Escrow checkpoint write failed for %s", task.repo_name, exc_info=True
                    )
                continue
            pending_tasks.append(task)

        # Layer B (Commit0 dispatch policy): Commit0 public expected pytest-id
        # counts are a cheap runtime proxy, but strict largest-first can fill all
        # solve lanes with long-tail repos before any short repo starts.
        pending_tasks = self._order_pending_tasks_for_dispatch(pending_tasks)

        def refresh_report_tasks() -> None:
            report.tasks = [
                completed_results[instance_id]
                for instance_id in ordered_instance_ids
                if instance_id in completed_results
            ]

        refresh_report_tasks()
        self._write_report_checkpoint(report, requested_task_ids, completed=False)

        solve_workers = self._task_worker_limit(len(pending_tasks))
        audit_workers = self._official_audit_worker_limit(len(pending_tasks))
        executor_workers = self._task_executor_worker_limit(len(pending_tasks))
        try:
            with self._outer_task_parallelism_context(solve_workers, audit_workers):
                if executor_workers == 1:
                    for task in pending_tasks:
                        self._raise_if_cancel_requested()
                        result = self._run_task_with_checkpoint(task)
                        completed_results[task.instance_id] = result
                        refresh_report_tasks()
                        self._write_report_checkpoint(report, requested_task_ids, completed=False)
                else:
                    with _interruptible_thread_pool(executor_workers) as executor:
                        futures = {
                            executor.submit(self._run_task_with_checkpoint, task): task.instance_id
                            for task in pending_tasks
                        }
                        for future in as_completed(futures):
                            self._raise_if_cancel_requested()
                            task_instance_id = futures[future]
                            completed_results[task_instance_id] = future.result()
                            refresh_report_tasks()
                            self._write_report_checkpoint(
                                report, requested_task_ids, completed=False
                            )
        except (KeyboardInterrupt, SystemExit) as exc:
            refresh_report_tasks()
            self._write_report_checkpoint(report, requested_task_ids, completed=False)
            self._mark_run_interrupted(report, requested_task_ids, exc)
            raise
        finally:
            self._close_commit0_docker_proxy_relays()

        self._write_report_checkpoint(report, requested_task_ids, completed=True)
        # Phase 1: persist the per-run reproducibility manifest. Written
        # at the END so docker_images / model_versions populated during
        # the run land in the JSON. Best-effort: a manifest write
        # failure must NEVER fail the published headline.
        self._write_run_manifest()
        # Phase 1: persist the fairness audit artifact iff configured.
        self._write_fairness_audit_artifact()
        return report

    def _raise_if_cancel_requested(self) -> None:
        if cli_cleanup_signal_requested():
            raise KeyboardInterrupt("Apex run cancellation requested by signal")

    def _init_run_manifest(self) -> None:
        """Capture the per-run reproducibility manifest.

        Guarded so multiple ``run()`` calls (or a unit test that calls
        the helper directly) reuse the existing manifest rather than
        clobbering its docker_images / model_versions accumulators.
        """
        if self.run_manifest is not None:
            return
        try:
            manifest = RunManifest.capture(
                apex_config=self.config,
                seed=getattr(self.config, "seed", None),
                additional_metadata={
                    "benchmark": "commit0",
                    "split": self.split,
                    "dataset_name": self.dataset_name,
                    "dataset_split": self.dataset_split,
                    "dataset_revision": self.dataset_revision,
                    "dataset_fallback_revisions": list(self.dataset_fallback_revisions),
                    "evidence_mode": "gold_suite_visible",
                    "suite_visibility": "visible_gold_test_suite",
                    "harness_name": COMMIT0_BENCHMARK_HARNESS_NAME,
                    "harness_version": COMMIT0_BENCHMARK_HARNESS_VERSION,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RunManifest.capture failed: %s", exc)
            return
        # Record the upstream commit0 harness up front; per-task docker
        # invocations will fill in docker_images and model_versions as
        # they fire.
        try:
            for name, version in detect_upstream_harness_versions().items():
                manifest.add_upstream_harness(name, version)
        except Exception:  # pragma: no cover - defensive
            pass
        # Always advertise the commit0 shared harness even when the
        # ``commit0`` distribution can't be auto-detected (e.g. running
        # against a vendored copy in tests).
        manifest.add_upstream_harness(
            COMMIT0_BENCHMARK_HARNESS_NAME, COMMIT0_BENCHMARK_HARNESS_VERSION
        )
        # Record the LLM aliases the run will use. Models are listed in
        # ``self.config.llm_configs``; record them as alias->model_id so
        # reviewers can audit which weights produced the headline.
        try:
            for entry in getattr(self.config, "llm_configs", []) or []:
                alias = getattr(entry, "alias", None) or getattr(entry, "name", None)
                model_id = getattr(entry, "model", None) or getattr(entry, "model_id", None)
                if alias and model_id:
                    manifest.add_model(str(alias), str(model_id))
        except Exception:  # pragma: no cover - defensive
            pass
        self.run_manifest = manifest

    def _write_run_manifest(self) -> None:
        if self.run_manifest is None:
            return
        # NB: the legacy ``run_manifest.json`` filename in ``output_dir``
        # is owned by ``apex.evaluation.run_artifacts`` (benchmark-policy
        # tracking + checkpointing). The new core manifest writes to a
        # sibling file so the two artifacts coexist; reviewers consult
        # the new ``apex_run_manifest.json`` for the per-run
        # reproducibility snapshot (apex git sha, docker digests, model
        # ids, upstream harness versions).
        try:
            self.run_manifest.write_to(self.output_dir / "apex_run_manifest.json")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RunManifest.write_to failed: %s", exc)

    def _fairness_audit_mode(self) -> FairnessAuditMode:
        raw = getattr(self.config.benchmark, "fairness_audit_mode", "off")
        if isinstance(raw, FairnessAuditMode):
            return raw
        try:
            return FairnessAuditMode(str(raw).lower())
        except ValueError:
            logger.warning("Unknown fairness_audit_mode=%r; defaulting to OFF.", raw)
            return FairnessAuditMode.OFF

    def _init_fairness_audit_aggregator(self) -> None:
        if self.fairness_audit_aggregator is not None:
            return
        if self._fairness_audit_mode() == FairnessAuditMode.OFF:
            return
        self.fairness_audit_aggregator = FairnessAuditAggregator()

    def _write_fairness_audit_artifact(self) -> None:
        aggregator = self.fairness_audit_aggregator
        if aggregator is None:
            return
        try:
            aggregator.write_to(self.output_dir)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("FairnessAuditAggregator.write_to failed: %s", exc)

    def _populate_core_failure_classification(
        self,
        evaluation: "Commit0Evaluation",
        exc_or_text: Any,
        *,
        phase: str = "",
    ) -> None:
        """Run the core ``classify_failure`` and stash the result.

        ``exc_or_text`` may be an Exception or a raw stderr/stdout
        string. Phase hints (``pre_install``, ``install``, ``baseline``,
        ``test_execution``, ``scoring``) drive ambiguous-pattern
        routing — see ``apex.core.failure_classifier`` docstring for
        the full taxonomy.
        """
        try:
            if isinstance(exc_or_text, BaseException):
                stderr = str(exc_or_text)
            else:
                stderr = str(exc_or_text or "")
            result = _core_classify_failure(
                stderr=stderr,
                stdout="",
                returncode=int(getattr(evaluation, "returncode", 1) or 1),
                context={"phase": phase} if phase else None,
            )
            evaluation.failure_class = result.failure_class.value
            evaluation.failure_classification = result.to_dict()
        except Exception:  # pragma: no cover - defensive
            pass

    def _record_fairness_audit_delta(
        self,
        *,
        task: "Commit0Task",
        apex_private_eval: "Commit0Evaluation",
        upstream_audit_eval: "Commit0Evaluation",
    ) -> None:
        """Record a per-task fairness delta when the audit is enabled.

        This is a thin glue function: when ``fairness_audit_mode`` is
        OFF (the default for production runs) it is a no-op. Otherwise
        it constructs the two scorers, runs the audit, and adds the
        result to the aggregator. Failures here MUST NOT fail the
        published headline; they're surfaced as warnings only.
        """
        aggregator = self.fairness_audit_aggregator
        if aggregator is None:
            return
        try:
            from .scorers.commit0_private import Commit0PrivateScorer
            from .scorers.commit0_upstream import Commit0UpstreamScorer

            artifacts = {
                "apex_private": apex_private_eval,
                "upstream": upstream_audit_eval,
            }
            shared_image_digest: Optional[str] = None
            if self.run_manifest is not None:
                images = getattr(self.run_manifest, "docker_images", {}) or {}
                if images:
                    # First recorded image is the runtime image used for
                    # both scorers (we don't currently use a different
                    # image for the audit step).
                    first_tag = next(iter(images.keys()), None)
                    if first_tag:
                        shared_image_digest = f"{first_tag}={images[first_tag]}"
            extra_notes: list[str] = []
            if self._fairness_audit_mode() == FairnessAuditMode.UPSTREAM_ONLY:
                extra_notes.append(
                    "fairness_audit_mode=UPSTREAM_ONLY; APEX-private scores are diagnostic-only."
                )
            delta = run_fairness_audit(
                task,
                artifacts,
                Commit0PrivateScorer(),
                Commit0UpstreamScorer(),
                shared_image_digest=shared_image_digest,
                extra_notes=extra_notes,
            )
            aggregator.add_task(delta)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("fairness audit recording failed for %s: %s", task, exc)

    def _run_task_with_checkpoint(self, task: Commit0Task) -> Commit0TaskResult:
        solve_slot: Optional[_TaskSolveSlot] = None
        previous_solve_slot = getattr(self._task_lane_state, "solve_slot", None)
        if self._task_solve_semaphore is not None:
            self._note_solve_waiter_started()
            try:
                self._task_solve_semaphore.acquire()
            except BaseException:
                self._note_solve_waiter_cancelled()
                raise
            self._note_solve_slot_acquired()
            solve_slot = _TaskSolveSlot(
                self._task_solve_semaphore,
                on_release=self._note_solve_slot_released,
            )
            self._task_lane_state.solve_slot = solve_slot
        try:
            ensure_clean_directory_for_task(self._task_output_dir(task), completed=False)
            ensure_clean_directory_for_task(self._task_workspace_dir(task), completed=False)
            result = self._run_task(task)
            write_task_checkpoint(self._task_output_dir(task), result.to_dict())
            return result
        finally:
            if solve_slot is not None:
                solve_slot.release()
                if previous_solve_slot is None:
                    try:
                        delattr(self._task_lane_state, "solve_slot")
                    except AttributeError:
                        pass
                else:
                    self._task_lane_state.solve_slot = previous_solve_slot

    def _release_current_task_solve_slot_for_audit(self) -> bool:
        solve_slot = getattr(self._task_lane_state, "solve_slot", None)
        if isinstance(solve_slot, _TaskSolveSlot):
            return solve_slot.release()
        return False

    @contextmanager
    def _official_audit_lane(self) -> Iterator[None]:
        audit_semaphore = self._task_audit_semaphore
        if audit_semaphore is not None:
            audit_semaphore.acquire()
        try:
            yield
        finally:
            if audit_semaphore is not None:
                audit_semaphore.release()

    def _task_worker_limit(self, task_count: int) -> int:
        if task_count <= 0:
            return 1
        configured = max(int(self.config.benchmark.task_parallelism or 1), 1)
        return max(1, min(task_count, configured))

    def _official_audit_worker_limit(self, task_count: int) -> int:
        if task_count <= 0:
            return 0
        if not self.config.benchmark.commit0_official_audit_selected:
            return 0
        if (
            self.config.benchmark.commit0_primary_evaluation_backend
            != BenchmarkEvaluationBackend.LOCAL_PYTEST
        ):
            return 0
        configured = int(
            getattr(self.config.benchmark, "commit0_official_audit_parallelism", 1) or 0
        )
        if configured <= 0:
            return 0
        return min(task_count, configured)

    def _task_executor_worker_limit(self, task_count: int) -> int:
        solve_workers = self._task_worker_limit(task_count)
        audit_workers = self._official_audit_worker_limit(task_count)
        if task_count <= solve_workers or audit_workers <= 0:
            return solve_workers
        # Commit0 task futures wait on solve/audit lane semaphores. Starting one
        # lightweight future per task prevents audit-waiting futures from
        # occupying every executor thread and starving the solve lane.
        return task_count

    @contextmanager
    def _outer_task_parallelism_context(
        self,
        solve_workers: int,
        audit_workers: int,
    ) -> Iterator[None]:
        previous_solve_semaphore = self._task_solve_semaphore
        previous_audit_semaphore = self._task_audit_semaphore
        previous_outer_task_env = {
            _ACTIVE_OUTER_TASK_COUNT_ENV: os.environ.get(_ACTIVE_OUTER_TASK_COUNT_ENV),
            _WAITING_OUTER_TASK_COUNT_ENV: os.environ.get(_WAITING_OUTER_TASK_COUNT_ENV),
        }
        self._task_solve_semaphore = threading.BoundedSemaphore(max(1, int(solve_workers or 1)))
        self._task_audit_semaphore = (
            threading.BoundedSemaphore(max(1, int(audit_workers)))
            if int(audit_workers or 0) > 0
            else None
        )
        with self._task_solve_slot_lock:
            self._active_solve_task_count = 0
            self._waiting_solve_task_count = 0
        self._publish_outer_task_parallelism_state()
        try:
            yield
        finally:
            self._task_solve_semaphore = previous_solve_semaphore
            self._task_audit_semaphore = previous_audit_semaphore
            with self._task_solve_slot_lock:
                self._active_solve_task_count = 0
                self._waiting_solve_task_count = 0
            for env_name, previous_value in previous_outer_task_env.items():
                if previous_value is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = previous_value

    def _publish_outer_task_parallelism_state(self) -> None:
        with self._task_solve_slot_lock:
            active = max(0, int(self._active_solve_task_count))
            waiting = max(0, int(self._waiting_solve_task_count))
        if self._task_solve_semaphore is None:
            os.environ.pop(_ACTIVE_OUTER_TASK_COUNT_ENV, None)
            os.environ.pop(_WAITING_OUTER_TASK_COUNT_ENV, None)
            return
        os.environ[_ACTIVE_OUTER_TASK_COUNT_ENV] = str(active)
        os.environ[_WAITING_OUTER_TASK_COUNT_ENV] = str(waiting)

    def _note_solve_waiter_started(self) -> None:
        with self._task_solve_slot_lock:
            self._waiting_solve_task_count += 1
        self._publish_outer_task_parallelism_state()

    def _note_solve_slot_acquired(self) -> None:
        with self._task_solve_slot_lock:
            self._waiting_solve_task_count = max(0, self._waiting_solve_task_count - 1)
            self._active_solve_task_count += 1
        self._publish_outer_task_parallelism_state()

    def _note_solve_waiter_cancelled(self) -> None:
        with self._task_solve_slot_lock:
            self._waiting_solve_task_count = max(0, self._waiting_solve_task_count - 1)
        self._publish_outer_task_parallelism_state()

    def _note_solve_slot_released(self) -> None:
        with self._task_solve_slot_lock:
            self._active_solve_task_count = max(0, self._active_solve_task_count - 1)
        self._publish_outer_task_parallelism_state()

    def _task_expected_test_count_hint(self, task: Commit0Task) -> int:
        """Cheap, task-independent suite-size proxy from the public expected-ID
        inventory. Layer B: Commit0 exposes the expected pytest-id universe by
        repo name without running the task, so its size estimates suite runtime
        for dispatch/budget decisions. Returns 0 when the inventory is
        unavailable (those tasks then sort last, deterministically)."""
        cache = getattr(self, "_expected_test_count_hint_cache", None)
        if cache is None:
            cache = {}
            self._expected_test_count_hint_cache = cache
        if task.repo_name in cache:
            return cache[task.repo_name]
        try:
            count = len(_load_expected_test_ids(task.repo_name))
        except Exception:
            count = 0
        cache[task.repo_name] = count
        return count

    def _order_pending_tasks_for_dispatch(
        self, pending_tasks: list[Commit0Task]
    ) -> list[Commit0Task]:
        """Deterministic heavy/light Commit0 dispatch order.

        The public expected-ID inventory gives a task-independent suite-size
        proxy. Interleaving the largest and smallest known-size repos keeps
        long-tail repos started early while still allowing short repos to finish
        and free lanes quickly. Tasks without inventory hints remain last so an
        unknown size does not accidentally outrank a known quick task.
        """
        known: list[tuple[int, Commit0Task]] = []
        unknown: list[Commit0Task] = []
        for task in pending_tasks:
            size_hint = self._task_expected_test_count_hint(task)
            if size_hint > 0:
                known.append((size_hint, task))
            else:
                unknown.append(task)

        ranked_known = [
            task
            for _size, task in sorted(
                known,
                key=lambda item: (-item[0], item[1].instance_id),
            )
        ]
        ordered: list[Commit0Task] = []
        left = 0
        right = len(ranked_known) - 1
        while left <= right:
            ordered.append(ranked_known[left])
            left += 1
            if left <= right:
                ordered.append(ranked_known[right])
                right -= 1
        ordered.extend(sorted(unknown, key=lambda task: task.instance_id))
        return ordered

    def _task_output_dir(self, task: Commit0Task) -> Path:
        return self.output_dir / task.repo_name

    def _task_workspace_dir(self, task: Commit0Task) -> Path:
        return self.output_dir / "workspaces" / task.repo_name

    def _task_execution_sandbox_base(self, task: Commit0Task) -> Optional[Path]:
        if not (
            self._task_requires_linux_container(task)
            or self._commit0_docker_runtime_forced()
            or self._docker_fallback_available()
        ):
            return None
        configured_base = os.environ.get("APEX_COMMIT0_TASK_SANDBOX_BASE")
        if configured_base:
            sandbox_base = Path(configured_base).expanduser().resolve()
        else:
            output_parent = self.output_dir.parent.resolve()
            output_parent_text = str(output_parent)
            if sys.platform == "darwin" and (
                output_parent_text == "/private/var/folders"
                or output_parent_text.startswith("/private/var/folders/")
                or output_parent_text == "/var/folders"
                or output_parent_text.startswith("/var/folders/")
            ):
                output_parent = Path.cwd().resolve()
            sandbox_base = (output_parent / ".apex_commit0_task_sandboxes").resolve()
        sandbox_base.mkdir(parents=True, exist_ok=True)
        return sandbox_base

    def _commit0_min_free_disk_bytes(self) -> int:
        raw = getattr(self.config.benchmark, "commit0_min_free_disk_gb", 0) or 0
        try:
            gb = float(raw)
        except (TypeError, ValueError):
            gb = 0.0
        return max(0, int(gb * 1024 * 1024 * 1024))

    def _ensure_commit0_disk_headroom(self, tasks: Iterable[Commit0Task]) -> None:
        min_free_bytes = self._commit0_min_free_disk_bytes()
        if min_free_bytes <= 0:
            return
        sandbox_bases: set[Path] = set()
        for task in tasks:
            sandbox_base = self._task_execution_sandbox_base(task)
            if sandbox_base is not None:
                sandbox_bases.add(sandbox_base)
        if getattr(self.config.benchmark, "commit0_prune_stale_task_sandboxes", True):
            min_age_raw = getattr(
                self.config.benchmark,
                "commit0_stale_task_sandbox_min_age_seconds",
                1800,
            )
            try:
                min_age_seconds = max(0, float(min_age_raw))
            except (TypeError, ValueError):
                min_age_seconds = 1800.0
            for sandbox_base in sorted(sandbox_bases):
                for prune_base in self._commit0_task_sandbox_prune_bases(sandbox_base):
                    self._prune_stale_commit0_task_sandboxes(
                        prune_base,
                        min_age_seconds=min_age_seconds,
                    )

        checked_paths = [self.output_dir, *sorted(sandbox_bases)]
        low_space: list[str] = []
        for path in checked_paths:
            try:
                usage = shutil.disk_usage(path)
            except OSError as exc:
                low_space.append(f"{path}: disk usage unavailable ({exc})")
                continue
            if usage.free < min_free_bytes:
                low_space.append(f"{path}: {usage.free / (1024**3):.1f}GiB free")
        if low_space:
            required_gb = min_free_bytes / (1024**3)
            details = "; ".join(low_space)
            raise RuntimeError(
                "Commit0 disk preflight failed: "
                f"requires at least {required_gb:.1f}GiB free after stale "
                f"task-sandbox pruning, but observed {details}."
            )

    def _prune_stale_commit0_task_sandboxes(
        self,
        sandbox_base: Path,
        *,
        min_age_seconds: float,
    ) -> None:
        if not sandbox_base.exists():
            return
        now = time.time()
        try:
            children = list(sandbox_base.iterdir())
        except OSError:
            logger.debug(
                "Commit0 stale sandbox scan failed for %s",
                sandbox_base,
                exc_info=True,
            )
            return
        for child in children:
            if not child.is_dir() or not child.name.startswith("apex-commit0-"):
                continue
            try:
                age_seconds = now - child.stat().st_mtime
            except OSError:
                continue
            if age_seconds < min_age_seconds:
                continue
            if self._sandbox_has_live_process(child):
                continue
            # Commit0 Docker solve tasks materialize large per-task sandboxes;
            # interrupted runs can leave them behind and exhaust the host volume.
            shutil.rmtree(child, ignore_errors=True)
        try:
            if "_stale_" in sandbox_base.name:
                sandbox_base.rmdir()
        except OSError:
            pass

    def _commit0_task_sandbox_prune_bases(self, sandbox_base: Path) -> list[Path]:
        bases = [sandbox_base]
        try:
            stale_bases = sorted(sandbox_base.parent.glob(f"{sandbox_base.name}_stale_*"))
        except OSError:
            stale_bases = []
        for stale_base in stale_bases:
            if stale_base.is_dir():
                bases.append(stale_base)
        return bases

    def _sandbox_has_live_process(self, sandbox_root: Path) -> bool:
        marker = str(sandbox_root.resolve(strict=False))
        try:
            completed = subprocess.run(
                ["ps", "-axo", "pid=,command="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # If process inspection is unavailable, preserve the sandbox.
            return True
        current_pid = os.getpid()
        for line in completed.stdout.splitlines():
            if marker not in line:
                continue
            parts = line.strip().split(maxsplit=1)
            try:
                pid = int(parts[0])
            except (IndexError, ValueError):
                return True
            if pid != current_pid:
                return True
        return False

    def _build_task_execution_layout(self, task: Commit0Task) -> _TaskExecutionLayout:
        mkdtemp_kwargs: dict[str, str] = {"prefix": f"apex-commit0-{task.repo_name}-"}
        sandbox_base = self._task_execution_sandbox_base(task)
        if sandbox_base is not None:
            mkdtemp_kwargs["dir"] = str(sandbox_base)
        sandbox_root = Path(tempfile.mkdtemp(**mkdtemp_kwargs)).resolve()
        try:
            layout = _TaskExecutionLayout(
                sandbox_root=sandbox_root,
                repo_dir=sandbox_root / task.repo_name,
                runtime_dir=sandbox_root / ".runtime",
                task_output_dir=sandbox_root / "task_output",
                workspace_dir=sandbox_root / "workspaces",
            )
            layout.task_output_dir.mkdir(parents=True, exist_ok=True)
            layout.workspace_dir.mkdir(parents=True, exist_ok=True)
            layout.runtime_dir.mkdir(parents=True, exist_ok=True)
            self._validate_task_execution_layout(task, layout)
            return layout
        except Exception:
            shutil.rmtree(sandbox_root, ignore_errors=True)
            raise

    def _validate_task_execution_layout(
        self,
        task: Commit0Task,
        layout: _TaskExecutionLayout,
    ) -> None:
        sandbox_root = layout.sandbox_root.resolve(strict=False)
        persistent_root = self.output_dir.resolve(strict=False)
        for label, candidate in (
            ("repo_dir", layout.repo_dir),
            ("runtime_dir", layout.runtime_dir),
            ("task_output_dir", layout.task_output_dir),
            ("workspace_dir", layout.workspace_dir),
        ):
            resolved = candidate.resolve(strict=False)
            if not _path_is_relative_to(resolved, sandbox_root):
                raise RuntimeError(
                    f"Commit0 task {task.repo_name} uses non-sandboxed {label}: {resolved}"
                )
            if _path_is_relative_to(resolved, persistent_root):
                raise RuntimeError(
                    f"Commit0 task {task.repo_name} leaks persistent benchmark path via {label}: {resolved}"
                )

    def _task_process_markers(
        self,
        layout: _TaskExecutionLayout,
    ) -> list[str]:
        markers = [
            str(layout.sandbox_root.resolve(strict=False)),
            str(layout.repo_dir.resolve(strict=False)),
            str(layout.runtime_dir.resolve(strict=False)),
            str(layout.task_output_dir.resolve(strict=False)),
            str(layout.workspace_dir.resolve(strict=False)),
        ]
        return [marker for marker in dict.fromkeys(markers) if marker]

    def _collect_task_processes(
        self,
        markers: list[str],
    ) -> set[int]:
        if not markers:
            return set()
        snapshot = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if snapshot.returncode != 0:
            return set()

        current_pid = os.getpid()
        children_by_parent: dict[int, list[int]] = {}
        matched_roots: set[int] = set()
        for raw_line in snapshot.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue
            command = parts[2]
            children_by_parent.setdefault(ppid, []).append(pid)
            if pid == current_pid:
                continue
            if any(marker in command for marker in markers):
                matched_roots.add(pid)

        tracked = set(matched_roots)
        stack = list(matched_roots)
        while stack:
            parent = stack.pop()
            for child in children_by_parent.get(parent, []):
                if child in tracked or child == current_pid:
                    continue
                tracked.add(child)
                stack.append(child)
        return tracked

    def _cleanup_task_processes(
        self,
        layout: _TaskExecutionLayout,
    ) -> list[int]:
        tracked = self._collect_task_processes(self._task_process_markers(layout))
        if not tracked:
            return []

        for pid in sorted(tracked, reverse=True):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue

        deadline = time.time() + 1.0
        remaining = set(tracked)
        while remaining and time.time() < deadline:
            time.sleep(0.1)
            still_running: set[int] = set()
            for pid in remaining:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                else:
                    still_running.add(pid)
            remaining = still_running

        for pid in sorted(remaining, reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        return sorted(tracked)

    def _sync_task_output_artifacts(
        self,
        source_dir: Path,
        destination_dir: Path,
    ) -> None:
        if not source_dir.exists():
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        sync_failures: list[dict[str, str]] = []
        for source_path in sorted(source_dir.rglob("*")):
            relative = source_path.relative_to(source_dir)
            if not relative.parts:
                continue
            if _is_commit0_atomic_write_temp_artifact(relative):
                continue
            destination_path = destination_dir / relative
            if source_path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
            elif source_path.is_file():
                if relative.parts[-1] == "task_live_state.json":
                    payload = load_json_if_exists(source_path)
                    if isinstance(payload, dict):
                        # Merge the sandbox task live state into the persisted
                        # benchmark artifact so task-level progress reflects the
                        # orchestrator's rollout heartbeats instead of freezing
                        # at the runner-authored "phase=solving" marker.
                        write_task_live_state(destination_dir, payload)
                        self._refresh_live_run_state_from_task_states()
                    continue
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                # Commit0 progress streaming is read concurrently by operators;
                # publish complete files atomically so readers never see a
                # partially overwritten JSON/status artifact.
                try:
                    _copy_file_atomic(source_path, destination_path)
                except OSError as exc:
                    sync_failures.append(
                        {
                            "path": relative.as_posix(),
                            "error": str(exc)[:1000],
                        }
                    )
                    logger.warning(
                        "Skipping Commit0 task artifact during sync: %s -> %s: %s",
                        source_path,
                        destination_path,
                        exc,
                    )
        if sync_failures:
            try:
                atomic_write_json(
                    destination_dir / "artifact_sync_failures.json",
                    {"failures": sync_failures},
                )
            except OSError:
                logger.warning(
                    "Failed to persist Commit0 artifact sync failure diagnostics",
                    exc_info=True,
                )

    def _persist_task_workspaces(
        self,
        source_dir: Path,
        destination_dir: Path,
    ) -> None:
        if not source_dir.exists():
            return
        try:
            copy_tree(
                source_dir,
                destination_dir,
                dirs_exist_ok=True,
                ignore=_commit0_persisted_workspace_ignore,
            )
        except shutil.Error as exc:
            if not _copy_tree_error_is_only_missing_files(exc):
                raise
            # Commit0 task sandboxes can still have quarantined agent workers
            # finalizing diagnostic artifacts; vanished files are best-effort
            # workspace evidence and must not abort the benchmark report.
            logger.warning(
                "Persisted task workspaces with missing transient files: %s",
                exc,
            )
        for internal_dir in list(destination_dir.rglob(".locks")) + list(
            destination_dir.rglob("_pool")
        ):
            if internal_dir.is_dir():
                shutil.rmtree(internal_dir, ignore_errors=True)
        for rollout_dir in sorted(destination_dir.rglob("rollout_*")):
            if rollout_dir.is_dir():
                self._materialize_persisted_worktree(rollout_dir)

    def _remove_worktree_contents(
        self,
        worktree_dir: Path,
        *,
        keep_names: Optional[set[str]] = None,
    ) -> None:
        keep = keep_names or set()
        for child in list(worktree_dir.iterdir()):
            if child.name in keep:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    def _write_task_result_checkpoint_best_effort(
        self,
        task_output_dir: Path,
        task_result: Commit0TaskResult,
    ) -> None:
        try:
            write_task_checkpoint(task_output_dir, task_result.to_dict())
        except Exception as exc:  # pragma: no cover - defensive artifact path
            logger.warning(
                "Failed to persist Commit0 task checkpoint for %s before cleanup: %s",
                task_result.instance_id,
                exc,
            )

    def _materialize_persisted_worktree(self, worktree_dir: Path) -> None:
        self._materialize_gitfile_current_tree(worktree_dir)

    def _materialize_gitfile_current_tree(self, worktree_dir: Path) -> bool:
        git_entry = worktree_dir / ".git"
        if not git_entry.is_file():
            return False
        try:
            git_text = git_entry.read_text(errors="ignore").strip()
        except OSError:
            git_text = ""
        if not git_text.startswith("gitdir:"):
            return False
        try:
            git_entry.unlink()
        except OSError:
            return False
        commands = [
            ["git", "init"],
            ["git", "config", "user.email", "apex@example.com"],
            ["git", "config", "user.name", "APEX"],
            ["git", "add", "-A"],
            ["git", "commit", "--allow-empty", "-m", "materialize persisted rollout workspace"],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                cwd=str(worktree_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    "Failed to materialize persisted worktree %s via %s: %s",
                    worktree_dir,
                    " ".join(command),
                    (result.stderr or result.stdout or "").strip(),
                )
                return False
        return True

    def _materialize_commit0_audit_worktree_from_baseline(
        self,
        *,
        audit_worktree: Path,
        baseline_repo_dir: Optional[Path],
    ) -> bool:
        git_entry = audit_worktree / ".git"
        if not git_entry.is_file():
            return True
        try:
            git_text = git_entry.read_text(errors="ignore").strip()
        except OSError:
            git_text = ""
        if not git_text.startswith("gitdir:"):
            return True
        if baseline_repo_dir is None or not baseline_repo_dir.exists():
            return False

        snapshot_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{audit_worktree.name}.candidate.",
                dir=str(audit_worktree.parent),
            )
        )

        def ignore_git_metadata(_directory: str, names: list[str]) -> list[str]:
            return [name for name in names if name == ".git"]

        def run_git(command: list[str], timeout: int = 300) -> bool:
            result = run_process_command(command, cwd=audit_worktree, timeout=timeout)
            if result.returncode == 0:
                return True
            logger.warning(
                "Failed to materialize Commit0 audit worktree %s via %s: %s",
                audit_worktree,
                " ".join(command),
                (result.stderr or result.stdout or "").strip(),
            )
            return False

        try:
            copy_tree(
                audit_worktree,
                snapshot_dir,
                dirs_exist_ok=True,
                ignore=ignore_git_metadata,
            )
            self._remove_worktree_contents(audit_worktree)
            audit_worktree.mkdir(parents=True, exist_ok=True)
            if not run_git(["git", "init", "-b", "apex-base"], timeout=120):
                if not run_git(["git", "init"], timeout=120):
                    return False
                if not run_git(["git", "checkout", "-B", "apex-base"], timeout=120):
                    return False
            if not run_git(["git", "config", "user.email", "apex@example.com"], timeout=60):
                return False
            if not run_git(["git", "config", "user.name", "APEX"], timeout=60):
                return False

            # Commit0 official audit generates base..HEAD patches with GitPython;
            # sanitized gitfile copies therefore need a standalone baseline commit.
            export = run_process_command(
                [
                    "git",
                    "checkout-index",
                    "-a",
                    "-f",
                    f"--prefix={str(audit_worktree)}/",
                ],
                cwd=baseline_repo_dir,
                timeout=300,
            )
            if export.returncode != 0:
                logger.warning(
                    "Failed to export Commit0 baseline for audit worktree %s: %s",
                    audit_worktree,
                    (export.stderr or export.stdout or "").strip(),
                )
                return False

            tracked = run_process_command(
                ["git", "ls-files", "-z"],
                cwd=baseline_repo_dir,
                timeout=120,
            )
            tracked_files = [
                path
                for path in tracked.stdout.split("\0")
                if path
                and ((audit_worktree / path).exists() or (audit_worktree / path).is_symlink())
            ]
            if tracked.returncode == 0 and tracked_files:
                for start in range(0, len(tracked_files), 200):
                    chunk = tracked_files[start : start + 200]
                    if not run_git(["git", "add", "-f", "--", *chunk], timeout=300):
                        return False
            elif not run_git(["git", "add", "-A"], timeout=300):
                return False
            if not run_git(
                ["git", "commit", "--allow-empty", "-m", "APEX Commit0 baseline"],
                timeout=300,
            ):
                return False

            self._remove_worktree_contents(audit_worktree, keep_names={".git"})
            copy_tree(snapshot_dir, audit_worktree, dirs_exist_ok=True, ignore=ignore_git_metadata)
            return True
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    def _stable_task_worktree_path(
        self,
        task: Commit0Task,
        actual_path: Optional[str],
        *,
        internal_workspace_dir: Path,
        retain_worktrees: bool,
    ) -> Optional[str]:
        if not actual_path:
            return actual_path
        if not retain_worktrees:
            return None
        resolved_path = Path(actual_path).resolve(strict=False)
        internal_root = internal_workspace_dir.resolve(strict=False)
        if not _path_is_relative_to(resolved_path, internal_root):
            return str(resolved_path)
        relative = resolved_path.relative_to(internal_root)
        return str((self._task_workspace_dir(task) / relative).resolve(strict=False))

    def _load_checkpointed_task_result(
        self,
        task: Commit0Task,
    ) -> Optional[Commit0TaskResult]:
        checkpoint_path = task_result_path(self._task_output_dir(task))
        payload = load_json_if_exists(checkpoint_path)
        if payload is None:
            return None
        try:
            return Commit0TaskResult.from_dict(payload)
        except Exception as exc:
            logger.warning(
                "Ignoring corrupt Commit0 checkpoint for %s at %s: %s",
                task.repo_name,
                checkpoint_path,
                exc,
            )
            return None

    def _replay_escrow_results(self) -> dict[str, "EscrowRecord"]:
        """WS2A: best-effort replay of the confirmed-candidate escrow WAL.

        Returns ``{repo_name: best EscrowRecord}`` (replay() dedups by idempotency
        key and keeps the highest-(score, seq) record per task). Never fatal.
        """
        try:
            return EscrowStore(self.output_dir).replay()
        except Exception:  # noqa: BLE001 - escrow replay must never break a run
            logger.debug("Escrow replay failed", exc_info=True)
            return {}

    def _task_result_from_escrow(
        self,
        task: Commit0Task,
        escrow_by_task: dict[str, "EscrowRecord"],
    ) -> Optional[Commit0TaskResult]:
        """WS2A: recover a SOLVED result for ``task`` from an escrowed confirmed
        full-scope pass, or ``None``.

        Defense in depth: only a ``confirmed_full_scope_pass`` record whose
        payload re-confirms a genuine local full-scope pass
        (:func:`quick_verification_has_local_full_scope_pass`) is honored, so a
        spurious WAL record can never mint a false SOLVED headline.
        """
        record = escrow_by_task.get(task.repo_name)
        if record is None:
            return None
        if str(getattr(record, "kind", "")) != "confirmed_full_scope_pass":
            return None
        payload = dict(getattr(record, "payload", {}) or {})
        quick_verification = payload.get("quick_verification")
        if not quick_verification_has_local_full_scope_pass(
            quick_verification if isinstance(quick_verification, dict) else {}
        ):
            return None
        passed = 0
        if isinstance(quick_verification, dict):
            try:
                passed = int(quick_verification.get("passed") or 0)
            except (TypeError, ValueError):
                passed = 0
        if passed <= 0:
            passed = 1
        final = Commit0Evaluation(
            returncode=0,
            output="",
            passed=passed,
            failed=0,
            errors=0,
            skipped=0,
            total_tests=passed,
            scoring_source="escrow_replay",
            evaluation_backend="escrow_replay",
        )
        baseline = Commit0Evaluation(returncode=1)
        rollout_id = payload.get("rollout_id")
        try:
            selected_rollout_id = int(rollout_id) if rollout_id is not None else None
        except (TypeError, ValueError):
            selected_rollout_id = None
        logger.info(
            "Recovered SOLVED task %s from escrow candidate=%s seq=%s",
            task.repo_name,
            getattr(record, "candidate_id", ""),
            getattr(record, "seq", ""),
        )
        return Commit0TaskResult(
            task_name=task.repo_name,
            instance_id=task.instance_id,
            repo=task.repo,
            success=True,
            baseline_failed=True,
            final_tests_passed=True,
            baseline=baseline,
            final=final,
            orchestrator_success=True,
            candidate_found=True,
            selected_rollout_id=selected_rollout_id,
            selected_worktree_path=str(payload.get("worktree_path") or "") or None,
            final_candidate_id=getattr(record, "candidate_id", None),
            final_decision_source="escrow_replay",
            result_path=str(task_result_path(self._task_output_dir(task))),
            execution_metadata={
                "recovered_from_escrow": True,
                "escrow_seq": getattr(record, "seq", None),
                "escrow_score": getattr(record, "score", None),
                "escrow_candidate_id": getattr(record, "candidate_id", None),
            },
        )

    def _write_report_checkpoint(
        self,
        report: Commit0BenchmarkReport,
        requested_task_ids: list[str],
        *,
        completed: bool,
    ) -> None:
        report.updated_at = time.time()
        report.finished_at = report.updated_at if completed else 0.0
        update_run_manifest(
            self.output_dir,
            requested_task_ids=requested_task_ids,
            completed_task_ids=[task.instance_id for task in report.tasks],
            completed=completed,
            extra_updates={
                "config_payload": self.config.to_dict(),
                "environment_snapshot": capture_environment_snapshot(self.config),
                "prompt_template_fingerprints": build_prompt_template_fingerprints(),
                "dataset_name": report.dataset_name,
                "dataset_split": report.dataset_split,
                "dataset_revision": report.dataset_revision,
                "dataset_fallback_revisions": list(report.dataset_fallback_revisions),
                "split": report.split,
                "requested_repo_names": list(report.requested_repo_names),
                "execution": {
                    "entrypoint": "commit0-benchmark",
                    "args": {
                        "split": report.split,
                        "repos": list(report.requested_repo_names),
                        "limit": None,
                        "dataset_name": report.dataset_name,
                        "dataset_split": report.dataset_split,
                        "dataset_revision": report.dataset_revision,
                        "dataset_fallback_revisions": list(report.dataset_fallback_revisions),
                        "task_parallelism": self.config.benchmark.task_parallelism,
                        "commit0_official_audit_parallelism": (
                            self.config.benchmark.commit0_official_audit_parallelism
                        ),
                    },
                },
            },
        )
        report.run_manifest = load_run_manifest(self.output_dir) or report.run_manifest
        atomic_write_json(self.output_dir / "benchmark_report.json", report.to_dict())
        atomic_write_text(self.output_dir / "benchmark_report.md", report.to_markdown())
        completed_task_ids = [task.instance_id for task in report.tasks]
        active_task_ids = self._active_task_ids(
            requested_task_ids,
            completed_task_ids=completed_task_ids,
        )
        active_tasks = self._active_task_summaries(
            requested_task_ids,
            completed_task_ids=completed_task_ids,
        )
        atomic_write_json(
            self.output_dir / RUN_STATE_FILENAME,
            build_run_state(
                report_kind=report.report_kind,
                harness_name=report.harness_name,
                harness_version=report.harness_version,
                started_at=report.started_at,
                requested_task_ids=requested_task_ids,
                completed_task_ids=completed_task_ids,
                successful_tasks=sum(1 for task in report.tasks if task.success),
                failed_tasks=sum(1 for task in report.tasks if not task.success),
                completed=completed,
                metadata={
                    "config_source": report.config_source,
                    "dataset_name": report.dataset_name,
                    "dataset_split": report.dataset_split,
                    "dataset_revision": report.dataset_revision,
                    "dataset_fallback_revisions": list(report.dataset_fallback_revisions),
                    "split": report.split,
                    "model_config": copy.deepcopy(report.model_config),
                    "ablation_config": copy.deepcopy(report.ablation_config),
                    "active_task_ids": active_task_ids,
                    "active_tasks": active_tasks,
                },
            ),
        )

    def _active_task_ids(
        self,
        requested_task_ids: list[str],
        *,
        completed_task_ids: list[str],
    ) -> list[str]:
        return list(
            self._active_task_summaries(
                requested_task_ids,
                completed_task_ids=completed_task_ids,
            ).keys()
        )

    def _active_task_summaries(
        self,
        requested_task_ids: list[str],
        *,
        completed_task_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        completed = {str(task_id) for task_id in completed_task_ids}
        active: dict[str, dict[str, Any]] = {}
        now = time.time()
        for task_id in requested_task_ids:
            if task_id in completed:
                continue
            live_state = None
            task_output_rel = self._live_task_output_rel_by_id.get(task_id, task_id)
            for candidate_dir in (
                self.output_dir / task_output_rel,
                self.output_dir / task_id,
            ):
                live_state = load_json_if_exists(candidate_dir / "task_live_state.json")
                if isinstance(live_state, dict):
                    break
            if not isinstance(live_state, dict):
                continue
            if bool(live_state.get("terminal")):
                continue
            if str(live_state.get("status") or "") == "active":
                started_at = (
                    live_state.get("task_started_at")
                    or live_state.get("started_at")
                    or live_state.get("last_progress_at")
                )
                task_elapsed_seconds: Optional[float] = None
                if isinstance(started_at, (int, float)):
                    task_elapsed_seconds = max(0.0, now - float(started_at))
                last_progress_at = live_state.get("last_progress_at")
                seconds_since_last_progress: Optional[float] = None
                if isinstance(last_progress_at, (int, float)):
                    seconds_since_last_progress = max(0.0, now - float(last_progress_at))
                long_tail_threshold_seconds = 1800.0
                long_tail_slot_occupancy = (
                    task_elapsed_seconds is not None
                    and task_elapsed_seconds >= long_tail_threshold_seconds
                )
                active[task_id] = {
                    "task_id": live_state.get("task_id") or task_id,
                    "instance_id": live_state.get("instance_id") or task_id,
                    "phase": live_state.get("phase"),
                    "status": live_state.get("status"),
                    "stage": live_state.get("stage"),
                    "current_stage": live_state.get("current_stage"),
                    "current_rollout_id": live_state.get("current_rollout_id"),
                    "active_rollout_ids": live_state.get("active_rollout_ids"),
                    "active_rollout_count": live_state.get("active_rollout_count"),
                    "completed_rollout_count": live_state.get("completed_rollout_count"),
                    "error_rollout_count": live_state.get("error_rollout_count"),
                    "terminal_rollout_count": live_state.get("terminal_rollout_count"),
                    "total_rollout_count": live_state.get("total_rollout_count"),
                    "retry_count": live_state.get("retry_count"),
                    "retry_rollout_count": live_state.get("retry_rollout_count"),
                    "process_pid": live_state.get("process_pid"),
                    "last_progress_at": live_state.get("last_progress_at"),
                    "last_progress_source": live_state.get("last_progress_source"),
                    "task_started_at": started_at,
                    "task_elapsed_seconds": task_elapsed_seconds,
                    "seconds_since_last_progress": seconds_since_last_progress,
                    "slot_occupancy_seconds": task_elapsed_seconds,
                    "long_tail_slot_occupancy": long_tail_slot_occupancy,
                    "long_tail_threshold_seconds": long_tail_threshold_seconds,
                    "elapsed_seconds": task_elapsed_seconds,
                    "selected_candidate_id": live_state.get("selected_candidate_id"),
                    "selected_rollout_id": live_state.get("selected_rollout_id"),
                    "current_evaluation_phase": live_state.get("current_evaluation_phase"),
                }
        return active

    def _write_task_live_state(
        self,
        task_output_dir: Path,
        payload: dict[str, Any],
    ) -> Path:
        path = write_task_live_state(task_output_dir, payload)
        self._refresh_live_run_state_from_task_states()
        return path

    def _write_task_live_state_terminal(
        self,
        task_output_dir: Path,
        payload: dict[str, Any],
    ) -> Path:
        path = write_task_live_state_terminal(task_output_dir, payload)
        self._refresh_live_run_state_from_task_states()
        return path

    def _refresh_live_run_state_from_task_states(self) -> None:
        requested_task_ids = list(self._live_requested_task_ids or [])
        if not requested_task_ids:
            return
        state_path = self.output_dir / RUN_STATE_FILENAME
        state = load_json_if_exists(state_path)
        if not isinstance(state, dict):
            return
        if str(state.get("status") or "") == "completed":
            return
        completed_task_ids = [
            str(task_id)
            for task_id in (state.get("completed_task_ids") or [])
            if str(task_id or "").strip()
        ]
        active_tasks = self._active_task_summaries(
            requested_task_ids,
            completed_task_ids=completed_task_ids,
        )
        active_task_ids = list(active_tasks.keys())
        remaining_tasks = int(
            state.get("remaining_tasks")
            if state.get("remaining_tasks") is not None
            else max(0, len(requested_task_ids) - len(completed_task_ids))
        )
        state.update(
            {
                "updated_at": time.time(),
                "active_task_ids": active_task_ids,
                "active_tasks": active_tasks,
                "queued_task_count": max(0, remaining_tasks - len(active_task_ids)),
            }
        )
        metadata = state.get("metadata")
        if isinstance(metadata, dict):
            metadata["active_task_ids"] = active_task_ids
            metadata["active_tasks"] = active_tasks
        atomic_write_json(state_path, state)

    def _mark_run_interrupted(
        self,
        report: Commit0BenchmarkReport,
        requested_task_ids: list[str],
        exc: BaseException,
    ) -> None:
        now = time.time()
        termination: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "interrupted_at": now,
        }
        if isinstance(exc, SystemExit):
            termination["exit_code"] = exc.code
        update_run_manifest(
            self.output_dir,
            requested_task_ids=requested_task_ids,
            completed_task_ids=[task.instance_id for task in report.tasks],
            completed=False,
            extra_updates={
                "interrupted": True,
                "interrupted_at": now,
                "termination": termination,
            },
        )
        state_path = self.output_dir / RUN_STATE_FILENAME
        state = load_json_if_exists(state_path) or {}
        if isinstance(state, dict):
            state.update(
                {
                    "completed": False,
                    "interrupted": True,
                    "status": "stopped",
                    "updated_at": now,
                    "termination": termination,
                }
            )
            atomic_write_json(state_path, state)

    @staticmethod
    def _target_container_backend_preflight_key(
        item: Mapping[str, Any],
        *,
        scope: str = "",
    ) -> tuple[str, str, str, str]:
        return (
            str(scope or "").strip(),
            str(item.get("backend") or "").strip(),
            str(item.get("model") or "").strip(),
            str(item.get("command") or "").strip(),
        )

    @staticmethod
    def _target_container_backend_preflight_auth_scope() -> str:
        return "commit0-target-container-auth"

    @staticmethod
    def _target_container_backend_preflight_scope(task: Commit0Task) -> str:
        return (
            str(getattr(task, "instance_id", "") or "").strip()
            or str(getattr(task, "repo_name", "") or "").strip()
            or str(getattr(task, "repo", "") or "").strip()
        )

    @staticmethod
    def _commit0_optional_configured_cli_backends(config: ApexConfig) -> set[str]:
        raw_backends = getattr(config.benchmark, "commit0_optional_configured_cli_backends", [])
        if not isinstance(raw_backends, list):
            return set()
        return {
            str(getattr(backend, "value", backend) or "").strip()
            for backend in raw_backends
            if str(getattr(backend, "value", backend) or "").strip()
        }

    @staticmethod
    def _target_container_backend_preflight_failure_can_use_prior_success(
        reason: object,
    ) -> bool:
        text = str(reason or "").strip().lower()
        return (
            "did not complete target-container auth probe" in text
            or "did not complete target-container model-proxy smoke probe" in text
        )

    def _merge_target_container_backend_preflight_successes(
        self,
        snapshots: list[dict[str, Any]],
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        """Keep prior same-task health proof through transient auth-smoke timeouts."""

        merged: list[dict[str, Any]] = []
        with self._target_container_backend_preflight_lock:
            for item in snapshots:
                key = self._target_container_backend_preflight_key(item, scope=scope)
                if bool(item.get("healthy")) and all(key):
                    self._target_container_backend_preflight_healthy[key] = copy.deepcopy(item)
                    # Commit0 target-container auth is shared backend/runtime state;
                    # after per-container CLI lookup succeeds, auth-smoke timeouts
                    # are transient contention, not repo-specific unavailability.
                    auth_key = self._target_container_backend_preflight_key(
                        item,
                        scope=self._target_container_backend_preflight_auth_scope(),
                    )
                    if all(auth_key):
                        self._target_container_backend_preflight_healthy[auth_key] = copy.deepcopy(
                            item
                        )
            for item in snapshots:
                snapshot = dict(item)
                key = self._target_container_backend_preflight_key(snapshot, scope=scope)
                cached = self._target_container_backend_preflight_healthy.get(key)
                if (
                    cached is None
                    and not bool(snapshot.get("healthy"))
                    and self._target_container_backend_preflight_failure_can_use_prior_success(
                        snapshot.get("unavailable_reason")
                    )
                ):
                    auth_key = self._target_container_backend_preflight_key(
                        snapshot,
                        scope=self._target_container_backend_preflight_auth_scope(),
                    )
                    cached = self._target_container_backend_preflight_healthy.get(auth_key)
                if (
                    cached
                    and not bool(snapshot.get("healthy"))
                    and self._target_container_backend_preflight_failure_can_use_prior_success(
                        snapshot.get("unavailable_reason")
                    )
                ):
                    current_reason = snapshot.get("unavailable_reason")
                    snapshot = copy.deepcopy(cached)
                    snapshot["healthy"] = True
                    snapshot["unavailable_reason"] = ""
                    snapshot["preflight_cache_source"] = "prior_target_container_success"
                    snapshot["preflight_current_attempt_reason"] = current_reason
                merged.append(snapshot)
        return merged

    def _write_target_container_backend_preflight(
        self,
        *,
        task: Commit0Task,
        config: ApexConfig,
        task_output_dir: Path,
        persistent_task_output_dir: Path,
    ) -> dict[str, Any]:
        require_all = bool(config.benchmark.commit0_require_all_configured_cli_backends)
        attempts = _commit0_target_container_preflight_attempts()
        attempt_summaries: list[dict[str, Any]] = []
        configured_snapshots: list[dict[str, Any]] = []
        healthy: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        preflight_scope = self._target_container_backend_preflight_scope(task)
        optional_backends = self._commit0_optional_configured_cli_backends(config)
        for attempt_index in range(attempts):
            snapshots = build_allowed_backend_snapshots(config, refresh_health=True)
            current_configured_snapshots = [
                dict(item) for item in snapshots if bool(item.get("configured", True))
            ]
            configured_snapshots = self._merge_target_container_backend_preflight_successes(
                current_configured_snapshots,
                scope=preflight_scope,
            )
            healthy = [item for item in configured_snapshots if bool(item.get("healthy"))]
            unavailable = [item for item in configured_snapshots if not bool(item.get("healthy"))]
            required_unavailable = [
                item
                for item in unavailable
                if str(item.get("backend") or "").strip() not in optional_backends
            ]
            current_unavailable = [
                item for item in current_configured_snapshots if not bool(item.get("healthy"))
            ]
            attempt_summaries.append(
                {
                    "attempt": attempt_index + 1,
                    "healthy_backend_count": len(healthy),
                    "configured_backend_count": len(configured_snapshots),
                    "unavailable_backends": [
                        str(item.get("backend") or "").strip()
                        for item in unavailable
                        if str(item.get("backend") or "").strip()
                    ],
                    "current_attempt_unavailable_backends": [
                        str(item.get("backend") or "").strip()
                        for item in current_unavailable
                        if str(item.get("backend") or "").strip()
                    ],
                    "required_unavailable_backends": [
                        str(item.get("backend") or "").strip()
                        for item in required_unavailable
                        if str(item.get("backend") or "").strip()
                    ],
                }
            )
            if not required_unavailable or (healthy and not require_all):
                if not unavailable:
                    break
                # Commit0 target-container model transports can have brief
                # per-container startup races; retry before permanently pruning.
            if attempt_index + 1 < attempts and unavailable:
                time.sleep(min(2.0, 0.5 * float(attempt_index + 1)))
            else:
                break
        payload = {
            "task_id": task.repo_name,
            "instance_id": task.instance_id,
            "mode": "target_container_cli_backend_preflight",
            "generated_by_apex": True,
            "require_all_configured_backends": require_all,
            "optional_configured_backends": sorted(optional_backends),
            "preflight_scope": preflight_scope,
            "preflight_attempt_count": len(attempt_summaries),
            "preflight_attempts": attempt_summaries,
            "healthy_backend_count": len(healthy),
            "configured_backend_count": len(configured_snapshots),
            "backend_health": configured_snapshots,
            "unavailable_configured_backends": [
                {
                    "backend": item.get("backend"),
                    "model": item.get("model"),
                    "command": item.get("command"),
                    "reason": item.get("unavailable_reason"),
                }
                for item in unavailable
            ],
            "unavailable_required_configured_backends": [
                {
                    "backend": item.get("backend"),
                    "model": item.get("model"),
                    "command": item.get("command"),
                    "reason": item.get("unavailable_reason"),
                }
                for item in unavailable
                if str(item.get("backend") or "").strip() not in optional_backends
            ],
            "unavailable_optional_configured_backends": [
                {
                    "backend": item.get("backend"),
                    "model": item.get("model"),
                    "command": item.get("command"),
                    "reason": item.get("unavailable_reason"),
                }
                for item in unavailable
                if str(item.get("backend") or "").strip() in optional_backends
            ],
        }
        for directory in (task_output_dir, persistent_task_output_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "target_container_backend_preflight.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                logger.debug(
                    "Failed to write Commit0 target-container backend preflight artifact",
                    exc_info=True,
                )
        if not healthy:
            reasons = "; ".join(
                str(item.get("unavailable_reason") or item.get("backend") or "unknown")
                for item in configured_snapshots[:3]
            )
            raise RuntimeError(
                "No configured CLI backend passed the Commit0 target-container "
                f"preflight for {task.repo_name}: {reasons or 'no backends configured'}"
            )
        required_unavailable = [
            item
            for item in unavailable
            if str(item.get("backend") or "").strip() not in optional_backends
        ]
        if require_all and required_unavailable:
            failed = ", ".join(
                f"{item.get('backend') or 'unknown'}:{item.get('model') or 'unknown'}"
                for item in required_unavailable
            )
            reasons = "; ".join(
                str(
                    item.get("unavailable_reason")
                    or item.get("backend")
                    or "unknown unavailable backend"
                )
                for item in required_unavailable[:3]
            )
            raise RuntimeError(
                "Configured CLI backends failed the Commit0 target-container "
                f"preflight for {task.repo_name}: {failed}. {reasons}"
            )
        return payload

    @staticmethod
    def _llm_config_backend_value(llm_config: Any) -> str:
        backend = getattr(llm_config, "backend", "")
        return str(getattr(backend, "value", backend) or "").strip()

    @staticmethod
    def _llm_config_identity_payload(llm_config: Any, *, index: int) -> dict[str, Any]:
        return {
            "index": int(index),
            "backend": Commit0BenchmarkRunner._llm_config_backend_value(llm_config),
            "model": str(getattr(llm_config, "model", "") or ""),
            "cli_model_id": str(getattr(llm_config, "cli_model_id", "") or ""),
            "command": (
                str(getattr(llm_config, "resolved_cli_command", "") or "")
                if bool(getattr(llm_config, "is_cli_backend", False))
                else ""
            ),
        }

    @staticmethod
    def _anchor_candidate_matches_llm_config(
        candidate: dict[str, Any],
        llm_config: Any,
        *,
        index: int,
    ) -> bool:
        if "llm_config_index" in candidate:
            try:
                if int(candidate["llm_config_index"]) != int(index):
                    return False
            except (TypeError, ValueError):
                return False
        backend = str(candidate.get("backend") or "").strip()
        if backend and backend != Commit0BenchmarkRunner._llm_config_backend_value(llm_config):
            return False
        model = str(candidate.get("model") or "").strip()
        if model and model != str(getattr(llm_config, "model", "") or "").strip():
            return False
        cli_model_id = str(candidate.get("cli_model_id") or "").strip()
        if cli_model_id and cli_model_id != str(getattr(llm_config, "cli_model_id", "") or ""):
            return False
        return True

    @staticmethod
    def _anchor_candidate_for_llm_config(llm_config: Any) -> dict[str, Any]:
        backend = Commit0BenchmarkRunner._llm_config_backend_value(llm_config)
        model = str(getattr(llm_config, "model", "") or "").strip()
        candidate: dict[str, Any] = {
            "backend": backend,
            "model": model,
            "label": f"{backend}:{model}" if model else backend,
            "harness": "cli_agent",
        }
        cli_model_id = str(getattr(llm_config, "cli_model_id", "") or "").strip()
        if cli_model_id:
            candidate["cli_model_id"] = cli_model_id
        return candidate

    def _prune_unhealthy_target_container_backend_routes(
        self,
        *,
        task: Commit0Task,
        config: ApexConfig,
        preflight_payload: dict[str, Any],
        task_output_dir: Path,
        persistent_task_output_dir: Path,
    ) -> dict[str, Any]:
        old_llm_configs = list(getattr(config, "llm_configs", []) or [])
        backend_health = [
            dict(item)
            for item in list(preflight_payload.get("backend_health") or [])
            if isinstance(item, dict) and bool(item.get("configured", True))
        ]
        healthy_backends = {
            str(item.get("backend") or "").strip()
            for item in backend_health
            if bool(item.get("healthy"))
        }
        unavailable_backends = {
            str(item.get("backend") or "").strip()
            for item in backend_health
            if not bool(item.get("healthy"))
        }
        configured_backends = healthy_backends | unavailable_backends
        require_all = bool(preflight_payload.get("require_all_configured_backends"))
        optional_backends = {
            str(item or "").strip()
            for item in list(preflight_payload.get("optional_configured_backends") or [])
            if str(item or "").strip()
        } or self._commit0_optional_configured_cli_backends(config)
        optional_unavailable_backends = unavailable_backends & optional_backends
        required_unavailable_backends = unavailable_backends - optional_backends
        prunable_unavailable_backends = (
            optional_unavailable_backends if require_all else unavailable_backends
        )
        report: dict[str, Any] = {
            "task_id": task.repo_name,
            "instance_id": task.instance_id,
            "mode": "target_container_cli_backend_route_pruning",
            "generated_by_apex": True,
            "applied": False,
            "reason": "",
            "require_all_configured_backends": require_all,
            "optional_configured_backends": sorted(optional_backends),
            "healthy_backends": sorted(healthy_backends),
            "unavailable_backends": sorted(unavailable_backends),
            "required_unavailable_backends": sorted(required_unavailable_backends),
            "optional_unavailable_backends": sorted(optional_unavailable_backends),
            "original_llm_config_count": len(old_llm_configs),
            "pruned_llm_config_count": 0,
            "kept_llm_config_indices": list(range(len(old_llm_configs))),
            "pruned_llm_config_indices": [],
        }

        if require_all and required_unavailable_backends:
            report["reason"] = "require_all_configured_backends"
            self._write_target_container_backend_route_pruning_artifact(
                report,
                task_output_dir=task_output_dir,
                persistent_task_output_dir=persistent_task_output_dir,
            )
            return report
        if not prunable_unavailable_backends or not configured_backends:
            report["reason"] = "no_unavailable_configured_backends"
            self._write_target_container_backend_route_pruning_artifact(
                report,
                task_output_dir=task_output_dir,
                persistent_task_output_dir=persistent_task_output_dir,
            )
            return report

        keep_old_indices: list[int] = []
        pruned_old_indices: list[int] = []
        for index, llm_config in enumerate(old_llm_configs):
            backend = self._llm_config_backend_value(llm_config)
            if (
                bool(getattr(llm_config, "is_cli_backend", False))
                and backend in configured_backends
                and backend in prunable_unavailable_backends
            ):
                pruned_old_indices.append(index)
            else:
                keep_old_indices.append(index)

        if not pruned_old_indices:
            report["reason"] = "no_configured_routes_needed_pruning"
            self._write_target_container_backend_route_pruning_artifact(
                report,
                task_output_dir=task_output_dir,
                persistent_task_output_dir=persistent_task_output_dir,
            )
            return report
        if not keep_old_indices:
            report["reason"] = "all_routes_would_be_pruned"
            self._write_target_container_backend_route_pruning_artifact(
                report,
                task_output_dir=task_output_dir,
                persistent_task_output_dir=persistent_task_output_dir,
            )
            return report

        old_to_new = {old_index: new_index for new_index, old_index in enumerate(keep_old_indices)}
        old_stage_indices = dict(getattr(config.rollout, "scaffold_stage_llm_indices", {}) or {})
        new_stage_indices: dict[str, int] = {}
        dropped_stage_indices: dict[str, Any] = {}
        for stage_name, raw_index in old_stage_indices.items():
            try:
                old_index = int(raw_index)
            except (TypeError, ValueError):
                dropped_stage_indices[str(stage_name)] = raw_index
                continue
            if old_index in old_to_new:
                new_stage_indices[str(stage_name)] = old_to_new[old_index]
            else:
                dropped_stage_indices[str(stage_name)] = old_index

        old_profiles = [
            dict(profile)
            for profile in list(getattr(config.rollout, "llm_profiles", []) or [])
            if isinstance(profile, dict)
        ]
        new_profiles: list[dict[str, int]] = []
        old_profile_to_new: dict[int, int] = {}
        dropped_profile_indices: list[int] = []
        for profile_index, profile in enumerate(old_profiles):
            remapped_profile: dict[str, int] = {}
            drop_profile = False
            for key, raw_index in profile.items():
                try:
                    old_index = int(raw_index)
                except (TypeError, ValueError):
                    drop_profile = True
                    break
                if old_index not in old_to_new:
                    drop_profile = True
                    break
                remapped_profile[str(key)] = old_to_new[old_index]
            if drop_profile:
                dropped_profile_indices.append(profile_index)
                continue
            old_profile_to_new[profile_index] = len(new_profiles)
            new_profiles.append(remapped_profile)

        old_always_include = list(getattr(config.rollout, "always_include_profiles", []) or [])
        new_always_include: list[int] = []
        for raw_profile_index in old_always_include:
            try:
                old_profile_index = int(raw_profile_index)
            except (TypeError, ValueError):
                continue
            if old_profile_index in old_profile_to_new:
                new_always_include.append(old_profile_to_new[old_profile_index])
        if old_always_include and new_profiles and not new_always_include:
            new_always_include = [0]
        if not new_profiles:
            new_always_include = []

        old_anchor_candidates = [
            dict(candidate)
            for candidate in list(getattr(config.rollout, "standalone_anchor_candidates", []) or [])
            if isinstance(candidate, dict)
        ]
        new_anchor_candidates: list[dict[str, Any]] = []
        dropped_anchor_candidates: list[dict[str, Any]] = []
        for candidate in old_anchor_candidates:
            matched_kept_index: Optional[int] = None
            for old_index in keep_old_indices:
                if self._anchor_candidate_matches_llm_config(
                    candidate,
                    old_llm_configs[old_index],
                    index=old_index,
                ):
                    matched_kept_index = old_index
                    break
            if matched_kept_index is None:
                dropped_anchor_candidates.append(dict(candidate))
                continue
            remapped_candidate = dict(candidate)
            if "llm_config_index" in remapped_candidate:
                remapped_candidate["llm_config_index"] = old_to_new[matched_kept_index]
            new_anchor_candidates.append(remapped_candidate)

        if old_anchor_candidates and not new_anchor_candidates:
            for old_index in keep_old_indices:
                llm_config = old_llm_configs[old_index]
                if not bool(getattr(llm_config, "is_cli_backend", False)):
                    continue
                new_anchor_candidates.append(self._anchor_candidate_for_llm_config(llm_config))

        old_anchor_profile_index = int(
            max(0, int(getattr(config.rollout, "standalone_anchor_profile_index", 0) or 0))
        )
        if old_profile_to_new:
            new_anchor_profile_index = old_profile_to_new.get(old_anchor_profile_index, 0)
        elif old_anchor_profile_index in old_to_new:
            new_anchor_profile_index = old_to_new[old_anchor_profile_index]
        else:
            new_anchor_profile_index = 0

        planning_llm_indices_before = {
            "planner_llm_index": getattr(config.planning, "planner_llm_index", None),
            "preplanner_llm_index": getattr(config.planning, "preplanner_llm_index", None),
        }
        planning_llm_indices_after: dict[str, Optional[int]] = {}
        for attr_name, raw_index in planning_llm_indices_before.items():
            if not isinstance(raw_index, int):
                planning_llm_indices_after[attr_name] = raw_index
                continue
            if raw_index in old_to_new:
                remapped_index = old_to_new[raw_index]
            else:
                remapped_index = 0
            planning_llm_indices_after[attr_name] = remapped_index

        old_portfolio_seed_profile_count = int(
            getattr(config.rollout, "portfolio_seed_profile_count", 0) or 0
        )
        # Commit0 target-runtime CLIs are optional portfolio inputs; unavailable
        # target-container routes are recorded and skipped instead of consuming rollouts.
        config.llm_configs = [old_llm_configs[index] for index in keep_old_indices]
        config.rollout.scaffold_stage_llm_indices = new_stage_indices
        config.rollout.llm_profiles = new_profiles
        config.rollout.portfolio_seed_profile_count = min(
            old_portfolio_seed_profile_count,
            len(new_profiles),
        )
        config.rollout.always_include_profiles = new_always_include
        config.rollout.standalone_anchor_candidates = new_anchor_candidates
        config.rollout.standalone_anchor_profile_index = new_anchor_profile_index
        config.planning.planner_llm_index = planning_llm_indices_after["planner_llm_index"]
        config.planning.preplanner_llm_index = planning_llm_indices_after["preplanner_llm_index"]

        report.update(
            {
                "applied": True,
                "reason": "unavailable_optional_configured_backends_pruned",
                "pruned_llm_config_count": len(pruned_old_indices),
                "kept_llm_config_indices": keep_old_indices,
                "pruned_llm_config_indices": pruned_old_indices,
                "remapped_llm_config_indices": {
                    str(old_index): new_index for old_index, new_index in old_to_new.items()
                },
                "kept_llm_configs": [
                    self._llm_config_identity_payload(
                        old_llm_configs[index],
                        index=old_to_new[index],
                    )
                    for index in keep_old_indices
                ],
                "pruned_llm_configs": [
                    self._llm_config_identity_payload(old_llm_configs[index], index=index)
                    for index in pruned_old_indices
                ],
                "scaffold_stage_llm_indices_before": old_stage_indices,
                "scaffold_stage_llm_indices_after": new_stage_indices,
                "dropped_scaffold_stage_llm_indices": dropped_stage_indices,
                "llm_profile_count_before": len(old_profiles),
                "llm_profile_count_after": len(new_profiles),
                "dropped_llm_profile_indices": dropped_profile_indices,
                "portfolio_seed_profile_count_before": old_portfolio_seed_profile_count,
                "portfolio_seed_profile_count_after": config.rollout.portfolio_seed_profile_count,
                "always_include_profiles_before": old_always_include,
                "always_include_profiles_after": new_always_include,
                "standalone_anchor_candidate_count_before": len(old_anchor_candidates),
                "standalone_anchor_candidate_count_after": len(new_anchor_candidates),
                "dropped_standalone_anchor_candidates": dropped_anchor_candidates,
                "standalone_anchor_profile_index_before": old_anchor_profile_index,
                "standalone_anchor_profile_index_after": new_anchor_profile_index,
                "planning_llm_indices_before": planning_llm_indices_before,
                "planning_llm_indices_after": planning_llm_indices_after,
            }
        )
        self._write_target_container_backend_route_pruning_artifact(
            report,
            task_output_dir=task_output_dir,
            persistent_task_output_dir=persistent_task_output_dir,
        )
        return report

    def _write_target_container_backend_route_pruning_artifact(
        self,
        report: dict[str, Any],
        *,
        task_output_dir: Path,
        persistent_task_output_dir: Path,
    ) -> None:
        for directory in (task_output_dir, persistent_task_output_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "target_container_backend_route_pruning.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                logger.debug(
                    "Failed to write Commit0 target-container backend route-pruning artifact",
                    exc_info=True,
                )

    def _run_task(self, task: Commit0Task) -> Commit0TaskResult:
        started = time.time()
        execution_layout = self._build_task_execution_layout(task)
        repo_dir = execution_layout.repo_dir
        task_output_dir = execution_layout.task_output_dir
        task_workspace_dir = execution_layout.workspace_dir
        runtime_dir = execution_layout.runtime_dir
        persistent_task_output_dir = self._task_output_dir(task)
        persistent_task_output_dir.mkdir(parents=True, exist_ok=True)
        persistent_workspace_dir = self._task_workspace_dir(task)
        retain_task_workspaces = bool(self.config.rollout.keep_worktrees)
        task_result: Optional[Commit0TaskResult] = None
        result: Optional[Any] = None
        phase = "pre_orchestrator"
        orchestrator_reached = False
        container_name: Optional[str] = None

        try:
            self._write_task_live_state(
                persistent_task_output_dir,
                {
                    "task_id": task.repo_name,
                    "instance_id": task.instance_id,
                    "phase": "prepare_repo",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            self._write_task_live_state(
                persistent_task_output_dir,
                {
                    "task_id": task.repo_name,
                    "instance_id": task.instance_id,
                    "phase": "baseline_eval",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            expected_test_ids = [
                test_id for test_id in _load_expected_test_ids(task.repo_name) if test_id
            ]
            if not expected_test_ids and _commit0_expected_id_scoring_required(self.config):
                # Commit0 gold scoring is defined by the expected pytest-id inventory;
                # falling back to pytest summary would silently change the scored universe.
                baseline = Commit0Evaluation(
                    returncode=1,
                    output="Commit0 expected-test-id inventory unavailable.",
                    raw_returncode=1,
                    scoring_source="commit0_test_ids",
                    evaluation_backend=COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST,
                    expected_test_coverage={
                        "expected_test_count": 0,
                        "matched_expected_test_count": 0,
                        "missing_expected_test_count": 0,
                        "coverage_preserved": False,
                        "inventory_unavailable": True,
                    },
                    diagnostics={
                        "harness_failure": True,
                        "expected_test_inventory_unavailable": True,
                        "scoring_universe": "expected_test_ids",
                    },
                )
                _commit0_evaluation_decision(baseline)
                return self._emit_preflight_skip(
                    task=task,
                    category="expected_test_inventory_unavailable",
                    details={
                        "expected_test_count": 0,
                        "scoring_universe": "expected_test_ids",
                    },
                    baseline=baseline,
                    task_output_dir=task_output_dir,
                    persistent_task_output_dir=persistent_task_output_dir,
                    started=started,
                )
            (
                env,
                baseline,
                venv_python,
                container_name,
                used_docker_fallback,
            ) = self._prepare_and_baseline_with_docker_fallback(
                task,
                repo_dir,
                runtime_dir,
                task_output_dir,
                expected_test_ids,
            )
            if container_name:
                _install_commit0_prepared_runtime_container_guards(
                    container_name=container_name,
                    docker_bin=shutil.which("docker") or "docker",
                    docker_env=_resolve_docker_sdk_env(),
                    container_venv=str(env.get("APEX_COMMIT0_CONTAINER_VENV") or ""),
                )
            rollout_report_file = _APEX_ROLLOUT_REPORT_FILENAME
            rollout_python_executable = str(venv_python)
            if env.get("APEX_COMMIT0_DOCKER_CONTAINER"):
                # Commit0 Docker rollouts execute through target-runtime shims inside
                # the container, so the prompt must not expose the host docker wrapper.
                rollout_python_executable = "python"
            rollout_test_command = self._build_test_command(
                task,
                python_executable=rollout_python_executable,
                report_file=rollout_report_file,
                expected_test_ids=expected_test_ids if expected_test_ids else None,
                repo_dir=repo_dir if expected_test_ids else None,
                # Commit0/Python solve-phase pytest is the agent/QV command; it
                # is bounded by outer task lanes, not all possible rollout slots.
                xdist_context="rollout",
            )
            (task_output_dir / "baseline_metrics.json").write_text(
                json.dumps(baseline.to_dict(), indent=2)
            )
            if used_docker_fallback:
                (task_output_dir / "docker_fallback.json").write_text(
                    json.dumps(
                        {
                            "fallback_triggered": True,
                            "container_name": container_name,
                            "reason": "host_baseline_failure_or_signature",
                        },
                        indent=2,
                    )
                )

            preflight_skip = self._preflight_block_or_none(
                task=task,
                baseline=baseline,
                baseline_eval_dir=task_output_dir / "baseline_eval",
                expected_test_ids=expected_test_ids,
                task_output_dir=task_output_dir,
                persistent_task_output_dir=persistent_task_output_dir,
                started=started,
                repo_dir=repo_dir,
            )
            if preflight_skip is not None:
                return preflight_skip

            config = copy.deepcopy(self.config)
            config.output_dir = str(task_output_dir)
            config.workspace_dir = str(task_workspace_dir)
            # Commit0 expected-ID pytest runner/plugin are ignored harness files;
            # seed/repair worktrees can lose them, so selector verification
            # re-materializes them outside candidate diffs.
            config.selection.verification_helper_files = [
                _APEX_EXPECTED_IDS_FILENAME,
                _APEX_EXPECTED_IDS_MIRROR_FILENAME,
                _APEX_EXPECTED_IDS_PLUGIN_FILENAME,
                _APEX_EXPECTED_IDS_RUNNER_FILENAME,
                _APEX_LOCAL_EVAL_REPORT_FILENAME,
            ]
            if env.get("APEX_COMMIT0_DOCKER_CONTAINER"):
                # Commit0 Linux fallback bind-mounts only the per-task sandbox root;
                # keep rollout worktrees inside that mounted root for in-container agents.
                config.rollout.shared_workspace = True
            target_cli_auth_mode = _commit0_target_cli_auth_mode(self.config)
            if target_cli_auth_mode.lower() in _COMMIT0_DOCKER_IMAGE_AUTH_MODES:
                # Commit0 workdir-only agent containers mount a single rollout worktree;
                # use one-commit snapshots so git status/diff work without exposing history.
                config.rollout.use_git_worktrees = False
                config.rollout.use_worktree_pool = False
                config.rollout.historyless_snapshots = True
            runtime_cli_env = _runtime_cli_env_overrides(env)
            # V2 anti-cheat: advertise the SOLVE-phase boundary and disable local
            # package caches. Source/network denial is structural: the Docker
            # container has already been moved to an internal network, with only
            # model transport exposed through the egress sidecar.
            solve_phase_env = _commit0_solve_phase_proxy_env(runtime_cli_env)
            runtime_cli_env.update(solve_phase_env)
            # Commit0/Python solve fact: exploratory agent pytest runs can emit
            # megabytes of tracebacks; final expected-ID scoring runs separately.
            runtime_cli_env = _commit0_apply_solve_phase_pytest_output_env(runtime_cli_env)
            # Keep docker-exec target-tool subprocesses under the same solve
            # network guard as the top-level CLI backend process.
            env.update(solve_phase_env)
            agent_target_tool_timeout = str(
                self._commit0_agent_target_tool_timeout_seconds(
                    task,
                    expected_test_count=len(expected_test_ids),
                )
            )
            runtime_cli_env[_COMMIT0_AGENT_COMMAND_TIMEOUT_ENV] = agent_target_tool_timeout
            env[_COMMIT0_AGENT_COMMAND_TIMEOUT_ENV] = agent_target_tool_timeout
            if target_cli_auth_mode:
                runtime_cli_env["APEX_TARGET_RUNTIME_CLI_AUTH_MODE"] = target_cli_auth_mode
                if target_cli_auth_mode.lower() in _COMMIT0_MODEL_PROXY_AUTH_MODES:
                    runtime_cli_env["APEX_TARGET_RUNTIME_REQUIRE_MODEL_PROXY"] = "1"
            if env.get("APEX_COMMIT0_DOCKER_CONTAINER"):
                if target_cli_auth_mode.lower() in (
                    _COMMIT0_DOCKER_IMAGE_AUTH_MODES
                    | _COMMIT0_DOCKER_SANDBOX_AUTH_MODES
                    | _COMMIT0_MODEL_PROXY_AUTH_MODES
                ):
                    # Commit0 gold solve fact: the official image plus a worktree-only
                    # bind mount gives agents normal Python/Linux tools without sibling checkouts.
                    image_runtime_env = _commit0_docker_exec_runtime_env(env)
                    image_runtime_env["PYTHONPATH"] = self._commit0_container_workdir_pythonpath(
                        task
                    )
                    agent_cli_bundle_host = str(
                        env.get("APEX_COMMIT0_AGENT_CLI_BUNDLE_HOST_ROOT") or ""
                    ).strip()
                    image_mounts: list[dict[str, str]] = []
                    if agent_cli_bundle_host:
                        image_mounts.append(
                            {
                                "source": agent_cli_bundle_host,
                                "target": _COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT,
                                "readonly": "true",
                            }
                        )
                    if (
                        self._commit0_pytest_xdist_worker_spec()
                        and task.repo_name not in self._commit0_pytest_xdist_disabled_repos
                    ):
                        xdist_vendor_mounts = self._commit0_pytest_xdist_vendor_mounts()
                        if xdist_vendor_mounts:
                            image_mounts.extend(xdist_vendor_mounts)
                            image_runtime_env["APEX_COMMIT0_PYTEST_XDIST_VENDOR_ROOT"] = (
                                _COMMIT0_PYTEST_XDIST_VENDOR_CONTAINER_ROOT
                            )
                        else:
                            self._commit0_pytest_xdist_disabled_repos.add(task.repo_name)
                    # Commit0 agent containers can reuse a cached superset CLI bundle;
                    # expose only configured binaries through the runtime PATH.
                    agent_cli_view_setup = ""
                    if agent_cli_bundle_host:
                        selected_binaries = tuple(
                            str(env.get("APEX_COMMIT0_AGENT_CLI_SELECTED_BINARIES") or "")
                            .strip()
                            .split()
                        )
                        agent_cli_view_setup = self._commit0_agent_cli_filtered_view_command(
                            source_bundle_container_path=(
                                _COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT
                            ),
                            filtered_bundle_container_path=(
                                _COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT
                            ),
                            selected_binaries=selected_binaries,
                        )
                    docker_runtime = docker_image_runtime(
                        image=str(env.get("APEX_COMMIT0_RUNTIME_IMAGE") or ""),
                        docker_workdir=_COMMIT0_DOCKER_WORKSPACE_ROOT,
                        # Commit0 official images are linux/amd64; pin fresh
                        # solve containers so host-arch warnings do not fail rollouts.
                        docker_platform=str(
                            env.get("APEX_COMMIT0_RUNTIME_PLATFORM")
                            or _COMMIT0_OFFICIAL_IMAGE_PLATFORM
                        ),
                        docker_network=str(env.get("APEX_COMMIT0_SOLVE_NETWORK") or "none"),
                        docker_bin=shutil.which("docker") or "docker",
                        docker_user=str(env.get("APEX_COMMIT0_DOCKER_USER") or ""),
                        docker_host_env=_resolve_docker_sdk_env(),
                        docker_env=image_runtime_env,
                        docker_mounts=image_mounts,
                        # Commit0 official uv images can symlink venv Python into /root/.local; fresh workdir-only containers must fix traversal before UID drop.
                        docker_root_setup_script=(
                            _commit0_official_image_root_setup_script(
                                str(
                                    env.get("APEX_COMMIT0_CONTAINER_VENV")
                                    or _COMMIT0_OFFICIAL_TESTBED_VENV
                                ),
                                repair_python_env=(
                                    _commit0_official_image_python_repair_required(task.repo_name)
                                ),
                            )
                            + "\n"
                            + agent_cli_view_setup
                        ),
                        description="commit0_official_image_workdir_runtime",
                    )
                else:
                    docker_runtime = docker_exec_runtime(
                        container_name=str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or ""),
                        host_workdir_root=repo_dir.parent,
                        container_workdir_root=_COMMIT0_DOCKER_WORKSPACE_ROOT,
                        docker_bin=shutil.which("docker") or "docker",
                        docker_env=_commit0_docker_exec_runtime_env(env),
                        # Commit0 Docker contexts may point at a Desktop/Colima socket;
                        # keep that host Docker-client setting outside the agent container.
                        docker_host_env=_resolve_docker_sdk_env(),
                        docker_user=str(env.get("APEX_COMMIT0_DOCKER_USER") or ""),
                        # Commit0 Docker solve-phase wraps /bin/sh for agent policy; APEX control-plane scans use the preserved real shell.
                        docker_control_shell="/bin/sh.apex-real",
                        # Commit0 Linux fallback uses a long-lived benchmark container;
                        # docker exec keeps scaffolded agents inside the evaluated repo sandbox.
                        description="commit0_linux_docker_exec_runtime",
                    )
                target_tool_output_dir = task_workspace_dir / "_target_runtime_tools"
            else:
                docker_runtime = host_env_runtime(env, description="commit0_local_venv_runtime")
                target_tool_output_dir = task_output_dir / "target_runtime_tools"
            target_tool_env, target_tool_diagnostics = target_tool_env_overrides(
                workdir=repo_dir,
                output_dir=target_tool_output_dir,
                timeout_seconds=self._commit0_evaluation_timeout_seconds(task),
                # Commit0/Python execution fact: official full-suite scoring can
                # take much longer than exploratory agent-run pytest commands.
                agent_command_timeout_seconds=(
                    self._commit0_agent_target_tool_timeout_seconds(
                        task,
                        expected_test_count=len(expected_test_ids),
                    )
                ),
                # Commit0/Python fact: pytest/build output can be enormous; keep
                # command exit codes real while bounding agent-visible streams.
                output_capture_max_chars=_COMMIT0_AGENT_TARGET_OUTPUT_CAPTURE_MAX_CHARS,
                runtime=docker_runtime,
                label=f"commit0_{task.repo_name}",
                # Commit0 solve egress is container/proxy-enforced; do not turn
                # dependency/source attempts into command-text policy failures.
                command_policy_blocks=False,
                # Commit0 solve worktrees are flattened to a root commit before
                # rollout, so git history is structurally absent instead of
                # prompt/policy-blocked.
                git_history_policy="structurally_erased",
                # Commit0 solve containers leave the Docker bridge after setup;
                # source/package egress is denied by the sidecar network ACL.
                source_network_policy="structurally_denied",
                # Commit0 solve containers mount only the task sandbox and APEX
                # runtime bundle, so filesystem provenance is container-enforced.
                filesystem_boundary_policy="structurally_isolated",
            )
            runtime_cli_env.update(target_tool_env)
            try:
                egress_proxy_mappings = json.loads(
                    env.get(_COMMIT0_EGRESS_PROXY_MAPPINGS_ENV, "[]") or "[]"
                )
            except json.JSONDecodeError:
                egress_proxy_mappings = []
            target_tool_diagnostics["solve_phase_network_boundary"] = {
                "mode": (
                    "docker_internal_network_with_model_proxy_sidecar"
                    if env.get("APEX_COMMIT0_DOCKER_CONTAINER")
                    else "host_environment"
                ),
                "internal_network": env.get("APEX_COMMIT0_SOLVE_NETWORK", ""),
                "egress_proxy_alias": _COMMIT0_EGRESS_PROXY_ALIAS,
                "egress_proxy_mappings": egress_proxy_mappings,
                "physical_preflight": env.get(_COMMIT0_SOLVE_NETWORK_PREFLIGHT_ENV, ""),
                "command_policy_blocks": False,
                "filesystem_policy_blocks": False,
            }
            (task_output_dir / "target_runtime_tools.json").write_text(
                json.dumps(target_tool_diagnostics, indent=2) + "\n",
                encoding="utf-8",
            )
            apply_target_tool_env_to_apex_config(config, runtime_cli_env)
            configured_cli_backends = any(
                bool(getattr(llm_config, "is_cli_backend", False))
                for llm_config in list(getattr(config, "llm_configs", []) or [])
            )
            # A3: run backend preflight + route-pruning for ANY CLI-backend run,
            # not only docker-container runs. The host-venv CLI path is just as
            # vulnerable to dispatching a sampled rollout profile to a dead or
            # unauthenticated backend (web3.py run-33). Preflight is the only
            # enforcement that at least one healthy backend exists, and pruning's
            # ``all_routes_would_be_pruned`` guard is the defensive backstop, so
            # they stay coupled. ``probe_cli_backend_health`` works on the host
            # path (``_probe_cli_backend_health_on_host``) as well as in-container.
            if configured_cli_backends:
                self._write_task_live_state(
                    persistent_task_output_dir,
                    {
                        "task_id": task.repo_name,
                        "instance_id": task.instance_id,
                        "phase": "target_backend_preflight",
                        "status": "active",
                        "process_pid": os.getpid(),
                        "last_progress_at": time.time(),
                    },
                )
                preflight_payload = self._write_target_container_backend_preflight(
                    task=task,
                    config=config,
                    task_output_dir=task_output_dir,
                    persistent_task_output_dir=persistent_task_output_dir,
                )
                self._prune_unhealthy_target_container_backend_routes(
                    task=task,
                    config=config,
                    preflight_payload=preflight_payload,
                    task_output_dir=task_output_dir,
                    persistent_task_output_dir=persistent_task_output_dir,
                )
            keep_worktrees = config.rollout.keep_worktrees
            config.rollout.keep_worktrees = True
            _apply_task_complexity_rollout_budget(
                config,
                expected_test_count=len(expected_test_ids),
            )
            benchmark_metadata = {
                "benchmark_name": "commit0",
                "protect_visible_test_files": True,
                "evidence_mode": "gold_suite_visible",
                "test_suite_evidence_mode": "gold_suite_visible",
                # WS3E: stable per-task identity for cross-solve episodic memory
                # (only consumed when RolloutConfig.enable_cross_solve_episodic_memory).
                "instance_id": task.instance_id,
                "task_id": task.instance_id,
            }
            if expected_test_ids:
                benchmark_metadata["expected_test_count"] = len(expected_test_ids)
                benchmark_metadata["expected_test_ids"] = list(expected_test_ids)
                benchmark_metadata["expected_test_ids_file"] = _APEX_EXPECTED_IDS_FILENAME
                benchmark_metadata["test_inventory_source"] = "commit0_public_test_inventory"
                collection_command = _commit0_pytest_collection_command(rollout_test_command)
                if collection_command:
                    benchmark_metadata["test_inventory_collection_command"] = collection_command
                # TIER 2 (T2.4): per-module expected-id mapper for the planner's
                # decomposition-scale module groups. Layer-B closure: it carries
                # the node-id<->package ecosystem fact so Layer-A orchestration
                # never needs it. The planner reads this locally and never
                # serializes it.
                benchmark_metadata["module_group_expected_id_mapper"] = (
                    _make_module_group_expected_id_mapper(task.repo_name, expected_test_ids)
                )
                # Preferred over the per-group mapper above: a single global
                # partition that keeps the per-group subsets DISJOINT (the mapper
                # double-assigns when groups co-own a subpackage). The planner
                # prefers this when present and falls back to the mapper.
                benchmark_metadata["module_group_expected_id_partitioner"] = (
                    _make_module_group_expected_id_partitioner(task.repo_name, expected_test_ids)
                )
            # Phase A.1 (Decisive-Edge): when the orchestrator routes
            # through the V5 in-container agent surface (per
            # ``BenchmarkConfig.default_agent_mode``), the per-task
            # Linux runtime image is the ContainerSupervisor target.
            # ``_linux_runtime_container_image`` already resolves the
            # pinned digest when one is recorded in
            # ``configs/docker_image_digests.json``.
            try:
                benchmark_metadata["docker_image"] = self._linux_runtime_container_image(task)
            except Exception:
                # Image resolution should never fail benchmark setup;
                # the V5 dispatch will fall back to the host shim.
                pass

            orchestrator = ApexOrchestrator(config)
            self._write_task_live_state(
                persistent_task_output_dir,
                {
                    "task_id": task.repo_name,
                    "instance_id": task.instance_id,
                    "phase": "solving",
                    "status": "active",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                },
            )
            issue_description = task.build_issue_description(
                rollout_test_command,
                expected_test_count=len(expected_test_ids) if expected_test_ids else None,
                expected_test_ids=expected_test_ids if expected_test_ids else None,
            )
            # Capture the realized prompt content hash so reviewers can
            # audit fairness claims (e.g. "no required-test names leaked
            # into the prompt"). Stored alongside the per-task result so
            # comparisons across runs can verify identical prompt surface.
            (task_output_dir / "issue_prompt.json").write_text(
                json.dumps(
                    _artifact_safe_issue_prompt_payload(
                        issue_description=issue_description,
                        test_command=rollout_test_command,
                    ),
                    indent=2,
                )
            )
            with _stream_task_output_artifacts(
                self._sync_task_output_artifacts,
                task_output_dir,
                persistent_task_output_dir,
                interval_seconds=0.75,
            ):
                phase = "orchestrator"
                orchestrator_reached = True
                result = orchestrator.solve(
                    repo_path=str(repo_dir),
                    issue_description=issue_description,
                    test_command=rollout_test_command,
                    benchmark_metadata=benchmark_metadata,
                )
            phase = "finalization"
            _scrub_commit0_run_artifacts(task_output_dir)
            self._sync_task_output_artifacts(task_output_dir, persistent_task_output_dir)

            final = Commit0Evaluation(returncode=1, output=result.explanation or "")
            final_tests_passed = False
            failure_reason = None
            selected_rollout_id = result.selected_rollout_id
            selected_worktree_path = result.selected_worktree_path
            orchestrator_identity = (
                CandidateIdentity.from_worktree(
                    task_id=task.instance_id,
                    origin_rollout_id=result.selected_rollout_id,
                    worktree_path=result.selected_worktree_path,
                    selection_stage="orchestrator_nomination",
                )
                if result.selected_rollout_id is not None
                else None
            )
            selected_identity: Optional[CandidateIdentity] = None
            benchmark_rescored_candidate_id: Optional[str] = None
            official_audit_candidate_id: Optional[str] = None
            final_decision_source = "no_candidate"
            official_audit = None
            authorized_orchestrator_nomination = (
                self._apex_result_authorizes_orchestrator_nomination(result)
            )
            ignored_benchmark_rescore_candidate: Optional[dict[str, Any]] = None
            rollout_summaries = getattr(result, "rollout_summaries", None)
            external_scoring_candidates = getattr(
                result,
                "external_scoring_candidates",
                None,
            )
            summary_by_rollout = {
                int(summary.get("rollout_id")): summary
                for summary in (rollout_summaries or [])
                if isinstance(summary, dict) and isinstance(summary.get("rollout_id"), int)
            }
            for scoring_candidate in external_scoring_candidates or []:
                if not isinstance(scoring_candidate, dict) or not isinstance(
                    scoring_candidate.get("rollout_id"),
                    int,
                ):
                    continue
                rollout_id = int(scoring_candidate["rollout_id"])
                summary_by_rollout[rollout_id] = {
                    **summary_by_rollout.get(rollout_id, {}),
                    **scoring_candidate,
                }
            protected_test_files = self._load_protected_visible_test_files(
                task_output_dir,
                fallback_expected_test_ids=expected_test_ids,
            )
            incomplete_test_files = self._load_incomplete_visible_test_files(task_output_dir)
            has_primary_rollout_nomination = isinstance(result.selected_rollout_id, int)
            primary_summary = (
                summary_by_rollout.get(result.selected_rollout_id)
                if has_primary_rollout_nomination
                else None
            )
            primary_authoritative_scoring_request = (
                self._rollout_summary_authorizes_benchmark_confirmation(primary_summary)
            )
            primary_preferred_only_scoring_nomination = (
                self._rollout_summary_authorizes_preferred_only_scoring(primary_summary)
            )
            candidate = self._select_best_rollout_candidate(
                task=task,
                workspace_dir=task_workspace_dir,
                task_output_dir=task_output_dir,
                rollout_summaries=rollout_summaries,
                external_scoring_candidates=external_scoring_candidates,
                preferred_rollout_id=result.selected_rollout_id,
                protected_test_files=protected_test_files,
                incomplete_test_files=incomplete_test_files,
                baseline_repo_dir=repo_dir,
                python_executable=str(venv_python),
                env=env,
                expected_test_ids=expected_test_ids,
                use_expected_test_scoring=bool(expected_test_ids),
                restrict_to_preferred_rollout=bool(
                    authorized_orchestrator_nomination or primary_preferred_only_scoring_nomination
                ),
            )
            diagnostic_score_only_candidates = self._load_diagnostic_score_only_candidates(
                task_output_dir
            )
            benchmark_confirmed_primary_nomination = bool(
                candidate is not None
                and has_primary_rollout_nomination
                and candidate.rollout_id == result.selected_rollout_id
                and primary_authoritative_scoring_request
                and _commit0_evaluation_success(candidate.evaluation)
            )
            if candidate is not None and (
                not (authorized_orchestrator_nomination or benchmark_confirmed_primary_nomination)
                or candidate.rollout_id != result.selected_rollout_id
            ):
                ignored_benchmark_rescore_candidate = {
                    "rollout_id": candidate.rollout_id,
                    "preferred_rollout_id": result.selected_rollout_id,
                    "reason": (
                        "benchmark_rescore_is_diagnostic_only_without_matching_"
                        "authorized_orchestrator_nomination"
                    ),
                    "evaluation": candidate.evaluation.to_dict(),
                    "worktree_path": str(candidate.worktree_path),
                }
                (task_output_dir / "ignored_benchmark_rescore_candidate.json").write_text(
                    json.dumps(ignored_benchmark_rescore_candidate, indent=2) + "\n",
                    encoding="utf-8",
                )
                candidate = None
            if candidate is not None:
                self._write_task_live_state(
                    persistent_task_output_dir,
                    {
                        "task_id": task.repo_name,
                        "instance_id": task.instance_id,
                        "phase": "final_eval",
                        "status": "active",
                        "process_pid": os.getpid(),
                        "last_progress_at": time.time(),
                        "selected_rollout_id": candidate.rollout_id,
                        "current_evaluation_phase": (
                            "orchestrator_nomination"
                            if authorized_orchestrator_nomination
                            else "benchmark_rescore"
                        ),
                        "_clear_keys": _COMMIT0_TASK_ROLLOUT_LIVE_STATE_KEYS,
                    },
                )
                selected_rollout_id = candidate.rollout_id
                selected_worktree_path = str(candidate.worktree_path)
                selected_identity = CandidateIdentity.from_worktree(
                    task_id=task.instance_id,
                    origin_rollout_id=candidate.rollout_id,
                    worktree_path=candidate.worktree_path,
                    selection_stage=(
                        "orchestrator_nomination"
                        if authorized_orchestrator_nomination
                        else "benchmark_rescore"
                    ),
                )
                if authorized_orchestrator_nomination:
                    final_decision_source = "orchestrator_nomination"
                else:
                    benchmark_rescored_candidate_id = selected_identity.candidate_id
                    final_decision_source = "benchmark_rescore"
                final = candidate.evaluation
                final_tests_passed = _commit0_evaluation_success(final)
                if not final_tests_passed:
                    failure_reason = final.output or result.explanation
            elif authorized_orchestrator_nomination:
                selected_path = Path(result.selected_worktree_path)
                selected_summary = (
                    summary_by_rollout.get(result.selected_rollout_id)
                    if isinstance(result.selected_rollout_id, int)
                    else None
                )
                safe_selected_path, protected_edit_reason = (
                    self._prepare_visible_test_safe_worktree(
                        task=task,
                        baseline_repo_dir=repo_dir,
                        candidate_worktree=selected_path,
                        artifacts_dir=task_output_dir / "selected_rollout_eval",
                        protected_test_files=protected_test_files,
                        incomplete_test_files=incomplete_test_files,
                        rollout_summary=selected_summary,
                    )
                )
                if protected_edit_reason:
                    failure_reason = "Rejected selected rollout due to " + protected_edit_reason
                    final = Commit0Evaluation(returncode=1, output=failure_reason)
                    selected_rollout_id = None
                    selected_worktree_path = None
                    (task_output_dir / "selected_rollout_eval").mkdir(parents=True, exist_ok=True)
                    (task_output_dir / "selected_rollout_eval" / "policy_rejection.txt").write_text(
                        failure_reason
                    )
                else:
                    selected_path = safe_selected_path or selected_path
                    selected_eval_dir = task_output_dir / "selected_rollout_eval"
                    selected_identity = CandidateIdentity.from_worktree(
                        task_id=task.instance_id,
                        origin_rollout_id=result.selected_rollout_id,
                        worktree_path=selected_path,
                        selection_stage="orchestrator_nomination",
                    )
                    selected_changed_files = self._candidate_changed_files(
                        selected_path,
                        (
                            None
                            if safe_selected_path is not None
                            and safe_selected_path != Path(result.selected_worktree_path)
                            else selected_summary
                        ),
                    )
                    gate = self._candidate_quality_gate(
                        task=task,
                        candidate_worktree=selected_path,
                        artifacts_dir=selected_eval_dir,
                        changed_files=selected_changed_files,
                        python_executable=str(venv_python),
                        env=env,
                        rollout_summary=selected_summary,
                        incomplete_test_files=incomplete_test_files,
                    )
                    if gate.get("status") == "failed":
                        gate, failure_reason = self._record_quality_gate_rejection(
                            gate=gate,
                            artifacts_dir=selected_eval_dir,
                            reasons=list(gate.get("reasons") or ["quality_gate_failed"]),
                        )
                        final = self._quality_gate_rejection_evaluation(
                            reasons=list(gate.get("reasons") or ["quality_gate_failed"])
                        )
                        final_tests_passed = False
                        final_decision_source = "quality_gate_rejected"
                    else:
                        final_decision_source = "orchestrator_nomination"
                        self._write_task_live_state(
                            persistent_task_output_dir,
                            {
                                "task_id": task.repo_name,
                                "instance_id": task.instance_id,
                                "phase": "final_eval",
                                "status": "active",
                                "process_pid": os.getpid(),
                                "last_progress_at": time.time(),
                                "selected_rollout_id": result.selected_rollout_id,
                                "current_evaluation_phase": "orchestrator_nomination",
                                "_clear_keys": _COMMIT0_TASK_ROLLOUT_LIVE_STATE_KEYS,
                            },
                        )
                        final = self.evaluate_repo(
                            task,
                            selected_path,
                            artifacts_dir=selected_eval_dir,
                            label="selected",
                            python_executable=str(venv_python),
                            env=env,
                            expected_test_ids=expected_test_ids,
                        )
                        final_tests_passed = _commit0_evaluation_success(final)
                        if self._expected_coverage_collapsed(final):
                            gate, failure_reason = self._record_quality_gate_rejection(
                                gate=gate,
                                artifacts_dir=selected_eval_dir,
                                reasons=["expected_coverage_collapsed"],
                                evaluation=final,
                            )
                            final = self._quality_gate_rejection_evaluation(
                                reasons=list(gate.get("reasons") or ["quality_gate_failed"]),
                                evaluation=final,
                            )
                            final_tests_passed = False
                            final_decision_source = "quality_gate_rejected"
                        elif not final_tests_passed:
                            failure_reason = final.output or result.explanation
            else:
                if diagnostic_score_only_candidates:
                    final_decision_source = "candidate_invalid_for_submission"
                    failure_reason = "Candidate invalid for submission; diagnostic score only."
                else:
                    failure_reason = result.explanation or "No worktree selected."

            audit_worktree = Path(selected_worktree_path) if selected_worktree_path else None
            if final_decision_source != "quality_gate_rejected" and self._should_run_official_audit(
                final, audit_worktree
            ):
                self._write_task_live_state(
                    persistent_task_output_dir,
                    {
                        "task_id": task.repo_name,
                        "instance_id": task.instance_id,
                        "phase": "final_eval",
                        "status": "active",
                        "process_pid": os.getpid(),
                        "last_progress_at": time.time(),
                        "selected_rollout_id": selected_rollout_id,
                        "current_evaluation_phase": "official_audit_queued",
                        "_clear_keys": _COMMIT0_TASK_ROLLOUT_LIVE_STATE_KEYS,
                    },
                )
                released_solve_slot = self._release_current_task_solve_slot_for_audit()
                if released_solve_slot:
                    logger.info(
                        "Repo '%s' released Commit0 solve slot before official audit",
                        task.repo_name,
                    )
                with self._official_audit_lane():
                    # B5: run the audit with a transient-teardown-flake re-audit
                    # budget so a non-deterministic teardown ERROR (e.g. Twisted
                    # DirtyReactorAggregateError) cannot zero an otherwise-clean
                    # official audit. The loop only re-runs a green-except-teardown
                    # result and never re-runs a genuine scored failure.
                    official_audit = self._commit0_reaudit_until_stable(
                        task=task,
                        audit_worktree=audit_worktree,
                        task_output_dir=task_output_dir,
                        persistent_task_output_dir=persistent_task_output_dir,
                        selected_rollout_id=selected_rollout_id,
                    )
                (task_output_dir / "official_audit_metrics.json").write_text(
                    json.dumps(official_audit.to_dict(), indent=2)
                )
                # When the official Docker audit produced a usable result it
                # is the source of truth for the published metric — local
                # pytest runs on the host and can disagree with the audit
                # (different pytest plugins, missing extras, divergent
                # collection). Promote audit numbers over local when the
                # audit ran without an internal harness error AND it
                # actually executed tests. Don't overwrite when the audit
                # itself crashed or produced no signal.
                if _commit0_official_audit_usable(official_audit):
                    # Phase 1.1: preserve the APEX-private score in the
                    # audit's diagnostics so reviewers can see what the
                    # private path would have published. ``final`` at
                    # this point is the APEX-private evaluation; copy
                    # its key numbers BEFORE we overwrite it with the
                    # audit result.
                    apex_private_pass_rate = float(final.pass_rate)
                    apex_private_passed = int(final.passed)
                    apex_private_failed = int(final.failed)
                    apex_private_errors = int(final.errors)
                    apex_private_score_source = str(final.score_source)
                    official_audit.diagnostics["score_apex_private"] = {
                        "pass_rate": apex_private_pass_rate,
                        "passed": apex_private_passed,
                        "failed": apex_private_failed,
                        "errors": apex_private_errors,
                        "score_source": apex_private_score_source,
                    }
                    # Phase 1: side-by-side fairness audit. When
                    # configured, record the per-task delta between the
                    # APEX-private and the upstream-canonical scorers.
                    self._record_fairness_audit_delta(
                        task=task,
                        apex_private_eval=final,
                        upstream_audit_eval=official_audit,
                    )
                    if selected_identity is not None:
                        official_audit_candidate_id = selected_identity.candidate_id
                    apex_private_success = _commit0_evaluation_success(final)
                    official_audit_success = _commit0_evaluation_success(official_audit)
                    if apex_private_success and not official_audit_success:
                        official_audit.diagnostics["official_audit_disagrees"] = True
                        final_decision_source = "official_audit_disagreement"
                    else:
                        final_decision_source = "official_audit"
                    if official_audit_success and not apex_private_success:
                        official_audit.diagnostics["official_audit_rescued_private_failure"] = True
                    final = official_audit
                    final_tests_passed = official_audit_success
                    if final_tests_passed:
                        failure_reason = None
                    elif not failure_reason:
                        failure_reason = (
                            "Official audit reports failing/missing tests; "
                            "see official_audit_metrics.json."
                        )
                elif _commit0_evaluation_success(final):
                    apex_private_pass_rate = float(final.pass_rate)
                    failure_reason = (
                        "Official audit did not produce a usable scored result; "
                        "refusing to publish APEX-private success."
                    )
                    final_decision_source = "official_audit_unusable"
                    final = Commit0Evaluation(
                        returncode=1,
                        output=failure_reason,
                        raw_returncode=1,
                        failed=1,
                        total_tests=1,
                        scoring_source=str(final.scoring_source or "pytest_summary"),
                        evaluation_backend=str(final.evaluation_backend or "unknown"),
                        score_source="official_audit_required",
                        diagnostics={
                            "official_audit_unusable": True,
                            "score_apex_private": {
                                "pass_rate": apex_private_pass_rate,
                                "passed": int(final.passed),
                                "failed": int(final.failed),
                                "errors": int(final.errors),
                                "score_source": str(final.score_source),
                            },
                            "official_audit": official_audit.to_dict(),
                        },
                    )
                    final_tests_passed = False

            (task_output_dir / "final_metrics.json").write_text(
                json.dumps(final.to_dict(), indent=2)
            )
            (task_output_dir / "commit0_task.json").write_text(
                json.dumps(_artifact_safe_commit0_task_payload(task), indent=2)
            )

            if not keep_worktrees:
                shutil.rmtree(task_workspace_dir, ignore_errors=True)

            reported_orchestrator_worktree_path = self._stable_task_worktree_path(
                task,
                result.selected_worktree_path,
                internal_workspace_dir=task_workspace_dir,
                retain_worktrees=retain_task_workspaces,
            )
            reported_selected_worktree_path = self._stable_task_worktree_path(
                task,
                selected_worktree_path,
                internal_workspace_dir=task_workspace_dir,
                retain_worktrees=retain_task_workspaces,
            )
            final_candidate_id = (
                selected_identity.candidate_id
                if selected_identity is not None
                else (
                    orchestrator_identity.candidate_id
                    if orchestrator_identity is not None and selected_rollout_id is not None
                    else None
                )
            )
            final_patch_id = (
                selected_identity.patch_id
                if selected_identity is not None
                else worktree_patch_hash(selected_worktree_path)
            )
            candidate_identity_payload = (
                selected_identity.to_dict()
                if selected_identity is not None
                else (orchestrator_identity.to_dict() if orchestrator_identity is not None else {})
            )
            execution_metadata = extract_apex_execution_metadata(result)
            candidate_scorecard = load_json_if_exists(
                task_output_dir / "rollout_evals" / "candidate_scorecard.json"
            )
            if isinstance(candidate_scorecard, dict):
                execution_metadata["benchmark_candidate_scorecard"] = candidate_scorecard
                execution_metadata["standalone_anchor_results"] = list(
                    candidate_scorecard.get("standalone_anchor_results") or []
                )
            if diagnostic_score_only_candidates:
                execution_metadata["diagnostic_score_only_candidates"] = (
                    diagnostic_score_only_candidates
                )
            if ignored_benchmark_rescore_candidate is not None:
                execution_metadata["ignored_benchmark_rescore_candidate"] = (
                    ignored_benchmark_rescore_candidate
                )
            task_result = Commit0TaskResult(
                task_name=task.repo_name,
                instance_id=task.instance_id,
                repo=task.repo,
                success=final_tests_passed,
                baseline_failed=baseline.returncode != 0,
                final_tests_passed=final_tests_passed,
                baseline=baseline,
                final=final,
                orchestrator_success=bool(result.success or benchmark_rescored_candidate_id),
                candidate_found=bool(selected_worktree_path or selected_rollout_id is not None),
                orchestrator_selected_rollout_id=result.selected_rollout_id,
                orchestrator_selected_worktree_path=reported_orchestrator_worktree_path,
                selected_rollout_id=selected_rollout_id,
                selected_worktree_path=reported_selected_worktree_path,
                orchestrator_nomination_candidate_id=(
                    orchestrator_identity.candidate_id if orchestrator_identity else None
                ),
                orchestrator_nomination_rollout_id=result.selected_rollout_id,
                benchmark_rescored_candidate_id=benchmark_rescored_candidate_id,
                official_audit_candidate_id=official_audit_candidate_id,
                final_candidate_id=final_candidate_id,
                final_patch_id=final_patch_id or None,
                final_decision_source=final_decision_source,
                candidate_identity=candidate_identity_payload,
                total_tokens=result.total_tokens,
                duration_seconds=time.time() - started,
                result_path=str(task_result_path(persistent_task_output_dir)),
                failure_reason=failure_reason,
                official_audit=official_audit,
                execution_metadata=execution_metadata,
            )
            self._write_task_result_checkpoint_best_effort(
                persistent_task_output_dir,
                task_result,
            )
            self._write_task_live_state_terminal(
                persistent_task_output_dir,
                {
                    "task_id": task.repo_name,
                    "instance_id": task.instance_id,
                    "phase": "completed",
                    "status": "completed",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "success": task_result.success,
                    "final_tests_passed": task_result.final_tests_passed,
                    "selected_rollout_id": task_result.selected_rollout_id,
                },
            )
            for trace_output_dir in (task_output_dir, persistent_task_output_dir):
                append_benchmark_task_outcome_trace(
                    self.config,
                    output_dir=trace_output_dir,
                    benchmark_name="commit0",
                    task_id=task.instance_id,
                    task_success=task_result.success,
                    orchestrator_reached=orchestrator_reached,
                    orchestrator_success=task_result.orchestrator_success,
                    baseline_failed=task_result.baseline_failed,
                    baseline_pass_rate=task_result.baseline.pass_rate,
                    final_pass_rate=task_result.final.pass_rate,
                    candidate_found=task_result.candidate_found,
                    selected_rollout_id=task_result.selected_rollout_id,
                    skipped=task_result.skipped,
                    skip_category=task_result.skip_category,
                    duration_seconds=task_result.duration_seconds,
                )
            return task_result
        except Exception as exc:
            output = str(exc)
            traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            skipped, skip_category = self._classify_prepare_error(exc)
            if phase == "finalization" and isinstance(exc, FileNotFoundError):
                output = f"candidate_materialization_failed: {output}"
                skipped = False
                skip_category = "candidate_materialization_failed"
            _scrub_commit0_run_artifacts(task_output_dir)
            _scrub_commit0_run_artifacts(persistent_task_output_dir)
            task_payload = json.dumps(_artifact_safe_commit0_task_payload(task), indent=2)
            for output_dir in (task_output_dir, persistent_task_output_dir):
                (output_dir / "commit0_task.json").write_text(task_payload)
            failure = Commit0Evaluation(returncode=1, output=output)
            # Phase 1: also consult the new core classifier so the per-task
            # ``failure_class`` / ``failure_classification`` fields are
            # populated. The categorical ``skip_category`` string above
            # remains the back-compat headline; the new classification
            # is additive structured metadata.
            self._populate_core_failure_classification(failure, exc, phase="pre_install")
            for output_dir in (task_output_dir, persistent_task_output_dir):
                (output_dir / "prepare_error.txt").write_text(output)
                (output_dir / "prepare_error_traceback.txt").write_text(traceback_text)
            # Make prepare/baseline failures visible in the live log so
            # operators can see WHY a repo never reached the orchestrator
            # stage. Without this line the run silently scores these as
            # 0% and the only signal is a one-line entry buried in the
            # final report's failure_clusters.
            log_summary = output.splitlines()[0] if output else "<no message>"
            if phase == "finalization":
                logger.warning(
                    "Repo '%s' failed during finalization (skipped=%s category=%s): %s",
                    task.repo_name,
                    skipped,
                    skip_category or "unclassified",
                    log_summary[:300],
                )
            else:
                logger.warning(
                    "Repo '%s' did not reach orchestrator stage (skipped=%s category=%s): %s",
                    task.repo_name,
                    skipped,
                    skip_category or "unclassified",
                    log_summary[:300],
                )
            preserved_tokens = 0
            preserved_rollout_id: Optional[int] = None
            preserved_worktree_path: Optional[str] = None
            if result is not None:
                try:
                    preserved_tokens = int(getattr(result, "total_tokens", 0) or 0)
                except (TypeError, ValueError):
                    preserved_tokens = 0
                raw_rollout_id = getattr(result, "selected_rollout_id", None)
                if isinstance(raw_rollout_id, int):
                    preserved_rollout_id = raw_rollout_id
                raw_worktree_path = str(getattr(result, "selected_worktree_path", "") or "").strip()
                if raw_worktree_path:
                    preserved_worktree_path = raw_worktree_path
            task_result = Commit0TaskResult(
                task_name=task.repo_name,
                instance_id=task.instance_id,
                repo=task.repo,
                success=False,
                baseline_failed=False,
                final_tests_passed=False,
                baseline=failure,
                final=failure,
                orchestrator_success=bool(getattr(result, "success", False))
                if result is not None
                else False,
                candidate_found=bool(preserved_worktree_path),
                orchestrator_selected_rollout_id=preserved_rollout_id,
                orchestrator_selected_worktree_path=preserved_worktree_path,
                selected_rollout_id=preserved_rollout_id,
                selected_worktree_path=preserved_worktree_path,
                total_tokens=preserved_tokens,
                duration_seconds=time.time() - started,
                result_path=str(task_result_path(persistent_task_output_dir)),
                failure_reason=output,
                skipped=skipped,
                skip_category=skip_category,
            )
            self._write_task_result_checkpoint_best_effort(
                persistent_task_output_dir,
                task_result,
            )
            self._write_task_live_state_terminal(
                persistent_task_output_dir,
                {
                    "task_id": task.repo_name,
                    "instance_id": task.instance_id,
                    "phase": "completed",
                    "status": "error",
                    "process_pid": os.getpid(),
                    "last_progress_at": time.time(),
                    "success": False,
                    "failure_reason": output,
                    "skipped": skipped,
                    "skip_category": skip_category,
                    "selected_rollout_id": preserved_rollout_id,
                },
            )
            for trace_output_dir in (task_output_dir, persistent_task_output_dir):
                append_benchmark_task_outcome_trace(
                    self.config,
                    output_dir=trace_output_dir,
                    benchmark_name="commit0",
                    task_id=task.instance_id,
                    task_success=task_result.success,
                    orchestrator_reached=orchestrator_reached,
                    orchestrator_success=False,
                    baseline_failed=task_result.baseline_failed,
                    baseline_pass_rate=task_result.baseline.pass_rate,
                    final_pass_rate=task_result.final.pass_rate,
                    candidate_found=task_result.candidate_found,
                    selected_rollout_id=task_result.selected_rollout_id,
                    skipped=task_result.skipped,
                    skip_category=task_result.skip_category,
                    duration_seconds=task_result.duration_seconds,
                )
            return task_result
        finally:
            PROCESS_REGISTRY.kill(task.instance_id, signum=signal.SIGTERM)
            self._cleanup_task_processes(execution_layout)
            if container_name:
                self._cleanup_linux_runtime_container(container_name)
            self._sync_task_output_artifacts(task_output_dir, persistent_task_output_dir)
            if retain_task_workspaces:
                self._persist_task_workspaces(task_workspace_dir, persistent_workspace_dir)
            else:
                shutil.rmtree(persistent_workspace_dir, ignore_errors=True)
            if task_result is not None and task_result.success:
                for field_name in (
                    "orchestrator_selected_worktree_path",
                    "selected_worktree_path",
                ):
                    reported_path = getattr(task_result, field_name, None)
                    if reported_path and not Path(reported_path).exists():
                        setattr(task_result, field_name, None)
            shutil.rmtree(execution_layout.sandbox_root, ignore_errors=True)

    def evaluate_repo(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        artifacts_dir: Path,
        label: str,
        python_executable: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        expected_test_ids: Optional[list[str]] = None,
        timeout_seconds: Optional[int] = None,
        use_expected_test_scoring: bool = True,
        process_task_id: Optional[str] = None,
    ) -> Commit0Evaluation:
        if not use_expected_test_scoring and env is not None and python_executable is not None:
            return self._evaluate_repo_locally(
                task,
                repo_dir,
                artifacts_dir=artifacts_dir,
                python_executable=python_executable,
                env=env,
                expected_test_ids=None,
                timeout_seconds=timeout_seconds,
                use_expected_test_scoring=False,
                process_task_id=process_task_id,
            )
        if (
            self.config.benchmark.commit0_primary_evaluation_backend
            == BenchmarkEvaluationBackend.LOCAL_PYTEST
            and env is not None
            and python_executable is not None
        ):
            return self._evaluate_repo_locally(
                task,
                repo_dir,
                artifacts_dir=artifacts_dir,
                python_executable=python_executable,
                env=env,
                expected_test_ids=expected_test_ids,
                timeout_seconds=timeout_seconds,
                use_expected_test_scoring=use_expected_test_scoring,
                process_task_id=process_task_id,
            )

        return self._evaluate_repo_official(
            task,
            repo_dir,
            artifacts_dir=artifacts_dir,
            label=label,
            timeout_seconds=timeout_seconds,
            process_task_id=process_task_id,
        )

    def _should_run_official_audit(
        self,
        primary_evaluation: Commit0Evaluation,
        worktree_path: Optional[Path],
    ) -> bool:
        if worktree_path is None:
            return False
        if (
            self.config.benchmark.commit0_primary_evaluation_backend
            != BenchmarkEvaluationBackend.LOCAL_PYTEST
        ):
            return False
        if (
            primary_evaluation.evaluation_backend
            == COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER
        ):
            return False
        if not self.config.benchmark.commit0_official_audit_selected:
            return False
        if not worktree_path.exists():
            return False
        if not self.config.benchmark.commit0_official_audit_only_if_primary_passes:
            return True
        if _commit0_evaluation_success(primary_evaluation):
            return True
        # Commit0 gold harness fact: a scored expected-ID failure cannot publish,
        # but parser/harness failures mean the private scorer did not arbitrate.
        return _commit0_runner_health(primary_evaluation) in {
            RunnerHealth.PARSER_ERROR,
            RunnerHealth.HARNESS_FAILURE,
        }

    def _docker_fallback_available(self) -> bool:
        """True iff we can sensibly retry a failed task in a Docker container.

        Already on Linux: Docker would just be the same env again.
        Already in a Docker container: the failure is structural, not env.
        Docker CLI missing on the host: nothing to fall back to.
        Configured off: opt-out for users who can't / don't want to use Docker.
        """
        if sys.platform == "linux":
            return False
        if shutil.which("docker") is None:
            return False
        if not bool(getattr(self.config.benchmark, "commit0_docker_fallback_on_failure", True)):
            return False
        return True

    def _commit0_docker_runtime_forced(self) -> bool:
        mode = (
            str(
                getattr(self.config.benchmark, "commit0_docker_runtime_mode", "fallback")
                or "fallback"
            )
            .strip()
            .lower()
        )
        return mode in {"always", "force", "forced", "required", "require"}

    def _commit0_prepare_in_linux_container_first(self, task: Commit0Task) -> bool:
        # Commit0 max-quality runs isolate agentic CLIs in the same Docker runtime
        # that executes Python tests; host_env remains available for fallback-mode runs.
        return self._task_requires_linux_container(task) or self._commit0_docker_runtime_forced()

    def _baseline_signals_host_env_failure(
        self,
        baseline: Commit0Evaluation,
        baseline_eval_dir: Path,
        expected_test_ids: list[str],
        repo_dir: Optional[Path] = None,
    ) -> Optional[str]:
        """Return a category string when the baseline result smells like a
        host-environment limitation that a clean Linux container would fix.

        The categories mirror ``_preflight_block_or_none`` so the same
        signatures we'd otherwise skip on instead trigger a Docker retry.

        Phase 1: in addition to the historical category-string return,
        this method now populates the new core
        ``failure_class``/``failure_classification`` fields on the
        baseline evaluation so the env-skip ledger and
        ``classify_failure``-aware downstream tooling can see the
        structured verdict.
        """
        if not expected_test_ids:
            return None
        plugin_markers = (
            "AttributeError: module 'pytest' has no attribute",
            "pytest_metadata",
            "pytest_cov",
            "load_setuptools_entrypoints",
        )
        report_path = baseline_eval_dir / "report.json"
        report: dict[str, Any] = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                report = {}
        summary = (report or {}).get("summary") or {}
        collected = int(summary.get("collected") or 0)
        local_module_roots = (
            _commit0_python_local_module_roots(repo_dir) if repo_dir is not None else []
        )
        local_import_surface = _commit0_output_mentions_local_python_import(
            baseline.output,
            local_module_roots=local_module_roots,
        )
        # A native/compiled-extension build failure (numpy/scipy/cython/fortran)
        # is an ENV limitation even when the failing import names a local module
        # whose C extension never built — keep it env-classified (Docker retry),
        # do NOT silence it as an APEX source gap.
        native_build = _commit0_output_has_native_build_signature(baseline.output)
        category: Optional[str] = None
        if collected == 0 and any(m in baseline.output for m in plugin_markers):
            category = "pytest_plugin_incompat"
        else:
            collectors_failed = [
                collector
                for collector in (report or {}).get("collectors") or []
                if isinstance(collector, dict) and str(collector.get("outcome") or "") == "failed"
            ]
            if collected == 0 and collectors_failed:
                category = (
                    None
                    if (local_import_surface and not native_build)
                    else "baseline_collection_gap"
                )
            elif baseline.returncode == 4 and collected == 0:
                category = (
                    None
                    if (local_import_surface and not native_build)
                    else "baseline_pytest_usage_error"
                )
        if category is not None or local_import_surface:
            self._populate_core_failure_classification(baseline, baseline.output, phase="baseline")
            if (
                local_import_surface
                and not native_build
                and baseline.failure_class == _CoreFailureClass.ENV_INSTALL.value
            ):
                # Python import errors name top-level modules; local-root failures are source
                # import-surface blockers, not missing external dependency installs.
                baseline.failure_class = _CoreFailureClass.APEX_MISS.value
                baseline.failure_classification = {
                    "failure_class": _CoreFailureClass.APEX_MISS.value,
                    "confidence": 0.82,
                    "matched_pattern": ",".join(local_module_roots[:8]),
                    "reason": (
                        "baseline import failure targets a local Python module root; "
                        "treat as source import-surface blocker, not environment install"
                    ),
                }
        return category

    def _reset_repo_and_runtime_for_retry(
        self,
        repo_dir: Path,
        runtime_dir: Path,
    ) -> None:
        """Wipe the broken host-mode state so a Docker retry starts clean."""
        # Drop the broken host venv. The Docker variant builds a fresh one
        # inside the container's mount of runtime_dir.
        try:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        except Exception:
            pass
        runtime_dir.mkdir(parents=True, exist_ok=True)
        # Reset the worktree so the second prepare_repo's git clone /
        # checkout doesn't trip over half-applied state.
        try:
            shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception:
            pass
        repo_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_and_baseline(
        self,
        task: Commit0Task,
        repo_dir: Path,
        runtime_dir: Path,
        task_output_dir: Path,
        expected_test_ids: list[str],
        *,
        force_linux_container: Optional[bool] = None,
    ) -> tuple[dict[str, str], Commit0Evaluation, Path]:
        """Prepare the repo and capture baseline; return (env, baseline, venv_python).

        Used by ``_run_task`` so the prepare/baseline/retry logic lives in
        one place rather than being interleaved with orchestrator setup.
        Container cleanup on internal failure is the caller's responsibility
        — read ``env.get("APEX_COMMIT0_DOCKER_CONTAINER")`` if ``_prepare_repo``
        succeeded but a later step raised.
        """
        env: dict[str, str] = {}
        venv_python = Path()
        try:
            env = self._prepare_repo(
                task,
                repo_dir,
                runtime_dir,
                force_linux_container=force_linux_container,
            )
            venv_python = Path(env["VIRTUAL_ENV"]) / "bin" / "python"
            baseline = self.evaluate_repo(
                task,
                repo_dir,
                artifacts_dir=task_output_dir / "baseline_eval",
                label="baseline",
                python_executable=str(venv_python),
                env=env,
                expected_test_ids=expected_test_ids,
                timeout_seconds=self._commit0_baseline_evaluation_timeout_seconds(task),
            )
        except subprocess.TimeoutExpired as exc:
            if "VIRTUAL_ENV" not in env:
                container_name = str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or "")
                if container_name:
                    self._cleanup_linux_runtime_container(container_name)
                raise
            venv_python = Path(env["VIRTUAL_ENV"]) / "bin" / "python"
            baseline = self._baseline_timeout_evaluation(
                task=task,
                task_output_dir=task_output_dir,
                expected_test_ids=expected_test_ids,
                timeout=exc,
            )
        except Exception:
            container_name = str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or "")
            if container_name:
                self._cleanup_linux_runtime_container(container_name)
            raise
        return env, baseline, venv_python

    def _baseline_timeout_evaluation(
        self,
        *,
        task: Commit0Task,
        task_output_dir: Path,
        expected_test_ids: list[str],
        timeout: subprocess.TimeoutExpired,
    ) -> Commit0Evaluation:
        baseline_eval_dir = task_output_dir / "baseline_eval"
        baseline_eval_dir.mkdir(parents=True, exist_ok=True)
        output_parts = [
            f"Baseline evaluation timed out after {timeout.timeout}s.",
            str(timeout.output or ""),
            str(timeout.stderr or ""),
        ]
        output = normalize_terminal_output("\n".join(part for part in output_parts if part)).strip()
        (baseline_eval_dir / "baseline_timeout.txt").write_text(output + "\n")
        expected_count = len(expected_test_ids or [])
        evaluation = Commit0Evaluation(
            returncode=124,
            output=output,
            report_path=str(baseline_eval_dir / "report.json"),
            passed=0,
            failed=0,
            errors=0,
            skipped=0,
            total_tests=expected_count,
            scoring_source="commit0_test_ids" if expected_count else "pytest_summary",
            evaluation_backend="baseline_timeout",
            expected_test_coverage={
                "expected_test_count": expected_count,
                "observed_expected_test_count": 0,
                "missing_expected_test_count": expected_count,
                "missing_expected_test_ids": list(expected_test_ids or [])[:50],
            }
            if expected_count
            else {},
            diagnostics={
                "timeout": {
                    "phase": "baseline_eval",
                    "timeout_seconds": timeout.timeout,
                    "task": task.instance_id,
                },
                "baseline_timeout_attempt_anyway": True,
            },
            score_source="baseline_timeout",
        )
        self._populate_core_failure_classification(
            evaluation,
            output,
            phase="baseline",
        )
        try:
            decision = evaluation.contract_decision()
            evaluation.decision = decision.to_dict()
        except Exception:
            pass
        return evaluation

    def _prepare_and_baseline_with_docker_fallback(
        self,
        task: Commit0Task,
        repo_dir: Path,
        runtime_dir: Path,
        task_output_dir: Path,
        expected_test_ids: list[str],
    ) -> tuple[dict[str, str], Commit0Evaluation, Path, Optional[str], bool]:
        """Try host prepare+baseline; on failure or host-env signature,
        retry inside a Linux Docker container.

        Returns ``(env, baseline, venv_python, container_name, used_docker_fallback)``.
        Container teardown is the caller's responsibility (handled by the
        existing finally clause in ``_run_task``).
        """
        already_docker_first = self._commit0_prepare_in_linux_container_first(task)
        host_failure: Optional[Exception] = None
        try:
            env, baseline, venv_python = self._prepare_and_baseline(
                task,
                repo_dir,
                runtime_dir,
                task_output_dir,
                expected_test_ids,
                force_linux_container=True if already_docker_first else None,
            )
        except Exception as exc:
            host_failure = exc
            env = {}
            baseline = None  # type: ignore[assignment]
            venv_python = Path()

        # Decide whether a Docker retry is warranted. If the first attempt
        # was already a Linux container (per ``_task_requires_linux_container``),
        # there is no fallback — propagate the original failure / baseline.
        if host_failure is None and baseline is not None:
            host_signature = self._baseline_signals_host_env_failure(
                baseline,
                task_output_dir / "baseline_eval",
                expected_test_ids,
                repo_dir,
            )
            if (
                host_signature is None
                or already_docker_first
                or not self._docker_fallback_available()
            ):
                container_name = str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or "") or None
                return env, baseline, venv_python, container_name, False
            logger.warning(
                "Repo '%s' baseline shows host-env signature (%s); retrying in Docker container.",
                task.repo_name,
                host_signature,
            )
        elif host_failure is not None:
            if already_docker_first or not self._docker_fallback_available():
                raise host_failure
            logger.warning(
                "Repo '%s' host prepare/baseline failed (%s); retrying in Docker container.",
                task.repo_name,
                str(host_failure)[:300],
            )

        # Reset state and re-attempt under Docker.
        self._reset_repo_and_runtime_for_retry(repo_dir, runtime_dir)
        try:
            env, baseline, venv_python = self._prepare_and_baseline(
                task,
                repo_dir,
                runtime_dir,
                task_output_dir,
                expected_test_ids,
                force_linux_container=True,
            )
        except Exception as docker_exc:
            # Container is teared down by the outer finally either way; we
            # raise the more informative composite exception.
            if host_failure is not None:
                raise RuntimeError(
                    f"Both host and Docker prepare failed; host={host_failure}; docker={docker_exc}"
                ) from docker_exc
            raise

        container_name = str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or "") or None
        return env, baseline, venv_python, container_name, True

    def _preflight_block_or_none(
        self,
        *,
        task: Commit0Task,
        baseline: Commit0Evaluation,
        baseline_eval_dir: Path,
        expected_test_ids: list[str],
        task_output_dir: Path,
        persistent_task_output_dir: Path,
        started: float,
        repo_dir: Optional[Path] = None,
    ) -> Optional[Commit0TaskResult]:
        """Cheap, post-baseline checks that short-circuit before rollouts launch.

        Catches conditions where launching the orchestrator is guaranteed
        to produce no usable patch — e.g. baseline pytest collected zero of
        the expected test IDs (broken seed / pytest plugin incompat), or
        the entire pytest invocation produced a usage error. Saves tens of
        rollouts per affected repo on hard repos like jedi (105M tokens
        previously wasted) and the structural-blocker cluster (scrapy /
        statsmodels / paramiko / joblib / imbalanced-learn).

        The whole gate is opt-in via ``APEX_PREFLIGHT_BLOCK_ENABLED`` —
        early benchmark runs over-blocked tasks the agent could in fact
        recover, so the default is now off and operators flip the flag
        when they know the historical signature applies.
        """
        if not _env_flag_enabled("APEX_PREFLIGHT_BLOCK_ENABLED"):
            return None
        if not expected_test_ids:
            return None

        report_path = baseline_eval_dir / "report.json"
        report_payload: Optional[dict[str, Any]] = None
        if report_path.exists():
            try:
                report_payload = json.loads(report_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                report_payload = None
        summary = (report_payload or {}).get("summary") or {}
        collected = int(summary.get("collected") or 0)
        collectors_failed = [
            collector
            for collector in (report_payload or {}).get("collectors") or []
            if isinstance(collector, dict) and str(collector.get("outcome") or "") == "failed"
        ]

        category: Optional[str] = None
        details: dict[str, Any] = {
            "expected_test_count": len(expected_test_ids),
            "baseline_collected": collected,
            "baseline_returncode": baseline.returncode,
        }

        # Plugin / pytest version incompat — common on jedi (pytest_metadata)
        # and web3.py (pytest_cov). Pytest emits an internal error before
        # collection begins, so summary.collected == 0 and the output blob
        # carries an AttributeError for the pytest module.
        plugin_incompat_markers = (
            "AttributeError: module 'pytest' has no attribute",
            "pytest_metadata",
            "pytest_cov",
            "load_setuptools_entrypoints",
        )
        stashkey_incompat = (
            "attributeerror: module 'pytest' has no attribute 'stashkey'"
            in str(baseline.output or "").lower()
        )
        if any(marker in baseline.output for marker in plugin_incompat_markers) and (
            collected == 0 or stashkey_incompat
        ):
            category = "pytest_plugin_incompat"
            details["plugin_marker_seen"] = next(
                marker for marker in plugin_incompat_markers if marker in baseline.output
            )

        # Baseline collected zero expected tests AND failed to import at
        # least one test file — no rollout can recover when the seed code
        # can't be imported by pytest. Distinct from the "all tests fail"
        # case (which has collected > 0 and is genuinely solvable).
        # Tightened: only block when none of the expected_test_ids refer to
        # a test file the agent could plausibly create (i.e. every
        # expected nodeid lives inside a *.py path that already exists or
        # whose parent directory exists). Otherwise the agent has a clear
        # remediation path — write the missing test file — and blocking
        # would deny the rollout that work.
        if category is None and collected == 0 and collectors_failed:
            agent_can_create_test_file = self._expected_ids_have_creatable_paths(
                repo_dir=repo_dir,
                expected_test_ids=expected_test_ids,
            )
            if not agent_can_create_test_file:
                category = "baseline_collection_gap"
                details["failed_collectors"] = [
                    str(collector.get("nodeid")) for collector in collectors_failed[:8]
                ]

        # Pytest usage error (rc=4) with no tests collected — config or
        # plugin error, agent can't fix it.
        if category is None and baseline.returncode == 4 and collected == 0:
            category = "baseline_pytest_usage_error"

        if category is None:
            return None

        return self._emit_preflight_skip(
            task=task,
            category=category,
            details=details,
            baseline=baseline,
            task_output_dir=task_output_dir,
            persistent_task_output_dir=persistent_task_output_dir,
            started=started,
        )

    def _expected_ids_have_creatable_paths(
        self,
        *,
        repo_dir: Optional[Path],
        expected_test_ids: list[str],
    ) -> bool:
        """Return True when at least one expected test file can plausibly be
        authored by the agent (the parent directory exists in the seed
        repo). Used to refuse over-blocking pre-flight at
        ``baseline_collection_gap``: if the agent has somewhere to drop
        the missing test file, the rollout is worth running.
        """

        if repo_dir is None or not expected_test_ids:
            return False
        try:
            repo_root = Path(repo_dir).resolve()
        except Exception:
            return False
        for nodeid in expected_test_ids:
            head = str(nodeid or "").partition("::")[0].strip()
            if not head:
                continue
            try:
                candidate = (repo_root / head).resolve()
                # Bail on path traversal — only consider paths inside the
                # repo. ``Path.is_relative_to`` is 3.9+, fall back to a
                # string check for safety.
                try:
                    inside = candidate.is_relative_to(repo_root)
                except AttributeError:
                    inside = str(candidate).startswith(str(repo_root))
                if not inside:
                    continue
            except Exception:
                continue
            if candidate.exists():
                return True
            parent = candidate.parent
            if parent.exists() and parent.is_dir():
                return True
        return False

    def _emit_preflight_skip(
        self,
        *,
        task: Commit0Task,
        category: str,
        details: dict[str, Any],
        baseline: Commit0Evaluation,
        task_output_dir: Path,
        persistent_task_output_dir: Path,
        started: float,
    ) -> Commit0TaskResult:
        payload = {
            "category": category,
            "task_id": task.repo_name,
            "instance_id": task.instance_id,
            **details,
        }
        for output_dir in (task_output_dir, persistent_task_output_dir):
            (output_dir / "preflight_block.json").write_text(json.dumps(payload, indent=2))
        logger.warning(
            "Repo '%s' pre-flight blocked (category=%s): %s",
            task.repo_name,
            category,
            json.dumps({k: v for k, v in details.items() if k != "failed_collectors"}),
        )
        failure_reason = f"preflight:{category}"
        final = Commit0Evaluation(
            returncode=baseline.returncode,
            output=failure_reason,
            raw_returncode=baseline.raw_returncode,
            scoring_source=baseline.scoring_source,
            evaluation_backend=baseline.evaluation_backend,
            expected_test_coverage=copy.deepcopy(baseline.expected_test_coverage),
            score_source=baseline.score_source,
            diagnostics={
                **copy.deepcopy(baseline.diagnostics),
                "preflight_skip": copy.deepcopy(payload),
            },
            evaluation_contract=copy.deepcopy(baseline.evaluation_contract),
        )
        _commit0_evaluation_decision(final)
        task_result = Commit0TaskResult(
            task_name=task.repo_name,
            instance_id=task.instance_id,
            repo=task.repo,
            success=False,
            baseline_failed=baseline.returncode != 0,
            final_tests_passed=False,
            baseline=baseline,
            final=final,
            duration_seconds=time.time() - started,
            result_path=str(task_result_path(persistent_task_output_dir)),
            failure_reason=failure_reason,
            skipped=True,
            skip_category=category,
        )
        self._write_task_result_checkpoint_best_effort(
            persistent_task_output_dir,
            task_result,
        )
        self._write_task_live_state_terminal(
            persistent_task_output_dir,
            {
                "task_id": task.repo_name,
                "instance_id": task.instance_id,
                "phase": "completed",
                "status": "skipped",
                "process_pid": os.getpid(),
                "last_progress_at": time.time(),
                "success": False,
                "skipped": True,
                "skip_category": category,
                "failure_reason": failure_reason,
            },
        )
        return task_result

    def _load_protected_visible_test_files(
        self,
        task_output_dir: Path,
        *,
        fallback_expected_test_ids: Optional[list[str]] = None,
    ) -> list[str]:
        protected: list[str] = []
        payload = load_json_if_exists(task_output_dir / "issue_plan.json")
        if isinstance(payload, dict):
            protected.extend(protected_test_files_from_context(payload))
        if fallback_expected_test_ids:
            # Commit0 gold scoring fact: the expected-ID inventory is the scored
            # visible-test universe, so every expected test file remains
            # protected even when the planner focused on a narrower subset.
            protected.extend(
                protected_test_files_from_context(
                    {
                        "evaluation_constraints": {
                            "expected_test_ids": list(fallback_expected_test_ids),
                        }
                    }
                )
            )
        return list(dict.fromkeys(path for path in protected if path))

    def _load_incomplete_visible_test_files(
        self,
        task_output_dir: Path,
    ) -> list[str]:
        payload = load_json_if_exists(task_output_dir / "issue_plan.json")
        if not isinstance(payload, dict):
            return []
        return incomplete_test_files_from_context(payload)

    def _protected_test_edit_reason(
        self,
        worktree_path: Path,
        *,
        protected_test_files: list[str],
        incomplete_test_files: Optional[list[str]] = None,
        baseline_repo_dir: Optional[Path] = None,
        rollout_summary: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        protected = {
            str(path).strip() for path in list(protected_test_files or []) if str(path).strip()
        }
        if not protected:
            return None

        summary_changed = self._rollout_summary_changed_files(rollout_summary)
        changed_files = list(dict.fromkeys(summary_changed + list_git_changed_files(worktree_path)))
        incomplete = {
            str(path).strip() for path in list(incomplete_test_files or []) if str(path).strip()
        }
        offending: list[str] = []
        for path in sorted(path for path in changed_files if path in protected):
            if baseline_repo_dir is not None:
                baseline_path = baseline_repo_dir / path
                candidate_path = worktree_path / path
                try:
                    baseline_text = baseline_path.read_text(errors="replace")
                    candidate_text = candidate_path.read_text(errors="replace")
                except OSError:
                    offending.append(path)
                    continue
                analysis = analyze_visible_test_edit(
                    rel_path=path,
                    baseline_text=baseline_text,
                    candidate_text=candidate_text,
                    allow_placeholder_completion=path in incomplete,
                )
                if analysis.action in {"allow", "restore"}:
                    continue
            offending.append(path)
        if not offending:
            return None
        return "protected visible test edits: " + ", ".join(offending[:8])

    def _prepare_visible_test_safe_worktree(
        self,
        *,
        task: Optional[Commit0Task] = None,
        baseline_repo_dir: Optional[Path],
        candidate_worktree: Path,
        artifacts_dir: Path,
        protected_test_files: list[str],
        incomplete_test_files: Optional[list[str]] = None,
        rollout_summary: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[Path], Optional[str]]:
        summary_changed = self._rollout_summary_changed_files(rollout_summary)
        if baseline_repo_dir is not None and baseline_repo_dir.exists():
            filtered_summary_changed = []
            for rel_path in summary_changed:
                baseline_path = baseline_repo_dir / rel_path
                candidate_path = candidate_worktree / rel_path
                try:
                    if (
                        baseline_path.is_file()
                        and candidate_path.is_file()
                        and baseline_path.read_bytes() == candidate_path.read_bytes()
                    ):
                        continue
                except OSError:
                    pass
                filtered_summary_changed.append(rel_path)
            summary_changed = filtered_summary_changed
        changed_files = list(
            dict.fromkeys(summary_changed + list_git_changed_files(candidate_worktree))
        )
        sanitizer_dir = artifacts_dir / "patch_hygiene"
        original_candidate_worktree = candidate_worktree
        collection_critical_paths = self._candidate_collection_critical_edit_paths(
            candidate_worktree,
            changed_files,
            incomplete_test_files=incomplete_test_files,
        )[1]
        dependency_artifact_paths = self._candidate_dependency_artifact_paths(
            task=task,
            candidate_worktree=candidate_worktree,
        )
        candidate_worktree, patch_manifest, sanitizer_actions = sanitize_candidate_worktree(
            candidate_worktree=candidate_worktree,
            baseline_repo_dir=baseline_repo_dir,
            changed_files=changed_files,
            artifacts_dir=sanitizer_dir,
            evidence_mode="gold_suite_visible",
            incomplete_test_files=incomplete_test_files or (),
            collection_critical_paths=collection_critical_paths,
            protected_test_files=protected_test_files or (),
            dependency_artifact_paths=dependency_artifact_paths,
        )
        if sanitizer_actions:
            sanitizer_dir.mkdir(parents=True, exist_ok=True)
            (sanitizer_dir / "patch_manifest.json").write_text(
                json.dumps(patch_manifest.to_dict(), indent=2)
            )
            (sanitizer_dir / "patch_hygiene_actions.txt").write_text(
                "\n".join(sanitizer_actions) + "\n"
            )
            # Commit0 visible-gold adapter fact: non-solution visible-test edits
            # are stripped/restored here and the resulting source-only worktree
            # is rescored; the final published patch must be clean, not the raw
            # rollout worktree.
            if patch_manifest.vendored_upstream_artifacts:
                return (
                    None,
                    "vendored upstream/source artifacts modified: "
                    + ", ".join(patch_manifest.vendored_upstream_artifacts[:8]),
                )
            if candidate_worktree != original_candidate_worktree and not (
                self._materialize_commit0_audit_worktree_from_baseline(
                    audit_worktree=candidate_worktree,
                    baseline_repo_dir=baseline_repo_dir,
                )
            ):
                return None, "unable to materialize sanitized Commit0 audit worktree"
            changed_files = list_git_changed_files(candidate_worktree)
            if not changed_files and not (candidate_worktree / ".git").exists():
                changed_files = list(patch_manifest.solution_files)

        protected = {
            str(path).strip() for path in list(protected_test_files or []) if str(path).strip()
        }
        if not protected:
            return candidate_worktree, None
        protected_changed = sorted(path for path in changed_files if path in protected)
        if not protected_changed:
            return candidate_worktree, None
        if baseline_repo_dir is None or not baseline_repo_dir.exists():
            return (
                None,
                "protected visible test edits: " + ", ".join(protected_changed[:8]),
            )

        incomplete = {
            str(path).strip() for path in list(incomplete_test_files or []) if str(path).strip()
        }
        analyses: dict[str, VisibleTestEditDisposition] = {}
        for path in protected_changed:
            baseline_path = baseline_repo_dir / path
            candidate_path = candidate_worktree / path
            try:
                baseline_text = baseline_path.read_text(errors="replace")
                candidate_text = candidate_path.read_text(errors="replace")
            except OSError:
                analyses[path] = VisibleTestEditDisposition(
                    action="restore",
                    reason=f"{path} could not be read for protected visible-test analysis.",
                )
                continue
            analyses[path] = analyze_visible_test_edit(
                rel_path=path,
                baseline_text=baseline_text,
                candidate_text=candidate_text,
                allow_placeholder_completion=path in incomplete,
            )

        reject_paths = [
            path
            for path, analysis in analyses.items()
            if analysis.action not in {"allow", "sanitize", "restore"}
        ]
        if reject_paths:
            return (
                None,
                "protected visible test edits: " + ", ".join(sorted(reject_paths)[:8]),
            )

        if all(
            analysis.action == "allow" for analysis in analyses.values() if analysis is not None
        ):
            return candidate_worktree, None

        sanitized_root = artifacts_dir / "sanitized_worktree"
        if sanitized_root.exists():
            shutil.rmtree(sanitized_root, ignore_errors=True)
        copy_tree(candidate_worktree, sanitized_root)

        restored_files: list[str] = []
        sanitized_files: list[str] = []
        for path, analysis in analyses.items():
            target = sanitized_root / path
            if analysis.action == "allow":
                continue
            if analysis.action == "sanitize" and analysis.sanitized_text:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(analysis.sanitized_text)
                sanitized_files.append(path)
                continue
            baseline_path = baseline_repo_dir / path
            if baseline_path.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline_path, target)
            else:
                target.unlink(missing_ok=True)
            restored_files.append(path)

        policy_reason = self._protected_test_edit_reason(
            sanitized_root,
            protected_test_files=protected_test_files,
            incomplete_test_files=incomplete_test_files,
            baseline_repo_dir=baseline_repo_dir,
            rollout_summary=rollout_summary,
        )
        if policy_reason:
            return None, policy_reason

        notes: list[str] = []
        if restored_files:
            notes.append(
                "Restored protected visible-test edits to baseline: "
                + ", ".join(sorted(restored_files))
            )
        if sanitized_files:
            notes.append(
                "Sanitized incomplete visible-test files to placeholder-only completions: "
                + ", ".join(sorted(sanitized_files))
            )
        if notes:
            (artifacts_dir / "policy_sanitization.txt").write_text("\n".join(notes) + "\n")
        if not self._materialize_commit0_audit_worktree_from_baseline(
            audit_worktree=sanitized_root,
            baseline_repo_dir=baseline_repo_dir,
        ):
            return None, "unable to materialize sanitized Commit0 audit worktree"
        return sanitized_root, None

    def _evaluate_repo_locally(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        artifacts_dir: Path,
        python_executable: str,
        env: dict[str, str],
        expected_test_ids: Optional[list[str]] = None,
        timeout_seconds: Optional[int] = None,
        use_expected_test_scoring: bool = True,
        process_task_id: Optional[str] = None,
    ) -> Commit0Evaluation:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifacts_dir / "report.json"
        repo_report_path = repo_dir / _APEX_LOCAL_EVAL_REPORT_FILENAME
        output_path = artifacts_dir / "test_output.txt"
        progress_path = artifacts_dir / "evaluation_progress.json"
        _exclude_commit0_harness_helpers_from_git_status(repo_dir)
        for stale_path in (report_path, repo_report_path):
            try:
                stale_path.unlink()
            except FileNotFoundError:
                pass

        # Commit0 Docker local eval runs pytest in the repo/container workdir; host artifact paths are not always mount-visible.
        command = self._build_test_command(
            task,
            python_executable=python_executable,
            report_file=_APEX_LOCAL_EVAL_REPORT_FILENAME,
            expected_test_ids=expected_test_ids,
            repo_dir=repo_dir,
            xdist_context="scoring",
        )
        effective_timeout = timeout_seconds or self._commit0_evaluation_timeout_seconds(task)
        evaluation_started_at = time.time()
        atomic_write_json(
            progress_path,
            {
                "task_id": task.instance_id,
                "repo": task.repo_name,
                "phase": "evaluation",
                "status": "running",
                "command": command,
                "timeout_seconds": effective_timeout,
                "started_at": evaluation_started_at,
                "updated_at": evaluation_started_at,
                "output_path": str(output_path),
                "report_path": str(report_path),
                "workspace_report_path": str(repo_report_path),
            },
        )
        try:
            result = self._run_command(
                repo_dir,
                f"{command} > {shlex.quote(str(output_path))} 2>&1",
                env=env,
                timeout=effective_timeout,
                task_id=process_task_id or task.instance_id,
            )
        except subprocess.TimeoutExpired as exc:
            telemetry = getattr(exc, "apex_process_telemetry", None)
            atomic_write_json(
                progress_path,
                {
                    "task_id": task.instance_id,
                    "repo": task.repo_name,
                    "phase": "evaluation",
                    "status": "timeout",
                    "command": command,
                    "timeout_seconds": effective_timeout,
                    "started_at": evaluation_started_at,
                    "updated_at": time.time(),
                    "elapsed_seconds": time.time() - evaluation_started_at,
                    "output_path": str(output_path),
                    "report_path": str(report_path),
                    "workspace_report_path": str(repo_report_path),
                    "timeout_output": str(exc.output or "")[:2000],
                    "timeout_stderr": str(exc.stderr or "")[:2000],
                    "process_telemetry": telemetry if isinstance(telemetry, dict) else {},
                },
            )
            try:
                repo_report_path.unlink()
            except FileNotFoundError:
                pass
            raise
        if repo_report_path.exists():
            shutil.copyfile(repo_report_path, report_path)
            try:
                repo_report_path.unlink()
            except FileNotFoundError:
                pass
        evaluation = self._collect_evaluation(
            task=task,
            repo_dir=artifacts_dir,
            command_result=_CommandResult(
                returncode=result.returncode,
                output=_read_text_if_exists(output_path) or result.output,
            ),
            report_file="report.json",
            expected_test_ids_for_scoring=expected_test_ids,
            use_expected_test_scoring=use_expected_test_scoring,
        )
        evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
        evaluation.report_path = str(report_path)
        atomic_write_json(
            progress_path,
            {
                "task_id": task.instance_id,
                "repo": task.repo_name,
                "phase": "evaluation",
                "status": "completed",
                "command": command,
                "timeout_seconds": effective_timeout,
                "started_at": evaluation_started_at,
                "updated_at": time.time(),
                "elapsed_seconds": time.time() - evaluation_started_at,
                "output_path": str(output_path),
                "report_path": str(report_path),
                "workspace_report_path": str(repo_report_path),
                "returncode": evaluation.returncode,
                "passed": evaluation.passed,
                "failed": evaluation.failed,
                "errors": evaluation.errors,
                "total_tests": evaluation.total_tests,
            },
        )
        return evaluation

    def _run_single_official_audit(
        self,
        *,
        task: Commit0Task,
        audit_worktree: Path,
        artifacts_dir: Path,
        label: str,
    ) -> Commit0Evaluation:
        """Run one official-audit attempt, converting a crash into a usable
        ENV/harness-flavoured evaluation rather than propagating the exception."""
        try:
            return self._evaluate_repo_official(
                task,
                audit_worktree,
                artifacts_dir=artifacts_dir,
                label=label,
            )
        except Exception as exc:  # noqa: BLE001 - audit crash is a result, not fatal
            return Commit0Evaluation(
                returncode=1,
                output=str(exc),
                evaluation_backend=COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER,
            )

    def _commit0_reaudit_until_stable(
        self,
        *,
        task: Commit0Task,
        audit_worktree: Path,
        task_output_dir: Path,
        persistent_task_output_dir: Path,
        selected_rollout_id: Any,
        changed_files: Optional[list[str]] = None,
    ) -> Commit0Evaluation:
        """B5: run the official audit, re-running it while it is a transient
        teardown-flake (failed==0, scored ERRORS with a known teardown signature,
        coverage preserved) until a stable/clean outcome or the budget is spent.

        A single decisive clean success ends the loop; a genuine scored failure
        (``failed > 0`` / non-teardown error) never triggers a re-run, so this can
        only ever recover a non-deterministic teardown ERROR — never hide a real
        regression. Per-attempt metrics are persisted and a ``transient_audit_reaudit``
        diagnostic records the full attempt history on the published evaluation.
        """
        budget = max(
            1, int(getattr(self.config.benchmark, "commit0_transient_audit_rerun_budget", 3) or 1)
        )
        require_stable = max(
            1,
            int(getattr(self.config.benchmark, "commit0_transient_audit_require_stable", 2) or 1),
        )
        attempts: list[Commit0Evaluation] = []
        for attempt in range(1, budget + 1):
            try:
                self._write_task_live_state(
                    persistent_task_output_dir,
                    {
                        "task_id": task.repo_name,
                        "instance_id": task.instance_id,
                        "phase": "final_eval",
                        "status": "active",
                        "process_pid": os.getpid(),
                        "last_progress_at": time.time(),
                        "selected_rollout_id": selected_rollout_id,
                        "current_evaluation_phase": (
                            "official_audit"
                            if attempt == 1
                            else f"official_audit_reaudit_{attempt}"
                        ),
                        "_clear_keys": _COMMIT0_TASK_ROLLOUT_LIVE_STATE_KEYS,
                    },
                )
            except Exception:  # noqa: BLE001 - live-state write is best-effort
                pass
            artifacts_dir = task_output_dir / (
                "official_audit_eval" if attempt == 1 else f"official_audit_eval_reaudit_{attempt}"
            )
            label = "official-audit" if attempt == 1 else f"official-audit-reaudit-{attempt}"
            result = self._run_single_official_audit(
                task=task,
                audit_worktree=audit_worktree,
                artifacts_dir=artifacts_dir,
                label=label,
            )
            attempts.append(result)
            if attempt > 1:
                try:
                    (task_output_dir / f"official_audit_metrics_reaudit_{attempt}.json").write_text(
                        json.dumps(result.to_dict(), indent=2)
                    )
                except Exception:  # noqa: BLE001 - artifact write is best-effort
                    pass
            # A genuine outcome (clean success OR a real scored failure) is
            # decisive. Retry only audit-side failures with no score-bearing
            # signal: known teardown leaks or zero-signal harness/parser crashes.
            if not (
                _commit0_audit_error_is_transient_teardown_flake(result)
                or _commit0_audit_error_is_transient_harness_failure(result)
            ):
                break
            # Still transient audit infrastructure: keep going until the budget
            # runs out (a re-run may surface the true scored state).
        clean_success = next(
            (
                candidate
                for candidate in attempts
                if _commit0_evaluation_success(candidate)
                and not _commit0_audit_error_is_transient_teardown_flake(candidate)
            ),
            None,
        )
        chosen = clean_success or attempts[-1]
        chosen.diagnostics["transient_audit_reaudit"] = {
            "attempts": len(attempts),
            "budget": budget,
            "require_stable": require_stable,
            "resolved_clean_success": clean_success is not None,
            "per_attempt_success": [
                _commit0_evaluation_success(candidate) for candidate in attempts
            ],
            "per_attempt_transient_flake": [
                _commit0_audit_error_is_transient_teardown_flake(candidate)
                for candidate in attempts
            ],
            "per_attempt_transient_harness_failure": [
                _commit0_audit_error_is_transient_harness_failure(candidate)
                for candidate in attempts
            ],
        }
        # WS2B (NDFF live): when the re-audit budget is exhausted and the chosen
        # result is STILL a teardown flake, stamp it NON_DETERMINISTIC so the
        # scoring rollup can carve it out of the headline (a flaky gold test must
        # never charge an APEX miss). This is the canonical NDFF classification
        # call, made live. DeFlaker change-coverage / single-nodeid isolation are
        # not available in the coverage-off Docker official audit, so the in-loop
        # signal is the teardown-marker verdict; ``changed_files`` is threaded for
        # the host path where coverage is available.
        if clean_success is None and _commit0_audit_error_is_transient_teardown_flake(chosen):
            verdict = classify_oracle_failure(
                failed=int(chosen.failed),
                errors=int(chosen.errors),
                passed=int(chosen.passed),
                output=chosen.output,
                coverage_preserved=(chosen.expected_test_coverage or {}).get("coverage_preserved"),
                failing_test_covered_files=None,
                changed_files=changed_files,
            )
            if verdict.is_flaky:
                chosen.failure_class = verdict.failure_class.value
                chosen.failure_classification = verdict.to_dict()
        return chosen

    def _evaluate_repo_official(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        artifacts_dir: Path,
        label: str,
        timeout_seconds: Optional[int] = None,
        process_task_id: Optional[str] = None,
    ) -> Commit0Evaluation:
        branch = self._prepare_official_evaluation_branch(task, repo_dir, label)
        test_ids = task.test_dir or "tests/"
        log_dir = (
            artifacts_dir / "logs" / "pytest" / task.repo_name / branch / _hash_string(test_ids)
        )
        command_result = self._run_official_commit0_evaluation(
            task=task,
            repo_dir=repo_dir,
            branch=branch,
            test_ids=test_ids,
            artifacts_dir=artifacts_dir,
            timeout_seconds=timeout_seconds,
            process_task_id=process_task_id,
        )
        evaluation = self._collect_evaluation(
            task=task,
            repo_dir=log_dir,
            command_result=command_result,
            report_file="report.json",
        )
        evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER
        evaluation.report_path = str(log_dir / "report.json")
        # Phase 1.1: official upstream docker harness is the source of
        # truth for the published number. Tag the score_source so
        # downstream consumers can distinguish audit-derived numbers
        # from the APEX-private local pytest path.
        evaluation.score_source = "upstream_audit"
        return evaluation

    def _run_official_commit0_evaluation(
        self,
        *,
        task: Commit0Task,
        repo_dir: Path,
        branch: str,
        test_ids: str,
        artifacts_dir: Path,
        timeout_seconds: Optional[int] = None,
        process_task_id: Optional[str] = None,
    ) -> "_CommandResult":
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output = ""
        returncode = 0
        docker_env = _resolve_docker_sdk_env()
        self._ensure_commit0_official_runtime_image_available(
            task,
            self._commit0_official_runtime_image_tag(task),
            docker_env=docker_env,
        )
        self._cleanup_official_eval_container(task.repo_name, docker_env)
        with self._official_eval_repo_dir(
            task, repo_dir, artifacts_dir=artifacts_dir, label=branch
        ) as official_repo_dir:
            runner_kwargs = self._official_commit0_runner_kwargs(
                task=task,
                official_repo_dir=official_repo_dir,
                branch=branch,
                test_ids=test_ids,
                timeout_seconds=timeout_seconds,
            )
            inner_timeout = int(runner_kwargs["timeout"])
            outer_timeout = inner_timeout + COMMIT0_OFFICIAL_EVALUATION_SUBPROCESS_GRACE_SECONDS
            runner_script = _commit0_official_runner_script()
            invocation_payload = {
                "runner": "commit0.harness.run_pytest_ids",
                "execution": "bounded_subprocess",
                "python_executable": sys.executable,
                "outer_timeout_seconds": outer_timeout,
                "patch_transport": "binary_full_index",
                "patch_preapply_cleanup": "untracked_added_paths",
                "termination": "hard_exit_after_upstream_runner_exit",
                "kwargs": runner_kwargs,
            }
            (artifacts_dir / "official_runner_invocation.json").write_text(
                json.dumps(invocation_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            try:
                # Commit0 official local evaluation imports GitPython, which can
                # wedge git cat-file batch helpers; isolate it in a bounded
                # subprocess tree so benchmark audit cannot freeze APEX.
                completed = run_process_command(
                    [sys.executable, "-c", runner_script, json.dumps(runner_kwargs)],
                    cwd=artifacts_dir,
                    env=docker_env,
                    timeout=outer_timeout,
                    task_id=process_task_id or task.instance_id,
                )
                returncode = int(completed.returncode)
                output = normalize_terminal_output(
                    "\n".join(part for part in (completed.stdout, completed.stderr) if part)
                ).strip()
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                telemetry = getattr(exc, "apex_process_telemetry", None)
                output = normalize_terminal_output(
                    "\n".join(
                        part
                        for part in (
                            f"Official Commit0 evaluation timed out after {outer_timeout}s.",
                            str(exc.output or ""),
                            str(exc.stderr or ""),
                        )
                        if part
                    )
                ).strip()
                timeout_payload = {
                    "task_id": task.instance_id,
                    "repo": task.repo_name,
                    "timeout_seconds": outer_timeout,
                    "inner_timeout_seconds": inner_timeout,
                    "output_excerpt": output[:4000],
                    "process_telemetry": telemetry if isinstance(telemetry, dict) else {},
                }
                (artifacts_dir / "official_runner_timeout.json").write_text(
                    json.dumps(timeout_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            finally:
                self._cleanup_official_eval_container(task.repo_name, docker_env)

        log_dir = (
            artifacts_dir / "logs" / "pytest" / task.repo_name / branch / _hash_string(test_ids)
        )
        output_parts = [
            output,
            _read_text_if_exists(log_dir / "test_output.txt"),
            _read_text_if_exists(log_dir / "run_pytest.log"),
        ]
        combined_output = "\n\n".join(part for part in output_parts if part).strip()
        return _CommandResult(returncode=returncode, output=combined_output)

    def _official_commit0_runner_kwargs(
        self,
        *,
        task: Commit0Task,
        official_repo_dir: Path,
        branch: str,
        test_ids: str,
        timeout_seconds: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "base_dir": str(official_repo_dir.parent),
            "repo_or_repo_dir": str(official_repo_dir),
            "branch": branch,
            "test_ids": test_ids,
            "coverage": False,
            "backend": "local",
            "timeout": timeout_seconds or self._commit0_evaluation_timeout_seconds(task),
            "num_cpus": COMMIT0_OFFICIAL_EVALUATION_NUM_CPUS,
            "rebuild_image": False,
            "verbose": 0,
            # Commit0 dropped some gold-suite rows (notably pytest) from the primary
            # dataset revision after 2024-09-22 (see COMMIT0_DEFAULT_DATASET_FALLBACK_REVISIONS).
            # APEX's task creation recovers those rows from the pinned fallback
            # revision, but commit0.run_pytest_ids.main loads the latest mutable
            # revision and matches the spec by repo name -- for a dropped repo the
            # iteration finds nothing and aborts with "No spec available", scoring a
            # real candidate as audit_inconclusive. Pass the SAME revision universe
            # the task was built from so the audit can resolve the spec. These
            # APEX-only keys are popped by the runner wrapper before commit0.main runs.
            "_apex_dataset_primary_revision": self.dataset_revision,
            "_apex_dataset_fallback_revisions": list(self.dataset_fallback_revisions),
        }

    def _cleanup_official_eval_container(
        self,
        repo_name: str,
        docker_env: Optional[dict[str, str]] = None,
    ) -> None:
        if shutil.which("docker") is None:
            return
        run_process_command(
            ["docker", "rm", "-f", f"commit0.eval.{repo_name}"],
            env=docker_env,
            timeout=120,
        )

    def _prepare_official_evaluation_branch(
        self,
        task: Commit0Task,
        repo_dir: Path,
        label: str,
    ) -> str:
        branch_result = self._run_git_for_official_audit(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
            timeout=60,
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
        if not branch:
            branch = _build_commit0_eval_branch_name(label)
            self._run_checked_git_for_official_audit(
                ["git", "checkout", "-B", branch],
                cwd=repo_dir,
                timeout=120,
            )

        dirty_entries = self._git_dirty_entries(repo_dir)
        unwanted_entries = [
            (status_code, rel_path)
            for status_code, rel_path in dirty_entries
            if not self._is_commit0_snapshot_path(task, status_code, rel_path)
        ]
        self._discard_commit0_snapshot_artifacts(repo_dir, unwanted_entries)

        snapshot_entries = [
            (status_code, rel_path)
            for status_code, rel_path in self._git_dirty_entries(repo_dir)
            if self._is_commit0_snapshot_path(task, status_code, rel_path)
        ]
        snapshot_paths = list(dict.fromkeys(rel_path for _, rel_path in snapshot_entries))
        if snapshot_paths:
            self._run_checked_git_for_official_audit(
                ["git", "add", "-A", "--", *snapshot_paths], cwd=repo_dir, timeout=300
            )
            staged = self._run_git_for_official_audit(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(repo_dir),
                timeout=120,
            )
            if staged.returncode == 1:
                commit = self._run_git_for_official_audit(
                    ["git", "commit", "-m", f"APEX benchmark snapshot ({label})"],
                    cwd=repo_dir,
                    timeout=300,
                )
                if commit.returncode != 0:
                    commit_output = (commit.stdout + commit.stderr).strip().lower()
                    if "nothing to commit" not in commit_output:
                        raise RuntimeError(commit_output or "git commit failed")

        return branch

    def _run_git_for_official_audit(
        self,
        command: list[str],
        *,
        cwd: Path | str,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        # Commit0 official Docker/GitPython can leave stale worktree index.lock
        # files, so audit branch prep retries and prunes stale Git locks.
        last_result: subprocess.CompletedProcess[str] | None = None
        retry_delays = list(COMMIT0_OFFICIAL_GIT_LOCK_RETRY_DELAYS_SECONDS)
        for attempt in range(len(retry_delays) + 2):
            result = run_process_command(command, cwd=cwd, timeout=timeout)
            last_result = result
            if result.returncode == 0:
                return result

            output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            lock_paths = _git_index_lock_paths_from_output(output, cwd=cwd)
            if not lock_paths:
                return result

            recovered = self._recover_stale_official_git_index_locks(lock_paths)
            if recovered:
                continue

            if attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
            return result
        return last_result or run_process_command(command, cwd=cwd, timeout=timeout)

    def _run_checked_git_for_official_audit(
        self,
        command: list[str],
        *,
        cwd: Path | str,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        result = self._run_git_for_official_audit(command, cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                (result.stdout + result.stderr).strip() or f"Command failed: {command}"
            )
        return result

    def _recover_stale_official_git_index_locks(self, lock_paths: list[Path]) -> bool:
        recovered = False
        now = time.time()
        for lock_path in lock_paths:
            if lock_path.name != "index.lock":
                continue
            try:
                stat_result = lock_path.stat()
            except FileNotFoundError:
                continue
            except OSError:
                logger.debug("Unable to stat Git index lock %s", lock_path, exc_info=True)
                continue
            age_seconds = now - float(stat_result.st_mtime)
            if age_seconds < COMMIT0_OFFICIAL_GIT_INDEX_LOCK_STALE_SECONDS:
                continue
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                logger.debug("Unable to unlink stale Git index lock %s", lock_path, exc_info=True)
                continue
            logger.warning(
                "Removed stale Git index lock before Commit0 official audit: %s",
                lock_path,
            )
            recovered = True
        return recovered

    def _git_dirty_entries(self, repo_dir: Path) -> list[tuple[str, str]]:
        status = self._run_git_for_official_audit(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            timeout=120,
        )
        if status.returncode != 0:
            return []
        entries: list[tuple[str, str]] = []
        for line in status.stdout.splitlines():
            rel_path = normalize_changed_path(parse_porcelain_path(line))
            if not rel_path:
                continue
            entries.append((line[:2], rel_path))
        return entries

    def _is_commit0_snapshot_path(
        self,
        task: Commit0Task,
        status_code: str,
        rel_path: str,
    ) -> bool:
        if is_ignored_change_path(rel_path):
            return False
        if status_code != "??":
            return True
        if self._is_commit0_untracked_project_path(task, rel_path):
            return True
        return False

    def _is_commit0_untracked_project_path(self, task: Commit0Task, rel_path: str) -> bool:
        normalized = normalize_changed_path(rel_path)
        if not normalized:
            return False
        if any(
            self._path_within_root(normalized, root)
            for root in (
                normalize_changed_path(task.src_dir),
                normalize_changed_path(task.test_dir),
            )
            if root
        ):
            return True
        basename = Path(normalized).name.lower()
        if basename in _COMMIT0_UNTRACKED_SNAPSHOT_FILENAMES:
            return True
        if basename.startswith("requirements") and basename.endswith((".txt", ".in")):
            return True
        if basename.startswith("constraints") and basename.endswith(".txt"):
            return True
        return False

    def _discard_commit0_snapshot_artifacts(
        self,
        repo_dir: Path,
        entries: list[tuple[str, str]],
    ) -> None:
        tracked_paths = sorted({path for status_code, path in entries if status_code != "??"})
        untracked_paths = sorted({path for status_code, path in entries if status_code == "??"})
        if tracked_paths:
            self._run_checked_git_for_official_audit(
                ["git", "restore", "--staged", "--worktree", "--source=HEAD", "--", *tracked_paths],
                cwd=repo_dir,
                timeout=300,
            )
        if untracked_paths:
            self._run_checked_git_for_official_audit(
                ["git", "clean", "-fd", "--", *untracked_paths],
                cwd=repo_dir,
                timeout=300,
            )

    def _path_within_root(self, rel_path: str, root: str) -> bool:
        normalized_path = normalize_changed_path(rel_path)
        normalized_root = normalize_changed_path(root)
        if not normalized_root:
            return False
        return normalized_path == normalized_root or normalized_path.startswith(
            normalized_root + "/"
        )

    def _task_requires_linux_container(self, task: Commit0Task) -> bool:
        # Phase 4 10.N: order matters here — when the task's pre_install
        # contains apt-get / apt commands, we need the Linux container
        # path even on Linux hosts (the host's apt may be unavailable in
        # CI sandboxes; routing through our pinned bookworm container
        # keeps prepare reproducible).
        if any(_requires_linux_package_install(command) for command in task.pre_install):
            return True
        if sys.platform == "linux":
            return False
        # Explicit allow-list of repos whose tests need Linux semantics
        # the macOS sandbox can't provide (real PTY for pexpect, OpenSSL
        # pkg-config bindings for tlslite-ng, network-attached fixture
        # downloads for pypdf, cryptography extras for dnspython). The
        # auto-detection above only catches `apt-get` pre-installs;
        # these repos have neither but still fail badly on macOS.
        return task.repo_name in _COMMIT0_FORCE_LINUX_CONTAINER_REPOS

    def _linux_runtime_container_name(self, task: Commit0Task) -> str:
        slug = _slugify_output_component(task.repo_name).replace("_", "-")
        return f"apex-commit0-{slug}-{os.getpid()}-{time.time_ns()}"

    def _linux_runtime_network_name(self, container_name: str) -> str:
        return f"{container_name}-net"

    def _linux_runtime_egress_proxy_container_name(self, container_name: str) -> str:
        return f"{container_name}-egress"

    def _linux_runtime_container_label_args(self, task: Commit0Task) -> list[str]:
        return docker_label_args(
            apex_docker_labels(
                run_id=self.output_dir.name,
                task_id=task.instance_id,
                benchmark="commit0",
                owner_pid=os.getpid(),
            )
        )

    def _commit0_owner_pid_is_alive(self, owner_pid: str) -> bool:
        try:
            pid = int(str(owner_pid or "").strip())
        except ValueError:
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _commit0_owner_pid_from_docker_resource_name(self, name: str) -> str:
        parts = str(name or "").strip().split("-")
        if len(parts) < 5 or parts[:2] != ["apex", "commit0"]:
            return ""
        index = -3 if parts[-1] in {"egress", "net"} else -2
        try:
            owner_pid = parts[index]
        except IndexError:
            return ""
        return owner_pid if owner_pid.isdigit() else ""

    def _commit0_docker_resource_owner_is_alive(
        self,
        name: str,
        labels: Mapping[str, str],
    ) -> bool:
        owner_pid = str(labels.get(APEX_OWNER_PID_LABEL) or "").strip()
        if not owner_pid:
            owner_pid = self._commit0_owner_pid_from_docker_resource_name(name)
        if not owner_pid:
            return False
        return self._commit0_owner_pid_is_alive(owner_pid)

    def _cleanup_stale_commit0_docker_containers(
        self,
        docker_env: dict[str, str],
        *,
        network_name: str | None = None,
    ) -> list[str]:
        command = [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Labels}}",
        ]
        if network_name:
            command.extend(["--filter", f"network={network_name}"])
        else:
            command.extend(["--filter", f"label={APEX_BENCHMARK_LABEL}=commit0"])
        result = run_process_command(command, env=docker_env, timeout=120)
        if result.returncode != 0:
            return []

        stale_ids: list[str] = []
        stale_names: list[str] = []
        for raw_line in result.stdout.splitlines():
            container_id, _, remainder = raw_line.partition("\t")
            container_name, _, label_text = remainder.partition("\t")
            container_id = container_id.strip()
            container_name = container_name.strip()
            if not container_id:
                continue
            labels = parse_docker_label_string(label_text)
            is_commit0_resource = (
                labels.get(APEX_BENCHMARK_LABEL) == "commit0"
                or container_name.startswith("apex-commit0-")
                or bool(network_name)
            )
            if not is_commit0_resource:
                continue
            if self._commit0_docker_resource_owner_is_alive(container_name, labels):
                continue
            stale_ids.append(container_id)
            stale_names.append(container_name or container_id)

        if not stale_ids:
            return []
        # Commit0 Docker fact: interrupted solve runs can leave both named task
        # containers and anonymous sidecar endpoints attached to internal networks.
        cleanup = run_process_command(
            ["docker", "rm", "-f", *stale_ids],
            env=docker_env,
            timeout=120,
        )
        if cleanup.returncode != 0:
            return []
        return [f"container:{name}" for name in stale_names]

    def _cleanup_stale_commit0_solve_networks(self, docker_env: dict[str, str]) -> list[str]:
        """Remove labeled Commit0 solve resources whose Apex owner PID is gone."""

        removed = self._cleanup_stale_commit0_docker_containers(docker_env)
        result = run_process_command(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                "label=apex.benchmark=commit0",
                "--format",
                "{{.Name}}\t{{.Labels}}",
            ],
            env=docker_env,
            timeout=120,
        )
        if result.returncode != 0:
            return removed
        for raw_line in result.stdout.splitlines():
            name, _, label_text = raw_line.partition("\t")
            network_name = name.strip()
            if not (network_name.startswith("apex-commit0-") and network_name.endswith("-net")):
                continue
            labels = parse_docker_label_string(label_text)
            if self._commit0_docker_resource_owner_is_alive(network_name, labels):
                continue
            removed.extend(
                self._cleanup_stale_commit0_docker_containers(
                    docker_env,
                    network_name=network_name,
                )
            )
            cleanup = run_process_command(
                ["docker", "network", "rm", network_name],
                env=docker_env,
                timeout=120,
            )
            if cleanup.returncode == 0:
                removed.append(f"network:{network_name}")
        return removed

    def _commit0_official_runtime_image_tag(self, task: Commit0Task) -> str:
        # Commit0 official local audit evaluates wentingzhao/<repo>:v0; rollouts
        # must use that same image so local validation cannot drift by Python/deps.
        repo_name = str(task.repo_name or task.repo.split("/")[-1]).strip()
        return f"{_COMMIT0_OFFICIAL_IMAGE_NAMESPACE}/{repo_name}:v0"

    def _commit0_task_repo_instance(self, task: Commit0Task) -> dict[str, Any]:
        return {
            "repo": task.repo,
            "base_commit": task.base_commit,
            "reference_commit": task.reference_commit,
            "setup": {
                "python": task.python_version,
                "pre_install": list(task.pre_install) if task.pre_install else None,
                "packages": list(task.packages) if task.packages else None,
                "pip_packages": list(task.pip_packages) if task.pip_packages else None,
                "install": task.install_command or None,
            },
            "test": {
                "test_cmd": task.test_cmd,
                "test_dir": task.test_dir,
            },
            "src_dir": task.src_dir,
        }

    def _docker_image_exists_locally(
        self,
        image_ref: str,
        *,
        docker_env: Optional[dict[str, str]] = None,
    ) -> bool:
        if not image_ref or shutil.which("docker") is None:
            return False
        result = run_process_command(
            ["docker", "image", "inspect", image_ref],
            env=docker_env,
            timeout=120,
        )
        return result.returncode == 0

    def _docker_pull_image(
        self,
        image_ref: str,
        *,
        docker_env: Optional[dict[str, str]] = None,
    ) -> bool:
        if not image_ref or shutil.which("docker") is None:
            return False
        result = run_process_command(
            ["docker", "pull", image_ref],
            env=docker_env,
            timeout=self._commit0_runtime_setup_timeout_seconds(),
        )
        if result.returncode == 0:
            return True
        logger.info(
            "[commit0] official runtime image pull failed for %s: %s",
            image_ref,
            normalize_terminal_output((result.stdout + result.stderr).strip())[:500],
        )
        return False

    def _prepare_commit0_local_runtime_spec(
        self,
        spec: Any,
        task: Commit0Task,
    ) -> Any:
        repo_script_list = getattr(spec, "repo_script_list", None)
        repo_directory = str(getattr(spec, "repo_directory", "/testbed") or "/testbed")
        if not isinstance(repo_script_list, list):
            return spec

        # Commit0's base Dockerfile installs uv via the current Astral script,
        # which places uv in /root/.local/bin; Docker RUN does not load profiles.
        uv_path_export = 'export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"'
        if uv_path_export not in repo_script_list:
            repo_script_list.insert(0, uv_path_export)

        clone_index = next(
            (
                index
                for index, command in enumerate(repo_script_list)
                if isinstance(command, str) and command.startswith("git clone ")
            ),
            None,
        )
        if clone_index is not None:
            fetch_commands: list[str] = []
            for commit in dict.fromkeys(
                str(value or "").strip() for value in (task.reference_commit, task.base_commit)
            ):
                if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
                    continue
                # Commit0 task commits are fixed SHAs; fetch by SHA before
                # reset because a default clone may not advertise every commit.
                fetch_commands.append(
                    f"cd {shlex.quote(repo_directory)} && "
                    f"(git cat-file -e {commit}^{{commit}} 2>/dev/null || "
                    f"git fetch origin {commit} --depth=1)"
                )
            for command in reversed(fetch_commands):
                if command not in repo_script_list:
                    repo_script_list.insert(clone_index + 1, command)
        return spec

    def _require_docker_image_after_build(
        self,
        image_ref: str,
        build_dir: Path,
        *,
        docker_env: Optional[dict[str, str]] = None,
    ) -> None:
        if self._docker_image_exists_locally(image_ref, docker_env=docker_env):
            return
        raise RuntimeError(
            "Commit0 local image build completed without producing "
            f"{image_ref}; see {build_dir / 'build_image.log'}"
        )

    def _build_commit0_official_runtime_image(
        self,
        task: Commit0Task,
        image_ref: str,
        *,
        docker_env: Optional[dict[str, str]] = None,
    ) -> None:
        docker_env = docker_env or _resolve_docker_sdk_env()
        try:
            import docker
            from commit0.harness.docker_build import build_image
            from commit0.harness.spec import make_spec
        except ImportError as exc:
            raise RuntimeError(
                "Commit0 official Docker image is unavailable and the commit0/docker "
                "build dependencies are not installed."
            ) from exc

        # Commit0 image publication is incomplete for some public repos (for
        # example commit-0/pytest); build the canonical repo image locally from
        # the same Commit0 spec and tag it with the official harness ref.
        spec = self._prepare_commit0_local_runtime_spec(
            make_spec(self._commit0_task_repo_instance(task)),
            task,
        )
        client = docker.from_env(environment=docker_env)
        build_root = (
            self.output_dir
            / "_commit0_runtime_image_builds"
            / _slugify_output_component(task.repo_name)
        )
        base_build_dir = build_root / spec.base_image_key.replace(":", "__")
        repo_build_dir = build_root / image_ref.replace("/", "__").replace(":", "__")
        if not self._docker_image_exists_locally(spec.base_image_key, docker_env=docker_env):
            base_build_dir.mkdir(parents=True, exist_ok=True)
            build_image(
                spec.base_image_key,
                {},
                spec.base_dockerfile,
                spec.platform,
                client,
                base_build_dir,
            )
            self._require_docker_image_after_build(
                spec.base_image_key,
                base_build_dir,
                docker_env=docker_env,
            )
        repo_build_dir.mkdir(parents=True, exist_ok=True)
        build_image(
            image_ref,
            {"setup.sh": spec.setup_script},
            spec.repo_dockerfile,
            spec.platform,
            client,
            repo_build_dir,
        )
        self._require_docker_image_after_build(
            image_ref,
            repo_build_dir,
            docker_env=docker_env,
        )

    def _ensure_commit0_official_runtime_image_available(
        self,
        task: Commit0Task,
        image_ref: str,
        *,
        docker_env: Optional[dict[str, str]] = None,
    ) -> None:
        if not image_ref or "@sha256:" in image_ref or shutil.which("docker") is None:
            return
        docker_env = docker_env or _resolve_docker_sdk_env()
        if self._docker_image_exists_locally(image_ref, docker_env=docker_env):
            return
        if self._docker_pull_image(image_ref, docker_env=docker_env):
            return
        self._build_commit0_official_runtime_image(
            task,
            image_ref,
            docker_env=docker_env,
        )

    def _commit0_container_repo_root(self, task: Commit0Task) -> str:
        repo_name = str(task.repo_name or task.repo.split("/")[-1]).strip()
        return f"{_COMMIT0_DOCKER_WORKSPACE_ROOT}/{repo_name}"

    def _commit0_container_repo_pythonpath(self, task: Commit0Task) -> str:
        repo_root = self._commit0_container_repo_root(task)
        entries: list[str] = []
        src_root = str(task.src_root or "").strip().strip("/")
        if src_root:
            entries.append(f"{repo_root}/{src_root}")
        entries.append(repo_root)
        return ":".join(dict.fromkeys(entries))

    def _commit0_container_workdir_pythonpath(
        self,
        task: Commit0Task,
        *,
        container_workdir: str = "/workspace",
    ) -> str:
        root = str(container_workdir or "/workspace").rstrip("/") or "/workspace"
        entries: list[str] = []
        src_root = str(task.src_root or "").strip().strip("/")
        if src_root:
            entries.append(f"{root}/{src_root}")
        entries.append(root)
        return ":".join(dict.fromkeys(entries))

    def _linux_runtime_container_image(self, task: Commit0Task) -> str:
        """Return the docker image ref to launch the Linux runtime in.

        Resolved through :mod:`apex.core.docker_pinning` so the image is
        pinned to a sha256 digest when one is recorded in
        ``configs/docker_image_digests.json``. Unknown / unpinned tags
        fall through to the bare tag with a logged warning. The
        resolution is also recorded onto ``self.run_manifest`` so the
        per-run manifest captures exactly which image was used.
        """
        tag = self._commit0_official_runtime_image_tag(task)
        self._ensure_commit0_official_runtime_image_available(
            task,
            tag,
            docker_env=_resolve_docker_sdk_env(),
        )
        manifest = getattr(self, "run_manifest", None)
        try:
            resolved = _resolve_docker_image(tag, record_to_manifest=manifest)
        except Exception:
            return tag
        return resolved.image_ref

    def _cleanup_linux_runtime_container(
        self,
        container_name: str,
        docker_env: Optional[dict[str, str]] = None,
    ) -> None:
        if not container_name or shutil.which("docker") is None:
            return
        resolved_env = docker_env if docker_env is not None else _resolve_docker_sdk_env()
        sidecar_name = self._linux_runtime_egress_proxy_container_name(container_name)
        network_name = self._linux_runtime_network_name(container_name)
        run_process_command(
            ["docker", "rm", "-f", container_name],
            env=resolved_env,
            timeout=120,
        )
        run_process_command(
            ["docker", "rm", "-f", sidecar_name],
            env=resolved_env,
            timeout=120,
        )
        run_process_command(
            ["docker", "network", "rm", network_name],
            env=resolved_env,
            timeout=120,
        )

    def _install_repo_test_extras_best_effort(
        self,
        repo_dir: Path,
        env: dict[str, str],
        uv_pip: str,
    ) -> None:
        """B4: install declared test/dev extras so collection-time deps resolve.

        Each extra is installed independently with ``check=False`` so a missing or
        broken extra never fails repo preparation. A diagnostic records the outcome.
        """
        extras = _repo_test_extra_names(repo_dir)
        if not extras:
            return
        installed: list[str] = []
        for extra in extras:
            command = _rewrite_pip_command(f"pip install -e .[{extra}]", uv_pip)
            try:
                result = self._run_command(
                    repo_dir,
                    command,
                    env=env,
                    timeout=self._commit0_dependency_install_timeout_seconds(),
                    check=False,
                )
            except Exception:  # noqa: BLE001 - extras install is best-effort
                continue
            if getattr(result, "returncode", 1) == 0:
                installed.append(extra)
        if installed:
            self._record_diagnostic(
                repo_dir,
                "prepare_install_test_extras",
                {"installed_extras": installed, "candidate_extras": extras},
            )

    def _commit0_docker_memory_limit(self) -> str:
        """B3: memory cgroup limit for the shared per-task runtime container."""
        value = str(getattr(self.config.benchmark, "commit0_docker_memory_limit", "") or "").strip()
        configured = value or "8g"
        return self._commit0_docker_memory_limit_for_parallelism(configured)

    def _commit0_docker_memory_limit_for_parallelism(self, configured: str) -> str:
        configured_bytes = _parse_docker_size_bytes(configured)
        colima_memory_bytes = self._commit0_colima_configured_memory_bytes()
        if configured_bytes is None or colima_memory_bytes is None:
            return configured
        workers = max(1, int(getattr(self.config.benchmark, "task_parallelism", 1) or 1))
        reserve_bytes = max(2 * 1024**3, int(colima_memory_bytes * 0.10))
        available_bytes = max(1024**3, colima_memory_bytes - reserve_bytes)
        per_container_cap = max(1024**3, available_bytes // workers)
        if configured_bytes <= per_container_cap:
            return configured
        # Layer B: Commit0 Docker starts one long-lived runtime container per
        # solve slot; Colima VM memory is shared, so cap cgroups by task slots.
        return _format_docker_size_bytes(per_container_cap)

    def _commit0_colima_configured_memory_bytes(self) -> Optional[int]:
        config_path = Path.home() / ".colima" / "default" / "colima.yaml"
        try:
            text = config_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            match = re.match(r"^\s*memory\s*:\s*(?P<value>\d+)\s*(?:#.*)?$", line)
            if match:
                return int(match.group("value")) * 1024**3
        return None

    def _commit0_docker_shm_size(self) -> str:
        """B3: /dev/shm size for the shared per-task runtime container."""
        value = str(getattr(self.config.benchmark, "commit0_docker_shm_size", "") or "").strip()
        return value or "2g"

    def _docker_runtime_env_args(
        self,
        container_venv: str,
        env: Optional[dict[str, str]] = None,
    ) -> list[str]:
        args = [
            "-e",
            f"VIRTUAL_ENV={container_venv}",
            "-e",
            (
                "PATH="
                f"{container_venv}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        ]
        source_env = env or {}
        for key in _COMMIT0_DOCKER_RUNTIME_PASSTHROUGH_ENV_KEYS:
            value = str(source_env.get(key) or "").strip()
            if value:
                args.extend(["-e", f"{key}={value}"])
        return args

    def _commit0_docker_container_cwd(
        self,
        cwd: Path,
        env: Optional[dict[str, str]],
    ) -> str:
        source_env = env or {}
        host_root_raw = str(source_env.get(_COMMIT0_DOCKER_HOST_WORKDIR_ROOT_ENV) or "").strip()
        if not host_root_raw:
            return str(cwd)
        container_root = (
            str(
                source_env.get(_COMMIT0_DOCKER_CONTAINER_WORKDIR_ROOT_ENV)
                or _COMMIT0_DOCKER_WORKSPACE_ROOT
            )
            .strip()
            .rstrip("/")
            or _COMMIT0_DOCKER_WORKSPACE_ROOT
        )
        try:
            host_root = Path(host_root_raw).expanduser().resolve(strict=False)
            cwd_path = Path(cwd).expanduser().resolve(strict=False)
        except OSError:
            host_root = Path(host_root_raw).expanduser().absolute()
            cwd_path = Path(cwd).expanduser().absolute()
        if cwd_path == host_root:
            return container_root
        try:
            suffix = cwd_path.relative_to(host_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Commit0 Docker command cwd is outside task sandbox: {cwd_path}"
            ) from exc
        return f"{container_root}/{suffix.as_posix()}"

    def _run_docker_shell_command(
        self,
        *,
        container_name: str,
        cwd: Path,
        command: str,
        timeout: int,
        container_venv: str,
        env: Optional[dict[str, str]] = None,
        check: bool = False,
        docker_env: Optional[dict[str, str]] = None,
        task_id: str | None = None,
    ) -> "_CommandResult":
        result = run_process_command(
            [
                "docker",
                "exec",
                "-i",
                "-w",
                self._commit0_docker_container_cwd(cwd, env),
                *self._docker_runtime_env_args(container_venv, env),
                container_name,
                "bash",
                "-lc",
                command,
            ],
            env=docker_env if docker_env is not None else _resolve_docker_sdk_env(),
            timeout=timeout,
            task_id=task_id,
        )
        output = normalize_terminal_output(result.stdout + result.stderr).strip()
        if check and result.returncode != 0:
            raise RuntimeError(output or f"Command failed: {command}")
        return _CommandResult(returncode=result.returncode, output=output)

    def _commit0_agent_container_user(self) -> str:
        try:
            uid = os.getuid()
            gid = os.getgid()
        except AttributeError:
            return ""
        if uid <= 0:
            return ""
        return f"{uid}:{gid}"

    def _ensure_commit0_docker_proxy_relay(
        self,
        target_host: str,
        target_port: int,
    ) -> _Commit0DockerProxyRelay:
        key = (target_host, target_port)
        with self._commit0_docker_proxy_relay_lock:
            relay = self._commit0_docker_proxy_relays.get(key)
            if relay is not None:
                return relay
            server = _Commit0ProxyTCPServer(("0.0.0.0", 0), _Commit0ProxyRelayHandler)
            server.apex_target = key  # type: ignore[attr-defined]
            server.apex_allowed_hosts = _commit0_model_transport_hosts_from_env()  # type: ignore[attr-defined]
            thread = threading.Thread(
                target=server.serve_forever,
                name=f"commit0-docker-proxy-{target_port}",
                daemon=True,
            )
            thread.start()
            relay = _Commit0DockerProxyRelay(
                server=server,
                thread=thread,
                target_host=target_host,
                target_port=target_port,
            )
            self._commit0_docker_proxy_relays[key] = relay
            return relay

    def _commit0_colima_ssh_config_path(self) -> Path:
        return Path.home() / ".colima" / "_lima" / "colima" / "ssh.config"

    def _commit0_docker_should_use_colima_proxy_tunnel(self) -> bool:
        config_path = self._commit0_colima_ssh_config_path()
        if not config_path.is_file() or shutil.which("ssh") is None:
            return False
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False
        return result.returncode == 0 and result.stdout.strip().lower() == "colima"

    def _commit0_colima_proxy_port(self, target_port: int) -> int:
        base = 20000 + ((os.getpid() + target_port) % 20000)
        return min(base, 59999)

    def _ensure_commit0_colima_proxy_tunnel(
        self,
        target_host: str,
        target_port: int,
    ) -> _Commit0ColimaProxyTunnel:
        key = (target_host, target_port)
        with self._commit0_docker_proxy_relay_lock:
            tunnel = self._commit0_colima_proxy_tunnels.get(key)
            if tunnel is not None and tunnel.process.poll() is None:
                return tunnel
            listen_port = self._commit0_colima_proxy_port(target_port)
            command = [
                "ssh",
                "-F",
                str(self._commit0_colima_ssh_config_path()),
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-R",
                f"127.0.0.1:{listen_port}:{target_host}:{target_port}",
                "lima-colima",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.5)
            if process.poll() is not None:
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().strip()
                raise RuntimeError(stderr or "failed to start Colima proxy tunnel")
            tunnel = _Commit0ColimaProxyTunnel(
                process=process,
                listen_port=listen_port,
                target_host=target_host,
                target_port=target_port,
            )
            self._commit0_colima_proxy_tunnels[key] = tunnel
            return tunnel

    def _close_commit0_docker_proxy_relays(self) -> None:
        with self._commit0_docker_proxy_relay_lock:
            relays = list(self._commit0_docker_proxy_relays.values())
            self._commit0_docker_proxy_relays.clear()
            tunnels = list(self._commit0_colima_proxy_tunnels.values())
            self._commit0_colima_proxy_tunnels.clear()
        for relay in relays:
            relay.close()
        for tunnel in tunnels:
            tunnel.close()

    def _commit0_docker_proxy_value(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        parsed = urlsplit(raw)
        if parsed.scheme and parsed.hostname in _COMMIT0_LOOPBACK_PROXY_HOSTS:
            target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            relay = self._ensure_commit0_docker_proxy_relay(parsed.hostname, target_port)
            credentials = ""
            if parsed.username:
                credentials = parsed.username
                if parsed.password:
                    credentials += f":{parsed.password}"
                credentials += "@"
            replacement = f"{credentials}host.docker.internal:{relay.listen_port}"
            return urlunsplit(
                (
                    parsed.scheme,
                    replacement,
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
        return raw

    def _commit0_agent_proxy_address(self, proxy_env: dict[str, str]) -> str:
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            parsed = urlsplit(str(proxy_env.get(key) or "").strip())
            host = parsed.hostname
            if not host:
                continue
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            if parsed.port is not None:
                return f"{host}:{parsed.port}"
            return host
        return ""

    def _commit0_docker_proxy_env(self) -> dict[str, str]:
        proxy_env: dict[str, str] = {}
        for key in _COMMIT0_DOCKER_PROXY_ENV_KEYS:
            value = self._commit0_docker_proxy_value(os.environ.get(key, ""))
            if value:
                proxy_env[key] = value
        if proxy_env:
            for key in _COMMIT0_DOCKER_NO_PROXY_ENV_KEYS:
                value = str(os.environ.get(key) or "").strip()
                if value:
                    proxy_env[key] = value
        return proxy_env

    def _commit0_docker_model_proxy_value(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        parsed = urlsplit(raw)
        if parsed.username or parsed.password:
            raise RuntimeError(
                "Commit0 Docker model proxy URLs must not embed credentials; "
                "authenticate the host-side proxy service instead."
            )
        return self._commit0_docker_proxy_value(raw)

    def _commit0_docker_model_proxy_env(self) -> dict[str, str]:
        model_proxy_env: dict[str, str] = {}
        for key in _COMMIT0_DOCKER_MODEL_PROXY_ENV_KEYS:
            value = self._commit0_docker_model_proxy_value(os.environ.get(key, ""))
            if value:
                model_proxy_env[key] = value
        return model_proxy_env

    def _commit0_claude_provider_configured(self) -> bool:
        for key in _COMMIT0_CLAUDE_PROVIDER_ENV_KEYS:
            if str(os.environ.get(key) or "").strip():
                return True
        for llm_config in list(getattr(self.config, "llm_configs", []) or []):
            backend = getattr(llm_config, "backend", "")
            backend_value = getattr(backend, "value", backend)
            if str(backend_value or "") != "claude_cli":
                continue
            if str(getattr(llm_config, "base_url", "") or "").strip():
                return True
            overrides = getattr(llm_config, "cli_env_overrides", {}) or {}
            for key in _COMMIT0_CLAUDE_PROVIDER_ENV_KEYS:
                if str(overrides.get(key) or "").strip():
                    return True
        return False

    def _commit0_docker_claude_plugboard_env(
        self,
        proxy_env: dict[str, str],
        model_proxy_env: dict[str, str],
    ) -> dict[str, str]:
        target_cli_auth_mode = _commit0_target_cli_auth_mode(self.config).lower()
        if target_cli_auth_mode in _COMMIT0_HOST_CLI_AUTH_MODES:
            return {}
        if target_cli_auth_mode in _COMMIT0_MODEL_PROXY_AUTH_MODES:
            return {}
        if any(
            str(model_proxy_env.get(key) or os.environ.get(key) or "").strip()
            for key in _COMMIT0_CLAUDE_MODEL_PROXY_ENV_KEYS
        ):
            return {}
        if self._commit0_claude_provider_configured():
            return {}
        if not self._commit0_agent_proxy_address(proxy_env):
            return {}
        # Meta Claude Code in scrubbed Commit0 Linux Docker uses Plugboard V2 via
        # the benchmark-routed X2P proxy; Plugboard rejects Claude's advisor beta.
        return {
            "ANTHROPIC_BASE_URL": _COMMIT0_CLAUDE_PLUGBOARD_V2_BASE_URL,
            "ANTHROPIC_API_KEY": _COMMIT0_CLAUDE_PLUGBOARD_PLACEHOLDER_API_KEY,
            "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        }

    def _commit0_docker_context_is_colima(self, docker_env: dict[str, str]) -> bool:
        docker_host = str(docker_env.get("DOCKER_HOST") or os.environ.get("DOCKER_HOST") or "")
        if ".colima" in docker_host:
            return True
        docker_bin = shutil.which("docker") or "docker"
        result = run_process_command(
            [docker_bin, "context", "show"],
            env=docker_env,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "colima"

    def _commit0_colima_root_available_bytes(self, docker_env: dict[str, str]) -> Optional[int]:
        if shutil.which("colima") is None:
            return None
        if not self._commit0_docker_context_is_colima(docker_env):
            return None
        result = run_process_command(
            ["colima", "ssh", "--", "df", "-Pk", "/"],
            env=docker_env,
            timeout=20,
        )
        if result.returncode != 0:
            logger.debug(
                "Commit0 Colima root-space probe failed: %s",
                normalize_terminal_output(result.stdout + result.stderr).strip(),
            )
            return None
        return _parse_posix_df_available_bytes(result.stdout)

    def _ensure_commit0_docker_root_has_space(
        self,
        task: Commit0Task,
        docker_env: dict[str, str],
    ) -> None:
        # Commit0 Docker-on-Colima stores containerd snapshots on the VM root
        # filesystem; if root is near full, runc fails mid-rollout before APEX can
        # run tests or cleanup. Probe it before launching each task container.
        minimum = _commit0_min_docker_root_free_bytes()
        if minimum <= 0:
            return
        available = self._commit0_colima_root_available_bytes(docker_env)
        if available is None or available >= minimum:
            return
        raise RuntimeError(
            "Commit0 Docker runtime root filesystem is too full for "
            f"{task.repo_name}: available={_format_bytes_gib(available)}, "
            f"required>={_format_bytes_gib(minimum)}. Increase Colima root disk "
            "(`colima stop && colima start --root-disk 80`) or free Docker VM "
            "storage before running Commit0 gold Docker tasks."
        )

    def _commit0_docker_proxy_run_args(
        self,
        proxy_env: dict[str, str],
        model_proxy_env: Optional[dict[str, str]] = None,
    ) -> list[str]:
        endpoint_env = {**dict(proxy_env or {}), **dict(model_proxy_env or {})}
        # Commit0 SETUP uses Docker bridge networking for image/bootstrap
        # installs. SOLVE rewrites these values to the internal-network sidecar.
        egress_env = _commit0_solve_phase_proxy_env(endpoint_env)
        needs_add_host = any("host.docker.internal" in value for value in endpoint_env.values())
        endpoint_env.update(egress_env)
        if not endpoint_env:
            return []
        args: list[str] = []
        if needs_add_host:
            args.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in sorted(endpoint_env.items()):
            args.extend(["-e", f"{key}={value}"])
        return args

    @staticmethod
    def _commit0_docker_bind_source_missing(output: str) -> bool:
        lowered = str(output or "").lower()
        return (
            'invalid mount config for type "bind"' in lowered
            and "bind source path does not exist" in lowered
        )

    def _commit0_refresh_docker_bind_source(
        self,
        source: Path,
        *,
        docker_env: dict[str, str],
        container_image: str,
        task_id: str,
    ) -> bool:
        source = source.resolve(strict=False)
        parent = source.parent.resolve(strict=False)
        if not parent.is_dir():
            return False
        container_parent = "/mnt/apex-bind-parent"
        container_child = f"{container_parent}/{source.name}"
        # Commit0/Docker Desktop: a read-only, no-network parent mount can
        # refresh the VM file-sharing view without exposing sibling sandboxes to
        # the actual solve container, which still direct-mounts only `source`.
        result = run_process_command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                *self._commit0_official_image_platform_args(),
                "--mount",
                f"type=bind,source={parent},target={container_parent},readonly",
                container_image,
                "sh",
                "-lc",
                f"test -d {shlex.quote(container_child)}",
            ],
            env=docker_env,
            timeout=60,
            task_id=task_id,
        )
        return result.returncode == 0

    def _commit0_official_image_platform_args(self) -> list[str]:
        # Commit0 official Docker images are linux/amd64; pin docker-run platform
        # so host architecture warnings cannot be mistaken for agent failures.
        return ["--platform", _COMMIT0_OFFICIAL_IMAGE_PLATFORM]

    def _commit0_docker_bootstrap_retry_prelude(self) -> str:
        return (
            "_apex_retry() { "
            "local attempt=1; local status=0; "
            "while true; do "
            '"$@" && return 0; '
            "status=$?; "
            'if [ "${attempt}" -ge 3 ]; then return "${status}"; fi; '
            'sleep "$((attempt * 5))"; '
            'attempt="$((attempt + 1))"; '
            "done; "
            "}; "
        )

    def _commit0_apt_network_options(self, *, update: bool = False) -> str:
        options = [
            "-o Acquire::Retries=5",
            "-o Acquire::http::Timeout=30",
            "-o Acquire::https::Timeout=30",
        ]
        if update:
            options.append("-o APT::Update::Error-Mode=any")
        return " ".join(options)

    def _commit0_linux_runtime_bootstrap_command(self) -> str:
        apt_update_options = self._commit0_apt_network_options(update=True)
        apt_install_options = self._commit0_apt_network_options()
        # Commit0 audit images usually ship compiler primitives; avoid apt unless missing.
        return (
            "set -euo pipefail; "
            "export DEBIAN_FRONTEND=noninteractive; "
            f"{self._commit0_docker_bootstrap_retry_prelude()}"
            "_apex_has_build_toolchain() { "
            "command -v gcc >/dev/null && "
            "command -v g++ >/dev/null && "
            "command -v make >/dev/null && "
            "test -e /usr/include/stdio.h; "
            "}; "
            "_apex_apt_build_bootstrap() { "
            "if _apex_has_build_toolchain; then return 0; fi; "
            f"apt-get {apt_update_options} update && "
            f"apt-get {apt_install_options} install -y --no-install-recommends build-essential; "
            "}; "
            "_apex_retry _apex_apt_build_bootstrap; "
            "_apex_install_uv() { "
            "if command -v uv >/dev/null; then return 0; fi; "
            "if python -m pip --version >/dev/null 2>&1; then python -m pip install --upgrade uv; return $?; fi; "
            "if /usr/bin/python3 -m pip --version >/dev/null 2>&1; then /usr/bin/python3 -m pip install --upgrade uv; return $?; fi; "
            "if command -v pip3 >/dev/null; then pip3 install --upgrade uv; return $?; fi; "
            "echo 'APEX Commit0 runtime could not find pip to install uv' >&2; return 86; "
            "}; "
            "_apex_retry _apex_install_uv"
        )

    def _commit0_required_agent_cli_binaries(self) -> tuple[str, ...]:
        binaries: list[str] = []
        for llm_config in list(getattr(self.config, "llm_configs", []) or []):
            backend = getattr(llm_config, "backend", "")
            backend_value = str(getattr(backend, "value", backend) or "")
            binary = _COMMIT0_BACKEND_AGENT_CLI_BINARY.get(backend_value)
            if binary:
                binaries.append(binary)
            reviewer_backend = str(
                getattr(llm_config, "cli_tool_review_reviewer_backend", "") or ""
            )
            reviewer_binary = _COMMIT0_BACKEND_AGENT_CLI_BINARY.get(reviewer_backend)
            if reviewer_binary:
                binaries.append(reviewer_binary)
        if not binaries:
            binaries.extend(_COMMIT0_DEFAULT_AGENT_CLI_BINARIES)
        return tuple(dict.fromkeys(binaries))

    def _commit0_optional_agent_cli_binaries(self) -> tuple[str, ...]:
        # Commit0 max config fact: optional Avocado/OpenCode routes are eligible
        # only when a ready target-container CLI bundle already proves them local.
        binaries = [
            _COMMIT0_BACKEND_AGENT_CLI_BINARY[backend]
            for backend in sorted(self._commit0_optional_configured_cli_backends(self.config))
            if backend in _COMMIT0_BACKEND_AGENT_CLI_BINARY
        ]
        return tuple(dict.fromkeys(binaries))

    def _commit0_hard_required_agent_cli_binaries(self) -> tuple[str, ...]:
        llm_configs = list(getattr(self.config, "llm_configs", []) or [])
        if not llm_configs:
            return _COMMIT0_DEFAULT_AGENT_CLI_BINARIES
        binaries: list[str] = []
        primary = llm_configs[0]
        backend = getattr(primary, "backend", "")
        backend_value = str(getattr(backend, "value", backend) or "")
        binary = _COMMIT0_BACKEND_AGENT_CLI_BINARY.get(backend_value)
        if binary:
            binaries.append(binary)
        if bool(getattr(primary, "cli_tool_review_enabled", False)):
            reviewer_backend = str(getattr(primary, "cli_tool_review_reviewer_backend", "") or "")
            reviewer_binary = _COMMIT0_BACKEND_AGENT_CLI_BINARY.get(reviewer_backend)
            if reviewer_binary:
                binaries.append(reviewer_binary)
        if not binaries:
            return self._commit0_required_agent_cli_binaries()
        return tuple(dict.fromkeys(binaries))

    def _commit0_selected_agent_cli_binaries(
        self,
        *,
        required_binaries: Optional[tuple[str, ...]] = None,
        hard_required_binaries: Optional[tuple[str, ...]] = None,
    ) -> tuple[tuple[str, ...], set[str]]:
        requested_binaries = tuple(required_binaries or _COMMIT0_DEFAULT_AGENT_CLI_BINARIES)
        hard_binaries = tuple(
            hard_required_binaries if hard_required_binaries is not None else requested_binaries
        )
        selected_binaries = tuple(
            dict.fromkeys(
                binary
                for binary in (*requested_binaries, *hard_binaries)
                if binary in _COMMIT0_AGENT_CLI_NPM_PACKAGES
            )
        )
        if not selected_binaries:
            selected_binaries = _COMMIT0_DEFAULT_AGENT_CLI_BINARIES
        hard_binary_set = {
            binary for binary in hard_binaries if binary in _COMMIT0_AGENT_CLI_NPM_PACKAGES
        }
        if hard_required_binaries is None:
            hard_binary_set = set(selected_binaries)
        elif not hard_binary_set:
            hard_binary_set = {selected_binaries[0]}
        return selected_binaries, hard_binary_set

    def _commit0_agent_cli_bundle_cache_root(self) -> Path:
        override = str(os.environ.get("APEX_COMMIT0_AGENT_CLI_CACHE_DIR") or "").strip()
        if override:
            return Path(override).expanduser()
        # Commit0/Docker Desktop: freshly recreated hidden home caches can be
        # invisible to bind mounts, so keep the reusable CLI bundle beside the
        # benchmark output unless explicitly overridden.
        return self.output_dir.parent.resolve() / ".apex_commit0_agent_cli_cache"

    def _commit0_agent_cli_bundle_dir(
        self,
        *,
        required_binaries: Optional[tuple[str, ...]] = None,
        hard_required_binaries: Optional[tuple[str, ...]] = None,
    ) -> Path:
        selected_binaries, hard_binary_set = self._commit0_selected_agent_cli_binaries(
            required_binaries=required_binaries,
            hard_required_binaries=hard_required_binaries,
        )
        package_fingerprint = "|".join(
            f"{binary}={_COMMIT0_AGENT_CLI_NPM_PACKAGES[binary]}" for binary in selected_binaries
        )
        hard_fingerprint = ",".join(sorted(hard_binary_set))
        digest = hashlib.sha256(
            f"node={_COMMIT0_AGENT_CLI_NODE_VERSION}|{package_fingerprint}|hard={hard_fingerprint}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        names = "-".join(selected_binaries)
        safe_names = re.sub(r"[^A-Za-z0-9_.-]+", "-", names).strip("-") or "agent-cli"
        return (
            self._commit0_agent_cli_bundle_cache_root()
            / f"node-{_COMMIT0_AGENT_CLI_NODE_VERSION}-{safe_names}-{digest}"
        )

    @staticmethod
    def _commit0_agent_cli_bundle_ready(
        bundle_dir: Path,
        *,
        required_available_binaries: Iterable[str] = (),
    ) -> bool:
        ready_file = bundle_dir / ".ready"
        available_file = bundle_dir / "available_binaries"
        if not ready_file.is_file() or not available_file.is_file():
            return False
        required = {binary for binary in required_available_binaries if binary}
        if not required:
            return True
        try:
            available = {
                line.strip()
                for line in available_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            }
        except OSError:
            return False
        return required.issubset(available)

    def _find_ready_commit0_agent_cli_bundle_superset(
        self,
        *,
        required_available_binaries: Iterable[str],
    ) -> Optional[Path]:
        required = {binary for binary in required_available_binaries if binary}
        if not required:
            return None
        cache_root = self._commit0_agent_cli_bundle_cache_root()
        try:
            candidates = sorted(
                (path for path in cache_root.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None
        prefix = f"node-{_COMMIT0_AGENT_CLI_NODE_VERSION}-"
        for candidate in candidates:
            if not candidate.name.startswith(prefix):
                continue
            if self._commit0_agent_cli_bundle_ready(
                candidate,
                required_available_binaries=required,
            ):
                return candidate.resolve()
        return None

    def _commit0_agent_cli_bundle_build_command(
        self,
        *,
        selected_binaries: tuple[str, ...],
        hard_binary_set: set[str],
    ) -> str:
        bundle = shlex.quote(_COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT)
        node_version = shlex.quote(_COMMIT0_AGENT_CLI_NODE_VERSION)
        install_steps: list[str] = []
        for binary in selected_binaries:
            quoted_binary = shlex.quote(binary)
            quoted_package = shlex.quote(_COMMIT0_AGENT_CLI_NPM_PACKAGES[binary])
            if binary in hard_binary_set:
                install_steps.append(
                    f"_apex_install_agent_cli {quoted_binary} {quoted_package}; "
                    f"_apex_agent_cli_ready {quoted_binary} {quoted_package}; "
                    f"printf '%s\\n' {quoted_binary} >> \"${{available_file}}\"; "
                )
            else:
                install_steps.append(
                    f"if _apex_install_agent_cli {quoted_binary} {quoted_package} "
                    f"&& _apex_agent_cli_ready {quoted_binary} {quoted_package}; then "
                    f"printf '%s\\n' {quoted_binary} >> \"${{available_file}}\"; "
                    "else "
                    "status=$?; "
                    f'printf \'%s\\t%s\\n\' {quoted_binary} "$status" >> "${{unavailable_file}}"; '
                    f"printf 'optional agent CLI %s unavailable after shared bundle bootstrap "
                    f'(rc=%s)\\n\' {quoted_binary} "$status" >> "${{failure_log}}"; '
                    "fi; "
                )
        return (
            "set -euo pipefail; "
            f"bundle={bundle}; "
            f"node_version={node_version}; "
            'rm -rf "${bundle}/bin" "${bundle}/node" "${bundle}/npm-global" '
            '"${bundle}/npm-cache" "${bundle}/tmp"; '
            'mkdir -p "${bundle}/bin" "${bundle}/node" "${bundle}/npm-global" '
            '"${bundle}/npm-cache" "${bundle}/tmp" "${bundle}/logs"; '
            'available_file="${bundle}/available_binaries"; '
            'unavailable_file="${bundle}/unavailable_binaries"; '
            'failure_log="${bundle}/bootstrap_failures.log"; '
            ': > "${available_file}"; : > "${unavailable_file}"; : > "${failure_log}"; '
            f"{self._commit0_docker_bootstrap_retry_prelude()}"
            'arch="$(uname -m)"; '
            'case "$arch" in x86_64|amd64) node_arch=x64 ;; '
            "aarch64|arm64) node_arch=arm64 ;; "
            '*) echo "unsupported node arch: $arch" >&2; exit 86 ;; esac; '
            'node_name="node-v${node_version}-linux-${node_arch}"; '
            'cd "${bundle}/tmp"; '
            'curl -fsSLO "https://nodejs.org/dist/v${node_version}/SHASUMS256.txt"; '
            'curl -fsSLO "https://nodejs.org/dist/v${node_version}/${node_name}.tar.xz"; '
            'grep " ${node_name}.tar.xz$" SHASUMS256.txt | sha256sum -c -; '
            # Docker Desktop bind mounts can reject symlink entries created by tar;
            # extract Node without those symlinks and recreate them explicitly.
            'tar --exclude="${node_name}/bin/npm" --exclude="${node_name}/bin/npx" '
            '--exclude="${node_name}/bin/corepack" '
            '-xJf "${node_name}.tar.xz" -C "${bundle}/node" --strip-components=1; '
            'ln -sf ../lib/node_modules/npm/bin/npm-cli.js "${bundle}/node/bin/npm"; '
            'ln -sf ../lib/node_modules/npm/bin/npx-cli.js "${bundle}/node/bin/npx"; '
            'ln -sf ../lib/node_modules/corepack/dist/corepack.js "${bundle}/node/bin/corepack"; '
            'ln -sf "${bundle}/node/bin/node" "${bundle}/bin/node"; '
            'ln -sf "${bundle}/node/bin/npm" "${bundle}/bin/npm"; '
            'ln -sf "${bundle}/node/bin/npx" "${bundle}/bin/npx"; '
            'export PATH="${bundle}/bin:${bundle}/npm-global/bin:${PATH}"; '
            'export npm_config_cache="${bundle}/npm-cache"; '
            "_apex_install_agent_cli() { "
            'local binary="$1"; local package="$2"; '
            'echo "Installing shared APEX agent CLI ${binary} from ${package}" >&2; '
            '_apex_retry npm install -g --prefix "${bundle}/npm-global" '
            '--no-audit --no-fund --include=optional --omit=dev "${package}"; '
            "}; "
            "_apex_codex_native_package() { "
            'local package="$1"; local version="${package##*@}"; '
            'case "$node_arch" in '
            'x64) printf "%s\\n" "@openai/codex-linux-x64@npm:@openai/codex@${version}-linux-x64" ;; '
            'arm64) printf "%s\\n" "@openai/codex-linux-arm64@npm:@openai/codex@${version}-linux-arm64" ;; '
            "*) return 1 ;; esac; "
            "}; "
            "_apex_claude_native_package() { "
            'local package="$1"; local version="${package##*@}"; '
            'case "$node_arch" in '
            'x64) printf "%s\\n" "@anthropic-ai/claude-code-linux-x64@${version}" ;; '
            'arm64) printf "%s\\n" "@anthropic-ai/claude-code-linux-arm64@${version}" ;; '
            "*) return 1 ;; esac; "
            "}; "
            "_apex_repair_codex_native_package() { "
            'local package="$1"; local native_package; '
            'native_package="$(_apex_codex_native_package "${package}")" || return 1; '
            'echo "Installing shared APEX Codex native package ${native_package}" >&2; '
            '_apex_retry npm install -g --prefix "${bundle}/npm-global" '
            '--no-audit --no-fund --include=optional --omit=dev "${native_package}"; '
            "}; "
            "_apex_repair_claude_native_package() { "
            'local package="$1"; local native_package; '
            'native_package="$(_apex_claude_native_package "${package}")" || return 1; '
            'echo "Installing shared APEX Claude Code native package ${native_package}" >&2; '
            '_apex_retry npm install -g --prefix "${bundle}/npm-global" '
            '--no-audit --no-fund --include=optional --omit=dev "${native_package}"; '
            "}; "
            "_apex_agent_cli_ready() { "
            'local binary="$1"; local package="$2"; local log_file="${bundle}/logs/${binary}_version.log"; '
            'command -v "${binary}" >/dev/null || return 1; '
            # Commit0 Docker/QEMU can transiently segfault native npm CLI version probes.
            'if _apex_retry "${binary}" --version >"${log_file}" 2>&1; then '
            'ln -sf "${bundle}/npm-global/bin/${binary}" "${bundle}/bin/${binary}"; return 0; fi; '
            "status=$?; "
            'if [ "${binary}" = "codex" ] && grep -q "Missing optional dependency @openai/codex-linux-" "${log_file}"; then '
            '_apex_repair_codex_native_package "${package}" && _apex_retry "${binary}" --version >"${log_file}" 2>&1 '
            '&& ln -sf "${bundle}/npm-global/bin/${binary}" "${bundle}/bin/${binary}" && return 0; '
            "status=$?; "
            "fi; "
            'if [ "${binary}" = "claude" ] && grep -q "claude native binary not installed" "${log_file}"; then '
            '_apex_repair_claude_native_package "${package}" && _apex_retry "${binary}" --version >"${log_file}" 2>&1 '
            '&& ln -sf "${bundle}/npm-global/bin/${binary}" "${bundle}/bin/${binary}" && return 0; '
            "status=$?; "
            "fi; "
            'cat "${log_file}" >&2 || true; '
            'return "${status}"; '
            "}; "
            f"{''.join(install_steps)}"
            # Commit0 Docker Desktop fact: npm packages can leave transient or
            # platform-specific symlinks in the host bind mount; chmod files and
            # dirs directly so optional OpenCode-family bundles do not fail the
            # whole reproducible CLI surface on a symlink traversal race.
            'find "${bundle}" -type d -exec chmod a+rx {} +; '
            'find "${bundle}" -type f -exec chmod a+rX {} +; '
            'printf "%s\\n" "ready" > "${bundle}/.ready"'
        )

    def _ensure_commit0_agent_cli_bundle(
        self,
        *,
        docker_env: dict[str, str],
        container_image: str,
        docker_proxy_env: Optional[dict[str, str]] = None,
        docker_model_proxy_env: Optional[dict[str, str]] = None,
        required_binaries: Optional[tuple[str, ...]] = None,
        hard_required_binaries: Optional[tuple[str, ...]] = None,
        optional_binaries: Optional[tuple[str, ...]] = None,
    ) -> Path:
        selected_binaries, hard_binary_set = self._commit0_selected_agent_cli_binaries(
            required_binaries=required_binaries,
            hard_required_binaries=hard_required_binaries,
        )
        optional_binary_set = {
            binary
            for binary in (optional_binaries or ())
            if binary in _COMMIT0_AGENT_CLI_NPM_PACKAGES
        }
        optional_soft_binary_set = optional_binary_set.difference(hard_binary_set)
        bundle_dir = self._commit0_agent_cli_bundle_dir(
            required_binaries=required_binaries,
            hard_required_binaries=hard_required_binaries,
        ).resolve()
        with self._commit0_agent_cli_bundle_lock:
            if self._commit0_agent_cli_bundle_ready(bundle_dir):
                return bundle_dir
            # Commit0 setup may run with source/package egress denied; a ready
            # cached superset bundle can satisfy the current smaller CLI set.
            superset_bundle = self._find_ready_commit0_agent_cli_bundle_superset(
                required_available_binaries=selected_binaries,
            )
            if superset_bundle is not None:
                return superset_bundle
            if optional_soft_binary_set:
                all_hard_bundle = self._commit0_agent_cli_bundle_dir(
                    required_binaries=selected_binaries,
                    hard_required_binaries=selected_binaries,
                ).resolve()
                if self._commit0_agent_cli_bundle_ready(
                    all_hard_bundle,
                    required_available_binaries=selected_binaries,
                ):
                    return all_hard_bundle
                build_required_binaries = tuple(
                    binary for binary in selected_binaries if binary not in optional_soft_binary_set
                )
                if not build_required_binaries:
                    build_required_binaries = tuple(
                        binary for binary in selected_binaries if binary in hard_binary_set
                    ) or tuple(selected_binaries)
                selected_binaries, hard_binary_set = self._commit0_selected_agent_cli_binaries(
                    required_binaries=build_required_binaries,
                    hard_required_binaries=tuple(hard_required_binaries or ()),
                )
                bundle_dir = self._commit0_agent_cli_bundle_dir(
                    required_binaries=build_required_binaries,
                    hard_required_binaries=tuple(hard_required_binaries or ()),
                ).resolve()
                if self._commit0_agent_cli_bundle_ready(bundle_dir):
                    return bundle_dir
            shutil.rmtree(bundle_dir, ignore_errors=True)
            bundle_dir.mkdir(parents=True, exist_ok=True)
            # Commit0 Docker images run as linux/amd64 under QEMU on Apple Silicon;
            # build the reusable Node/CLI bundle inside that ecosystem but store it
            # on the host bind mount so per-repo containers do not fill overlayfs.
            command = self._commit0_agent_cli_bundle_build_command(
                selected_binaries=selected_binaries,
                hard_binary_set=hard_binary_set,
            )
            result = run_process_command(
                [
                    "docker",
                    "run",
                    "--rm",
                    *self._commit0_official_image_platform_args(),
                    # Commit0 agent CLI bundle provisioning is harness setup,
                    # not solve-time agent execution; do not inherit solve/eval
                    # HTTP proxies that intentionally deny package hosts.
                    "--mount",
                    (
                        "type=bind,"
                        f"source={bundle_dir},"
                        f"target={_COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT}"
                    ),
                    container_image,
                    "bash",
                    "-lc",
                    command,
                ],
                env=docker_env,
                timeout=self._commit0_runtime_setup_timeout_seconds(),
                task_id="commit0-agent-cli-bundle",
            )
            if result.returncode != 0:
                shutil.rmtree(bundle_dir, ignore_errors=True)
                raise RuntimeError(
                    normalize_terminal_output((result.stdout + result.stderr).strip())
                    or "Failed to build shared Commit0 agent CLI bundle"
                )
            if not self._commit0_agent_cli_bundle_ready(bundle_dir):
                shutil.rmtree(bundle_dir, ignore_errors=True)
                raise RuntimeError(
                    "Shared Commit0 agent CLI bundle did not produce readiness metadata"
                )
        return bundle_dir

    def _commit0_agent_cli_filtered_view_command(
        self,
        *,
        source_bundle_container_path: str,
        filtered_bundle_container_path: str,
        selected_binaries: Iterable[str],
        available_file: str = "",
        unavailable_file: str = "",
        failure_log: str = "",
    ) -> str:
        selected_binary_tuple = tuple(
            dict.fromkeys(binary for binary in selected_binaries if binary)
        )
        allowed_binaries = shlex.quote(" " + " ".join(selected_binary_tuple) + " ")
        source_bundle = shlex.quote(source_bundle_container_path.rstrip("/"))
        filtered_bundle = shlex.quote(filtered_bundle_container_path.rstrip("/"))
        available_sink = (
            f'printf "%s\\n" "${{binary}}" >> {shlex.quote(available_file)}; '
            if available_file
            else ""
        )
        unavailable_sink = (
            f'printf "%s\\n" "${{binary}}" >> {shlex.quote(unavailable_file)}; '
            if unavailable_file
            else ""
        )
        failure_sink = (
            f'printf "%s\\n" "missing shared agent CLI ${{binary}}" >> {shlex.quote(failure_log)}; '
            if failure_log
            else ""
        )
        missing_binary_branch = (
            "else " + unavailable_sink + failure_sink + "fi; "
            if unavailable_sink or failure_sink
            else "fi; "
        )
        return (
            f"bundle_dir={source_bundle}; "
            f"filtered_bundle_dir={filtered_bundle}; "
            f"allowed_binaries={allowed_binaries}; "
            'if [ -x "${bundle_dir}/bin/node" ] && [ -r "${bundle_dir}/available_binaries" ]; then '
            'if [ "${filtered_bundle_dir}" != "${bundle_dir}" ]; then '
            'rm -rf "${filtered_bundle_dir}/bin"; '
            "fi; "
            'mkdir -p "${filtered_bundle_dir}/bin"; '
            'ln -sf "${bundle_dir}/bin/node" "${filtered_bundle_dir}/bin/node"; '
            'ln -sf "${bundle_dir}/bin/npm" "${filtered_bundle_dir}/bin/npm"; '
            'ln -sf "${bundle_dir}/bin/npx" "${filtered_bundle_dir}/bin/npx"; '
            'ln -sf "${filtered_bundle_dir}/bin/node" /usr/local/bin/node; '
            'ln -sf "${filtered_bundle_dir}/bin/npm" /usr/local/bin/npm; '
            'ln -sf "${filtered_bundle_dir}/bin/npx" /usr/local/bin/npx; '
            "while IFS= read -r binary; do "
            '[ -n "${binary}" ] || continue; '
            'case "${allowed_binaries}" in *" ${binary} "*) ;; *) continue ;; esac; '
            'if [ -x "${bundle_dir}/bin/${binary}" ]; then '
            'ln -sf "${bundle_dir}/bin/${binary}" "${filtered_bundle_dir}/bin/${binary}"; '
            'ln -sf "${filtered_bundle_dir}/bin/${binary}" "/usr/local/bin/${binary}"; '
            f"{available_sink}"
            f"{missing_binary_branch}"
            'done < "${bundle_dir}/available_binaries"; '
            "else "
            'echo "mounted agent CLI bundle is missing node or available_binaries" >&2; '
            "exit 86; "
            "fi; "
        )

    def _commit0_agent_cli_bootstrap_command(
        self,
        *,
        agent_user: str = "",
        required_binaries: Optional[tuple[str, ...]] = None,
        hard_required_binaries: Optional[tuple[str, ...]] = None,
        agent_cli_bundle_container_path: str = "",
        filtered_agent_cli_bundle_container_path: str = (_COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT),
    ) -> str:
        node_version = shlex.quote(_COMMIT0_AGENT_CLI_NODE_VERSION)
        selected_binaries, hard_binary_set = self._commit0_selected_agent_cli_binaries(
            required_binaries=required_binaries,
            hard_required_binaries=hard_required_binaries,
        )
        available_file = "/tmp/apex_agent_cli_available_binaries"
        unavailable_file = "/tmp/apex_agent_cli_unavailable_binaries"
        failure_log = "/tmp/apex_agent_cli_bootstrap_failures.log"
        install_steps = [
            (
                f": > {shlex.quote(available_file)}; "
                f": > {shlex.quote(unavailable_file)}; "
                f": > {shlex.quote(failure_log)}; "
            )
        ]
        for binary in selected_binaries:
            quoted_binary = shlex.quote(binary)
            quoted_package = shlex.quote(_COMMIT0_AGENT_CLI_NPM_PACKAGES[binary])
            if binary in hard_binary_set:
                install_steps.append(
                    f"_apex_install_agent_cli {quoted_binary} {quoted_package}; "
                    f"_apex_agent_cli_ready {quoted_binary} {quoted_package}; "
                    f"printf '%s\\n' {quoted_binary} >> {shlex.quote(available_file)}; "
                )
            else:
                install_steps.append(
                    f"if _apex_install_agent_cli {quoted_binary} {quoted_package} "
                    f"&& _apex_agent_cli_ready {quoted_binary} {quoted_package}; then "
                    f"printf '%s\\n' {quoted_binary} >> {shlex.quote(available_file)}; "
                    "else "
                    "status=$?; "
                    f"printf '%s\\t%s\\n' {quoted_binary} \"$status\" >> "
                    f"{shlex.quote(unavailable_file)}; "
                    f"printf 'optional agent CLI %s unavailable after npm bootstrap "
                    f'(rc=%s)\\n\' {quoted_binary} "$status" >> '
                    f"{shlex.quote(failure_log)}; "
                    "fi; "
                )
        user_setup = ""
        if agent_user:
            uid, _sep, gid = agent_user.partition(":")
            if uid and gid:
                user_setup = (
                    f"agent_uid={shlex.quote(uid)}; agent_gid={shlex.quote(gid)}; "
                    'if ! getent group "${agent_gid}" >/dev/null; then '
                    'groupadd -g "${agent_gid}" apexagent; fi; '
                    'if ! getent passwd "${agent_uid}" >/dev/null; then '
                    'useradd -m -u "${agent_uid}" -g "${agent_gid}" -s /bin/bash apexagent; '
                    "fi; "
                )
        bundle_setup = ""
        if agent_cli_bundle_container_path:
            hard_checks = []
            for binary in sorted(hard_binary_set):
                quoted_binary = shlex.quote(binary)
                hard_checks.append(f"_apex_agent_cli_hard_check {quoted_binary}; ")
            filtered_view = self._commit0_agent_cli_filtered_view_command(
                source_bundle_container_path=agent_cli_bundle_container_path,
                filtered_bundle_container_path=filtered_agent_cli_bundle_container_path,
                selected_binaries=selected_binaries,
                available_file=available_file,
                unavailable_file=unavailable_file,
                failure_log=failure_log,
            )
            bundle_setup = (
                f": > {shlex.quote(available_file)}; "
                f": > {shlex.quote(unavailable_file)}; "
                f": > {shlex.quote(failure_log)}; "
                f"{filtered_view}"
                'if [ -r "${bundle_dir}/unavailable_binaries" ]; then '
                f'cat "${{bundle_dir}}/unavailable_binaries" >> {shlex.quote(unavailable_file)}; '
                "fi; "
                'if [ -r "${bundle_dir}/bootstrap_failures.log" ]; then '
                f'cat "${{bundle_dir}}/bootstrap_failures.log" >> {shlex.quote(failure_log)}; '
                "fi; "
                f"{''.join(hard_checks)}"
                "exit 0; "
            )
        return (
            "set -euo pipefail; "
            "export DEBIAN_FRONTEND=noninteractive; "
            f"{user_setup}"
            f"{self._commit0_docker_bootstrap_retry_prelude()}"
            "_apex_agent_cli_hard_check() { "
            'local binary="$1"; local log_file="/tmp/apex_agent_cli_${binary}_version.log"; '
            'command -v "${binary}" >/dev/null || '
            '(echo "required shared agent CLI missing: ${binary}" >&2; return 86); '
            # Commit0 Docker/QEMU can transiently segfault native npm CLI version probes.
            'if _apex_retry "${binary}" --version >"${log_file}" 2>&1; then return 0; fi; '
            'cat "${log_file}" >&2 || true; return 86; '
            "}; "
            f"{bundle_setup}"
            # Python bookworm images include curl/xz/CA certs; install only if absent.
            "_apex_has_agent_cli_download_prereqs() { "
            "command -v curl >/dev/null && "
            "command -v xz >/dev/null && "
            "test -r /etc/ssl/certs/ca-certificates.crt; "
            "}; "
            "_apex_apt_agent_cli_bootstrap() { "
            "if _apex_has_agent_cli_download_prereqs; then return 0; fi; "
            f"apt-get {self._commit0_apt_network_options(update=True)} update && "
            f"apt-get {self._commit0_apt_network_options()} install -y --no-install-recommends ca-certificates curl xz-utils; "
            "}; "
            "_apex_retry _apex_apt_agent_cli_bootstrap; "
            'arch="$(uname -m)"; '
            'case "$arch" in x86_64|amd64) node_arch=x64 ;; '
            "aarch64|arm64) node_arch=arm64 ;; "
            '*) echo "unsupported node arch: $arch" >&2; exit 86 ;; esac; '
            f"node_version={node_version}; "
            'node_name="node-v${node_version}-linux-${node_arch}"; '
            'node_dir="/opt/${node_name}"; '
            'if [ ! -x "${node_dir}/bin/node" ]; then '
            'tmp_dir="$(mktemp -d)"; '
            "trap 'rm -rf \"${tmp_dir}\"' EXIT; "
            'cd "${tmp_dir}"; '
            'curl -fsSLO "https://nodejs.org/dist/v${node_version}/SHASUMS256.txt"; '
            'curl -fsSLO "https://nodejs.org/dist/v${node_version}/${node_name}.tar.xz"; '
            'grep " ${node_name}.tar.xz$" SHASUMS256.txt | sha256sum -c -; '
            'tar -xJf "${node_name}.tar.xz" -C /opt; '
            "fi; "
            'ln -sf "${node_dir}/bin/node" /usr/local/bin/node; '
            'ln -sf "${node_dir}/bin/npm" /usr/local/bin/npm; '
            'ln -sf "${node_dir}/bin/npx" /usr/local/bin/npx; '
            "_apex_install_agent_cli() { "
            'local binary="$1"; local package="$2"; '
            'echo "Installing APEX agent CLI ${binary} from ${package}" >&2; '
            # Commit0 Docker/QEMU can segfault optional npm CLI installs; require only primary/reviewer CLIs.
            '_apex_retry npm install -g --prefix /usr/local --no-audit --no-fund --include=optional --omit=dev "${package}"; '
            "}; "
            "_apex_codex_native_package() { "
            'local package="$1"; local version="${package##*@}"; '
            'case "${node_arch}" in '
            'x64) printf "%s\\n" "@openai/codex-linux-x64@npm:@openai/codex@${version}-linux-x64" ;; '
            'arm64) printf "%s\\n" "@openai/codex-linux-arm64@npm:@openai/codex@${version}-linux-arm64" ;; '
            "*) return 1 ;; "
            "esac; "
            "}; "
            "_apex_claude_native_package() { "
            'local package="$1"; local version="${package##*@}"; '
            'case "${node_arch}" in '
            'x64) printf "%s\\n" "@anthropic-ai/claude-code-linux-x64@${version}" ;; '
            'arm64) printf "%s\\n" "@anthropic-ai/claude-code-linux-arm64@${version}" ;; '
            "*) return 1 ;; "
            "esac; "
            "}; "
            "_apex_repair_codex_native_package() { "
            'local package="$1"; local native_package; '
            'native_package="$(_apex_codex_native_package "${package}")" || return 1; '
            'echo "Installing APEX Codex native package ${native_package}" >&2; '
            # Commit0 Linux images need Codex's platform package; npm may skip optional aliases.
            '_apex_retry npm install -g --prefix /usr/local --no-audit --no-fund --include=optional --omit=dev "${native_package}"; '
            "}; "
            "_apex_repair_claude_native_package() { "
            'local package="$1"; local native_package; '
            'native_package="$(_apex_claude_native_package "${package}")" || return 1; '
            'echo "Installing APEX Claude Code native package ${native_package}" >&2; '
            # Commit0 Linux images need Claude Code's platform package when npm skips optional deps.
            '_apex_retry npm install -g --prefix /usr/local --no-audit --no-fund --include=optional --omit=dev "${native_package}"; '
            "}; "
            "_apex_agent_cli_ready() { "
            'local binary="$1"; local package="$2"; local log_file="/tmp/apex_agent_cli_${binary}_version.log"; '
            'command -v "${binary}" >/dev/null || return 1; '
            # Commit0 Docker/QEMU can transiently segfault native npm CLI version probes.
            'if _apex_retry "${binary}" --version >"${log_file}" 2>&1; then return 0; fi; '
            "status=$?; "
            'if [ "${binary}" = "codex" ] && grep -q "Missing optional dependency @openai/codex-linux-" "${log_file}"; then '
            '_apex_repair_codex_native_package "${package}" && _apex_retry "${binary}" --version >"${log_file}" 2>&1 && return 0; '
            "status=$?; "
            "fi; "
            'if [ "${binary}" = "claude" ] && grep -q "claude native binary not installed" "${log_file}"; then '
            '_apex_repair_claude_native_package "${package}" && _apex_retry "${binary}" --version >"${log_file}" 2>&1 && return 0; '
            "status=$?; "
            "fi; "
            'cat "${log_file}" >&2 || true; '
            'return "${status}"; '
            "}; "
            # Gemini CLI requires Node >=20; exact npm packages keep the
            # Commit0 Docker runtime's agent surface reproducible.
            f"{''.join(install_steps)}"
        )

    def _write_linux_runtime_wrapper(
        self,
        target: Path,
        *,
        docker_bin: str,
        docker_env: dict[str, str],
        container_name: str,
        container_venv: str,
        host_sandbox_root: Path,
        container_sandbox_root: str,
        command_tokens: list[str],
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        host_root = str(host_sandbox_root.resolve(strict=False))
        container_root = str(container_sandbox_root).rstrip("/") or "/workspace"
        lines = [
            "#!/bin/sh",
            "set -eu",
        ]
        docker_host = str(docker_env.get("DOCKER_HOST") or "").strip()
        if docker_host:
            lines.append(f"export DOCKER_HOST={shlex.quote(docker_host)}")
        lines.append("cwd=$(pwd)")
        lines.extend(
            [
                f"host_root={shlex.quote(host_root)}",
                f"container_root={shlex.quote(container_root)}",
                'case "$cwd" in',
                '  "$host_root") container_cwd="$container_root" ;;',
                '  "$host_root"/*) suffix=${cwd#"$host_root"/}; container_cwd="$container_root/$suffix" ;;',
                '  *) echo "APEX docker runtime wrapper refused cwd outside task sandbox: $cwd" >&2; exit 126 ;;',
                "esac",
            ]
        )
        # Commit0 Docker wrappers narrow PATH; Desktop/Colima may install docker outside /usr/bin.
        docker_exec = f'{shlex.quote(docker_bin or "docker")} exec -i -w "$container_cwd"'
        docker_exec += f" -e {shlex.quote(f'VIRTUAL_ENV={container_venv}')}"
        docker_exec += " -e " + shlex.quote(
            (
                "PATH="
                f"{container_venv}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )
        )
        for key in _COMMIT0_DOCKER_RUNTIME_PASSTHROUGH_ENV_KEYS:
            docker_exec += f' -e "{key}=${{{key}-}}"'
        docker_exec += f" {shlex.quote(container_name)}"
        docker_exec += " " + " ".join(shlex.quote(token) for token in command_tokens)
        docker_exec += ' "$@"'
        lines.append(f"exec {docker_exec}")
        target.write_text("\n".join(lines) + "\n")
        target.chmod(0o755)

    def _create_commit0_internal_solve_network(
        self,
        *,
        task: Commit0Task,
        container_name: str,
        docker_env: dict[str, str],
    ) -> str:
        network_name = self._linux_runtime_network_name(container_name)
        # Commit0 Docker fact: interrupted solve runs leave dead-owner internal
        # networks behind; reclaim them before consuming another subnet.
        self._cleanup_stale_commit0_solve_networks(docker_env)
        # Commit0 Docker solve phase: an internal Docker network removes direct
        # internet/source access without turning agent commands into policy failures.
        create_command = [
            "docker",
            "network",
            "create",
            "--internal",
            *self._linux_runtime_container_label_args(task),
            network_name,
        ]
        result = run_process_command(
            create_command,
            env=docker_env,
            timeout=120,
            task_id=task.instance_id,
        )
        if result.returncode != 0:
            output = normalize_terminal_output((result.stdout + result.stderr).strip())
            if _commit0_docker_network_pool_exhausted(output):
                removed = self._cleanup_stale_commit0_solve_networks(docker_env)
                if removed:
                    result = run_process_command(
                        create_command,
                        env=docker_env,
                        timeout=120,
                        task_id=task.instance_id,
                    )
                    if result.returncode == 0:
                        return network_name
                    output = normalize_terminal_output((result.stdout + result.stderr).strip())
            raise RuntimeError(
                output or f"Failed to create Commit0 internal solve network {network_name}"
            )
        return network_name

    def _start_commit0_egress_proxy_sidecar(
        self,
        *,
        task: Commit0Task,
        container_name: str,
        container_image: str,
        network_name: str,
        docker_env: dict[str, str],
        mappings: tuple[_Commit0EgressProxyMapping, ...],
    ) -> None:
        if not mappings:
            return
        sidecar_name = self._linux_runtime_egress_proxy_container_name(container_name)
        script = _commit0_egress_proxy_sidecar_script(
            mappings,
            allowed_hosts=_commit0_model_transport_hosts_from_env(),
        )
        launch = run_process_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                sidecar_name,
                *self._linux_runtime_container_label_args(task),
                "--security-opt",
                "no-new-privileges=true",
                *self._commit0_official_image_platform_args(),
                "--add-host",
                "host.docker.internal:host-gateway",
                container_image,
                "python",
                "-u",
                "-c",
                script,
            ],
            env=docker_env,
            timeout=180,
            task_id=task.instance_id,
        )
        if launch.returncode != 0:
            raise RuntimeError(
                normalize_terminal_output((launch.stdout + launch.stderr).strip())
                or f"Failed to start Commit0 egress proxy sidecar {sidecar_name}"
            )
        connect = run_process_command(
            [
                "docker",
                "network",
                "connect",
                "--alias",
                _COMMIT0_EGRESS_PROXY_ALIAS,
                network_name,
                sidecar_name,
            ],
            env=docker_env,
            timeout=120,
            task_id=task.instance_id,
        )
        if connect.returncode != 0:
            raise RuntimeError(
                normalize_terminal_output((connect.stdout + connect.stderr).strip())
                or f"Failed to attach Commit0 egress sidecar to {network_name}"
            )

    def _move_commit0_agent_to_internal_solve_network(
        self,
        *,
        task: Commit0Task,
        container_name: str,
        network_name: str,
        docker_env: dict[str, str],
    ) -> None:
        connect = run_process_command(
            ["docker", "network", "connect", network_name, container_name],
            env=docker_env,
            timeout=120,
            task_id=task.instance_id,
        )
        if connect.returncode != 0:
            raise RuntimeError(
                normalize_terminal_output((connect.stdout + connect.stderr).strip())
                or f"Failed to attach Commit0 agent container to {network_name}"
            )
        disconnect = run_process_command(
            ["docker", "network", "disconnect", "-f", "bridge", container_name],
            env=docker_env,
            timeout=120,
            task_id=task.instance_id,
        )
        if disconnect.returncode != 0:
            raise RuntimeError(
                normalize_terminal_output((disconnect.stdout + disconnect.stderr).strip())
                or "Failed to detach Commit0 agent container from Docker bridge"
            )

    def _verify_commit0_solve_network_boundary(
        self,
        *,
        task: Commit0Task,
        container_name: str,
        docker_env: dict[str, str],
        proxy_env: Mapping[str, str],
    ) -> str:
        """Verify Commit0 solve containers cannot reach package/source egress."""

        probe_script = r"""
import os
import socket
import sys
from urllib.parse import urlsplit


def proxy_endpoint(value):
    text = str(value or "").strip()
    if not text:
        return None
    parsed = urlsplit(text if "://" in text else "//" + text)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, int(port)


failures = []
for host in ("pypi.org", "github.com"):
    sock = socket.socket()
    sock.settimeout(5.0)
    try:
        sock.connect((host, 443))
    except OSError as exc:
        print(f"direct_external_denied {host}:443 {type(exc).__name__}: {exc}")
    else:
        failures.append(f"direct external connect succeeded: {host}:443")
    finally:
        try:
            sock.close()
        except OSError:
            pass

proxy_value = ""
for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
    proxy_value = os.environ.get(key, "")
    if proxy_value:
        break
endpoint = proxy_endpoint(proxy_value)
if endpoint is None:
    print("proxy_source_denied skipped: no solve proxy endpoint")
else:
    try:
        sock = socket.create_connection(endpoint, timeout=5.0)
        try:
            sock.sendall(
                b"CONNECT pypi.org:443 HTTP/1.1\r\n"
                b"Host: pypi.org:443\r\n"
                b"Connection: close\r\n\r\n"
            )
            response = sock.recv(256)
        finally:
            sock.close()
    except OSError as exc:
        failures.append(f"proxy source-denial probe could not reach sidecar: {exc}")
    else:
        status = response.split(b"\r\n", 1)[0].decode("latin1", "replace")
        print(f"proxy_source_denied_status {status}")
        if b"403 Forbidden" not in response:
            failures.append(f"proxy source request was not denied: {status!r}")

if failures:
    print("APEX_COMMIT0_SOLVE_NETWORK_BOUNDARY_FAILED")
    for failure in failures:
        print(failure)
    sys.exit(1)
print("APEX_COMMIT0_SOLVE_NETWORK_BOUNDARY_OK")
""".strip()
        command = ["docker", "exec", "-i"]
        for key in _COMMIT0_DOCKER_PROXY_ENV_KEYS:
            value = str(proxy_env.get(key) or "")
            if value:
                command.extend(["-e", f"{key}={value}"])
        command.extend([container_name, "python", "-c", probe_script])
        result = run_process_command(
            command,
            env=docker_env,
            timeout=60,
            task_id=task.instance_id,
        )
        output = normalize_terminal_output((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            raise RuntimeError(
                output or "Commit0 solve-network boundary preflight failed without output"
            )
        if "APEX_COMMIT0_SOLVE_NETWORK_BOUNDARY_OK" not in output:
            raise RuntimeError(
                output or "Commit0 solve-network boundary preflight did not report success"
            )
        return output

    def _build_linux_docker_runtime_env(
        self,
        task: Commit0Task,
        repo_dir: Path,
        runtime_dir: Path,
    ) -> dict[str, str]:
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            raise RuntimeError(
                f"Task '{task.repo_name}' requires Linux container runtime, but docker is unavailable."
            )

        runtime_dir.mkdir(parents=True, exist_ok=True)
        host_venv_dir = runtime_dir / ".venv"
        host_venv_bin = host_venv_dir / "bin"
        host_venv_bin.mkdir(parents=True, exist_ok=True)

        docker_env = _resolve_docker_sdk_env()
        container_name = self._linux_runtime_container_name(task)
        container_image = self._linux_runtime_container_image(task)
        container_venv = _COMMIT0_OFFICIAL_TESTBED_VENV
        agent_user = self._commit0_agent_container_user()
        sandbox_root = repo_dir.parent.resolve(strict=False)
        container_sandbox_root = _COMMIT0_DOCKER_WORKSPACE_ROOT
        try:
            # Commit0 Docker bind-mounts this task-only sandbox as /workspace;
            # CLI helpers may run under non-host UIDs, so the mount root must be traversable.
            sandbox_root.chmod(0o755)
        except OSError:
            logger.debug("Failed to make Commit0 Docker sandbox traversable", exc_info=True)
        # Host X2P proxies often bind loopback only; SETUP uses host-gateway
        # relays, then SOLVE rewrites these endpoints to the egress sidecar.
        setup_docker_proxy_env = self._commit0_docker_proxy_env()
        # Agent model proxies can be host-loopback services; Commit0 Docker
        # rewrites them into the task container exactly like HTTP proxy endpoints.
        setup_docker_model_proxy_env = self._commit0_docker_model_proxy_env()
        (
            docker_proxy_env,
            docker_model_proxy_env,
            egress_proxy_mappings,
        ) = _commit0_build_egress_proxy_plan(
            setup_docker_proxy_env,
            setup_docker_model_proxy_env,
        )
        docker_claude_plugboard_env = self._commit0_docker_claude_plugboard_env(
            docker_proxy_env,
            docker_model_proxy_env,
        )
        required_agent_cli_binaries = self._commit0_required_agent_cli_binaries()
        hard_required_agent_cli_binaries = self._commit0_hard_required_agent_cli_binaries()
        optional_agent_cli_binaries = self._commit0_optional_agent_cli_binaries()

        logger.info(
            "Routing Commit0 task '%s' through Linux docker runtime container '%s'.",
            task.repo_name,
            container_name,
        )

        started = False
        try:
            self._cleanup_linux_runtime_container(container_name, docker_env)
            self._ensure_commit0_docker_root_has_space(task, docker_env)
            agent_cli_bundle = self._ensure_commit0_agent_cli_bundle(
                docker_env=docker_env,
                container_image=container_image,
                docker_proxy_env=setup_docker_proxy_env,
                docker_model_proxy_env=setup_docker_model_proxy_env,
                required_binaries=required_agent_cli_binaries,
                hard_required_binaries=hard_required_agent_cli_binaries,
                optional_binaries=optional_agent_cli_binaries,
            )
            launch_command = [
                "docker",
                "run",
                "-d",
                # B3: do NOT use --rm. A large shared-container run (e.g. the
                # 3612-test pytest-on-pytest suite) can OOM-kill the container;
                # with --rm the kernel auto-removes it, after which every later
                # rollout that exec's into the named container hits "No such
                # container" and the whole repo cascades to failures. We keep
                # the container around and force-remove it ourselves via
                # _cleanup_linux_runtime_container (docker rm -f), which is wired
                # into the pre-launch sweep and every finally/failure path, so a
                # stopped/OOM-killed container is still reclaimed.
                "--name",
                container_name,
                *self._linux_runtime_container_label_args(task),
                "--security-opt",
                "no-new-privileges=true",
                # B3: give the shared per-task container real memory + shm
                # headroom. The default 64MB /dev/shm and an unbounded cgroup
                # let a big test suite either thrash shared memory (xdist,
                # multiprocessing, large fixtures) or get OOM-reaped mid-run.
                "--memory",
                self._commit0_docker_memory_limit(),
                "--memory-swap",
                self._commit0_docker_memory_limit(),
                "--shm-size",
                self._commit0_docker_shm_size(),
                *self._commit0_official_image_platform_args(),
                *self._commit0_docker_proxy_run_args(
                    setup_docker_proxy_env,
                    setup_docker_model_proxy_env,
                ),
                "--mount",
                # Commit0 Docker bind-mounts only the task sandbox at a stable
                # container path so agents cannot address sibling host artifacts.
                f"type=bind,source={sandbox_root},target={container_sandbox_root}",
                "--mount",
                # Commit0 Docker Desktop has a small overlayfs; agent CLIs are
                # prebuilt once in a host-backed Linux bundle and mounted read-only;
                # the solve runtime exposes only a filtered configured-binary view.
                (
                    "type=bind,"
                    f"source={agent_cli_bundle},"
                    f"target={_COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT},"
                    "readonly"
                ),
                "-w",
                container_sandbox_root,
                container_image,
                "sleep",
                "infinity",
            ]
            launch = run_process_command(
                launch_command,
                env=docker_env,
                timeout=180,
                task_id=task.instance_id,
            )
            # Commit0 Docker Desktop fact: freshly recreated host bind-source
            # directories can lag in the VM file-sharing view for a few seconds.
            if (
                launch.returncode != 0
                and sandbox_root.exists()
                and self._commit0_docker_bind_source_missing(launch.stdout + launch.stderr)
            ):
                self._commit0_refresh_docker_bind_source(
                    sandbox_root,
                    docker_env=docker_env,
                    container_image=container_image,
                    task_id=task.instance_id,
                )
                for delay_seconds in (0.0, 1.0, 3.0):
                    if delay_seconds:
                        time.sleep(delay_seconds)
                    launch = run_process_command(
                        launch_command,
                        env=docker_env,
                        timeout=180,
                        task_id=task.instance_id,
                    )
                    if launch.returncode == 0 or not self._commit0_docker_bind_source_missing(
                        launch.stdout + launch.stderr
                    ):
                        break
            if launch.returncode != 0:
                raise RuntimeError(
                    normalize_terminal_output((launch.stdout + launch.stderr).strip())
                    or f"Failed to start docker runtime container {container_name}"
                )
            started = True

            bootstrap = run_process_command(
                [
                    "docker",
                    "exec",
                    "-i",
                    container_name,
                    "bash",
                    "-lc",
                    self._commit0_linux_runtime_bootstrap_command(),
                ],
                env=docker_env,
                timeout=self._commit0_runtime_setup_timeout_seconds(),
                task_id=task.instance_id,
            )
            if bootstrap.returncode != 0:
                raise RuntimeError(
                    normalize_terminal_output((bootstrap.stdout + bootstrap.stderr).strip())
                    or f"Failed to bootstrap docker runtime for {task.repo_name}"
                )

            agent_cli_bootstrap = run_process_command(
                [
                    "docker",
                    "exec",
                    "-i",
                    container_name,
                    "bash",
                    "-lc",
                    self._commit0_agent_cli_bootstrap_command(
                        agent_user=agent_user,
                        required_binaries=required_agent_cli_binaries,
                        hard_required_binaries=hard_required_agent_cli_binaries,
                        agent_cli_bundle_container_path=(
                            _COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT
                        ),
                        filtered_agent_cli_bundle_container_path=(
                            _COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT
                        ),
                    ),
                ],
                env=docker_env,
                timeout=self._commit0_runtime_setup_timeout_seconds(),
                task_id=task.instance_id,
            )
            if agent_cli_bootstrap.returncode != 0:
                raise RuntimeError(
                    normalize_terminal_output(
                        (agent_cli_bootstrap.stdout + agent_cli_bootstrap.stderr).strip()
                    )
                    or f"Failed to install agent CLIs in docker runtime for {task.repo_name}"
                )

            if _commit0_official_image_python_repair_required(task.repo_name):
                # Commit0 cookiecutter image fact: installed dependency sources can
                # be NUL-filled in /testbed/.venv, so repair before solve egress is
                # closed and before agents/pytest use the venv.
                python_env_repair = run_process_command(
                    [
                        "docker",
                        "exec",
                        "-i",
                        container_name,
                        "bash",
                        "-lc",
                        _commit0_official_image_python_repair_command(container_venv),
                    ],
                    env=docker_env,
                    timeout=self._commit0_dependency_install_timeout_seconds(),
                    task_id=task.instance_id,
                )
                if python_env_repair.returncode != 0:
                    raise RuntimeError(
                        normalize_terminal_output(
                            (python_env_repair.stdout + python_env_repair.stderr).strip()
                        )
                        or f"Failed to repair Commit0 Python runtime for {task.repo_name}"
                    )

            _lock_down_commit0_container_testbed(
                container_name=container_name,
                docker_bin=docker_bin,
                docker_env=docker_env,
                container_venv=container_venv,
            )
            network_name = self._create_commit0_internal_solve_network(
                task=task,
                container_name=container_name,
                docker_env=docker_env,
            )
            self._start_commit0_egress_proxy_sidecar(
                task=task,
                container_name=container_name,
                container_image=container_image,
                network_name=network_name,
                docker_env=docker_env,
                mappings=egress_proxy_mappings,
            )
            self._move_commit0_agent_to_internal_solve_network(
                task=task,
                container_name=container_name,
                network_name=network_name,
                docker_env=docker_env,
            )
            # Commit0 solve fact: package/source egress must be physically denied
            # by the internal Docker network and sidecar before agents run.
            solve_network_preflight = self._verify_commit0_solve_network_boundary(
                task=task,
                container_name=container_name,
                docker_env=docker_env,
                proxy_env=docker_proxy_env,
            )

            container_python = f"{container_venv}/bin/python"
            wrappers = {
                "python": [container_python],
                "python3": [container_python],
                f"python{task.python_version}": [container_python],
                "pip": [container_python, "-m", "pip"],
                "pip3": [container_python, "-m", "pip"],
                "pytest": [container_python, "-m", "pytest"],
                "coverage": [container_python, "-m", "coverage"],
                "uv": ["uv"],
            }
            for command_name, command_tokens in wrappers.items():
                self._write_linux_runtime_wrapper(
                    host_venv_bin / command_name,
                    docker_bin=docker_bin,
                    docker_env=docker_env,
                    container_name=container_name,
                    container_venv=container_venv,
                    host_sandbox_root=sandbox_root,
                    container_sandbox_root=container_sandbox_root,
                    command_tokens=command_tokens,
                )

            (host_venv_bin / "activate").write_text(
                "\n".join(
                    [
                        "#!/bin/sh",
                        f"VIRTUAL_ENV={shlex.quote(str(host_venv_dir))}",
                        'PATH="$VIRTUAL_ENV/bin:$PATH"',
                        "export VIRTUAL_ENV PATH",
                    ]
                )
                + "\n"
            )
            (host_venv_bin / "activate").chmod(0o755)
            (host_venv_dir / "pyvenv.cfg").write_text(
                "\n".join(
                    [
                        "implementation = apex-docker-wrapper",
                        f"version_info = {task.python_version}",
                        f"virtualenv = {container_venv}",
                    ]
                )
                + "\n"
            )

            from apex.core.subprocess_utils import build_command_env

            env = build_command_env()
            env["VIRTUAL_ENV"] = str(host_venv_dir)
            env["PATH"] = f"{host_venv_bin}{os.pathsep}{env.get('PATH', '')}"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
            env.update(docker_proxy_env)
            env.update(docker_model_proxy_env)
            env.update(docker_claude_plugboard_env)
            env["PYTHONPATH"] = self._commit0_container_repo_pythonpath(task)
            agent_proxy_address = self._commit0_agent_proxy_address(docker_proxy_env)
            if agent_proxy_address:
                env["X2P_AGENT_PROXY_ADDRESS"] = agent_proxy_address
            env["APEX_COMMIT0_DOCKER_CONTAINER"] = container_name
            env["APEX_COMMIT0_CONTAINER_VENV"] = container_venv
            env["APEX_COMMIT0_AGENT_CLI_BUNDLE_HOST_ROOT"] = str(agent_cli_bundle)
            env["APEX_COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT"] = (
                _COMMIT0_AGENT_CLI_RAW_BUNDLE_CONTAINER_ROOT
            )
            env["APEX_COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT"] = (
                _COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT
            )
            selected_agent_cli_binaries, _ = self._commit0_selected_agent_cli_binaries(
                required_binaries=required_agent_cli_binaries,
                hard_required_binaries=hard_required_agent_cli_binaries,
            )
            env["APEX_COMMIT0_AGENT_CLI_SELECTED_BINARIES"] = " ".join(selected_agent_cli_binaries)
            env["APEX_COMMIT0_OFFICIAL_AUDIT_IMAGE"] = "1"
            env["APEX_COMMIT0_RUNTIME_IMAGE"] = container_image
            env["APEX_COMMIT0_RUNTIME_PLATFORM"] = _COMMIT0_OFFICIAL_IMAGE_PLATFORM
            env["APEX_COMMIT0_SOLVE_NETWORK"] = network_name
            env[_COMMIT0_SOLVE_NETWORK_PREFLIGHT_ENV] = solve_network_preflight
            env[_COMMIT0_EGRESS_PROXY_MAPPINGS_ENV] = _commit0_egress_proxy_mappings_json(
                egress_proxy_mappings
            )
            env["APEX_COMMIT0_CONTAINER_REPO_ROOT"] = self._commit0_container_repo_root(task)
            # Commit0 Docker mounts each task sandbox at /workspace, so
            # adapter-owned docker exec calls must translate host cwd paths.
            env[_COMMIT0_DOCKER_HOST_WORKDIR_ROOT_ENV] = str(sandbox_root)
            env[_COMMIT0_DOCKER_CONTAINER_WORKDIR_ROOT_ENV] = container_sandbox_root
            if agent_user:
                env["APEX_COMMIT0_DOCKER_USER"] = agent_user
            return env
        except Exception:
            if started:
                self._cleanup_linux_runtime_container(container_name, docker_env)
            raise

    def _git_clone_with_retry(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        timeout: int,
        max_attempts: int = 4,
    ) -> None:
        # ADDITIVE: a perturbed-variant task has no GitHub repo — its source is the
        # emitted LOCAL git mirror.  Go straight to the local-mirror fallback and
        # skip GitHub entirely (a real clone would fail non-transiently and abort).
        if _is_perturbed_task(task):
            if self._try_local_clone_fallback(task, repo_dir, timeout=timeout):
                return
            raise RuntimeError(
                f"perturbed variant {task.repo_name}: no local mirror found under "
                f"{_perturbed_mirror_roots()} (run scripts/perturb_commit0.py to emit it)"
            )
        # Transient HTTP proxy/CONNECT failures (Meta X2P agent flapping under
        # parallel benchmark load) account for the dominant failure mode in
        # historical full-suite runs: clones briefly fail with "Proxy CONNECT
        # aborted due to timeout", which permanently zeros the repo. Retry
        # with backoff so a transient blip costs minutes, not a failed score.
        url = f"https://github.com/{task.repo}.git"
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            shutil.rmtree(repo_dir, ignore_errors=True)
            try:
                self._run_process(
                    ["git", "clone", url, str(repo_dir)],
                    timeout=timeout,
                )
                if repo_dir.exists():
                    self._ensure_task_commit_objects_available(
                        task,
                        repo_dir,
                        timeout=timeout,
                    )
                if attempt > 1:
                    logger.info(
                        "Clone retry succeeded for repo=%s on attempt %d/%d",
                        task.repo_name,
                        attempt,
                        max_attempts,
                    )
                return
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                transient = (
                    "proxy connect" in message
                    or "could not resolve host" in message
                    or "connection reset" in message
                    or "connection refused" in message
                    or "operation timed out" in message
                    or "early eof" in message
                    or "rpc failed" in message
                    or "the requested url returned error: 5" in message
                    or "ssl_read" in message
                    or "ssl: tlsv1_alert" in message
                    or "timed out" in message
                )
                # Non-transient errors abort immediately. Transient
                # errors at the final attempt fall through to the local
                # mirror fallback below — re-raise only when we have NO
                # fallback to try (handled at the end of the loop).
                if not transient:
                    raise
                if attempt == max_attempts:
                    break
                backoff = min(60, 5 * (2 ** (attempt - 1)))
                logger.warning(
                    "Clone attempt %d/%d failed for repo=%s (transient): %s — retrying in %ds",
                    attempt,
                    max_attempts,
                    task.repo_name,
                    str(exc).splitlines()[0][:200],
                    backoff,
                )
                time.sleep(backoff)
        # Network exhausted — try local mirrors before declaring defeat.
        # Each entry in ``commit0_local_repo_roots`` is treated as a parent
        # dir; the standalone-commit0 prepare scripts also drop mirrors
        # under ``$TMPDIR/standalone-commit0-<repo>-*`` which we glob for as
        # a last resort.
        if self._try_local_clone_fallback(task, repo_dir, timeout=timeout):
            return
        if last_error is not None:
            raise last_error

    def _ensure_task_commit_objects_available(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        timeout: int,
    ) -> None:
        """Fetch dataset commit objects that a normal clone did not advertise."""

        # Commit0/Python harness fact: some task base SHAs are fetchable by
        # object ID but are not advertised by the repo's default refs, so a
        # normal clone can miss the exact tree the dataset asks us to evaluate.
        for sha in (task.base_commit, task.reference_commit):
            commit = str(sha or "").strip()
            if not commit:
                continue
            present = run_process_command(
                ["git", "cat-file", "-e", f"{commit}^{{tree}}"],
                cwd=repo_dir,
                timeout=60,
            )
            if present.returncode == 0:
                continue
            fetched = run_process_command(
                ["git", "fetch", "origin", commit, "--depth=1"],
                cwd=repo_dir,
                timeout=timeout,
            )
            if fetched.returncode != 0:
                logger.info(
                    "[commit0] unable to fetch dataset commit object repo=%s sha=%s: %s",
                    task.repo_name,
                    commit[:12],
                    ((fetched.stdout or "") + (fetched.stderr or "")).strip()[:240],
                )

    def _try_local_clone_fallback(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        timeout: int,
    ) -> bool:
        """Try to clone *task* from each configured local mirror.

        Phase 1.8 safety check: after a successful clone, attempt the
        same ``git checkout -B apex-base task.base_commit`` that
        ``_prepare_repo`` will perform later, then assert that
        ``git rev-parse HEAD`` equals ``task.base_commit``. A stale
        mirror (hasn't been refreshed since the dataset rev moved) will
        succeed at clone but produce a different SHA on checkout — we
        reject it with a ``mirror_rejected_stale`` log line so the
        caller falls through to the next candidate (and eventually a
        fresh GitHub clone).
        """
        candidates: list[Path] = []
        configured_roots = list(
            getattr(self.config.benchmark, "commit0_local_repo_roots", []) or []
        )
        # ADDITIVE: perturbed variants declare their emitted mirror parent dir in
        # the sidecar so they resolve without any config change (vanilla unaffected
        # — these roots only contain ``<repo>_perturbed`` mirrors).
        for _proot in _perturbed_mirror_roots():
            if _proot not in configured_roots:
                configured_roots.append(_proot)
        for root in configured_roots:
            try:
                root_path = Path(root).expanduser()
            except Exception:
                continue
            candidate = root_path / task.repo_name
            if candidate.exists():
                candidates.append(candidate)
        # Fallback glob — picks up the standalone Commit0 prep script's
        # default tempdir layout: ``<TMPDIR>/standalone-commit0-<repo>-*``.
        try:
            tmp_root = Path(tempfile.gettempdir())
            for candidate in tmp_root.glob(f"standalone-commit0-{task.repo_name}-*"):
                if candidate.is_dir():
                    candidates.append(candidate)
        except Exception:
            pass
        for candidate in candidates:
            if not (candidate / ".git").exists():
                continue
            shutil.rmtree(repo_dir, ignore_errors=True)
            try:
                self._run_process(
                    [
                        "git",
                        "clone",
                        "--no-hardlinks",
                        str(candidate),
                        str(repo_dir),
                    ],
                    timeout=timeout,
                )
            except Exception as exc:
                logger.warning(
                    "Local clone fallback at %s failed for repo=%s: %s",
                    candidate,
                    task.repo_name,
                    str(exc).splitlines()[0][:200] if str(exc) else "",
                )
                continue
            # Phase 1.8: stale-mirror rejection. We pre-emptively run
            # the same checkout that ``_prepare_repo`` will issue and
            # verify the resolved SHA. A mirror that hasn't been
            # refreshed will either (a) fail the checkout entirely
            # because the SHA is unknown locally, or (b) succeed but
            # land on a different SHA because the ref points elsewhere.
            # In both cases we reject and try the next candidate.
            base_commit = (task.base_commit or "").strip()
            if base_commit:
                if not self._verify_local_mirror_base_commit(
                    repo_dir=repo_dir,
                    candidate=candidate,
                    base_commit=base_commit,
                    task=task,
                ):
                    shutil.rmtree(repo_dir, ignore_errors=True)
                    continue
            logger.info(
                "[commit0] clone fallback: used local mirror at %s",
                candidate,
            )
            return True
        return False

    def _verify_local_mirror_base_commit(
        self,
        *,
        repo_dir: Path,
        candidate: Path,
        base_commit: str,
        task: Commit0Task,
    ) -> bool:
        """Return True iff *repo_dir* can checkout *base_commit* cleanly.

        Runs ``git checkout -B apex-base <base_commit>`` followed by
        ``git rev-parse HEAD`` and compares against *base_commit*. On
        any mismatch (or if either git command fails), logs
        ``mirror_rejected_stale`` and returns False so the caller can
        clean up and try the next mirror.
        """
        try:
            self._run_process(
                ["git", "checkout", "-B", "apex-base", base_commit],
                cwd=repo_dir,
                timeout=300,
            )
        except Exception as exc:
            logger.warning(
                "[commit0] mirror_rejected_stale: mirror=%s repo=%s "
                "base_commit=%s checkout failed: %s",
                candidate,
                task.repo_name,
                base_commit,
                str(exc).splitlines()[0][:200] if str(exc) else "<no message>",
            )
            return False
        try:
            head_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "[commit0] mirror_rejected_stale: mirror=%s repo=%s rev-parse HEAD failed: %s",
                candidate,
                task.repo_name,
                exc,
            )
            return False
        head_sha = (head_result.stdout or "").strip()
        if head_result.returncode != 0 or head_sha != base_commit:
            logger.warning(
                "[commit0] mirror_rejected_stale: mirror=%s repo=%s "
                "expected base_commit=%s, got HEAD=%s",
                candidate,
                task.repo_name,
                base_commit,
                head_sha or "<empty>",
            )
            return False
        return True

    def _prepare_repo(
        self,
        task: Commit0Task,
        repo_dir: Path,
        runtime_dir: Path,
        *,
        force_linux_container: Optional[bool] = None,
    ) -> dict[str, str]:
        self._git_clone_with_retry(
            task,
            repo_dir,
            timeout=self._commit0_repo_clone_timeout_seconds(),
        )
        self._run_process(
            ["git", "checkout", "-B", "apex-base", task.base_commit],
            cwd=repo_dir,
            timeout=300,
        )
        if repo_dir.exists():
            # Commit0 repos such as web3.py may keep visible-test fixtures in
            # git submodules; populate gitlinks before history flattening turns
            # initialized submodules into ordinary rootless baseline files.
            self._sync_repo_submodules_if_present(task, repo_dir)
            self._scrub_benchmark_repo_history(
                repo_dir,
                preserve_refs={"refs/heads/apex-base"},
            )
        self._run_process(
            ["git", "config", "user.email", "apex@example.com"], cwd=repo_dir, timeout=60
        )
        self._run_process(["git", "config", "user.name", "APEX"], cwd=repo_dir, timeout=60)
        self._sync_repo_submodules_if_present(task, repo_dir)

        use_linux_container = (
            force_linux_container
            if force_linux_container is not None
            else self._commit0_prepare_in_linux_container_first(task)
        )
        env = (
            self._build_linux_docker_runtime_env(task, repo_dir, runtime_dir)
            if use_linux_container
            else self._build_runtime_env(task, repo_dir, runtime_dir)
        )
        venv_python = Path(env["VIRTUAL_ENV"]) / "bin" / "python"
        uv_pip = (
            f"{shlex.quote(str(Path(env['VIRTUAL_ENV']) / 'bin' / 'uv'))} pip"
            if use_linux_container
            else _resolve_uv_pip_command()
        )
        official_audit_image_runtime = (
            use_linux_container and env.get("APEX_COMMIT0_OFFICIAL_AUDIT_IMAGE") == "1"
        )

        if official_audit_image_runtime:
            xdist_prepared = False
            if self._commit0_pytest_xdist_worker_spec():
                # Commit0/Python xdist acceleration: official repo images may
                # omit pytest-xdist unless the upstream command already needs it.
                xdist_prepared = self._ensure_commit0_official_image_xdist(
                    task=task,
                    repo_dir=repo_dir,
                    env=env,
                )
            # Commit0 repo images already contain the benchmark dependency venv;
            # skip broad dependency recreation so local validation stays close
            # to the official image environment.
            atomic_write_json(
                runtime_dir / "official_audit_runtime.json",
                {
                    "repo": task.repo_name,
                    "image": env.get("APEX_COMMIT0_RUNTIME_IMAGE"),
                    "container_venv": env.get("APEX_COMMIT0_CONTAINER_VENV"),
                    "container_repo_root": env.get("APEX_COMMIT0_CONTAINER_REPO_ROOT"),
                    "skipped_prepare_install": True,
                    "pytest_xdist_prepared": xdist_prepared,
                    "rationale": (
                        "Commit0 rollout validation uses the official audit image "
                        "dependency environment instead of recreating a local venv."
                    ),
                },
            )
        else:
            for command in task.pre_install:
                if use_linux_container:
                    self._run_docker_shell_command(
                        container_name=str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or ""),
                        cwd=repo_dir,
                        command=_normalize_linux_package_command(command),
                        env=env,
                        timeout=self._commit0_runtime_setup_timeout_seconds(),
                        check=True,
                        container_venv=str(env.get("APEX_COMMIT0_CONTAINER_VENV") or ""),
                    )
                else:
                    self._run_command(
                        repo_dir,
                        command,
                        env=env,
                        timeout=self._commit0_runtime_setup_timeout_seconds(),
                        check=True,
                    )

            for requirements_file in task.packages:
                self._run_command(
                    repo_dir,
                    f"{uv_pip} install -r {shlex.quote(requirements_file)}",
                    env=env,
                    timeout=self._commit0_dependency_install_timeout_seconds(),
                    check=True,
                )
            for package in task.pip_packages:
                self._run_command(
                    repo_dir,
                    f"{uv_pip} install {shlex.quote(package)}",
                    env=env,
                    timeout=self._commit0_dependency_install_timeout_seconds(),
                    check=True,
                )

            self._ensure_test_dependencies(task, repo_dir, env, venv_python, uv_pip)
            # Validate the shared runtime before the editable project install can
            # introduce import side effects from an intentionally incomplete repo.
            self._run_environment_smoke_tests(repo_dir, env, venv_python, uv_pip)
            install_command = _rewrite_pip_command(task.install_command, uv_pip)
            try:
                self._run_command(
                    repo_dir,
                    install_command,
                    env=env,
                    timeout=self._commit0_dependency_install_timeout_seconds(),
                    check=True,
                )
            except RuntimeError as install_error:
                # Phase 4 10.L: ``pip install -e .`` failures from a broken
                # build_editable hook (PEP 660 metadata generation) shouldn't
                # be treated as fatal — the agent can still patch the source
                # tree (we set up PYTHONPATH to point at it) and pytest will
                # import directly without relying on the editable shim.
                # Strict scope: only swallow when (a) the failure looks like a
                # build_editable / metadata-generation crash AND (b) the
                # command is a single editable install.
                if not _looks_like_editable_install(
                    install_command
                ) or not _looks_like_build_editable_failure(str(install_error)):
                    raise
                logger.warning(
                    "[commit0] prepare_install_skipped_editable: %s",
                    str(install_error).splitlines()[0][:240] if str(install_error) else "",
                )
                self._record_diagnostic(
                    repo_dir,
                    "prepare_install_skipped_editable",
                    {
                        "install_command": install_command,
                        "error": str(install_error)[:1500],
                    },
                )
            # B4: best-effort install of declared test/dev extras so collection-time
            # dependencies (pytest plugins, fixtures, test runners) resolve. Missing
            # or broken extras must never fail repo preparation, so each is installed
            # independently with check=False.
            self._install_repo_test_extras_best_effort(repo_dir, env, uv_pip)
            # V4c anti-cheat: now that the editable project install has run, assert
            # the target package imports from inside repo_dir (not a pre-installed
            # real copy in site-packages); uninstall + reinstall editable if it
            # resolves outside the repo. Best-effort — never fail preparation.
            try:
                self._run_environment_smoke_tests(
                    repo_dir,
                    env,
                    venv_python,
                    uv_pip,
                    task=task,
                    check_editable_target=True,
                )
            except Exception as editable_check_exc:  # noqa: BLE001
                logger.warning(
                    "[commit0] v4c_editable_check_skipped for repo=%s: %s",
                    task.repo_name,
                    editable_check_exc,
                )
        self._run_process(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, timeout=120)
        self._run_process(["git", "clean", "-fdx"], cwd=repo_dir, timeout=120)
        self._sync_repo_submodules_if_present(task, repo_dir)
        # Phase 4 10.M repo-specific shims (reusing the same prepare-stage
        # extension point so order matters: shims run AFTER install so they
        # can target a populated venv, and AFTER git clean so dropped
        # files survive the next test run).
        try:
            _apply_commit0_repo_shims(task, repo_dir)
        except Exception as shim_exc:
            logger.warning(
                "[commit0] repo shims raised for repo=%s: %s",
                task.repo_name,
                shim_exc,
            )
        # Anti-cheat / no-upstream-leak: strip version + upstream-locator
        # breadcrumbs from the working tree and COMMIT them onto the base branch
        # so every forked worktree inherits a sanitized container. A commit0 task
        # is "implement from the visible tests"; the container must not reveal which
        # real release it is or where to fetch it. Best-effort: never break prepare.
        try:
            from apex_omega.eval.repo_sanitize import scrub_upstream_identifiers

            scrub_report = scrub_upstream_identifiers(repo_dir)
            logger.info(
                "[commit0] upstream_scrub repo=%s versions=%d urls=%d deleted=%d "
                "tags=%d remotes=%d committed=%s kept_pkg_version=%s",
                task.repo_name,
                len(scrub_report.get("version_files", [])),
                len(scrub_report.get("url_files", [])),
                len(scrub_report.get("deleted", [])),
                scrub_report.get("tags_removed", 0),
                len(scrub_report.get("remotes_removed", [])),
                scrub_report.get("committed"),
                scrub_report.get("kept_package_version"),
            )
        except Exception as scrub_exc:  # noqa: BLE001 - scrub must not break prepare
            logger.warning(
                "[commit0] upstream scrub skipped for repo=%s: %s",
                task.repo_name,
                scrub_exc,
            )
        return env

    def _ensure_commit0_official_image_xdist(
        self,
        *,
        task: Commit0Task,
        repo_dir: Path,
        env: dict[str, str],
    ) -> bool:
        container_name = str(env.get("APEX_COMMIT0_DOCKER_CONTAINER") or "")
        container_venv = str(env.get("APEX_COMMIT0_CONTAINER_VENV") or "").rstrip("/")
        if not container_name or not container_venv:
            return False
        python_bin = shlex.quote(f"{container_venv}/bin/python")
        probe = self._run_docker_shell_command(
            container_name=container_name,
            cwd=repo_dir,
            command=f"{python_bin} -c 'import xdist.plugin'",
            env=env,
            timeout=60,
            container_venv=container_venv,
            task_id=task.instance_id,
        )
        if probe.returncode == 0:
            return False
        try:
            self._copy_host_pytest_xdist_to_commit0_container(
                task=task,
                repo_dir=repo_dir,
                env=env,
                container_name=container_name,
                container_venv=container_venv,
            )
            verify = self._run_docker_shell_command(
                container_name=container_name,
                cwd=repo_dir,
                command=f"{python_bin} -c 'import xdist.plugin'",
                env=env,
                timeout=60,
                container_venv=container_venv,
                task_id=task.instance_id,
            )
            if verify.returncode == 0:
                return True
            raise RuntimeError(verify.output or "pytest-xdist import still failed")
        except Exception as exc:  # noqa: BLE001 - accelerator must not break prepare
            logger.warning(
                "[commit0] disabling pytest-xdist for %s: %s",
                task.repo_name,
                str(exc).splitlines()[0][:240] if str(exc) else "<unknown>",
            )
            self._commit0_pytest_xdist_disabled_repos.add(task.repo_name)
            return False

    def _copy_host_pytest_xdist_to_commit0_container(
        self,
        *,
        task: Commit0Task,
        repo_dir: Path,
        env: dict[str, str],
        container_name: str,
        container_venv: str,
    ) -> None:
        python_bin = shlex.quote(f"{container_venv}/bin/python")
        site_probe = self._run_docker_shell_command(
            container_name=container_name,
            cwd=repo_dir,
            command=(
                f"{python_bin} -c "
                + shlex.quote('import sysconfig; print(sysconfig.get_paths()["purelib"])')
            ),
            env=env,
            timeout=60,
            container_venv=container_venv,
            task_id=task.instance_id,
        )
        if site_probe.returncode != 0:
            raise RuntimeError(site_probe.output or "unable to locate container site-packages")
        site_packages = ""
        for line in reversed(site_probe.output.splitlines()):
            stripped = line.strip()
            if stripped.startswith("/"):
                site_packages = stripped
                break
        if not site_packages:
            raise RuntimeError("container site-packages path was empty")

        vendor_paths = _host_pytest_xdist_vendor_paths()
        if not vendor_paths:
            raise RuntimeError("host pytest-xdist vendor paths were unavailable")
        tmp_root = "/tmp/apex_pytest_xdist_vendor"
        self._run_docker_shell_command(
            container_name=container_name,
            cwd=repo_dir,
            command=f"rm -rf {shlex.quote(tmp_root)} && mkdir -p {shlex.quote(tmp_root)}",
            env=env,
            timeout=60,
            container_venv=container_venv,
            task_id=task.instance_id,
        )
        for source in vendor_paths:
            destination = f"{container_name}:{tmp_root}/{source.name}"
            result = run_process_command(
                ["docker", "cp", str(source), destination],
                env=_resolve_docker_sdk_env(),
                timeout=120,
                task_id=task.instance_id,
            )
            if result.returncode != 0:
                output = normalize_terminal_output(result.stdout + result.stderr).strip()
                raise RuntimeError(output or f"docker cp failed for {source}")
        self._run_docker_shell_command(
            container_name=container_name,
            cwd=repo_dir,
            command=(
                f"mkdir -p {shlex.quote(site_packages)} && "
                f"cp -a {shlex.quote(tmp_root)}/. {shlex.quote(site_packages)}/"
            ),
            env=env,
            timeout=120,
            check=True,
            container_venv=container_venv,
            task_id=task.instance_id,
        )

    def _commit0_pytest_xdist_vendor_mounts(self) -> list[dict[str, str]]:
        mounts: list[dict[str, str]] = []
        for source in _host_pytest_xdist_vendor_paths():
            mounts.append(
                {
                    "source": str(source),
                    "target": f"{_COMMIT0_PYTEST_XDIST_VENDOR_CONTAINER_ROOT}/{source.name}",
                    "readonly": "true",
                }
            )
        return mounts

    def _sync_repo_submodules_if_present(self, task: Commit0Task, repo_dir: Path) -> None:
        # Commit0 repos such as web3.py declare data submodules used by visible tests.
        result = sync_git_submodules(
            repo_dir,
            timeout=self._commit0_repo_clone_timeout_seconds(),
        )
        if result is None:
            return
        if result.returncode != 0:
            message = normalize_terminal_output(result.stdout + result.stderr).strip()
            raise RuntimeError(
                f"Commit0 submodule initialization failed for {task.repo_name}: "
                f"{message or 'no git output'}"
            )
        self._record_diagnostic(
            repo_dir,
            "submodules_initialized",
            {
                "repo": task.repo_name,
                "command": "git submodule update --init --recursive",
                "timeout_seconds": self._commit0_repo_clone_timeout_seconds(),
            },
        )

    def _scrub_benchmark_repo_history(
        self,
        repo_dir: Path,
        *,
        preserve_refs: set[str],
    ) -> None:
        """V1 anti-cheat: TRUE-flatten the checkout to a rootless base commit.

        Previously this only pruned refs/remotes/reflog + ``git gc``, which left
        every upstream commit reachable through ``apex-base`` HEAD's own parent
        chain — so ``git show <ancestor>:<stubbed_path>`` still recovered the
        gold implementation. The flatten ``rm -rf .git`` + ``git init`` makes the
        ancestry physically unreachable (no parent objects exist). ``preserve_refs``
        is now moot (the single ``apex-base`` root commit is the only ref) and is
        retained only for caller compatibility.
        """

        _flatten_repo_git_history(Path(repo_dir))

    def _git_lines(self, repo_dir: Path, args: list[str]) -> list[str]:
        command = "git " + " ".join(shlex.quote(arg) for arg in args)
        result = self._run_command(repo_dir, command, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(result.output or f"Command failed: {command}")
        return [line.strip() for line in result.output.splitlines() if line.strip()]

    def _record_diagnostic(
        self,
        repo_dir: Path,
        category: str,
        payload: dict[str, Any],
    ) -> None:
        """Record a non-fatal prepare-stage diagnostic alongside the repo.

        The ``.apex_prepare_diagnostics.json`` file is consumed by report
        rendering so operators can see what was skipped (e.g. an
        editable-install step swallowed by 10.L) without digging through
        process logs. Best-effort — missing dirs are tolerated.
        """

        try:
            target = Path(repo_dir) / ".apex_prepare_diagnostics.json"
            existing: list[Any]
            if target.exists():
                try:
                    existing = json.loads(target.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        existing = []
                except (json.JSONDecodeError, UnicodeDecodeError):
                    existing = []
            else:
                existing = []
            existing.append({"category": category, **payload})
            target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            logger.debug(
                "Failed to record diagnostic %s for repo_dir=%s",
                category,
                repo_dir,
                exc_info=True,
            )

    def _build_runtime_env(
        self,
        task: Commit0Task,
        repo_dir: Path,
        runtime_dir: Path,
    ) -> dict[str, str]:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        venv_dir = runtime_dir / ".venv"
        uv_cmd = _resolve_uv_command()
        self._run_process(
            [*uv_cmd, "venv", "--python", task.python_version, str(venv_dir)],
            cwd=repo_dir,
            timeout=self._commit0_runtime_setup_timeout_seconds(),
        )
        from apex.core.subprocess_utils import build_command_env

        env = build_command_env()
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        # Per-task sandbox dirs so concurrent rollouts (and concurrent tasks
        # under task_parallelism>1) don't race on /tmp or ~/.cache. Pylint
        # rollouts repeatedly clobbered each other on /var/folders/.../T/CONFIG
        # because pytest's tempfile.gettempdir() inherited the apex parent's
        # TMPDIR. Scoping HOME also prevents jinja/click/cookiecutter from
        # poisoning each other's ~/.cache.
        sandbox_home = (runtime_dir / "home").resolve()
        sandbox_tmp = (runtime_dir / "tmp").resolve()
        for path in (
            sandbox_home,
            sandbox_tmp,
            sandbox_home / ".cache",
            sandbox_home / ".config",
            sandbox_home / ".local" / "share",
            sandbox_home / ".local" / "state",
        ):
            path.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(sandbox_home)
        env["TMPDIR"] = str(sandbox_tmp)
        env["TEMP"] = str(sandbox_tmp)
        env["TMP"] = str(sandbox_tmp)
        env["XDG_CACHE_HOME"] = str(sandbox_home / ".cache")
        env["XDG_CONFIG_HOME"] = str(sandbox_home / ".config")
        env["XDG_DATA_HOME"] = str(sandbox_home / ".local" / "share")
        env["XDG_STATE_HOME"] = str(sandbox_home / ".local" / "state")
        return env

    def _ensure_test_dependencies(
        self,
        task: Commit0Task,
        repo_dir: Path,
        env: dict[str, str],
        venv_python: Path,
        uv_pip: str,
    ) -> None:
        required_packages = []
        module_package_pairs = [
            ("pytest", "pytest"),
            ("pytest_jsonreport", "pytest-json-report"),
            # setuptools 82 removed pkg_resources, which some Commit0 repos still import.
            ("pkg_resources", "setuptools<82"),
        ]
        # Pytest coverage plugins are only required when --cov / pytest-cov
        # appears in the repo's pytest command/config; installing absent
        # plugins unconditionally can force an avoidable PyPI fetch.
        for module_name, package_name in module_package_pairs:
            if not self._python_module_available(repo_dir, env, venv_python, module_name):
                required_packages.append(package_name)
        required_packages.extend(
            _infer_additional_test_packages(
                task.test_cmd,
                repo_root=repo_dir,
            )
        )
        if self._commit0_pytest_xdist_worker_spec():
            # Commit0/Python harness fact: pytest-xdist is only needed when the
            # adapter injects -n workers for local full-suite evaluation.
            required_packages.append("pytest-xdist")
        required_packages = list(dict.fromkeys(required_packages))
        if required_packages:
            package_args = " ".join(shlex.quote(package) for package in required_packages)
            self._run_command(
                repo_dir,
                f"{uv_pip} install {package_args}",
                env=env,
                timeout=self._commit0_dependency_install_timeout_seconds(),
                check=True,
            )

    def _run_environment_smoke_tests(
        self,
        repo_dir: Path,
        env: dict[str, str],
        venv_python: Path,
        uv_pip: str,
        *,
        task: Optional[Commit0Task] = None,
        check_editable_target: bool = False,
    ) -> None:
        if env.get("APEX_COMMIT0_DOCKER_CONTAINER"):
            probe_dir = (repo_dir.parent / ".smoke_probe").resolve()
            probe_dir.mkdir(parents=True, exist_ok=True)
        else:
            probe_dir = Path(tempfile.mkdtemp(prefix="apex-commit0-smoke-")).resolve()
        probe_env = dict(env)
        probe_env.pop("PYTHONPATH", None)
        try:
            for smoke_test in self._environment_smoke_tests(venv_python):
                result = self._run_command(
                    probe_dir,
                    smoke_test.probe,
                    env=probe_env,
                    timeout=60,
                )
                if result.returncode == 0:
                    continue

                if smoke_test.remediation_package:
                    self._run_command(
                        probe_dir,
                        f"{uv_pip} install {shlex.quote(smoke_test.remediation_package)}",
                        env=probe_env,
                        timeout=self._commit0_dependency_install_timeout_seconds(),
                        check=True,
                    )
                    retry = self._run_command(
                        probe_dir,
                        smoke_test.probe,
                        env=probe_env,
                        timeout=60,
                    )
                    if retry.returncode == 0:
                        continue
                    result = retry

                raise RuntimeError(
                    f"Environment smoke test failed ({smoke_test.name}): "
                    f"{result.output or smoke_test.probe}"
                )
            # V4c anti-cheat (editable-only check). Only meaningful AFTER the
            # editable project install, so the caller opts in via
            # ``check_editable_target=True`` from the post-install hook.
            if check_editable_target and task is not None:
                self._verify_editable_target_inside_repo(
                    task=task,
                    repo_dir=repo_dir,
                    env=env,
                    venv_python=venv_python,
                    uv_pip=uv_pip,
                    probe_dir=probe_dir,
                )
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def _verify_editable_target_inside_repo(
        self,
        *,
        task: Commit0Task,
        repo_dir: Path,
        env: dict[str, str],
        venv_python: Path,
        uv_pip: str,
        probe_dir: Path,
    ) -> None:
        """V4c anti-cheat: ensure the target package imports from inside repo_dir.

        A pre-installed *real* (non-editable) copy of the target distribution in
        ``site-packages`` would let pytest import the complete upstream
        implementation instead of the intentionally-stubbed repo source — i.e.
        the score reflects the installed package, not the candidate's edits.

        For each candidate import name (derived from ``src_dir``/``repo_name``)
        we resolve ``import <pkg>; print(__file__)`` from a probe dir OUTSIDE
        the repo (PYTHONPATH stripped so cwd/src can't shadow site-packages).
        If the resolved ``__file__`` is INSIDE ``repo_dir`` the install is
        already editable — nothing to do. If it resolves to a real copy in
        site-packages (outside repo_dir) AND an editable target genuinely exists
        in repo_dir, we uninstall that copy and reinstall ``-e .`` so imports
        bind to the candidate tree. The uninstall is gated on BOTH conditions so
        a legitimately namespaced dependency is never removed.
        """

        repo_dir = Path(repo_dir).resolve()
        quoted_python = shlex.quote(str(venv_python))
        probe_env = dict(env)
        probe_env.pop("PYTHONPATH", None)
        import_names = self._candidate_import_names(task)
        for import_name in import_names:
            # Only act on a package that actually lives in the repo tree (so the
            # editable target exists); otherwise this name is an external dep.
            if not self._target_package_exists_in_repo(task, repo_dir, import_name):
                continue
            probe = (
                f"{quoted_python} -c "
                f'"import {import_name} as _m; print(getattr(_m, \\"__file__\\", \\"\\") or \\"\\")"'
            )
            result = self._run_command(probe_dir, probe, env=probe_env, timeout=60)
            resolved = (result.output or "").strip().splitlines()
            resolved_path = resolved[-1].strip() if resolved else ""
            if result.returncode != 0 or not resolved_path:
                # Import failed entirely (e.g. C-extension build pending) — the
                # editable check can't conclude a cheat; leave it to pytest.
                continue
            try:
                resolved_abs = Path(resolved_path).resolve()
            except (OSError, ValueError):
                continue
            if self._path_is_within(resolved_abs, repo_dir):
                # Already importing from the candidate tree — editable/correct.
                continue
            # Import resolves OUTSIDE repo_dir => a non-editable real copy is
            # shadowing the stubbed source. Uninstall + reinstall editable.
            logger.warning(
                "[commit0] v4c_noneditable_target_detected: repo=%s import=%s "
                "resolved=%s (outside repo) -> reinstalling editable",
                task.repo_name,
                import_name,
                resolved_path,
            )
            self._record_diagnostic(
                repo_dir,
                "v4c_noneditable_target_reinstalled",
                {
                    "repo": task.repo_name,
                    "import_name": import_name,
                    "resolved_file": resolved_path,
                },
            )
            self._run_command(
                repo_dir,
                f"{quoted_python} -m pip uninstall -y {shlex.quote(import_name)}",
                env=env,
                timeout=self._commit0_dependency_install_timeout_seconds(),
            )
            reinstall = _rewrite_pip_command(task.install_command, uv_pip)
            if not _looks_like_editable_install(reinstall):
                reinstall = f"{uv_pip} install -e ."
            self._run_command(
                repo_dir,
                reinstall,
                env=env,
                timeout=self._commit0_dependency_install_timeout_seconds(),
            )
            # Only one reinstall is needed; subsequent names share the project.
            break

    def _target_package_exists_in_repo(
        self,
        task: Commit0Task,
        repo_dir: Path,
        import_name: str,
    ) -> bool:
        repo_dir = Path(repo_dir)
        roots = [repo_dir]
        if task.src_root:
            roots.insert(0, repo_dir / task.src_root)
        rel = import_name.replace(".", "/")
        for root in roots:
            package_path = root / rel
            module_path = package_path.with_suffix(".py")
            if package_path.exists() or module_path.exists():
                return True
        return False

    @staticmethod
    def _path_is_within(candidate: Path, parent: Path) -> bool:
        try:
            candidate = Path(candidate).resolve()
            parent = Path(parent).resolve()
        except (OSError, ValueError):
            return False
        if candidate == parent:
            return True
        return parent in candidate.parents

    def _environment_smoke_tests(
        self,
        venv_python: Path,
    ) -> list[_EnvironmentSmokeTest]:
        quoted_python = shlex.quote(str(venv_python))
        return [
            _EnvironmentSmokeTest(
                name="pytest_jsonreport_plugin",
                probe=(
                    f"{quoted_python} -c "
                    '"import pytest_jsonreport; import pytest_jsonreport.plugin"'
                ),
                remediation_package="pytest-json-report",
            ),
            _EnvironmentSmokeTest(
                name="pkg_resources_legacy_api",
                probe=(
                    f"{quoted_python} -c "
                    '"import pkg_resources; '
                    "dist = pkg_resources.get_distribution('setuptools'); "
                    'assert dist.version"'
                ),
                remediation_package="setuptools<82",
            ),
        ]

    def _python_module_available(
        self,
        repo_dir: Path,
        env: dict[str, str],
        venv_python: Path,
        module_name: str,
    ) -> bool:
        probe = self._run_command(
            repo_dir,
            f'{shlex.quote(str(venv_python))} -c "import {module_name}"',
            env=env,
            timeout=60,
        )
        return probe.returncode == 0

    def _resolve_test_runner_adapter(
        self,
        task: Commit0Task,
        repo_dir: Optional[Path],
    ) -> Optional[Any]:
        """Pick the right TestRunnerAdapter for this task.

        Currently every commit0 task is pytest, so the explicit hint
        wins; but the lookup goes through the registry so a future
        polyglot benchmark only has to set a different framework field
        to swap in jest / go-test / cargo-test / junit / etc.
        """
        try:
            from ..core.test_runners import adapter_for_task
        except ImportError:
            return None
        framework_hint = getattr(task, "framework", None) or "pytest"
        workspace = repo_dir if repo_dir is not None else Path(".")
        return adapter_for_task(framework_hint, workspace)

    def _build_test_command(
        self,
        task: Commit0Task,
        python_executable: str,
        report_file: str,
        *,
        expected_test_ids: Optional[list[str]] = None,
        repo_dir: Optional[Path] = None,
        xdist_context: str = "candidate_eval",
    ) -> str:
        # Local candidate evaluation runs inside copied rollout worktrees while
        # the task venv was installed from the base checkout. Prepending the
        # current worktree import roots keeps editable-install imports pointed at
        # the candidate workspace so local selection tracks the patch that would
        # be applied in-place by the official harness.
        python_path_entries = ["."]
        if task.src_root:
            python_path_entries.insert(0, task.src_root)
        python_path = ":".join(dict.fromkeys(python_path_entries))

        ids_filter_prefix = ""
        if expected_test_ids and repo_dir is None:
            # Without a repo_dir we can't stage the ids file used by the
            # post-run Commit0 scorer, so refuse instead of producing a
            # misleading unscored full-suite run.
            raise RuntimeError(
                "expected_test_ids set without repo_dir; refusing to run without ids file"
            )
        use_expected_id_scoring = bool(expected_test_ids) and repo_dir is not None
        if use_expected_id_scoring:
            assert repo_dir is not None
            _stage_expected_ids_filter(repo_dir, expected_test_ids or [])
            ids_filter_prefix = (
                f"export {_APEX_EXPECTED_IDS_ENV_VAR}="
                f'"{_APEX_EXPECTED_IDS_FILENAME}" && '
                'export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}" && '
            )
            test_cmd = task.test_cmd.strip()
            if _commit0_pytest_command_needs_test_dir(test_cmd, task.test_dir):
                test_cmd = f"{test_cmd} {shlex.quote(task.test_dir)}"
            original_test_cmd = test_cmd
            test_cmd = _rewrite_pytest_command(
                test_cmd,
                python_executable,
                disable_plugin_autoload=False,
            )
            test_cmd = self._maybe_add_commit0_pytest_xdist(
                task,
                test_cmd,
                original_command=original_test_cmd,
                xdist_context=xdist_context,
            )
            # Commit0 official audit runs the full test directory, then scores
            # only expected IDs. Local validation must preserve that execution
            # context so non-scored tests cannot hide order/global-state
            # regressions that later turn an expected test red.
            extras: list[str] = [
                "-p pytest_jsonreport.plugin",
                *_expected_id_pytest_plugin_args(task, repo_dir),
            ]
        else:
            test_cmd = task.test_cmd.strip()
            if _commit0_pytest_command_needs_test_dir(test_cmd, task.test_dir):
                test_cmd = f"{test_cmd} {shlex.quote(task.test_dir)}"
            original_test_cmd = test_cmd
            test_cmd = _rewrite_pytest_command(test_cmd, python_executable)
            test_cmd = self._maybe_add_commit0_pytest_xdist(
                task,
                test_cmd,
                original_command=original_test_cmd,
                xdist_context=xdist_context,
            )
            extras = []

        parsed_test_cmd = parse_pytest_command(test_cmd)
        option_tokens = list(parsed_test_cmd.option_tokens) if parsed_test_cmd else []
        pending_option_tokens = option_tokens + shlex.split(" ".join(extras))
        if not _pytest_command_loads_plugin(
            pending_option_tokens,
            "pytest_jsonreport.plugin",
        ):
            extras.append("-p pytest_jsonreport.plugin")
        if not _pytest_command_has_exact_option(option_tokens, "--json-report"):
            extras.append("--json-report")
        added_report_file_option = False
        if not _pytest_command_has_exact_option(option_tokens, "--json-report-file"):
            extras.append(f"--json-report-file={shlex.quote(report_file)}")
            added_report_file_option = True
        if not _pytest_command_has_exact_option(
            option_tokens,
            "--continue-on-collection-errors",
        ):
            extras.append("--continue-on-collection-errors")
        # Pytest's .pytest_cache stores last-run failures and parametrize
        # ids; carrying it across rollouts can cause non-deterministic
        # ordering or stale skips. Wipe before each evaluation.
        if not _pytest_command_has_exact_option(option_tokens, "--cache-clear"):
            extras.append("--cache-clear")

        # O4/NEW-I7: pytest applies the repo's config ``addopts`` at runtime, so a
        # plugin option there (e.g. pydantic's ``--memray``) makes the autoload-
        # disabled scoring run exit rc=4 ("unrecognized arguments") BEFORE
        # collection unless its plugin is loadable. Strip only the unloadable
        # plugin options (keeping core + loadable ones) by overriding addopts via
        # ``-o addopts=...``. The importability probe runs against the scoring venv
        # immediately before invocation, and every stripped option is logged.
        if repo_dir is not None:
            addopts_override = self._commit0_addopts_strip_override(
                repo_dir=repo_dir,
                python_executable=python_executable,
                task=task,
            )
            if addopts_override is not None:
                extras.append(addopts_override)

        if extras:
            test_cmd = f"{test_cmd} {' '.join(extras)}"
        test_cmd = _bound_commit0_pytest_output(test_cmd)
        report_dir_prefix = ""
        if added_report_file_option:
            report_parent = Path(report_file).parent
            if str(report_parent) not in {"", "."}:
                # Commit0/Python solve fact: pytest-json-report does not
                # guarantee parent-directory creation for report_file paths.
                report_dir_prefix = f"mkdir -p {shlex.quote(str(report_parent))} && "
        return (
            f'{report_dir_prefix}export PYTHONPATH="{python_path}:$PYTHONPATH" && '
            f"{ids_filter_prefix}{test_cmd}"
        )

    def _commit0_addopts_strip_override(
        self,
        *,
        repo_dir: Path,
        python_executable: str,
        task: Commit0Task,
    ) -> Optional[str]:
        """Build an ``-o addopts=...`` override stripping unloadable plugin opts.

        Returns ``None`` when there is nothing to strip (no repo addopts, or every
        plugin option is loadable) so the scoring command is unchanged in the common
        case. When at least one plugin option is unloadable, returns a single
        ``-o addopts=<surviving tokens>`` token (or ``-o addopts=`` to clear it) and
        logs every stripped option for auditability.
        """
        addopts_tokens = _read_repo_addopts_tokens(repo_dir)
        if not addopts_tokens:
            return None

        venv_python = Path(python_executable)

        def _is_importable(module: str) -> bool:
            try:
                probe = self._run_command(
                    repo_dir,
                    f'{shlex.quote(str(venv_python))} -c "import {module}"',
                    timeout=60,
                )
            except Exception:  # pragma: no cover - probe must never crash scoring
                return False
            return getattr(probe, "returncode", 1) == 0

        kept, stripped = _strip_unimportable_plugin_addopts(
            addopts_tokens,
            is_module_importable=_is_importable,
        )
        if not stripped:
            return None
        for option in stripped:
            logger.warning(
                "[commit0][addopts-strip] task=%s repo=%s dropped unloadable plugin "
                "addopts option %r (plugin not importable in scoring venv); pytest "
                "would otherwise exit rc=4 before collection",
                getattr(task, "task_id", getattr(task, "instance_id", "?")),
                repo_dir,
                option,
            )
        return f"-o addopts={shlex.quote(' '.join(kept))}"

    def _commit0_pytest_xdist_worker_spec(
        self,
        *,
        xdist_context: str = "candidate_eval",
    ) -> Optional[str]:
        raw = getattr(self.config.benchmark, "commit0_pytest_xdist_workers", "") or ""
        text = str(raw).strip().lower()
        if text in {"", "0", "false", "off", "none", "no"}:
            return None
        if text in {"auto", "true", "yes"}:
            # Commit0/Python pytest-xdist worker counts are per pytest process;
            # fair-share "auto" avoids CPU oversubscription under outer task lanes.
            return str(
                _commit0_pytest_xdist_fair_share_worker_count(
                    self.config,
                    xdist_context=xdist_context,
                )
            )
        if text in {"pytest-auto", "pytest_auto", "literal-auto", "literal_auto"}:
            return "auto"
        if text == "logical":
            return "logical"
        if text == "max":
            return str(
                _commit0_pytest_xdist_fair_share_worker_count(
                    self.config,
                    xdist_context=xdist_context,
                )
            )
        try:
            workers = int(text)
        except ValueError:
            logger.warning(
                "[commit0] ignoring invalid commit0_pytest_xdist_workers=%r",
                raw,
            )
            return None
        if workers <= 0:
            return None
        return str(workers)

    def _commit0_pytest_xdist_dist_mode(self) -> Optional[str]:
        raw = getattr(self.config.benchmark, "commit0_pytest_xdist_dist", "") or ""
        text = str(raw).strip()
        if not text or text.lower() in {"false", "off", "none", "no"}:
            return None
        return text

    def _maybe_add_commit0_pytest_xdist(
        self,
        task: Commit0Task,
        command: str,
        *,
        original_command: str,
        xdist_context: str = "candidate_eval",
    ) -> str:
        workers = self._commit0_pytest_xdist_worker_spec(xdist_context=xdist_context)
        if not workers:
            return command
        if task.repo_name in self._commit0_pytest_xdist_disabled_repos:
            return command
        parsed = parse_pytest_command(command)
        if parsed is None:
            return command
        option_tokens = _canonicalize_pytest_xdist_plugin_tokens(list(parsed.option_tokens))
        if not _pytest_command_loads_plugin(option_tokens, "xdist"):
            # Commit0/Python pytest-randomly checks hasplugin("xdist") to seed
            # workers; loading "xdist.plugin" directly skips that hook.
            option_tokens.extend(["-p", "xdist"])
        if not _pytest_command_has_xdist_workers(option_tokens):
            option_tokens.extend(["-n", workers])
        if not _pytest_command_has_xdist_dist(option_tokens):
            dist_mode = (
                _pytest_xdist_dist_from_command(original_command)
                or self._commit0_pytest_xdist_dist_mode()
            )
            if dist_mode:
                option_tokens.extend(["--dist", dist_mode])
        rewritten = type(parsed)(
            shell_prefix_tokens=parsed.shell_prefix_tokens,
            env_prefix_tokens=parsed.env_prefix_tokens,
            invocation_tokens=parsed.invocation_tokens,
            option_tokens=tuple(option_tokens),
            target_tokens=parsed.target_tokens,
        )
        return render_pytest_command(rewritten, disable_plugin_autoload=False)

    def _collect_evaluation(
        self,
        task: Commit0Task,
        repo_dir: Path,
        command_result: "_CommandResult",
        report_file: str,
        expected_test_ids_for_scoring: Optional[list[str]] = None,
        use_expected_test_scoring: bool = True,
    ) -> Commit0Evaluation:
        report_path = repo_dir / report_file
        # Phase 1.2: track the shell rc separately so we can compare it
        # against the pytest-json report.exitcode and emit a diagnostic
        # warning when they disagree (regardless of whether we choose to
        # apply the APEX-private rewrite).
        shell_rc = int(command_result.returncode)
        evaluation = Commit0Evaluation(
            returncode=shell_rc,
            output=normalize_terminal_output(command_result.output),
            raw_returncode=shell_rc,
            report_path=str(report_path),
        )
        evaluation.score_source = "shell_rc"
        expected_test_ids: list[str] = []
        if use_expected_test_scoring:
            source_ids = (
                expected_test_ids_for_scoring
                if expected_test_ids_for_scoring is not None
                else _load_expected_test_ids(task.repo_name)
            )
            expected_test_ids = [test_id for test_id in source_ids if test_id]
        if (
            use_expected_test_scoring
            and not expected_test_ids
            and _commit0_expected_id_scoring_required(self.config)
        ):
            # Commit0 gold scoring is defined by the expected pytest-id inventory;
            # a raw pytest summary is diagnostic only when that inventory is missing.
            evaluation.returncode = 1
            evaluation.scoring_source = "commit0_test_ids"
            evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
            evaluation.total_tests = 0
            evaluation.passed = 0
            evaluation.failed = 0
            evaluation.errors = 0
            evaluation.skipped = 0
            evaluation.expected_test_coverage = {
                "expected_test_count": 0,
                "matched_expected_test_count": 0,
                "missing_expected_test_count": 0,
                "coverage_preserved": False,
                "inventory_unavailable": True,
            }
            evaluation.diagnostics["harness_failure"] = True
            evaluation.diagnostics["expected_test_inventory_unavailable"] = True
            evaluation.diagnostics["scoring_universe"] = "expected_test_ids"
            _commit0_evaluation_decision(evaluation)
            return evaluation
        if expected_test_ids:
            evaluation.scoring_source = "commit0_test_ids"
            evaluation.total_tests = len(expected_test_ids)
            evaluation.passed = 0
            evaluation.failed = evaluation.total_tests
            evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
            evaluation.expected_test_coverage = {
                "expected_test_count": evaluation.total_tests,
                "matched_expected_test_count": 0,
                "missing_expected_test_count": evaluation.total_tests,
                "coverage_preserved": False,
            }

        if not report_path.exists():
            if _commit0_returncode_is_native_crash(shell_rc):
                # P0 harness fix: a native interpreter crash (segfault/abort/signal)
                # with NO json report is an environment failure, not a real outcome.
                # Classify it as harness_failure -> INDETERMINATE (excluded) so a
                # crashed interpreter is never scored as a genuine 0. This MUST run
                # before the collection-failure-before-report branch: a crash can
                # truncate output so it coincidentally matches a collection marker,
                # which would otherwise be mis-scored as errors==total (a false zero).
                evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
                evaluation.passed = 0
                evaluation.failed = 0
                evaluation.errors = 0
                evaluation.skipped = 0
                evaluation.diagnostics["harness_failure"] = True
                evaluation.diagnostics["native_crash_returncode"] = int(shell_rc)
                evaluation.diagnostics["pytest_json_report_missing_reason"] = (
                    "native_interpreter_crash"
                )
                evaluation.diagnostics["scored_signal_count"] = 0
                if expected_test_ids:
                    evaluation.expected_test_coverage = {
                        "expected_test_count": evaluation.total_tests,
                        "matched_expected_test_count": 0,
                        "missing_expected_test_count": evaluation.total_tests,
                        "coverage_preserved": False,
                        "scoring_universe_unobserved": True,
                    }
                _commit0_evaluation_decision(evaluation)
                return evaluation
            if expected_test_ids and _pytest_output_indicates_collection_failure_before_report(
                evaluation.output
            ):
                # pytest-json-report may not write a report when pytest aborts during conftest/import collection; score that as a Commit0 collection failure.
                evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
                evaluation.passed = 0
                evaluation.failed = 0
                evaluation.errors = evaluation.total_tests
                evaluation.skipped = 0
                evaluation.expected_test_coverage = {
                    "expected_test_count": evaluation.total_tests,
                    "matched_expected_test_count": 0,
                    "missing_expected_test_count": evaluation.total_tests,
                    "skipped_expected_test_count": 0,
                    "coverage_preserved": False,
                    "collected_test_count": 0,
                    "collection_failed_before_report": True,
                }
                evaluation.diagnostics["pytest_json_report_missing_reason"] = (
                    "collection_failure_before_report"
                )
                evaluation.diagnostics["scored_signal_count"] = evaluation.total_tests
                _commit0_evaluation_decision(evaluation)
                return evaluation
            if _apply_expected_id_terminal_summary_fallback(
                evaluation,
                expected_test_count=len(expected_test_ids),
                reason="pytest_json_report_missing",
            ):
                return evaluation
            evaluation.diagnostics["parser_error"] = "pytest_json_report_missing"
            evaluation.diagnostics["scored_signal_count"] = 0
            if expected_test_ids:
                evaluation.failed = 0
                evaluation.errors = 0
                evaluation.passed = 0
                evaluation.expected_test_coverage = {
                    "expected_test_count": evaluation.total_tests,
                    "matched_expected_test_count": 0,
                    "missing_expected_test_count": evaluation.total_tests,
                    "coverage_preserved": False,
                    "scoring_universe_unobserved": True,
                }
            _commit0_evaluation_decision(evaluation)
            return evaluation

        # Try non-pytest adapter-mediated parsing first; the normal Commit0
        # pytest path reads the JSON report once below because large expected-id
        # reports can be expensive to parse repeatedly.
        adapter = self._resolve_test_runner_adapter(task, repo_dir)
        run_result = None
        if adapter is not None and getattr(adapter, "name", "") != "pytest":
            try:
                run_result = adapter.parse_report(report_path)
            except Exception:
                run_result = None
        if run_result is not None and run_result.outcomes:
            outcomes = dict(run_result.outcomes)
            report = load_pytest_json_report(report_path) or {}
        else:
            report = load_pytest_json_report(report_path)
            if report is None:
                if _apply_expected_id_terminal_summary_fallback(
                    evaluation,
                    expected_test_count=len(expected_test_ids),
                    reason="pytest_json_report_unreadable",
                ):
                    return evaluation
                evaluation.diagnostics["parser_error"] = "pytest_json_report_unreadable"
                evaluation.diagnostics["scored_signal_count"] = 0
                if expected_test_ids:
                    evaluation.failed = 0
                    evaluation.errors = 0
                    evaluation.passed = 0
                    evaluation.expected_test_coverage = {
                        "expected_test_count": evaluation.total_tests,
                        "matched_expected_test_count": 0,
                        "missing_expected_test_count": evaluation.total_tests,
                        "coverage_preserved": False,
                        "scoring_universe_unobserved": True,
                    }
                _commit0_evaluation_decision(evaluation)
                return evaluation
            tests = extract_pytest_report_tests(report)
            outcomes = _extract_pytest_report_outcomes(tests)

        if expected_test_ids:
            if not outcomes and _apply_expected_id_terminal_summary_fallback(
                evaluation,
                expected_test_count=len(expected_test_ids),
                reason="pytest_json_report_without_per_test_outcomes",
            ):
                return evaluation
            expected_coverage = summarize_expected_pytest_coverage(
                expected_test_ids,
                outcomes,
            )
            evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
            evaluation.passed = int(expected_coverage.get("passed") or 0)
            evaluation.errors = int(expected_coverage.get("errors") or 0)
            evaluation.skipped = int(expected_coverage.get("skipped") or 0)
            evaluation.failed = int(expected_coverage.get("failed") or 0)
            missing_expected = int(expected_coverage.get("missing_expected_test_count") or 0)
            evaluation.expected_test_coverage = {
                "expected_test_count": evaluation.total_tests,
                "matched_expected_test_count": int(
                    expected_coverage.get("matched_expected_test_count") or 0
                ),
                "missing_expected_test_count": missing_expected,
                "skipped_expected_test_count": evaluation.skipped,
                "coverage_preserved": not bool(missing_expected),
                "collected_test_count": len(outcomes),
            }
            extra_diagnostics = _pytest_extra_non_scored_diagnostics(
                repo_name=task.repo_name,
                outcomes=outcomes,
                expected_test_ids=expected_test_ids,
            )
            evaluation.diagnostics["extra_non_scored_tests"] = extra_diagnostics
            if extra_diagnostics.get("failed") or extra_diagnostics.get("errors"):
                evaluation.diagnostics["raw_returncode_non_scoring_reason"] = (
                    "extra non-scored pytest outcomes are diagnostic under commit0_test_ids scoring"
                )
            if _commit0_can_normalize_benign_extra_returncode(
                evaluation,
                extra_diagnostics,
            ):
                evaluation.returncode = 0
                evaluation.score_source = "commit0_benign_extra_normalized"
                evaluation.diagnostics["benign_extra_tests_ignored_for_returncode"] = {
                    "raw_returncode": shell_rc,
                    "failed": int(extra_diagnostics.get("benign_failed") or 0),
                    "errors": int(extra_diagnostics.get("benign_errors") or 0),
                    "sample_failures": list(extra_diagnostics.get("benign_sample_failures") or [])[
                        :20
                    ],
                }
            # Phase 1.2: gate the APEX-private exit-code rewrite. The
            # historical behaviour preferred pytest's own json-report
            # exitcode over the shell returncode (the wrapper script can
            # be killed by an outer signal so collection succeeded →
            # expected tests passed → but the host shell still flagged
            # rc=1; fabric and tornado lose passing scores otherwise).
            # That rewrite is now opt-in via
            # ``BenchmarkConfig.commit0_use_pytest_json_exitcode``.
            # When the rewrite is OFF (the default), we still detect the
            # disagreement and surface it in ``diagnostics`` so reviewers
            # can audit it without changing the headline number.
            report_exitcode = report.get("exitcode")
            if isinstance(report_exitcode, int) and not isinstance(report_exitcode, bool):
                if report_exitcode != shell_rc:
                    evaluation.diagnostics["pytest_returncode_disagrees_with_report"] = {
                        "shell_rc": shell_rc,
                        "report_exitcode": report_exitcode,
                    }
                rewrite_enabled = bool(
                    getattr(
                        self.config.benchmark,
                        "commit0_use_pytest_json_exitcode",
                        False,
                    )
                )
                if rewrite_enabled:
                    evaluation.returncode = report_exitcode
                    evaluation.score_source = "apex_private_pytest_json"
                elif evaluation.score_source != "commit0_benign_extra_normalized":
                    evaluation.score_source = "shell_rc"
            _commit0_evaluation_decision(evaluation)
            return evaluation

        summary = report.get("summary") or {}
        passed = (
            int(summary.get("passed", 0) or 0)
            + int(summary.get("xfailed", 0) or 0)
            + int(summary.get("xpassed", 0) or 0)
        )
        failed = int(summary.get("failed", 0) or 0)
        errors = int(summary.get("error", 0) or 0) + int(summary.get("errors", 0) or 0)
        skipped = int(summary.get("skipped", 0) or 0)
        total = passed + failed + errors
        evaluation.scoring_source = "pytest_summary"
        evaluation.evaluation_backend = COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST
        evaluation.passed = passed
        evaluation.failed = failed
        evaluation.errors = errors
        evaluation.skipped = skipped
        evaluation.total_tests = total
        _commit0_evaluation_decision(evaluation)
        return evaluation

    def _resolve_repo_filter(self, repos: Optional[list[str]]) -> Optional[set[str]]:
        if repos:
            return {repo.split("/")[-1] for repo in repos}
        if self.split == "all":
            return None
        if self.split == "lite":
            return set(COMMIT0_LITE_REPOS)
        return {self.split}

    def _run_process(
        self,
        command: list[str],
        cwd: Optional[Path] = None,
        timeout: int = 300,
        task_id: str | None = None,
    ) -> None:
        result = run_process_command(
            command,
            cwd=cwd,
            timeout=timeout,
            task_id=task_id,
        )
        if result.returncode != 0:
            raise RuntimeError(
                (result.stdout + result.stderr).strip() or f"Command failed: {command}"
            )

    def _run_command(
        self,
        cwd: Path,
        command: str,
        env: Optional[dict[str, str]] = None,
        timeout: int = 300,
        check: bool = False,
        task_id: str | None = None,
    ) -> "_CommandResult":
        result = run_shell_command(
            command,
            cwd,
            env=env,
            timeout=timeout,
            task_id=task_id,
        )
        output = normalize_terminal_output(result.stdout + result.stderr).strip()
        if check and result.returncode != 0:
            raise RuntimeError(output or f"Command failed: {command}")
        return _CommandResult(returncode=result.returncode, output=output)

    def _classify_prepare_error(self, exc: Exception) -> tuple[bool, Optional[str]]:
        """Classify a prepare-stage failure so the report counts it correctly.

        Returns ``(skipped, skip_category)``. When ``skipped`` is True, the
        repo is excluded from the runnable denominator instead of being
        scored as ``other_failure: 0%`` — APEX never had an opportunity
        to act, so penalising the model for upstream defects gives a
        misleading headline.

        Phase 1: this method retains its historical
        ``(skipped, skip_category)`` contract for backwards compat with
        callers that key off the categorical string. Internally we ALSO
        consult the new core ``classify_failure`` and store the rich
        classification on the per-task evaluation via
        :meth:`_classify_prepare_error_with_core`. Direct callers who
        need the structured classification should use that helper
        instead.
        """

        message = str(exc).lower()
        if "requires linux package installation" in message:
            return True, "unsupported_host"
        if "requires linux container runtime" in message:
            return True, "unsupported_host"
        # ``git clone`` exhausted retries against GitHub AND no local mirror
        # was usable. Distinct from agent failures — the harness never got
        # the source, so it's a network/infra issue rather than an APEX
        # regression. Retry orchestration can route these differently.
        if "git clone" in message and (
            "proxy connect" in message
            or "could not resolve host" in message
            or "connection reset" in message
            or "connection refused" in message
            or "operation timed out" in message
            or "early eof" in message
            or "rpc failed" in message
            or "ssl_read" in message
            or "the requested url returned error: 5" in message
        ):
            return True, "clone_network_failure"
        # Baseline pytest exceeded the 30-min wall-clock cap before APEX
        # could plan or act. Slow-test repos like tlslite-ng on macOS
        # without OpenSSL pkg-config bindings hit this regularly.
        if "timed out after" in message or "timeoutexpired" in message:
            return True, "baseline_timeout"
        # ``pip install -e .`` failed at the build-backend stage. The
        # repo's setup.py is broken on the declared Python version
        # (e.g. imapclient's ``_imapclient_version_string``); APEX
        # cannot patch a repo that won't install.
        # Phase 4 10.L exemption: pure ``build_editable`` /
        # ``setuptools.build_meta`` / ``metadata-generation-failed``
        # failures are now swallowed by ``_prepare_repo`` (the agent can
        # still patch the source tree directly). They should NOT bubble
        # up here as ``upstream_install_broken``. Real PEP 517 / build
        # backend errors that escape that swallow path are still fatal.
        if (
            "build backend returned an error" in message or "pep 517" in message
        ) and not _looks_like_build_editable_failure(message):
            return True, "upstream_install_broken"
        if "setuptools.build_meta" in message or "metadata-generation-failed" in message:
            # If the install was specifically the editable shape, treat
            # it as a runtime warning — the prepare path swallowed it.
            return False, None
        # ``pre_install`` script raised — the repo's own bootstrap
        # script is broken (e.g. babel's CLDR import depends on
        # Python 3.12+ syntax in its own source, but the task is
        # declared for 3.10).
        if "pre_install" in message or "pre-install" in message:
            return True, "upstream_pre_install_broken"
        # SyntaxError during install / import means the repo's source
        # cannot be parsed by the declared interpreter — again, an
        # upstream defect rather than an APEX failure.
        if "syntaxerror" in message:
            return True, "upstream_syntax_error"
        return False, None

    @contextmanager
    def _official_eval_repo_dir(
        self,
        task: Commit0Task,
        repo_dir: Path,
        *,
        artifacts_dir: Path,
        label: str,
    ) -> Iterator[Path]:
        resolved_repo_dir = repo_dir.resolve()
        if task.repo_name in resolved_repo_dir.name:
            yield resolved_repo_dir
            return

        alias_root = artifacts_dir / "_repo_aliases"
        alias_root.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "eval"
        alias_path = alias_root / f"{task.repo_name}-{safe_label}"
        if alias_path.exists() or alias_path.is_symlink():
            try:
                if alias_path.is_dir() and not alias_path.is_symlink():
                    shutil.rmtree(alias_path, ignore_errors=True)
                else:
                    alias_path.unlink()
            except FileNotFoundError:
                pass
        try:
            alias_path.symlink_to(resolved_repo_dir, target_is_directory=True)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to create official evaluation alias for {resolved_repo_dir}: {exc}"
            ) from exc

        # Decision (Edit 4b): the alias is a symlink to the resolved repo,
        # so the expected-ids filter we wrote into the original repo_dir is
        # already visible via the alias path. As an additional guard we
        # mirror the filter into the alias path explicitly when the
        # original repo carries one — this is a no-op when the symlink
        # already exposes the file, but it covers the case where a future
        # caller passes a non-symlink alias dir. Smaller change than
        # rerouting cwd through ``resolved_repo_dir`` because the official
        # commit0 runner consumes ``base_dir.parent`` from the yielded
        # path directly.
        try:
            source_ids = resolved_repo_dir / _APEX_EXPECTED_IDS_FILENAME
            if source_ids.exists() and not (alias_path / _APEX_EXPECTED_IDS_FILENAME).exists():
                expected_ids = [
                    line.strip()
                    for line in source_ids.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                _stage_expected_ids_filter(alias_path, expected_ids)
        except Exception:
            logger.debug(
                "Failed to mirror expected-ids filter into alias dir %s",
                alias_path,
                exc_info=True,
            )

        try:
            yield alias_path
        finally:
            try:
                if alias_path.is_dir() and not alias_path.is_symlink():
                    shutil.rmtree(alias_path, ignore_errors=True)
                else:
                    alias_path.unlink()
            except FileNotFoundError:
                pass

    def _discover_rollout_worktree_paths(self, workspace_dir: Path) -> dict[int, Path]:
        """Return rollout worktrees keyed by rollout id.

        ApexOrchestrator creates a fresh ``apex_solve_*`` directory under the
        configured workspace and then places ``rollout_N`` worktrees inside it.
        Older callers and unit tests still use ``workspace/rollout_N``
        directly, so support both layouts.
        """

        discovered: list[tuple[int, float, Path]] = []
        for pattern in ("rollout_*", "apex_solve_*/rollout_*"):
            for worktree_path in workspace_dir.glob(pattern):
                if not worktree_path.is_dir():
                    continue
                rollout_id = _rollout_id_from_name(worktree_path.name)
                if rollout_id is None:
                    continue
                try:
                    mtime = worktree_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                discovered.append((rollout_id, mtime, worktree_path))

        worktrees: dict[int, Path] = {}
        for rollout_id, _mtime, worktree_path in sorted(
            discovered,
            key=lambda item: (item[0], item[1], str(item[2])),
        ):
            worktrees[rollout_id] = worktree_path
        return worktrees

    @staticmethod
    def _patch_header_token_path(raw_token: Any) -> str:
        token = str(raw_token or "").strip()
        if not token:
            return ""
        if "\t" in token:
            token = token.split("\t", 1)[0].strip()
        try:
            parts = shlex.split(token)
        except ValueError:
            parts = []
        if parts:
            token = parts[0]
        token = token.strip().strip('"')
        if token in {"", "/dev/null"}:
            return ""
        if token.startswith(("a/", "b/")):
            token = token[2:]
        return normalize_changed_path(token)

    @classmethod
    def _patch_text_changed_files(cls, patch_text: Any) -> list[str]:
        if not isinstance(patch_text, str) or not patch_text.strip():
            return []
        paths: list[str] = []

        def add(raw: Any) -> None:
            path = cls._patch_header_token_path(raw)
            if path:
                paths.append(path)

        for raw_line in patch_text.splitlines():
            line = raw_line.strip()
            if line.startswith("diff --git "):
                try:
                    parts = shlex.split(line)
                except ValueError:
                    parts = line.split()
                if len(parts) >= 4:
                    add(parts[2])
                    add(parts[3])
                continue
            if line.startswith(("--- ", "+++ ")):
                add(line[4:])
                continue
            if line.startswith("rename from "):
                add(line.removeprefix("rename from ").strip())
                continue
            if line.startswith("rename to "):
                add(line.removeprefix("rename to ").strip())
        return list(dict.fromkeys(paths))

    @classmethod
    def _rollout_summary_changed_files(
        cls,
        rollout_summary: Optional[dict[str, Any]],
    ) -> list[str]:
        if not isinstance(rollout_summary, dict):
            return []
        declared = [
            normalize_changed_path(str(path))
            for path in list(rollout_summary.get("changed_files") or [])
            if normalize_changed_path(str(path))
        ]
        patch_paths = cls._patch_text_changed_files(rollout_summary.get("patch"))
        return list(dict.fromkeys(declared + patch_paths))

    @staticmethod
    def _candidate_path_differs_from_baseline(
        candidate_worktree: Path,
        baseline_repo_dir: Optional[Path],
        rel_path: str,
    ) -> bool:
        if baseline_repo_dir is None or not baseline_repo_dir.exists():
            return True
        candidate_path = candidate_worktree / rel_path
        baseline_path = baseline_repo_dir / rel_path
        try:
            candidate_exists = candidate_path.exists() or candidate_path.is_symlink()
            baseline_exists = baseline_path.exists() or baseline_path.is_symlink()
            if not candidate_exists and not baseline_exists:
                return False
            if candidate_path.is_file() and baseline_path.is_file():
                return candidate_path.read_bytes() != baseline_path.read_bytes()
            return candidate_exists != baseline_exists or (
                candidate_path.is_dir() != baseline_path.is_dir()
            )
        except OSError:
            return True

    def _candidate_changed_files(
        self,
        worktree_path: Path,
        rollout_summary: Optional[dict[str, Any]] = None,
        *,
        baseline_repo_dir: Optional[Path] = None,
    ) -> list[str]:
        summary_changed = self._rollout_summary_changed_files(rollout_summary)
        if summary_changed and baseline_repo_dir is not None and baseline_repo_dir.exists():
            summary_changed = [
                path
                for path in summary_changed
                if self._candidate_path_differs_from_baseline(
                    worktree_path,
                    baseline_repo_dir,
                    path,
                )
            ]
        try:
            git_changed = list_git_changed_files(worktree_path)
        except Exception:
            git_changed = []
        return list(dict.fromkeys(summary_changed + git_changed))

    def _candidate_stub_findings(
        self,
        worktree_path: Path,
        changed_files: list[str],
    ) -> list[Any]:
        if not changed_files:
            return []
        try:
            from ..core.test_runners import detect_adapter
        except ImportError:
            detect_adapter = None
        try:
            adapter = detect_adapter(worktree_path) if detect_adapter is not None else None
            patterns = adapter.stub_patterns() if adapter is not None else []
            return scan_files_for_stubs(
                worktree_path,
                changed_files,
                adapter_stub_patterns=patterns,
            )
        except Exception:
            return []

    def _candidate_collection_critical_edit_paths(
        self,
        candidate_worktree: Path,
        changed_files: list[str],
        incomplete_test_files: Optional[list[str]] = None,
    ) -> tuple[str, list[str]]:
        try:
            from ..core.test_runners import detect_adapter
        except ImportError:
            return "", []
        try:
            adapter = detect_adapter(candidate_worktree)
            if adapter is None:
                return "", []
            # Pytest/Jest/etc. config and fixture files control the collected
            # scorer universe, so Commit0 gold candidates may not rewrite them.
            infrastructure = {
                normalize_changed_path(str(path))
                for path in adapter.infrastructure_paths(candidate_worktree)
                if normalize_changed_path(str(path))
            }
        except Exception:
            return "", []
        if not infrastructure:
            return getattr(adapter, "name", ""), []
        allowed = {
            normalize_changed_path(str(path))
            for path in list(incomplete_test_files or [])
            if normalize_changed_path(str(path))
        }
        changed = {
            normalize_changed_path(str(path))
            for path in list(changed_files or [])
            if normalize_changed_path(str(path))
        }
        violations = sorted(
            path for path in changed if path in infrastructure and path not in allowed
        )
        return str(getattr(adapter, "name", "") or ""), violations

    def _candidate_import_names(self, task: Commit0Task) -> list[str]:
        candidates: list[str] = []
        raw_src = task.src_dir.strip().strip("/")
        if raw_src:
            candidates.append(Path(raw_src).name)
        repo_name = task.repo_name.strip()
        if repo_name:
            candidates.append(repo_name)
            if "." in repo_name:
                candidates.append(repo_name.split(".", 1)[0])
        normalized: list[str] = []
        for raw_name in candidates:
            name = re.sub(r"[^0-9A-Za-z_]", "_", raw_name.strip())
            if not name or name[0].isdigit():
                continue
            if name not in normalized:
                normalized.append(name)
        return normalized

    def _candidate_import_roots(
        self,
        task: Commit0Task,
        candidate_worktree: Path,
    ) -> list[str]:
        roots = [""]
        if task.src_root:
            roots.insert(0, task.src_root)
        usable_roots: list[str] = []
        import_names = self._candidate_import_names(task)
        for root in roots:
            root_path = candidate_worktree / root if root else candidate_worktree
            if not root_path.exists():
                continue
            for import_name in import_names:
                package_path = root_path / import_name.replace(".", "/")
                module_path = package_path.with_suffix(".py")
                if package_path.exists() or module_path.exists():
                    usable_roots.append(root)
                    break
        return list(dict.fromkeys(usable_roots))

    def _candidate_dependency_artifact_paths(
        self,
        task: Optional[Commit0Task],
        candidate_worktree: Path,
    ) -> list[str]:
        """Return Commit0/Python dependency or pytest-harness root shadows."""

        dependency_roots: list[str] = ["pytest_jsonreport"]
        if task is not None:
            for requirement in list(getattr(task, "pip_packages", []) or []):
                dependency_roots.extend(_commit0_requirement_import_roots(str(requirement)))
            dependency_roots.extend(
                _repo_declared_requirement_import_roots(task, candidate_worktree)
            )
            for module in _repo_declared_pytest_plugin_dependency_modules(candidate_worktree):
                dependency_roots.append(str(module).split(".", 1)[0])
            for module in _repo_requested_pytest_option_plugin_modules(task, candidate_worktree):
                dependency_roots.append(str(module).split(".", 1)[0])

        repo_import_roots = set(self._candidate_import_names(task)) if task is not None else set()
        paths: list[str] = []
        for root in dependency_roots:
            normalized_root = normalize_changed_path(str(root or "").strip())
            if not normalized_root or normalized_root in repo_import_roots:
                continue
            # Commit0/Python fact: dependency projects can be root packages or
            # single-file modules; both are environment artifacts in gold patches.
            package_path = candidate_worktree / normalized_root
            module_path = candidate_worktree / f"{normalized_root}.py"
            if package_path.exists():
                paths.append(normalized_root)
            if module_path.exists():
                paths.append(f"{normalized_root}.py")
        return sorted(dict.fromkeys(paths))

    def _candidate_import_smoke(
        self,
        *,
        task: Commit0Task,
        candidate_worktree: Path,
        artifacts_dir: Path,
        python_executable: Optional[str],
        env: Optional[dict[str, str]],
    ) -> dict[str, Any]:
        import_names = self._candidate_import_names(task)
        import_roots = self._candidate_import_roots(task, candidate_worktree)
        if not import_names or not import_roots:
            return {"status": "skipped", "reason": "no_import_root"}

        smoke_env = dict(os.environ)
        if env:
            smoke_env.update(env)
        pythonpath_parts = [
            str((candidate_worktree / root).resolve())
            if root
            else str(candidate_worktree.resolve())
            for root in import_roots
        ]
        existing_pythonpath = smoke_env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        smoke_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        smoke_code = "\n".join(
            [
                "import importlib",
                f"names = {import_names!r}",
                "errors = []",
                "for name in names:",
                "    try:",
                "        importlib.import_module(name)",
                "        print(f'apex_import_smoke_ok:{name}')",
                "        raise SystemExit(0)",
                "    except Exception as exc:",
                "        errors.append((name, type(exc).__name__, str(exc)))",
                "print('apex_import_smoke_failed:%r' % (errors,))",
                "raise SystemExit(1)",
            ]
        )
        executable = python_executable or sys.executable
        command = f"{shlex.quote(executable)} -c {shlex.quote(smoke_code)}"
        timeout = max(30, min(int(self._commit0_evaluation_timeout_seconds(task)), 120))
        result = self._run_command(
            candidate_worktree,
            command,
            env=smoke_env,
            timeout=timeout,
            task_id=task.instance_id,
        )
        smoke_payload = {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "import_names": import_names,
            "import_roots": import_roots,
            "output": result.output,
        }
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "import_smoke.json").write_text(
            json.dumps(smoke_payload, indent=2),
            encoding="utf-8",
        )
        if result.output:
            (artifacts_dir / "import_smoke.txt").write_text(
                result.output,
                encoding="utf-8",
            )
        return smoke_payload

    def _candidate_quality_gate(
        self,
        *,
        task: Commit0Task,
        candidate_worktree: Path,
        artifacts_dir: Path,
        changed_files: list[str],
        python_executable: Optional[str],
        env: Optional[dict[str, str]],
        rollout_summary: Optional[dict[str, Any]] = None,
        incomplete_test_files: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        stub_findings = self._candidate_stub_findings(candidate_worktree, changed_files)
        collection_adapter, collection_critical_edits = (
            self._candidate_collection_critical_edit_paths(
                candidate_worktree,
                changed_files,
                incomplete_test_files=incomplete_test_files,
            )
        )
        import_smoke = self._candidate_import_smoke(
            task=task,
            candidate_worktree=candidate_worktree,
            artifacts_dir=artifacts_dir,
            python_executable=python_executable,
            env=env,
        )
        gate = {
            "status": "passed",
            "changed_files": changed_files,
            "stub_findings": [
                {
                    "path": getattr(finding, "path", ""),
                    "symbol": getattr(finding, "symbol", ""),
                    "reason": getattr(finding, "reason", ""),
                }
                for finding in stub_findings
            ],
            "import_smoke": import_smoke,
        }
        reasons: list[str] = []
        advisory_reasons: list[str] = []
        if not changed_files:
            # Commit0 visible-gold adapter fact: after stripping protected tests,
            # dependency shadows, and harness artifacts, a candidate must still
            # carry a source/project diff to be publishable.
            reasons.append("empty_candidate_diff")
        if collection_critical_edits:
            gate["collection_critical_adapter"] = collection_adapter
            gate["collection_critical_edit_paths"] = list(collection_critical_edits)
            reasons.append("collection_critical_edit")
        if stub_findings:
            # Commit0 visible-gold scoring is behavioral; residual source
            # stubs are selection/follow-up evidence unless scorer execution
            # shows they harm the scored objective.
            advisory_reasons.append("stub_residue")
        if import_smoke.get("status") == "failed":
            # Commit0 adapter fact: top-level import smoke is diagnostic only;
            # expected-id scoring runs the benchmark's actual import/test paths.
            advisory_reasons.append("import_smoke_failed")
        for reason in self._serialized_validity_quality_gate_reasons(rollout_summary):
            if reason not in reasons:
                reasons.append(reason)
        if advisory_reasons:
            gate["advisory_reasons"] = advisory_reasons
        if reasons:
            gate["status"] = "failed"
            gate["reasons"] = reasons
            if isinstance(rollout_summary, dict) and isinstance(
                rollout_summary.get("validity"),
                dict,
            ):
                gate["validity"] = copy.deepcopy(rollout_summary.get("validity") or {})
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "quality_gate.json").write_text(
            json.dumps(gate, indent=2),
            encoding="utf-8",
        )
        if stub_findings:
            summary = summarize_findings(stub_findings)
            if summary:
                (artifacts_dir / "stub_residue.txt").write_text(
                    summary + "\n",
                    encoding="utf-8",
                )
        return gate

    @staticmethod
    def _rollout_summary_has_clean_scored_verification(
        rollout_summary: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(rollout_summary, dict):
            return False
        validity = rollout_summary.get("validity")
        if isinstance(validity, dict):
            if validity.get("expected_coverage_preserved") is False:
                return False
            if int(validity.get("missing_expected_test_count") or 0) > 0:
                return False
            if validity.get("quick_verification_passed") is True:
                return True
        quick = rollout_summary.get("quick_verification")
        if not isinstance(quick, dict):
            return False
        if int(quick.get("returncode") or 0) != 0:
            return False
        if int(quick.get("failed") or 0) > 0 or int(quick.get("errors") or 0) > 0:
            return False
        if quick.get("coverage_preserved") is False:
            return False
        if int(quick.get("missing_expected_test_count") or 0) > 0:
            return False
        return True

    @staticmethod
    def _serialized_validity_quality_gate_reasons(
        rollout_summary: Optional[dict[str, Any]],
    ) -> list[str]:
        if not isinstance(rollout_summary, dict):
            return []
        validity = rollout_summary.get("validity")
        if not isinstance(validity, dict):
            return []
        reasons: list[str] = []
        # Commit0 gold fact: sampled/module quick verification can mark expected coverage
        # incomplete; only terminal coverage collapse should block full expected-ID scoring.
        if validity.get("coverage_collapse_terminal") is True:
            reasons.append("expected_coverage_collapsed")
        # Commit0 visible-gold adapter fact: protected visible-test edits are
        # restored from the baseline before scoring, so stale rollout metadata
        # must not reject the sanitized source-only worktree.
        # Commit0/Python fact: collection-critical helper edits are sanitized and
        # then rechecked against the materialized diff; stale serialized validity
        # must not reject a now-clean scorer worktree before that gate runs.
        if validity.get("provenance_violation") is True:
            reasons.append("provenance_violation")
        return reasons

    @staticmethod
    def _expected_coverage_collapsed(evaluation: Commit0Evaluation) -> bool:
        diagnostics = dict(evaluation.diagnostics or {})
        if diagnostics.get("parser_error") or diagnostics.get("harness_failure"):
            return False
        coverage = dict(evaluation.expected_test_coverage or {})
        return coverage.get("coverage_preserved") is False

    @staticmethod
    def _quality_gate_failure_reason(reasons: list[str]) -> str:
        return "Final candidate failed quality gate: " + ", ".join(
            reasons or ["quality_gate_failed"]
        )

    def _record_quality_gate_rejection(
        self,
        *,
        gate: dict[str, Any],
        artifacts_dir: Path,
        reasons: list[str],
        evaluation: Optional[Commit0Evaluation] = None,
    ) -> tuple[dict[str, Any], str]:
        merged_reasons = list(gate.get("reasons") or []) if gate.get("status") == "failed" else []
        for reason in reasons:
            if reason not in merged_reasons:
                merged_reasons.append(reason)
        gate = dict(gate)
        gate["status"] = "failed"
        gate["reasons"] = merged_reasons
        if evaluation is not None:
            gate["expected_test_coverage"] = copy.deepcopy(evaluation.expected_test_coverage or {})
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "quality_gate.json").write_text(
            json.dumps(gate, indent=2),
            encoding="utf-8",
        )
        (artifacts_dir / "quality_gate_rejection.txt").write_text(
            ", ".join(merged_reasons or ["quality_gate_failed"]),
            encoding="utf-8",
        )
        return gate, self._quality_gate_failure_reason(merged_reasons)

    def _quality_gate_rejection_evaluation(
        self,
        *,
        reasons: list[str],
        evaluation: Optional[Commit0Evaluation] = None,
    ) -> Commit0Evaluation:
        failure_reason = self._quality_gate_failure_reason(reasons)
        if evaluation is None:
            return Commit0Evaluation(returncode=1, output=failure_reason)
        coverage = dict(evaluation.expected_test_coverage or {})
        total_tests = int(coverage.get("expected_test_count") or evaluation.total_tests or 1)
        return Commit0Evaluation(
            returncode=1,
            output=failure_reason,
            passed=0,
            failed=max(total_tests, 1),
            total_tests=max(total_tests, 1),
            scoring_source=evaluation.scoring_source,
            evaluation_backend=evaluation.evaluation_backend,
            expected_test_coverage=copy.deepcopy(coverage),
            score_source=evaluation.score_source,
            diagnostics={
                **copy.deepcopy(evaluation.diagnostics or {}),
                "quality_gate_rejected": True,
                "quality_gate_reasons": list(reasons),
            },
        )

    def _candidate_scorecard_entry(
        self,
        *,
        task: Commit0Task,
        candidate: _CandidateFinalResult,
        rollout_summary: Optional[dict[str, Any]],
        selected_rollout_id: Optional[int],
    ) -> dict[str, Any]:
        summary = rollout_summary if isinstance(rollout_summary, dict) else {}
        search_metadata = (
            dict(summary.get("search_metadata") or {})
            if isinstance(summary.get("search_metadata"), dict)
            else {}
        )
        anchor_resolution = (
            dict(search_metadata.get("standalone_anchor_resolution") or {})
            if isinstance(search_metadata.get("standalone_anchor_resolution"), dict)
            else {}
        )
        anchor_candidate = (
            dict(anchor_resolution.get("candidate") or {})
            if isinstance(anchor_resolution.get("candidate"), dict)
            else {}
        )
        evaluation_payload = candidate.evaluation.to_dict()
        evaluation_payload.pop("output", None)
        standalone_label = (
            search_metadata.get("standalone_anchor_label") or anchor_candidate.get("label") or ""
        )
        return {
            "task_id": task.instance_id,
            "rollout_id": candidate.rollout_id,
            "selected": candidate.rollout_id == selected_rollout_id,
            "standalone_agent_anchor": bool(search_metadata.get("standalone_agent_anchor")),
            "standalone_anchor_label": standalone_label,
            "standalone_anchor_backend": (
                anchor_candidate.get("backend") or search_metadata.get("rollout_llm_backend")
            ),
            "standalone_anchor_model": (
                anchor_candidate.get("model") or search_metadata.get("rollout_llm_model")
            ),
            "pass_rate": candidate.evaluation.pass_rate,
            "passed": int(candidate.evaluation.passed),
            "failed": int(candidate.evaluation.failed),
            "errors": int(candidate.evaluation.errors),
            "total_tests": int(candidate.evaluation.total_tests),
            "scored_success": candidate.evaluation.scored_success,
            "evaluation_status": candidate.evaluation.evaluation_status,
            "evaluation": evaluation_payload,
            "quality_gate": copy.deepcopy(candidate.quality_gate),
        }

    def _write_candidate_scorecard(
        self,
        *,
        task: Commit0Task,
        task_output_dir: Path,
        candidates: list[_CandidateFinalResult],
        summary_by_rollout: dict[int, dict[str, Any]],
        selected_rollout_id: Optional[int],
    ) -> None:
        entries = [
            self._candidate_scorecard_entry(
                task=task,
                candidate=candidate,
                rollout_summary=summary_by_rollout.get(candidate.rollout_id),
                selected_rollout_id=selected_rollout_id,
            )
            for candidate in candidates
        ]
        scorecard = {
            "task_id": task.instance_id,
            "candidate_count": len(entries),
            "selected_rollout_id": selected_rollout_id,
            "candidates": entries,
            "standalone_anchor_results": [
                entry for entry in entries if entry.get("standalone_agent_anchor")
            ],
        }
        scorecard_dir = task_output_dir / "rollout_evals"
        scorecard_dir.mkdir(parents=True, exist_ok=True)
        (scorecard_dir / "candidate_scorecard.json").write_text(
            json.dumps(scorecard, indent=2),
            encoding="utf-8",
        )

    def _load_diagnostic_score_only_candidates(
        self,
        task_output_dir: Path,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        scorecard_dir = task_output_dir / "rollout_evals"
        if not scorecard_dir.is_dir():
            return entries
        for path in sorted(scorecard_dir.glob("rollout_*/diagnostic_score_only.json")):
            payload = load_json_if_exists(path)
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def _full_scope_quick_verification_candidate_evaluation(
        self,
        *,
        task: Commit0Task,
        rollout_id: int,
        rollout_summary: Optional[dict[str, Any]],
        expected_test_ids: Optional[list[str]],
        artifacts_dir: Path,
        candidate_worktree_unchanged: bool = True,
    ) -> Optional[Commit0Evaluation]:
        if not candidate_worktree_unchanged:
            # Commit0/Python fact: sanitizer-restored visible-test worktrees
            # must be rescored because rollout QV ran before sanitization.
            return None
        if not expected_test_ids or not isinstance(rollout_summary, dict):
            return None
        quick = rollout_summary.get("quick_verification")
        if not isinstance(quick, dict):
            return None
        if str(quick.get("scope") or "").strip() != "full_test_command":
            return None
        if bool(quick.get("timed_out")) or bool(quick.get("full_scope_timed_out")):
            return None
        if quick.get("returncode") not in (0, None):
            return None

        expected_count = len([test_id for test_id in expected_test_ids if test_id])
        if expected_count <= 0:
            return None

        def _count(key: str) -> int:
            value = quick.get(key)
            if isinstance(value, bool):
                return 0
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        failed = _count("failed")
        errors = _count("errors")
        skipped = _count("skipped")
        if failed or errors or skipped:
            return None
        passed = _count("passed")
        reported_expected = _count("expected_test_count")
        matched_expected = _count("matched_expected_test_count")
        missing_expected = _count("missing_expected_test_count")
        collected = _count("collected_test_count")
        if reported_expected != expected_count:
            return None
        if matched_expected < expected_count:
            return None
        if missing_expected != 0:
            return None
        if quick.get("coverage_preserved") is False:
            return None
        if collected and collected < expected_count:
            return None
        if passed and passed < expected_count:
            return None
        pass_rate = quick.get("pass_rate")
        if isinstance(pass_rate, (int, float)) and not isinstance(pass_rate, bool):
            if float(pass_rate) < 0.999:
                return None

        # Commit0/Python scoring fact: a clean full-scope quick verification
        # already ran the expected-ID pytest command; local candidate scoring can
        # reuse it while the separate official audit remains authoritative.
        coverage = {
            "expected_test_count": expected_count,
            "matched_expected_test_count": expected_count,
            "missing_expected_test_count": 0,
            "skipped_expected_test_count": 0,
            "coverage_preserved": True,
            "collected_test_count": collected or expected_count,
        }
        output = str(quick.get("output_excerpt") or "").strip()
        evaluation = Commit0Evaluation(
            returncode=0,
            output=output,
            raw_returncode=0,
            passed=expected_count,
            failed=0,
            errors=0,
            skipped=0,
            total_tests=expected_count,
            scoring_source="commit0_test_ids",
            evaluation_backend=COMMIT0_EVALUATION_BACKEND_LOCAL_PYTEST,
            expected_test_coverage=coverage,
            score_source="apex_private_pytest_json",
            diagnostics={
                "reused_full_scope_quick_verification": True,
                "reused_rollout_id": int(rollout_id),
                "reused_quick_verification_scope": "full_test_command",
            },
        )
        _commit0_evaluation_decision(evaluation)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "reused_full_scope_quick_verification.json").write_text(
            json.dumps(evaluation.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return evaluation

    def _select_best_rollout_candidate(
        self,
        task: Commit0Task,
        workspace_dir: Path,
        task_output_dir: Path,
        rollout_summaries: Optional[list[dict[str, Any]]] = None,
        external_scoring_candidates: Optional[list[dict[str, Any]]] = None,
        preferred_rollout_id: Optional[int] = None,
        protected_test_files: Optional[list[str]] = None,
        incomplete_test_files: Optional[list[str]] = None,
        baseline_repo_dir: Optional[Path] = None,
        python_executable: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        expected_test_ids: Optional[list[str]] = None,
        use_expected_test_scoring: bool = False,
        restrict_to_preferred_rollout: bool = False,
    ) -> Optional[_CandidateFinalResult]:
        summary_by_rollout = {
            int(summary.get("rollout_id")): summary
            for summary in (rollout_summaries or [])
            if isinstance(summary.get("rollout_id"), int)
        }
        manifest_worktrees: dict[int, Path] = {}
        for candidate in external_scoring_candidates or []:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("rollout_id"),
                int,
            ):
                continue
            rollout_id = int(candidate["rollout_id"])
            summary_by_rollout[rollout_id] = {
                **summary_by_rollout.get(rollout_id, {}),
                **candidate,
            }
            worktree_path = str(candidate.get("worktree_path") or "").strip()
            if not worktree_path:
                continue
            path = Path(worktree_path)
            if path.is_dir():
                manifest_worktrees[rollout_id] = path

        # Phase 4 10.J-b — raw-Codex floor: when the seed Codex profile
        # (profile_index == 0) has produced a rollout that passed strict
        # local verification, retain that as a tie-breaker. Do not return
        # early: Commit0 selection should still record and rank every
        # patched portfolio rollout using the benchmark scorer.
        codex_floor = self._raw_codex_floor_candidate(
            workspace_dir,
            summary_by_rollout,
        )
        codex_floor_rollout_id = codex_floor[0] if codex_floor is not None else None
        if codex_floor_rollout_id is not None:
            logger.warning(
                "Raw-Codex floor retained for %s: rollout=%s passed strict acceptance",
                task.instance_id,
                codex_floor_rollout_id,
            )

        rollout_paths: list[tuple[int, Path]] = []
        discovered_worktrees = self._discover_rollout_worktree_paths(workspace_dir)
        if manifest_worktrees:
            # Commit0 scorer handoff fact: external-scoring manifests are
            # additive hints from the orchestrator; a partial/stale handoff must
            # not hide materialized rollout worktrees that the adapter can still
            # score. Manifest paths win when both sources name the same rollout.
            discovered_worktrees = {**discovered_worktrees, **manifest_worktrees}
        for rollout_id, worktree_path in sorted(discovered_worktrees.items()):
            if summary_by_rollout and not self._is_evaluable_rollout_summary(
                summary_by_rollout.get(rollout_id)
            ):
                if self._rollout_summary_has_hard_validity_rejection(
                    summary_by_rollout.get(rollout_id)
                ):
                    rejection_dir = task_output_dir / "rollout_evals" / f"rollout_{rollout_id}"
                    rejection_dir.mkdir(parents=True, exist_ok=True)
                    reasons = self._serialized_validity_quality_gate_reasons(
                        summary_by_rollout.get(rollout_id)
                    )
                    (rejection_dir / "quality_gate_rejection.txt").write_text(
                        ", ".join(reasons or ["quality_gate_failed"]),
                        encoding="utf-8",
                    )
                continue
            rollout_paths.append((rollout_id, worktree_path))

        rollout_paths.sort(
            key=lambda item: self._rollout_candidate_sort_key(
                rollout_id=item[0],
                summary=summary_by_rollout.get(item[0]),
                preferred_rollout_id=preferred_rollout_id,
            ),
            reverse=True,
        )
        if restrict_to_preferred_rollout and preferred_rollout_id is not None:
            # Commit0 gold fact: final scoring must publish APEX's authorized
            # primary nomination; sibling rescoring is diagnostic and should not
            # spend full pytest suites on candidates that cannot be submitted.
            rollout_paths = [
                rollout_path
                for rollout_path in rollout_paths
                if rollout_path[0] == int(preferred_rollout_id)
            ]

        if not rollout_paths:
            return None

        def evaluate_candidate(
            rollout_id: int, worktree_path: Path
        ) -> Optional[_CandidateFinalResult]:
            rejection_dir = task_output_dir / "rollout_evals" / f"rollout_{rollout_id}"
            rejection_dir.mkdir(parents=True, exist_ok=True)
            rollout_summary = summary_by_rollout.get(rollout_id)
            try:
                safe_worktree_path, protected_edit_reason = (
                    self._prepare_visible_test_safe_worktree(
                        task=task,
                        baseline_repo_dir=baseline_repo_dir,
                        candidate_worktree=worktree_path,
                        artifacts_dir=rejection_dir,
                        protected_test_files=list(protected_test_files or []),
                        incomplete_test_files=incomplete_test_files,
                        rollout_summary=rollout_summary,
                    )
                )
                if protected_edit_reason:
                    (rejection_dir / "policy_rejection.txt").write_text(protected_edit_reason)
                    return None
                candidate_worktree = safe_worktree_path or worktree_path
                rollout_summary_for_changed_files = rollout_summary
                if (
                    safe_worktree_path is not None
                    and safe_worktree_path != worktree_path
                    and (baseline_repo_dir is None or not baseline_repo_dir.exists())
                ):
                    rollout_summary_for_changed_files = None
                changed_files = self._candidate_changed_files(
                    candidate_worktree,
                    rollout_summary_for_changed_files,
                    baseline_repo_dir=baseline_repo_dir,
                )
                quality_gate = self._candidate_quality_gate(
                    task=task,
                    candidate_worktree=candidate_worktree,
                    artifacts_dir=rejection_dir,
                    changed_files=changed_files,
                    python_executable=python_executable,
                    env=env,
                    rollout_summary=rollout_summary,
                    incomplete_test_files=incomplete_test_files,
                )
                if quality_gate.get("status") == "failed":
                    (rejection_dir / "quality_gate_rejection.txt").write_text(
                        ", ".join(quality_gate.get("reasons") or ["quality_gate_failed"]),
                        encoding="utf-8",
                    )
                    return None
                selection_expected_test_ids = (
                    expected_test_ids if use_expected_test_scoring else None
                )
                evaluation = self._full_scope_quick_verification_candidate_evaluation(
                    task=task,
                    rollout_id=rollout_id,
                    rollout_summary=rollout_summary,
                    expected_test_ids=selection_expected_test_ids,
                    artifacts_dir=rejection_dir,
                    candidate_worktree_unchanged=(
                        safe_worktree_path is None or safe_worktree_path == worktree_path
                    ),
                )
                if evaluation is None:
                    evaluation = self.evaluate_repo(
                        task,
                        candidate_worktree,
                        artifacts_dir=rejection_dir,
                        label=f"rollout_{rollout_id}",
                        python_executable=python_executable,
                        env=env,
                        expected_test_ids=selection_expected_test_ids,
                        use_expected_test_scoring=use_expected_test_scoring,
                        process_task_id=_commit0_candidate_eval_task_id(
                            task.instance_id,
                            rollout_id,
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "Ignoring rollout %s candidate evaluation failure for %s: %s",
                    rollout_id,
                    task.instance_id,
                    exc,
                )
                (rejection_dir / "candidate_error.txt").write_text(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                )
                return None
            if _commit0_runner_health(evaluation) in {
                RunnerHealth.HARNESS_FAILURE,
                RunnerHealth.PARSER_ERROR,
                RunnerHealth.ENVIRONMENT_FAILURE,
            }:
                return None
            if self._expected_coverage_collapsed(evaluation):
                self._record_quality_gate_rejection(
                    gate=quality_gate,
                    artifacts_dir=rejection_dir,
                    reasons=["expected_coverage_collapsed"],
                    evaluation=evaluation,
                )
                return None
            if isinstance(rollout_summary, dict) and rollout_summary.get("diagnostic_score_only"):
                diagnostic_payload = {
                    "rollout_id": rollout_id,
                    "diagnostic_score_only": True,
                    "evaluation": evaluation.to_dict(),
                }
                (rejection_dir / "diagnostic_score_only.json").write_text(
                    json.dumps(diagnostic_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return None
            return _CandidateFinalResult(
                rollout_id=rollout_id,
                worktree_path=candidate_worktree,
                evaluation=evaluation,
                changed_files=changed_files,
                stub_findings=list(quality_gate.get("stub_findings") or []),
                quality_gate=quality_gate,
            )

        candidates = []
        max_workers = self._rollout_evaluation_worker_limit(len(rollout_paths))
        preferred_rollout_path: Optional[tuple[int, Path]] = None
        if preferred_rollout_id is not None:
            for rollout_path in rollout_paths:
                if rollout_path[0] == int(preferred_rollout_id):
                    preferred_rollout_path = rollout_path
                    break

        def _candidate_has_perfect_scored_result(item: _CandidateFinalResult) -> bool:
            evaluation = item.evaluation
            coverage = evaluation.expected_test_coverage or {}
            decision = _commit0_evaluation_decision(evaluation)
            if not (
                item.quality_gate.get("status") == "passed"
                and coverage.get("coverage_preserved") is not False
                and decision.is_success
                and float(evaluation.pass_rate or 0.0) >= 1.0
                and int(evaluation.failed or 0) == 0
                and int(evaluation.errors or 0) == 0
                and int(evaluation.skipped or 0) == 0
                and not item.stub_findings
            ):
                return False
            if not use_expected_test_scoring:
                return True
            if evaluation.scoring_source != "commit0_test_ids":
                return False
            expected_count = len(expected_test_ids or [])
            if expected_count <= 0:
                return False

            def _coverage_count(key: str) -> int:
                value = coverage.get(key)
                if isinstance(value, bool):
                    return 0
                try:
                    return int(value or 0)
                except (TypeError, ValueError):
                    return 0

            reported_expected = _coverage_count("expected_test_count")
            matched_expected = _coverage_count("matched_expected_test_count")
            if matched_expected <= 0:
                matched_expected = _coverage_count("observed_expected_test_count")
            missing_expected = _coverage_count("missing_expected_test_count")
            return bool(
                coverage.get("coverage_preserved") is True
                and reported_expected == expected_count
                and matched_expected >= expected_count
                and missing_expected == 0
                and int(evaluation.total_tests or 0) == expected_count
            )

        def _can_stop_after_perfect_scored_result(item: _CandidateFinalResult) -> bool:
            return bool(
                use_expected_test_scoring
                and expected_test_ids
                and not self._should_use_official_audit_candidate_selection()
                and _candidate_has_perfect_scored_result(item)
            )

        def _write_candidate_scorecard_and_return(
            item: _CandidateFinalResult,
        ) -> _CandidateFinalResult:
            self._write_candidate_scorecard(
                task=task,
                task_output_dir=task_output_dir,
                candidates=candidates,
                summary_by_rollout=summary_by_rollout,
                selected_rollout_id=item.rollout_id,
            )
            return item

        def _cancel_parallel_candidate_evaluations(
            executor: ThreadPoolExecutor,
            futures: dict[Any, int],
            *,
            current_future: Any,
        ) -> None:
            # Commit0/Python scoring fact: strict perfect expected-ID scoring is
            # decisive, so sibling local pytest scorers only consume lanes.
            for pending, pending_rollout_id in list(futures.items()):
                if pending is current_future:
                    continue
                pending.cancel()
                process_task_id = _commit0_candidate_eval_task_id(
                    task.instance_id,
                    int(pending_rollout_id),
                )
                PROCESS_REGISTRY.kill(process_task_id, signum=signal.SIGTERM)
                PROCESS_REGISTRY.kill(process_task_id, signum=signal.SIGKILL)
            _abandon_interruptible_thread_pool(executor)

        if preferred_rollout_path is not None:
            preferred_candidate = evaluate_candidate(*preferred_rollout_path)
            if preferred_candidate is not None:
                candidates.append(preferred_candidate)
                if _can_stop_after_perfect_scored_result(preferred_candidate):
                    return _write_candidate_scorecard_and_return(preferred_candidate)
            rollout_paths = [
                rollout_path
                for rollout_path in rollout_paths
                if rollout_path != preferred_rollout_path
            ]
        elif (
            rollout_paths
            and use_expected_test_scoring
            and expected_test_ids
            and not self._should_use_official_audit_candidate_selection()
        ):
            top_rollout_path = rollout_paths[0]
            top_candidate = evaluate_candidate(*top_rollout_path)
            if top_candidate is not None:
                candidates.append(top_candidate)
                if _can_stop_after_perfect_scored_result(top_candidate):
                    return _write_candidate_scorecard_and_return(top_candidate)
            rollout_paths = [
                rollout_path for rollout_path in rollout_paths if rollout_path != top_rollout_path
            ]

        if max_workers == 1:
            for rollout_id, worktree_path in rollout_paths:
                candidate = evaluate_candidate(rollout_id, worktree_path)
                if candidate is not None:
                    candidates.append(candidate)
                    if _can_stop_after_perfect_scored_result(candidate):
                        return _write_candidate_scorecard_and_return(candidate)
        else:
            wave_size = max(1, int(max_workers))
            for wave_start in range(0, len(rollout_paths), wave_size):
                wave_paths = rollout_paths[wave_start : wave_start + wave_size]
                wave_candidates: dict[int, _CandidateFinalResult] = {}
                with _interruptible_thread_pool(min(wave_size, len(wave_paths))) as executor:
                    futures = {
                        executor.submit(evaluate_candidate, rollout_id, worktree_path): rollout_id
                        for rollout_id, worktree_path in wave_paths
                    }
                    for future in as_completed(futures):
                        rollout_id = futures[future]
                        try:
                            candidate = future.result()
                        except Exception as exc:
                            logger.warning(
                                "Ignoring rollout %s candidate future failure for %s: %s",
                                rollout_id,
                                task.instance_id,
                                exc,
                            )
                            rejection_dir = (
                                task_output_dir / "rollout_evals" / f"rollout_{rollout_id}"
                            )
                            rejection_dir.mkdir(parents=True, exist_ok=True)
                            (rejection_dir / "candidate_error.txt").write_text(
                                "".join(
                                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                                )
                            )
                            continue
                        if candidate is not None:
                            candidates.append(candidate)
                            wave_candidates[rollout_id] = candidate
                            if _can_stop_after_perfect_scored_result(candidate):
                                _cancel_parallel_candidate_evaluations(
                                    executor,
                                    futures,
                                    current_future=future,
                                )
                                return _write_candidate_scorecard_and_return(candidate)
                for rollout_id, _worktree_path in wave_paths:
                    candidate = wave_candidates.get(rollout_id)
                    if candidate is not None and _can_stop_after_perfect_scored_result(candidate):
                        return _write_candidate_scorecard_and_return(candidate)

        if not candidates:
            return None

        def _candidate_sort_key(item: _CandidateFinalResult) -> tuple:
            evaluation = item.evaluation
            coverage = evaluation.expected_test_coverage or {}
            decision = _commit0_evaluation_decision(evaluation)
            # Raw collected/passed signal breaks ties when the expected-set
            # scorer says two rollouts are equivalent. pylint rollout_2 had
            # passed=1878 vs rollout_1 passed=1877 on the broader pytest
            # surface but tied on the expected-set score; without this tier
            # the lower id won, which was strictly worse.
            collected = int(coverage.get("collected_test_count") or 0)
            return (
                item.quality_gate.get("status") == "passed",
                coverage.get("coverage_preserved") is not False,
                decision.is_success,
                evaluation.pass_rate,
                evaluation.passed,
                decision.is_candidate_viable,
                collected,
                -evaluation.failed,
                not bool(item.stub_findings),
                item.rollout_id == codex_floor_rollout_id,
                # Commit0 rescoring can produce benchmark-equivalent candidates; keep Apex's primary nomination on exact scorer ties.
                item.rollout_id == preferred_rollout_id,
                -item.rollout_id,
            )

        candidates.sort(key=_candidate_sort_key, reverse=True)
        candidates = self._official_audit_rerank_rollout_candidates_if_configured(
            task=task,
            candidates=candidates,
            task_output_dir=task_output_dir,
            sort_key=_candidate_sort_key,
        )
        candidates.sort(key=_candidate_sort_key, reverse=True)
        selected = candidates[0]
        self._write_candidate_scorecard(
            task=task,
            task_output_dir=task_output_dir,
            candidates=candidates,
            summary_by_rollout=summary_by_rollout,
            selected_rollout_id=selected.rollout_id,
        )
        return selected

    def _should_use_official_audit_candidate_selection(self) -> bool:
        return (
            self.config.benchmark.commit0_primary_evaluation_backend
            == BenchmarkEvaluationBackend.LOCAL_PYTEST
            and bool(self.config.benchmark.commit0_official_audit_selected)
            and bool(
                getattr(
                    self.config.benchmark,
                    "commit0_audit_candidate_selection",
                    False,
                )
            )
        )

    def _official_audit_rerank_rollout_candidates_if_configured(
        self,
        *,
        task: Commit0Task,
        candidates: list[_CandidateFinalResult],
        task_output_dir: Path,
        sort_key: Callable[[_CandidateFinalResult], tuple],
    ) -> list[_CandidateFinalResult]:
        if not candidates or not self._should_use_official_audit_candidate_selection():
            return candidates

        top_k_raw = getattr(
            self.config.benchmark,
            "commit0_audit_candidate_selection_top_k",
            3,
        )
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            top_k = 3
        selected_candidates = candidates if top_k <= 0 else candidates[:top_k]
        remaining_candidates = [] if top_k <= 0 else candidates[top_k:]
        reranked_candidates: list[_CandidateFinalResult] = []

        for candidate in selected_candidates:
            audit_dir = (
                task_output_dir
                / "rollout_evals"
                / f"rollout_{candidate.rollout_id}"
                / "official_candidate_eval"
            )
            audit_dir.mkdir(parents=True, exist_ok=True)
            try:
                official_evaluation = self._evaluate_repo_official(
                    task,
                    candidate.worktree_path,
                    artifacts_dir=audit_dir,
                    label=f"rollout-{candidate.rollout_id}-official-candidate",
                )
            except Exception as exc:
                official_evaluation = Commit0Evaluation(
                    returncode=1,
                    output=str(exc),
                    evaluation_backend=COMMIT0_EVALUATION_BACKEND_OFFICIAL_LOCAL_DOCKER,
                )
            (audit_dir / "candidate_official_metrics.json").write_text(
                json.dumps(official_evaluation.to_dict(), indent=2)
            )
            signal_count = _commit0_evaluation_signal_count(official_evaluation)
            official_usable = _commit0_official_audit_usable(official_evaluation)
            if not official_usable:
                candidate.evaluation.diagnostics["candidate_selection_audit"] = {
                    "official_audit_usable": False,
                    "official_returncode": official_evaluation.returncode,
                    "official_pass_rate": official_evaluation.pass_rate,
                    "official_signal_count": signal_count,
                    "official_harness_failure": not official_usable,
                }
                reranked_candidates.append(candidate)
                continue

            official_evaluation.diagnostics["score_apex_private"] = {
                "pass_rate": float(candidate.evaluation.pass_rate),
                "passed": int(candidate.evaluation.passed),
                "failed": int(candidate.evaluation.failed),
                "errors": int(candidate.evaluation.errors),
                "returncode": int(candidate.evaluation.returncode),
                "score_source": str(candidate.evaluation.score_source),
                "evaluation_backend": str(candidate.evaluation.evaluation_backend),
            }
            official_evaluation.diagnostics["candidate_selection_audit"] = {
                "official_audit_usable": True,
                "local_pass_rate": float(candidate.evaluation.pass_rate),
                "local_returncode": int(candidate.evaluation.returncode),
                "local_passed": int(candidate.evaluation.passed),
                "local_failed": int(candidate.evaluation.failed),
                "local_errors": int(candidate.evaluation.errors),
            }
            reranked_candidates.append(
                _CandidateFinalResult(
                    rollout_id=candidate.rollout_id,
                    worktree_path=candidate.worktree_path,
                    evaluation=official_evaluation,
                    changed_files=list(candidate.changed_files),
                    stub_findings=list(candidate.stub_findings),
                    quality_gate=dict(candidate.quality_gate),
                )
            )

        reranked_candidates.extend(remaining_candidates)
        reranked_candidates.sort(key=sort_key, reverse=True)
        return reranked_candidates

    @staticmethod
    def _rollout_summary_is_fallback_nomination(summary: Optional[dict[str, Any]]) -> bool:
        if not isinstance(summary, dict):
            return False
        verification = summary.get("verification")
        verification_payload = verification if isinstance(verification, dict) else {}
        if (
            summary.get("salvaged_for_external_scoring") is True
            or verification_payload.get("salvaged_for_external_scoring") is True
        ):
            return True
        selection_authority = summary.get("selection_authority") or verification_payload.get(
            "selection_authority"
        )
        if selection_authority in {
            "authoritative_scoring_nomination",
            "explicit_salvage_submission",
            "repair_seed_only",
        }:
            return True
        diagnostics = summary.get("selection_diagnostics")
        diagnostics_payload = diagnostics if isinstance(diagnostics, dict) else {}
        fallback = diagnostics_payload.get("fallback")
        if isinstance(fallback, dict) and fallback.get("reason"):
            return True
        return False

    @staticmethod
    def _rollout_summary_authorizes_benchmark_confirmation(
        summary: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(summary, dict):
            return False
        if summary.get("diagnostic_score_only") is True:
            return False
        # Commit0 publication contract: benchmark rescoring may diagnose a
        # fallback/salvage candidate, but it cannot turn that candidate into the
        # orchestrator's confident primary nomination.
        if Commit0BenchmarkRunner._rollout_summary_is_fallback_nomination(summary):
            return False
        validity = summary.get("validity")
        validity_payload = validity if isinstance(validity, dict) else {}
        if validity_payload.get("quality_gate_passed") is False:
            return False
        if validity_payload.get("backend_protocol_error") is True:
            return False
        if validity_payload.get("coverage_collapse_terminal") is True:
            return False
        if validity_payload.get("protected_tests_unchanged") is False:
            return False
        if validity_payload.get("eligible_for_external_scoring") is False:
            return False
        # Commit0/Python fact: collection-critical helper edits are sanitized and
        # rechecked by the final quality gate; stale serialized metadata alone is
        # not benchmark-confirmation authority.
        if validity_payload.get("provenance_violation") is True:
            return False
        verification = summary.get("verification")
        verification_payload = verification if isinstance(verification, dict) else {}
        if (
            summary.get("selected_for_submission") is True
            or summary.get("internally_accepted") is True
            or summary.get("officially_accepted") is True
            or verification_payload.get("selected_for_submission") is True
            or verification_payload.get("internally_accepted") is True
            or verification_payload.get("officially_accepted") is True
        ):
            return True
        if not validity_payload:
            return False
        if validity_payload.get("eligible_for_external_scoring") is not True:
            return False
        quick = summary.get("quick_verification")
        metadata = summary.get("search_metadata")
        if not isinstance(metadata, dict):
            metadata = summary
        return quick_verification_requires_authoritative_scoring(
            quick if isinstance(quick, dict) else {},
            metadata=metadata,
        )

    @staticmethod
    def _rollout_summary_authorizes_preferred_only_scoring(
        summary: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(summary, dict):
            return False
        if summary.get("diagnostic_score_only") is True:
            return False
        if Commit0BenchmarkRunner._rollout_summary_is_fallback_nomination(summary):
            return False
        validity = summary.get("validity")
        validity_payload = validity if isinstance(validity, dict) else {}
        if validity_payload.get("quality_gate_passed") is False:
            return False
        if validity_payload.get("backend_protocol_error") is True:
            return False
        if validity_payload.get("coverage_collapse_terminal") is True:
            return False
        if validity_payload.get("protected_tests_unchanged") is False:
            return False
        if validity_payload.get("eligible_for_external_scoring") is False:
            return False
        if validity_payload.get("provenance_violation") is True:
            return False
        verification = summary.get("verification")
        verification_payload = verification if isinstance(verification, dict) else {}
        if (
            summary.get("selected_for_submission") is True
            or summary.get("internally_accepted") is True
            or summary.get("officially_accepted") is True
            or verification_payload.get("selected_for_submission") is True
            or verification_payload.get("internally_accepted") is True
            or verification_payload.get("officially_accepted") is True
        ):
            return True
        if validity_payload.get("eligible_for_submission") is not True:
            return False
        quick = summary.get("quick_verification")
        # Commit0/Python scoring fact: reduced-scope quick verification is a
        # request for authoritative scoring, not authority to hide sibling
        # rollout candidates from final benchmark rescoring.
        return quick_verification_has_local_full_scope_pass(
            quick if isinstance(quick, dict) else {}
        )

    def _is_viable_rollout_summary(self, summary: Optional[dict[str, Any]]) -> bool:
        if not summary:
            return False
        if not summary.get("success"):
            return False
        return self._rollout_summary_has_patch(summary)

    def _apex_result_authorizes_orchestrator_nomination(self, result: Any) -> bool:
        selected_worktree_path = str(getattr(result, "selected_worktree_path", "") or "").strip()
        if not selected_worktree_path:
            return False
        if not bool(getattr(result, "success", False)):
            return False
        if not bool(getattr(result, "selected_for_submission", False)):
            return False
        # Layer B handoff guard: Commit0 final scoring must consume only APEX's
        # accepted primary nomination, unless an operator explicitly enabled the
        # legacy salvage path for an ablation.
        return bool(
            getattr(result, "internally_accepted", False) or getattr(result, "salvaged", False)
        )

    def _is_evaluable_rollout_summary(self, summary: Optional[dict[str, Any]]) -> bool:
        if not self._rollout_summary_has_patch(summary):
            return False
        if self._rollout_summary_has_hard_validity_rejection(summary):
            return False
        validity = summary.get("validity") if isinstance(summary, dict) else None
        # Commit0 gold scoring contract: an explicit external-scoring veto means
        # the expected-ID scorer cannot turn the candidate into a publication result.
        if isinstance(validity, dict) and validity.get("eligible_for_external_scoring") is False:
            return False
        return True

    @staticmethod
    def _rollout_summary_has_hard_validity_rejection(
        summary: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(summary, dict):
            return False
        verification = summary.get("verification")
        if isinstance(verification, dict):
            if verification.get("quality_gate_passed") is False:
                return True
            if verification.get("syntax_valid") is False:
                return True
            if verification.get("lint_clean") is False:
                return True
            prune_result = verification.get("prune_result")
            if isinstance(prune_result, dict) and prune_result.get("is_valid") is False:
                return True
        validity = summary.get("validity")
        if not isinstance(validity, dict):
            return False
        # Commit0 scoring fact: missing/stale advisory validity metadata must
        # not block the expected-ID scorer; only explicit hard defects do.
        return bool(
            validity.get("quality_gate_passed") is False
            or validity.get("backend_protocol_error") is True
            or validity.get("coverage_collapse_terminal") is True
            or validity.get("provenance_violation") is True
        )

    def _rollout_summary_has_patch(self, summary: Optional[dict[str, Any]]) -> bool:
        if not summary:
            return False
        patch = summary.get("patch")
        return isinstance(patch, str) and bool(patch.strip())

    def _raw_codex_floor_candidate(
        self,
        workspace_dir: Path,
        summary_by_rollout: dict[int, dict[str, Any]],
    ) -> Optional[tuple[int, Path]]:
        """Locate the raw-Codex (profile_index == 0) rollout that passed
        strict acceptance: returncode == 0 AND no failures/errors.

        Returns ``(rollout_id, worktree_path)`` or ``None``. The selector
        force-promotes this candidate over the diversity sort because the
        Codex backend is the highest-confidence signal in our portfolio:
        when it claims a strict pass, we rarely have evidence to override
        it.
        """

        worktree_paths = self._discover_rollout_worktree_paths(workspace_dir)
        for rollout_id, summary in sorted(summary_by_rollout.items()):
            if not isinstance(summary, dict):
                continue
            profile_index = summary.get("rollout_profile_index")
            if profile_index is None:
                # Fall back to the brief-level field if present.
                profile_index = summary.get("profile_index")
            if not isinstance(profile_index, int) or profile_index != 0:
                continue
            if not self._is_viable_rollout_summary(summary):
                continue
            verification = summary.get("verification") or {}
            if not isinstance(verification, dict):
                continue
            returncode = verification.get("returncode")
            if not (isinstance(returncode, int) and returncode == 0):
                continue
            passed = int(verification.get("passed") or 0)
            failed = int(verification.get("failed") or 0)
            errors = int(verification.get("errors") or 0)
            total = passed + failed + errors
            if total <= 0:
                continue
            if failed > 0 or errors > 0:
                continue
            worktree_path = worktree_paths.get(rollout_id)
            if worktree_path is None or not worktree_path.is_dir():
                continue
            return rollout_id, worktree_path
        return None

    def _rollout_candidate_sort_key(
        self,
        *,
        rollout_id: int,
        summary: Optional[dict[str, Any]],
        preferred_rollout_id: Optional[int],
    ) -> tuple[float, float, float, int, int]:
        verification = summary.get("verification") if isinstance(summary, dict) else None
        verification = verification if isinstance(verification, dict) else {}
        test_result = verification.get("test_result") if isinstance(verification, dict) else None
        test_result = test_result if isinstance(test_result, dict) else {}
        # Phase 2 10.O: smaller diffs win ties. The sort is reverse=True at
        # the call site, so larger values win — negate the changed-files
        # count so FEWER edited files outrank MORE on otherwise-equal keys.
        # Smaller patches reduce regression risk on any code-edit task.
        changed_file_count = (
            len(summary.get("changed_files") or []) if isinstance(summary, dict) else 0
        )
        return (
            1.0 if rollout_id == preferred_rollout_id else 0.0,
            float(verification.get("overall_score", 0.0) or 0.0),
            float(test_result.get("pass_rate", 0.0) or 0.0),
            -changed_file_count,
            -rollout_id,
        )

    def _rollout_evaluation_worker_limit(self, candidate_count: int) -> int:
        if candidate_count <= 0:
            return 1
        if (
            self.config.benchmark.commit0_primary_evaluation_backend
            == BenchmarkEvaluationBackend.OFFICIAL_DOCKER
        ):
            return 1
        configured = max(self.config.rollout.parallel_workers, 1)
        return max(1, min(candidate_count, configured))


def _rewrite_pip_command(command: str, pip_command: str) -> str:
    stripped = command.strip()
    if stripped.startswith("python -m pip "):
        return f"{pip_command} {stripped[len('python -m pip ') :]}"
    if stripped.startswith("pip "):
        return f"{pip_command} {stripped[len('pip ') :]}"
    raise RuntimeError(f"Unsupported install command for Commit0 benchmark: {command}")


def _looks_like_editable_install(command: str) -> bool:
    """Return True for ``pip install -e .`` / ``uv pip install -e .`` shapes.

    The shim only swallows install failures for editable installs; non-
    editable installs failing usually indicates a real dependency problem
    that would surface much later if ignored.
    """
    lowered = (command or "").lower().strip()
    if "install" not in lowered:
        return False
    if " -e " not in f" {lowered} " and not lowered.endswith(" -e ."):
        return False
    # Reject multi-package install commands. The intent is "single
    # editable install of the project" — anything more sophisticated is
    # an explicit operator choice we shouldn't override.
    tokens = shlex.split(lowered)
    install_idx = None
    for idx, tok in enumerate(tokens):
        if tok == "install":
            install_idx = idx
            break
    if install_idx is None:
        return False
    targets = [tok for tok in tokens[install_idx + 1 :] if not tok.startswith("-")]
    return targets == ["."] or targets == ["./"]


_BUILD_EDITABLE_FAILURE_MARKERS = (
    "setuptools.build_meta",
    "build_editable",
    "metadata-generation-failed",
)


def _looks_like_build_editable_failure(message: str) -> bool:
    text = (message or "").lower()
    return any(marker.lower() in text for marker in _BUILD_EDITABLE_FAILURE_MARKERS)


def _apply_commit0_repo_shims(task: "Commit0Task", repo_dir: Path) -> None:
    """Dispatch repo-specific shims after the install step (Phase 4 10.M).

    Each shim is defensive — failure to apply shouldn't fault the whole
    prepare. The shims live in :mod:`apex.evaluation.commit0_repo_shims`
    so the repo-specific logic stays out of this module's scope.
    """

    try:
        from .commit0_repo_shims import (
            ensure_convert_shim,
            ensure_ffmpeg_shim,
            prepare_filesystem_spec_s3fs,
            pypdf_install_followup_policy,
            rewrite_parsel_psutil_requirement,
            seed_babel_runtime_data,
            seed_filesystem_spec_runtime_version,
        )
    except Exception:
        logger.debug(
            "Commit0 repo shims module failed to import — skipping all shims",
            exc_info=True,
        )
        return

    repo_name = task.repo_name
    if repo_name == "babel":
        seed_babel_runtime_data(repo_dir)
    elif repo_name == "moviepy":
        ensure_ffmpeg_shim(repo_dir)
        ensure_convert_shim(repo_dir)
    elif repo_name == "parsel":
        rewrite_parsel_psutil_requirement(repo_dir)
    elif repo_name == "filesystem_spec":
        prepare_filesystem_spec_s3fs(repo_dir, [])
        seed_filesystem_spec_runtime_version(repo_dir)
    elif repo_name == "pypdf":
        pypdf_install_followup_policy(repo_dir)


def _requires_linux_package_install(command: str) -> bool:
    lowered = command.lower()
    return "apt-get" in lowered or re.search(r"(^|\s)apt\s+install(\s|$)", lowered) is not None


def _normalize_linux_package_command(command: str) -> str:
    normalized = command.strip()
    lowered = normalized.lower()
    if not _requires_linux_package_install(normalized):
        return normalized
    if re.search(r"\bapt-get\s+install\b", lowered):
        if re.search(r"\bapt-get\s+install\s+(?:-y|--yes)\b", lowered) is None:
            normalized = re.sub(
                r"\bapt-get\s+install\b",
                "apt-get install -y",
                normalized,
                count=1,
                flags=re.IGNORECASE,
            )
    elif re.search(r"(^|\s)apt\s+install\b", lowered):
        if re.search(r"\bapt\s+install\s+(?:-y|--yes)\b", lowered) is None:
            normalized = re.sub(
                r"\bapt\s+install\b",
                "apt install -y",
                normalized,
                count=1,
                flags=re.IGNORECASE,
            )
    if "debian_frontend=" not in lowered:
        normalized = f"DEBIAN_FRONTEND=noninteractive {normalized}"
    return normalized


def _rollout_id_from_name(name: str) -> Optional[int]:
    if not name.startswith("rollout_"):
        return None
    suffix = name[len("rollout_") :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _rewrite_pytest_command(
    command: str,
    python_executable: str,
    *,
    disable_plugin_autoload: bool = True,
) -> str:
    pytest_executable = Path(python_executable).with_name("pytest")
    pytest_invocation = (
        (str(pytest_executable),)
        if pytest_executable.exists()
        else (python_executable, "-m", "pytest")
    )
    parsed = parse_pytest_command(command)
    if parsed is None:
        stripped = command.strip()
        if stripped == "pytest" or stripped.startswith("pytest "):
            pytest_binary = (
                shlex.quote(str(pytest_executable))
                if pytest_executable.exists()
                else f"{shlex.quote(python_executable)} -m pytest"
            )
            suffix = stripped[len("pytest") :].strip()
            return f"{pytest_binary} {suffix}".strip()
        return stripped
    rewritten = type(parsed)(
        shell_prefix_tokens=parsed.shell_prefix_tokens,
        env_prefix_tokens=(
            _pytest_command_env_with_plugin_autoload_disabled(parsed.env_prefix_tokens)
            if disable_plugin_autoload
            else parsed.env_prefix_tokens
        ),
        invocation_tokens=pytest_invocation,
        option_tokens=parsed.option_tokens,
        target_tokens=parsed.target_tokens,
    )
    return render_pytest_command(
        rewritten,
        disable_plugin_autoload=False,
    )


def _pytest_command_env_with_plugin_autoload_disabled(
    env_prefix_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    if any(str(token).startswith("PYTEST_DISABLE_PLUGIN_AUTOLOAD=") for token in env_prefix_tokens):
        return env_prefix_tokens
    return (*env_prefix_tokens, "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")


def _commit0_pytest_command_needs_test_dir(command: str, test_dir: str) -> bool:
    test_dir = str(test_dir or "").strip()
    if not test_dir:
        return False
    parsed = parse_pytest_command(command)
    if parsed is None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if "--pyargs" in tokens:
            return False
        normalized_test_dir = _normalize_commit0_pytest_target_token(test_dir)
        return not any(
            _normalize_commit0_pytest_target_token(token) == normalized_test_dir for token in tokens
        )
    if _pytest_command_has_exact_option(list(parsed.option_tokens), "--pyargs"):
        return False
    if parsed.target_tokens:
        return False
    return True


def _normalize_commit0_pytest_target_token(token: str) -> str:
    text = str(token or "").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _pytest_command_has_exact_option(option_tokens: list[str], option: str) -> bool:
    normalized = str(option or "").strip()
    if not normalized:
        return False
    for token in option_tokens:
        text = str(token)
        if text == normalized:
            return True
        if normalized.startswith("--") and text.startswith(f"{normalized}="):
            return True
    return False


def _canonicalize_pytest_xdist_plugin_tokens(option_tokens: list[str]) -> list[str]:
    canonical: list[str] = []
    index = 0
    while index < len(option_tokens):
        token = str(option_tokens[index])
        if token == "-p" and index + 1 < len(option_tokens):
            plugin = str(option_tokens[index + 1]).strip()
            canonical.extend(
                ["-p", "xdist" if plugin == "xdist.plugin" else option_tokens[index + 1]]
            )
            index += 2
            continue
        if token.startswith("-p") and token[2:].strip() == "xdist.plugin":
            canonical.append("-pxdist")
            index += 1
            continue
        canonical.append(option_tokens[index])
        index += 1
    return canonical


def _pytest_command_loads_plugin(option_tokens: list[str], plugin_name: str) -> bool:
    normalized = str(plugin_name or "").strip()
    aliases = {normalized}
    if normalized in {"xdist", "xdist.plugin"}:
        aliases.add("xdist")
        aliases.add("xdist.plugin")
    if normalized in {"pytest_jsonreport", "pytest_jsonreport.plugin"}:
        aliases.add("pytest_jsonreport")
        aliases.add("pytest_jsonreport.plugin")
    index = 0
    while index < len(option_tokens):
        token = str(option_tokens[index])
        if token == "-p" and index + 1 < len(option_tokens):
            if str(option_tokens[index + 1]).strip() in aliases:
                return True
            index += 2
            continue
        if token.startswith("-p") and token[2:].strip() in aliases:
            return True
        index += 1
    return False


def _pytest_command_has_xdist_workers(option_tokens: list[str]) -> bool:
    for token in option_tokens:
        text = str(token)
        if text == "-n" or (text.startswith("-n") and len(text) > 2):
            return True
        if text == "--numprocesses" or text.startswith("--numprocesses="):
            return True
    return False


def _pytest_command_has_xdist_dist(option_tokens: list[str]) -> bool:
    return any(
        str(token) == "--dist" or str(token).startswith("--dist=") for token in option_tokens
    )


def _commit0_pytest_collection_command(command: str) -> str:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return ""
    option_tokens: list[str] = []
    index = 0
    while index < len(parsed.option_tokens):
        token = str(parsed.option_tokens[index])
        normalized = token.split("=", 1)[0]
        if token in {"-q", "--quiet", "-v", "-vv", "--tb=no", "--collect-only"}:
            index += 1
            continue
        if token == "-n":
            index += 2
            continue
        if token.startswith("-n") and len(token) > 2:
            index += 1
            continue
        if normalized in {"--numprocesses", "--dist"}:
            index += 1
            if "=" not in token and index < len(parsed.option_tokens):
                index += 1
            continue
        if token == "-p" and index + 1 < len(parsed.option_tokens):
            plugin = str(parsed.option_tokens[index + 1]).strip()
            if plugin in {"xdist", "xdist.plugin"}:
                index += 2
                continue
        if token.startswith("-p") and token[2:].strip() in {"xdist", "xdist.plugin"}:
            index += 1
            continue
        option_tokens.append(parsed.option_tokens[index])
        index += 1
    # Commit0/Python pytest-xdist does not reliably expose per-node collect-only
    # JSON, so expected-ID coverage sweeps use serial collection even when tests run in parallel.
    option_tokens.extend(["--collect-only", "-q"])
    rewritten = type(parsed)(
        shell_prefix_tokens=parsed.shell_prefix_tokens,
        env_prefix_tokens=parsed.env_prefix_tokens,
        invocation_tokens=parsed.invocation_tokens,
        option_tokens=tuple(option_tokens),
        target_tokens=parsed.target_tokens,
    )
    return render_pytest_command(rewritten, disable_plugin_autoload=False)


def _pytest_xdist_dist_from_command(command: str) -> Optional[str]:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return None
    option_tokens = list(parsed.option_tokens)
    for index, token in enumerate(option_tokens):
        text = str(token)
        if text == "--dist" and index + 1 < len(option_tokens):
            return str(option_tokens[index + 1]).strip() or None
        if text.startswith("--dist="):
            return text.split("=", 1)[1].strip() or None
    return None


def _bound_commit0_pytest_output(command: str) -> str:
    parsed = parse_pytest_command(command)
    if parsed is None:
        return command
    option_tokens = list(parsed.option_tokens)
    # Commit0 visible suites can have thousands of expected failures while an
    # agent is mid-repair; quiet pytest output keeps CLI model transport bounded
    # without changing collection, execution, JSON-report, or expected-ID scoring.
    if not _pytest_command_has_quiet_or_verbose_flag(option_tokens):
        option_tokens.append("-q")
    if not _pytest_command_has_traceback_style(option_tokens):
        option_tokens.append("--tb=short")
    rewritten = type(parsed)(
        shell_prefix_tokens=parsed.shell_prefix_tokens,
        env_prefix_tokens=parsed.env_prefix_tokens,
        invocation_tokens=parsed.invocation_tokens,
        option_tokens=tuple(option_tokens),
        target_tokens=parsed.target_tokens,
    )
    return render_pytest_command(rewritten, disable_plugin_autoload=False)


def _pytest_command_has_quiet_or_verbose_flag(option_tokens: list[str]) -> bool:
    for token in option_tokens:
        if token in {"--quiet", "--verbose"}:
            return True
        if token.startswith("-q") or token.startswith("-v"):
            return True
    return False


def _pytest_command_has_traceback_style(option_tokens: list[str]) -> bool:
    return any(token == "--tb" or token.startswith("--tb=") for token in option_tokens)


def _load_expected_test_ids(repo_name: str) -> list[str]:
    try:
        from commit0.harness.get_pytest_ids import main as get_pytest_ids
    except ImportError:
        return []

    try:
        return [test_id for test_id in get_pytest_ids(repo_name, verbose=0) if test_id]
    except Exception:
        return []


def _extract_pytest_report_outcomes(tests: list[dict[str, Any]]) -> dict[str, str]:
    return extract_pytest_report_outcomes(tests)


def _pytest_report_outcome(test: dict[str, Any]) -> str:
    return pytest_report_outcome(test)


def _normalize_pytest_outcome(value: Any) -> str:
    return normalize_pytest_outcome(value)


def _count_expected_test_outcomes(
    expected_test_ids: list[str],
    outcomes: dict[str, str],
) -> dict[str, int]:
    summary = summarize_expected_pytest_coverage(expected_test_ids, outcomes)
    return {
        "passed": int(summary.get("passed") or 0),
        "failed": int(summary.get("failed") or 0),
        "errors": int(summary.get("errors") or 0),
    }


def _parameterized_node_id_base(node_id: str) -> Optional[str]:
    return parameterized_node_id_base(node_id)


def _resolve_uv_command() -> list[str]:
    uv_binary = shutil.which("uv")
    if uv_binary:
        return [uv_binary]
    candidate_paths = [
        Path.home()
        / "Library"
        / "Python"
        / f"{sys.version_info.major}.{sys.version_info.minor}"
        / "bin"
        / "uv",
        Path.home() / ".local" / "bin" / "uv",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return [str(candidate)]
    module_cmd = [sys.executable, "-m", "uv"]
    if subprocess.run([*module_cmd, "--version"], capture_output=True).returncode == 0:
        return module_cmd
    raise RuntimeError(
        "Commit0 benchmarking requires uv. Install it with `python3 -m pip install --user uv`."
    )


def _resolve_uv_pip_command() -> str:
    return f"{shlex.join(_resolve_uv_command())} pip"


def _stage_commit0_prepared_runtime_python_policy(output_dir: Path) -> Path:
    # Commit0 solve-phase isolation is structural; keep this import hook empty
    # for compatibility with older staged runtime layouts.
    policy_dir = output_dir / _COMMIT0_PREPARED_RUNTIME_POLICY_DIRNAME
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "sitecustomize.py").write_text(
        _COMMIT0_PREPARED_RUNTIME_SITECUSTOMIZE,
        encoding="utf-8",
    )
    return policy_dir


def _prepend_commit0_target_runtime_docker_env_path(
    target_tool_env: dict[str, str],
    key: str,
    host_path: str | Path,
) -> str:
    context_path_raw = str(target_tool_env.get("APEX_TARGET_TOOL_CONTEXT") or "").strip()
    if not context_path_raw:
        return str(host_path)
    context_path = Path(context_path_raw)
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return str(host_path)
    runtime = dict(context.get("runtime") or {})
    if str(runtime.get("kind") or context.get("mode") or "") != "docker_exec":
        return str(host_path)

    # Layer B (Commit0 Docker solve phase): target-runtime helper files are
    # staged on the host bind mount, but agent subprocesses need container paths.
    container_path = target_runtime_path_for_file(target_tool_env, host_path)
    docker_env = {str(k): str(v) for k, v in dict(runtime.get("docker_env") or {}).items()}
    existing = str(docker_env.get(key) or "")
    docker_env[key] = container_path + (os.pathsep + existing if existing else "")
    runtime["docker_env"] = docker_env
    context["runtime"] = runtime
    try:
        context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return str(host_path)
    return container_path


def _install_commit0_prepared_runtime_container_guards(
    *,
    container_name: str,
    docker_bin: str,
    docker_env: dict[str, str],
    container_venv: str = "",
) -> None:
    # Commit0 Docker agents sometimes run shell snapshots with PATH reset to
    # /usr/local/bin; wrap container Python/pip there after harness install.
    if not str(container_name or "").strip():
        return
    result = run_process_command(
        [
            str(docker_bin or "docker"),
            "exec",
            "-u",
            "root",
            str(container_name),
            "sh",
            "-c",
            _commit0_prepared_runtime_container_guard_script(container_venv=container_venv),
        ],
        env=docker_env,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            normalize_terminal_output(result.stdout + result.stderr).strip()
            or "failed to install Commit0 prepared-runtime container guards"
        )


def _commit0_container_testbed_lockdown_script(*, container_venv: str = "") -> str:
    quoted_venv = shlex.quote(str(container_venv or ""))
    return f"""set -eu
testbed=/testbed
venv={quoted_venv}
if [ -d "$testbed" ]; then

# Commit0 official images keep the upstream checkout under /testbed and the
# runnable evaluator environment under /testbed/.venv. Solve-phase agents must
# execute the venv but must not read or list the upstream checkout as an oracle.
chmod 0755 "$testbed" 2>/dev/null || true
if [ -n "$venv" ] && [ -d "$venv" ]; then
  chmod a+rx "$venv" "$venv/bin" 2>/dev/null || true
  if [ -d "$venv/bin" ]; then
    find "$venv/bin" -maxdepth 1 -type f -exec chmod a+rx {{}} + 2>/dev/null || true
  fi
  # Commit0 uv images can symlink venv Python into /root/.local; make that
  # single interpreter target traversable without exposing the image checkout.
  for py in "$venv"/bin/python "$venv"/bin/python3 "$venv"/bin/python3.*; do
    [ -e "$py" ] || continue
    resolved="$(readlink -f "$py" 2>/dev/null || printf '%s\n' "$py")"
    [ -n "$resolved" ] || continue
    chmod a+rx "$resolved" 2>/dev/null || true
    parent="$(dirname "$resolved")"
    while [ -n "$parent" ] && [ "$parent" != "/" ]; do
      chmod a+x "$parent" 2>/dev/null || true
      parent="$(dirname "$parent")"
    done
  done
  xdist_vendor="${{APEX_COMMIT0_PYTEST_XDIST_VENDOR_ROOT:-}}"
  if [ -n "$xdist_vendor" ] && [ -d "$xdist_vendor" ] && [ -x "$venv/bin/python" ]; then
    site_packages="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    if [ -n "$site_packages" ]; then
      mkdir -p "$site_packages" 2>/dev/null || true
      cp -a "$xdist_vendor"/. "$site_packages"/ 2>/dev/null || true
    fi
  fi
fi

venv_name=
if [ -n "$venv" ] && [ "$(dirname "$venv")" = "$testbed" ]; then
  venv_name="$(basename "$venv")"
fi
if [ -n "$venv_name" ]; then
  # Commit0 images can carry large upstream checkouts; top-level no-execute
  # prevents traversal without recursively chmoding thousands of files.
  find "$testbed" -mindepth 1 -maxdepth 1 ! -name "$venv_name" -exec chmod go-rwx {{}} + 2>/dev/null || true
else
  # Commit0 images can carry large upstream checkouts; top-level no-execute
  # prevents traversal without recursively chmoding thousands of files.
  find "$testbed" -mindepth 1 -maxdepth 1 -exec chmod go-rwx {{}} + 2>/dev/null || true
fi
chmod 0711 "$testbed" 2>/dev/null || true
fi
"""


def _lock_down_commit0_container_testbed(
    *,
    container_name: str,
    docker_bin: str,
    docker_env: dict[str, str],
    container_venv: str = "",
) -> None:
    # Commit0 Docker fact: official images include the upstream checkout at
    # /testbed; only the evaluator venv is needed during solve-phase execution.
    if not str(container_name or "").strip():
        return
    result = run_process_command(
        [
            str(docker_bin or "docker"),
            "exec",
            "-u",
            "root",
            str(container_name),
            "sh",
            "-c",
            _commit0_container_testbed_lockdown_script(container_venv=container_venv),
        ],
        env=docker_env,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            normalize_terminal_output(result.stdout + result.stderr).strip()
            or "failed to lock down Commit0 /testbed source tree"
        )


def _prepend_env_path(env: dict[str, str], key: str, path: str | Path) -> None:
    value = str(path)
    existing = str(env.get(key) or "")
    env[key] = value + (os.pathsep + existing if existing else "")


# V1 anti-cheat: the new rootless flattened base commit.
_COMMIT0_FLATTENED_BASE_BRANCH = "apex-base"
_COMMIT0_FLATTENED_BASE_MESSAGE = "apex-base: Commit0 stubbed working tree (history erased)"
# V3 / V4a: build + bytecode artifacts removed before the flattened commit so they
# can never re-materialize stubbed source (decompiled .pyc) or carry a vendored
# build tree into the rootless base.
_COMMIT0_FLATTEN_SCRUB_DIR_NAMES = ("__pycache__", "build", "dist")
_COMMIT0_FLATTEN_SCRUB_FILE_SUFFIXES = (".pyc", ".pyo")
_COMMIT0_FLATTEN_SCRUB_DIR_SUFFIXES = (".egg-info",)


def _scrub_commit0_flatten_artifacts(repo_dir: Path) -> None:
    """Remove bytecode (V3) + build/dist/egg-info (V4a) before the flattened commit.

    Walks the working tree (excluding any ``.git`` admin dirs) and deletes
    ``__pycache__``/``build``/``dist``/``*.egg-info`` directories plus ``*.pyc``/
    ``*.pyo`` files. Best-effort per entry so a single permission error never
    aborts the flatten; the subsequent ``git add -A`` simply commits whatever
    survived. Anti-cheat rationale: decompilable bytecode and prebuilt
    distribution trees can leak the stubbed-out gold implementation, so they
    must not enter the rootless base commit.
    """

    repo_dir = Path(repo_dir)
    for root, dir_names, file_names in os.walk(repo_dir, topdown=True):
        # Never descend into git admin dirs (the flatten re-inits .git anyway).
        dir_names[:] = [name for name in dir_names if name != ".git"]
        root_path = Path(root)
        surviving_dirs: list[str] = []
        for name in dir_names:
            if name in _COMMIT0_FLATTEN_SCRUB_DIR_NAMES or name.endswith(
                _COMMIT0_FLATTEN_SCRUB_DIR_SUFFIXES
            ):
                shutil.rmtree(root_path / name, ignore_errors=True)
                continue
            surviving_dirs.append(name)
        # Don't recurse into directories we just deleted.
        dir_names[:] = surviving_dirs
        for name in file_names:
            if name.endswith(_COMMIT0_FLATTEN_SCRUB_FILE_SUFFIXES):
                try:
                    (root_path / name).unlink()
                except OSError:
                    pass


def _flatten_repo_git_history(repo_dir: Path) -> None:
    """V1 anti-cheat: erase all upstream git ancestry to a rootless base commit.

    The Commit0 checkout's ``apex-base`` HEAD keeps the full upstream commit
    chain reachable; rollout worktrees share the parent object DB, so an agent
    can ``git show <ancestor>:path`` and recover the exact gold implementation
    that was stubbed out. Closing that channel requires a *true* flatten — not
    ref/reflog pruning, which leaves the parent objects reachable through HEAD's
    own parent pointers.

    Sequence (preserves the working tree byte-for-byte):
      1. Scrub bytecode (V3) + build/dist/egg-info (V4a) artifacts.
      2. ``rm -rf .git`` (and any nested submodule ``.git`` dirs — e.g. web3.py —
         which the later ``git submodule update --init --recursive`` re-materializes
         from the tracked ``.gitmodules``).
      3. ``git init -q -b apex-base`` → a fresh, empty object DB / ref namespace.
      4. Configure ``user.email``/``user.name`` so the commit can be created in
         environments without global git identity.
      5. ``git add -A`` then a single ``apex-base`` commit (``--allow-empty`` so
         an empty stub tree still produces a valid rootless HEAD).

    Post-conditions (validated in tests): ``rev-list --all --count`` == 1,
    ``git show HEAD~1`` is fatal, the stub working tree is unchanged, and
    ``git worktree add`` succeeds against the new root commit. The new root SHA
    intentionally differs from ``task.base_commit``; that is safe because the
    official audit re-fetches the real base SHA from GitHub independently and
    consumes only the candidate's flat patch.
    """

    repo_dir = Path(repo_dir)

    # Step 1: V3 + V4a scrub (before .git is re-initialized so artifacts never
    # enter the rootless commit).
    _scrub_commit0_flatten_artifacts(repo_dir)

    # Step 2: erase the top-level .git and any nested submodule .git admin dirs.
    # web3.py and similar repos vendor submodules whose .git directories embed
    # their own upstream ancestry; the post-flatten ``git submodule update
    # --init --recursive`` re-materializes them cleanly from ``.gitmodules``.
    for git_path in _iter_repo_git_dirs(repo_dir):
        if git_path.is_dir() and not git_path.is_symlink():
            shutil.rmtree(git_path, ignore_errors=True)
        else:
            try:
                git_path.unlink()
            except OSError:
                pass

    def _git(args: list[str], *, timeout: int = 300) -> None:
        result = run_process_command(["git", *args], cwd=repo_dir, timeout=timeout)
        if result.returncode != 0:
            message = normalize_terminal_output(
                (result.stdout or "") + (result.stderr or "")
            ).strip()
            raise RuntimeError(message or f"git {' '.join(args)} failed in {repo_dir}")

    # Step 3-5: rootless re-init + identity + single flattened commit.
    _git(["init", "-q", "-b", _COMMIT0_FLATTENED_BASE_BRANCH], timeout=120)
    _git(["config", "user.email", "apex@example.com"], timeout=60)
    _git(["config", "user.name", "APEX"], timeout=60)
    _git(["add", "-A"], timeout=600)
    _git(
        [
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            _COMMIT0_FLATTENED_BASE_MESSAGE,
        ],
        timeout=300,
    )


def _iter_repo_git_dirs(repo_dir: Path) -> list[Path]:
    """Return the top-level ``.git`` plus every nested submodule ``.git`` entry.

    A submodule ``.git`` may be a directory (older git) or a file pointing at
    ``../.git/modules/...`` (newer git); both are returned so the flatten can
    remove them. The top-level ``.git`` is yielded last would-be irrelevant —
    order does not matter because each is removed independently.
    """

    repo_dir = Path(repo_dir)
    found: list[Path] = []
    top_git = repo_dir / ".git"
    if top_git.exists() or top_git.is_symlink():
        found.append(top_git)
    for root, dir_names, file_names in os.walk(repo_dir, topdown=True):
        root_path = Path(root)
        if root_path == repo_dir:
            # Top-level .git already captured; don't descend into it.
            dir_names[:] = [name for name in dir_names if name != ".git"]
            continue
        if ".git" in dir_names:
            found.append(root_path / ".git")
            dir_names[:] = [name for name in dir_names if name != ".git"]
        if ".git" in file_names:
            found.append(root_path / ".git")
    return found


def _commit0_solve_phase_proxy_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return solve-phase env hints without blocking commands by policy.

    The actual source/network denial is the Commit0 Docker internal network plus
    model-transport sidecar. This helper only disables local package caches and
    advertises the boundary for diagnostics.
    """

    overrides: dict[str, str] = {
        "PIP_NO_CACHE_DIR": "1",
        # Commit0 pytest commands load required plugins explicitly with -p;
        # autoloading site plugins can double-register pytest-json-report.
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        _COMMIT0_EGRESS_ALLOW_HOSTS_ENV: ",".join(_COMMIT0_EGRESS_ALLOW_HOSTS),
        _COMMIT0_EGRESS_DENY_HOSTS_ENV: ",".join(_COMMIT0_EGRESS_DENY_HOSTS),
    }
    return overrides


def _runtime_cli_env_overrides(env: dict[str, str]) -> dict[str, str]:
    keys = (
        "VIRTUAL_ENV",
        "PATH",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
        "X2P_AGENT_PROXY_ADDRESS",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_DISABLE_ADVISOR_TOOL",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
        "APEX_AGENT_MODEL_PROXY_URL",
        "APEX_HOST_MODEL_PROXY_URL",
        "APEX_CODEX_CLI_MODEL_PROXY_URL",
        "APEX_CODEX_MODEL_PROXY_URL",
        "APEX_CLAUDE_CLI_MODEL_PROXY_URL",
        "APEX_CLAUDE_MODEL_PROXY_URL",
        "APEX_GEMINI_CLI_MODEL_PROXY_URL",
        "APEX_GEMINI_MODEL_PROXY_URL",
        "APEX_OPENCODE_CLI_MODEL_PROXY_URL",
        "APEX_OPENCODE_MODEL_PROXY_URL",
        "APEX_METACODE_CLI_MODEL_PROXY_URL",
        "APEX_METACODE_MODEL_PROXY_URL",
    )
    return {key: str(env[key]) for key in keys if key in env and str(env[key]).strip()}


def _commit0_apply_solve_phase_pytest_output_env(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    updated.setdefault("PYTEST_ADDOPTS", _COMMIT0_SOLVE_PHASE_PYTEST_ADDOPTS)
    return updated


def _commit0_docker_exec_runtime_env(env: dict[str, str]) -> dict[str, str]:
    container_venv = str(env.get("APEX_COMMIT0_CONTAINER_VENV") or "").strip()
    agent_cli_bundle_root = str(
        env.get("APEX_COMMIT0_AGENT_CLI_BUNDLE_CONTAINER_ROOT") or ""
    ).strip()
    runtime_env: dict[str, str] = {}
    if container_venv:
        runtime_env["VIRTUAL_ENV"] = container_venv
        path_entries = [f"{container_venv}/bin"]
        if agent_cli_bundle_root:
            path_entries.append(f"{agent_cli_bundle_root.rstrip('/')}/bin")
        path_entries.extend(
            ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
        )
        runtime_env["PATH"] = ":".join(path_entries)
    for key in _COMMIT0_DOCKER_RUNTIME_PASSTHROUGH_ENV_KEYS:
        value = str(env.get(key) or "").strip()
        if value:
            runtime_env[key] = value
    return runtime_env


def _commit0_official_image_root_setup_script(
    container_venv: str,
    *,
    repair_python_env: bool = False,
) -> str:
    container_venv = str(container_venv or _COMMIT0_OFFICIAL_TESTBED_VENV)
    setup_steps: list[str] = []
    if repair_python_env:
        # Commit0 cookiecutter image fact: fresh solve/eval containers inherit
        # NUL-filled installed dependency files, so repair them before lockdown.
        setup_steps.append(_commit0_official_image_python_repair_command(container_venv))
    # Commit0 official images include an upstream checkout at /testbed; fresh
    # per-rollout agent containers must hide it just like long-lived containers.
    lockdown_script = _commit0_container_testbed_lockdown_script(
        container_venv=str(container_venv or _COMMIT0_OFFICIAL_TESTBED_VENV),
    ).strip()
    setup_steps.append(lockdown_script)
    return "\n".join(step for step in setup_steps if step).strip()


def _with_cli_env_overrides(llm_config: Any, env_overrides: dict[str, str]) -> Any:
    merged = dict(getattr(llm_config, "cli_env_overrides", {}) or {})
    merged.update(env_overrides)
    llm_config.cli_env_overrides = merged
    return llm_config


def _infer_additional_test_packages(
    test_command: str,
    repo_root: str | Path | None = None,
) -> list[str]:
    return infer_additional_pytest_packages(
        test_command,
        repo_root=repo_root,
    )


def _commit0_pytest_xdist_max_worker_count() -> int:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if callable(sched_getaffinity):
        try:
            affinity_count = len(sched_getaffinity(0))
        except (OSError, ValueError):
            affinity_count = 0
        if affinity_count > 0:
            return affinity_count
    try:
        cpu_count = os.cpu_count()
    except NotImplementedError:
        cpu_count = None
    return max(1, int(cpu_count or 1))


def _commit0_pytest_xdist_fair_share_worker_count(
    config: ApexConfig,
    *,
    xdist_context: str = "candidate_eval",
) -> int:
    raw_workers = _commit0_pytest_xdist_max_worker_count()
    benchmark = config.benchmark
    rollout = config.rollout
    task_lanes = max(1, int(getattr(benchmark, "task_parallelism", 1) or 1))
    audit_lanes = (
        max(1, int(getattr(benchmark, "commit0_official_audit_parallelism", 1) or 1))
        if bool(getattr(benchmark, "commit0_official_audit_selected", False))
        else 1
    )
    if str(xdist_context or "").strip().lower() in {
        "scoring",
        "local_scoring",
        "rollout",
        "quick_verification",
        "solve_phase",
    }:
        fair_share_lanes = max(task_lanes, audit_lanes)
    else:
        candidate_eval_lanes = max(1, int(getattr(rollout, "parallel_workers", 1) or 1))
        fair_share_lanes = max(task_lanes * candidate_eval_lanes, audit_lanes)
    # Commit0/Python pytest-xdist worker counts are per pytest process: rollout
    # quick checks share CPUs across nested task/rollout lanes, while local
    # scoring shares only across outer scorer/audit lanes.
    return max(1, (raw_workers + fair_share_lanes - 1) // fair_share_lanes)


def _host_pytest_xdist_vendor_paths() -> list[Path]:
    vendor_specs = (
        ("xdist", "pytest_xdist"),
        ("execnet", "execnet"),
    )
    paths: list[Path] = []
    seen: set[Path] = set()
    for module_name, dist_stem in vendor_specs:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return []
        if spec.submodule_search_locations:
            package_path = Path(next(iter(spec.submodule_search_locations))).resolve()
        elif spec.origin:
            package_path = Path(spec.origin).resolve()
        else:
            return []
        candidates = [package_path]
        site_dir = package_path.parent
        candidates.extend(sorted(site_dir.glob(f"{dist_stem}-*.dist-info")))
        candidates.extend(sorted(site_dir.glob(f"{dist_stem.replace('_', '-')}-*.dist-info")))
        for candidate in candidates:
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    return paths


def serialize_llm_configs(config: ApexConfig) -> list[dict[str, Any]]:
    return copy.deepcopy(config.to_dict().get("llm_configs", []))


def _load_commit0_official_runner():
    try:
        from commit0.harness.run_pytest_ids import main as run_pytest_ids
    except ImportError as exc:
        raise RuntimeError(
            "Commit0 benchmarking now uses the official local Docker evaluator. "
            "Install the required host packages with "
            "`python3 -m pip install --user commit0 GitPython docker modal strenum fastcore ghapi pre-commit`."
        ) from exc
    return run_pytest_ids


def _hash_string(value: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(value.encode("utf-8"))
    return sha256.hexdigest()[:22]


def _coerce_exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    try:
        return int(str(code))
    except (TypeError, ValueError):
        return 1


def _build_commit0_eval_branch_name(label: str) -> str:
    slug = _slugify_output_component(label).replace("_", "-")
    return f"apex-eval-{slug}-{time.time_ns()}"


def _resolve_docker_sdk_env() -> dict[str, str]:
    existing = {
        key: os.environ[key]
        for key in ("DOCKER_HOST", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
        if os.environ.get(key)
    }
    if existing.get("DOCKER_HOST"):
        return existing

    context_result = subprocess.run(
        ["docker", "context", "show"],
        capture_output=True,
        text=True,
    )
    context_name = context_result.stdout.strip()
    if context_result.returncode != 0 or not context_name:
        return {}

    inspect_result = subprocess.run(
        ["docker", "context", "inspect", context_name],
        capture_output=True,
        text=True,
    )
    if inspect_result.returncode != 0:
        return {}

    try:
        payload = json.loads(inspect_result.stdout)
    except json.JSONDecodeError:
        return {}
    if not payload:
        return {}

    docker_endpoint = (payload[0].get("Endpoints") or {}).get("docker") or {}
    host = str(docker_endpoint.get("Host") or "").strip()
    if not host:
        return {}
    return {"DOCKER_HOST": host}


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace").strip()


def default_commit0_output_dir(
    config: ApexConfig,
    run_kind: str,
    base_dir: str | Path | None = None,
) -> Path:
    if run_kind not in {"apex", "raw"}:
        raise ValueError(f"Unsupported Commit0 run kind: {run_kind}")

    llm_configs = serialize_llm_configs(config)
    primary = llm_configs[0] if llm_configs else {}
    backend = _slugify_output_component(str(primary.get("backend", "default")))
    model = _slugify_output_component(str(primary.get("model", "default")))
    output_root = Path(base_dir) if base_dir is not None else Path.cwd()
    return output_root / f".{run_kind}_commit0_{backend}_{model}"


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,  # 95% CI
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Used by the multi-seed CAID head-to-head reporter to attach honest
    confidence intervals to per-repo solve rates. Wilson is preferred
    over the normal-approximation interval because it remains
    well-defined at p=0 and p=1 — both of which actually appear in
    Commit0-Lite reports (some repos solve 0/3 seeds, some 3/3).

    Returns (lo, hi) bounded to [0.0, 1.0]. With trials=0 the interval
    degenerates to (0.0, 1.0), which is the right "no information"
    answer for downstream consumers.
    """
    if trials <= 0:
        return (0.0, 1.0)
    n = float(trials)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    halfwidth = (z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) ** 0.5) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return (round(lo, 4), round(hi, 4))


def aggregate_seed_reports(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-repo solve outcomes across N seed runs of the
    same Commit0 benchmark.

    Each entry in ``seed_reports`` is the parsed ``benchmark_report.json``
    from a single seed. We reduce to:

      * ``per_repo`` — for each repo seen across seeds: (successes,
        trials, mean_solve_rate, 95% Wilson CI).
      * ``mean_solve_rate`` and Wilson CI on the *aggregate* (sum across
        repos) — gives a single headline number with honest uncertainty.

    Robust to missing fields and partial seeds: a seed that didn't reach
    a given repo just doesn't contribute to that repo's denominator.
    """
    per_repo_outcomes: dict[str, dict[str, Any]] = {}
    seed_count = len(seed_reports)
    for report in seed_reports:
        tasks = list((report or {}).get("tasks") or [])
        for task in tasks:
            if not isinstance(task, dict):
                continue
            repo_name = str(
                task.get("repo") or task.get("repo_name") or task.get("instance_id") or ""
            ).strip()
            if not repo_name:
                continue
            entry = per_repo_outcomes.setdefault(
                repo_name,
                {"successes": 0, "trials": 0},
            )
            entry["trials"] += 1
            if bool(task.get("success")):
                entry["successes"] += 1

    aggregate_successes = 0
    aggregate_trials = 0
    per_repo: list[dict[str, Any]] = []
    for repo_name in sorted(per_repo_outcomes):
        entry = per_repo_outcomes[repo_name]
        s, n = int(entry["successes"]), int(entry["trials"])
        lo, hi = wilson_score_interval(s, n)
        per_repo.append(
            {
                "repo": repo_name,
                "successes": s,
                "trials": n,
                "mean_solve_rate": round(s / n, 4) if n > 0 else 0.0,
                "wilson_ci_low": lo,
                "wilson_ci_high": hi,
            }
        )
        aggregate_successes += s
        aggregate_trials += n

    aggregate_lo, aggregate_hi = wilson_score_interval(aggregate_successes, aggregate_trials)
    return {
        "seed_count": seed_count,
        "repo_count": len(per_repo),
        "aggregate_successes": aggregate_successes,
        "aggregate_trials": aggregate_trials,
        "aggregate_mean_solve_rate": (
            round(aggregate_successes / aggregate_trials, 4) if aggregate_trials > 0 else 0.0
        ),
        "aggregate_wilson_ci_low": aggregate_lo,
        "aggregate_wilson_ci_high": aggregate_hi,
        "per_repo": per_repo,
    }


def render_seed_aggregate_markdown(aggregate: dict[str, Any]) -> str:
    """Human-readable Markdown for the multi-seed aggregate.

    Column order: repo | successes/trials | mean | [Wilson CI lo, hi].
    The aggregate row at the bottom carries the headline numbers.
    """
    lines = [
        "# CAID Head-to-Head Aggregate (multi-seed Wilson 95% CIs)",
        "",
        f"- Seeds aggregated: {aggregate.get('seed_count', 0)}",
        f"- Unique repos: {aggregate.get('repo_count', 0)}",
        (
            f"- Aggregate solve rate: "
            f"{aggregate.get('aggregate_successes', 0)}/"
            f"{aggregate.get('aggregate_trials', 0)} "
            f"({100.0 * aggregate.get('aggregate_mean_solve_rate', 0.0):.1f}%) "
            f"[95% CI: {100.0 * aggregate.get('aggregate_wilson_ci_low', 0.0):.1f}%–"
            f"{100.0 * aggregate.get('aggregate_wilson_ci_high', 0.0):.1f}%]"
        ),
        "",
        "## Per-repo solve rates",
        "",
        "| Repo | successes/trials | mean | Wilson 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for entry in aggregate.get("per_repo") or []:
        lines.append(
            f"| {entry['repo']} | {entry['successes']}/{entry['trials']} | "
            f"{100.0 * entry['mean_solve_rate']:.1f}% | "
            f"[{100.0 * entry['wilson_ci_low']:.1f}%, {100.0 * entry['wilson_ci_high']:.1f}%] |"
        )
    return "\n".join(lines) + "\n"


def _format_model_config_summary(model_config: list[dict[str, Any]]) -> str:
    if not model_config:
        return "none"

    summaries = []
    for entry in model_config:
        backend = entry.get("backend", "unknown")
        model = entry.get("model", "default")
        timeout = entry.get("cli_timeout")
        if timeout is None:
            timeout = entry.get("timeout")
        if timeout is None:
            summaries.append(f"{backend}/{model}")
            continue
        summaries.append(f"{backend}/{model} (timeout={timeout}s)")
    return ", ".join(summaries)


def _slugify_output_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "default"


# Re-entrant lock so nested ``_temporary_environ(...) + _temporary_cwd(...)``
# inside the same thread doesn't deadlock; cross-thread serialization still
# holds.
_PROCESS_GLOBAL_MUTATION_LOCK = __import__("threading").RLock()


@contextmanager
def _temporary_environ(overrides: dict[str, str]):
    """Mutate process env for the duration of the with-block.

    Audit H10: serialize the mutation window via a module-level lock so
    concurrent task workers don't trample each other's overrides. The
    lock is held only across the brief env mutation; the underlying
    work runs WITHIN the with-block, but if multiple tasks contend on
    the same env we serialize them rather than risk a clobber.
    """

    with _PROCESS_GLOBAL_MUTATION_LOCK:
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@contextmanager
def _temporary_cwd(path: Path):
    """Mutate process cwd for the duration of the with-block.

    Audit H10: serialize via the same lock so parallel workers don't
    chdir on top of each other. ``os.chdir`` is process-global; the
    lock is the only safe way to use it under threads.
    """

    with _PROCESS_GLOBAL_MUTATION_LOCK:
        previous = Path.cwd()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(previous)


@dataclass
class _CommandResult:
    returncode: int
    output: str
