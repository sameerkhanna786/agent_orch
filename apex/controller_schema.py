"""
Typed controller action schema shared across planner, search, and rollout.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


# Back-compat shim: legacy controller payloads stored these flags as top-level
# keys on the search_policy dict (alongside `mode`, `origin`, etc.) instead of
# as typed attributes on `ControllerAction`. Promoted to typed fields below;
# this tuple stays only so legacy payloads can still be deserialized via
# `ControllerAction.from_search_policy`. Do not use in new code -- read the
# typed attributes (e.g. ``action.delegated_subtask``) instead.
_LEGACY_FLAG_KEYS = (
    "collection_error_fast_path",
    "planner_authored_subtasks",
    "delegated_subtask",
)


# Process-wide flag so we only warn once about legacy string-keyed inputs.
_LEGACY_FLAG_DEPRECATION_WARNED = False


def _warn_legacy_flag_usage(source: str) -> None:
    global _LEGACY_FLAG_DEPRECATION_WARNED
    if _LEGACY_FLAG_DEPRECATION_WARNED:
        return
    _LEGACY_FLAG_DEPRECATION_WARNED = True
    warnings.warn(
        (
            "ControllerAction received legacy string-keyed flag input via "
            f"{source}; promote callers to the typed fields "
            "collection_error_fast_path / planner_authored_subtasks / "
            "delegated_subtask on ControllerAction."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def _reset_legacy_flag_warning_state() -> None:
    """Test helper -- resets the once-per-process warning flag."""
    global _LEGACY_FLAG_DEPRECATION_WARNED
    _LEGACY_FLAG_DEPRECATION_WARNED = False


@dataclass
class EditSpan:
    file_path: str
    symbol: str = ""
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "symbol": self.symbol,
            "start_line": int(self.start_line or 0),
            "end_line": int(self.end_line or 0),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditSpan":
        return cls(
            file_path=str(data.get("file_path") or ""),
            symbol=str(data.get("symbol") or ""),
            start_line=int(data.get("start_line") or 0),
            end_line=int(data.get("end_line") or 0),
        )


def _coerce_edit_spans(values: Iterable[Any]) -> list[EditSpan]:
    spans: list[EditSpan] = []
    seen: set[tuple[str, str, int, int]] = set()
    for value in values:
        if isinstance(value, EditSpan):
            span = value
        elif isinstance(value, dict):
            span = EditSpan.from_dict(value)
        else:
            continue
        key = (
            str(span.file_path or "").strip(),
            str(span.symbol or "").strip(),
            int(span.start_line or 0),
            int(span.end_line or 0),
        )
        if not key[0] or key in seen:
            continue
        spans.append(span)
        seen.add(key)
    return spans


@dataclass
class ControllerAction:
    """Canonical typed description of a planner-visible controller action."""

    kind: str = "rollout_brief"
    mode: str = "surgical"
    origin: str = "heuristic"
    regime_state: str = ""
    verification_focus: str = "targeted_validation"
    graph_target_kind: str = ""
    graph_target_family: str = ""
    allocator_arm: str = ""
    file_paths: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    edit_spans: list[EditSpan] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Promoted-from-legacy typed flags. These are the single source of truth;
    # the same names also live in `flags` for back-compat with readers that
    # still use the dict view.
    collection_error_fast_path: bool = False
    planner_authored_subtasks: bool = False
    delegated_subtask: bool = False

    def __post_init__(self) -> None:
        # Mirror typed flags into the dict view so existing dict-based readers
        # (e.g. ``action.flags.get("collection_error_fast_path")``) keep
        # working with the same value as the typed attribute.
        self._sync_flags_view()

    def _sync_flags_view(self) -> None:
        """Mirror typed boolean fields into the `flags` dict view."""
        if self.collection_error_fast_path:
            self.flags["collection_error_fast_path"] = True
        if self.planner_authored_subtasks:
            self.flags["planner_authored_subtasks"] = True
        if self.delegated_subtask:
            self.flags["delegated_subtask"] = True

    def to_dict(self) -> dict[str, Any]:
        # Ensure typed -> dict mirror is up to date even if attributes were
        # mutated after construction.
        self._sync_flags_view()
        return {
            "kind": self.kind,
            "mode": self.mode,
            "origin": self.origin,
            "regime_state": self.regime_state,
            "verification_focus": self.verification_focus,
            "graph_target_kind": self.graph_target_kind,
            "graph_target_family": self.graph_target_family,
            "allocator_arm": self.allocator_arm,
            "file_paths": list(self.file_paths),
            "test_ids": list(self.test_ids),
            "symbols": list(self.symbols),
            "edit_spans": [span.to_dict() for span in self.edit_spans],
            "flags": {str(name): bool(value) for name, value in dict(self.flags or {}).items()},
            "metadata": dict(self.metadata or {}),
            "collection_error_fast_path": bool(self.collection_error_fast_path),
            "planner_authored_subtasks": bool(self.planner_authored_subtasks),
            "delegated_subtask": bool(self.delegated_subtask),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerAction":
        flags_input = {
            str(name): bool(value)
            for name, value in dict(data.get("flags") or {}).items()
            if str(name)
        }
        # Resolve each promoted flag with priority:
        #   typed field on payload  >  payload["flags"][name]  >  False
        # If the payload only carried the value via the legacy `flags` dict
        # (i.e. the typed key is absent from `data` AND the dict entry is
        # truthy) emit a deprecation warning so callers can migrate to
        # writing the typed field. Don't warn on round-trips of payloads we
        # ourselves emitted (which always include the typed key).
        legacy_used = False
        resolved_typed: dict[str, bool] = {}
        for key in _LEGACY_FLAG_KEYS:
            if key in data:
                resolved_typed[key] = bool(data.get(key))
            else:
                legacy_value = bool(flags_input.get(key, False))
                resolved_typed[key] = legacy_value
                if legacy_value:
                    legacy_used = True
        if legacy_used:
            _warn_legacy_flag_usage("ControllerAction.from_dict (flags dict)")
        return cls(
            kind=str(data.get("kind") or "rollout_brief"),
            mode=str(data.get("mode") or "surgical"),
            origin=str(data.get("origin") or "heuristic"),
            regime_state=str(data.get("regime_state") or ""),
            verification_focus=str(data.get("verification_focus") or "targeted_validation"),
            graph_target_kind=str(data.get("graph_target_kind") or ""),
            graph_target_family=str(data.get("graph_target_family") or ""),
            allocator_arm=str(data.get("allocator_arm") or ""),
            file_paths=_dedupe_strings(list(data.get("file_paths") or [])),
            test_ids=_dedupe_strings(list(data.get("test_ids") or [])),
            symbols=_dedupe_strings(list(data.get("symbols") or [])),
            edit_spans=_coerce_edit_spans(list(data.get("edit_spans") or [])),
            flags=flags_input,
            metadata=dict(data.get("metadata") or {}),
            collection_error_fast_path=resolved_typed["collection_error_fast_path"],
            planner_authored_subtasks=resolved_typed["planner_authored_subtasks"],
            delegated_subtask=resolved_typed["delegated_subtask"],
        )

    def to_search_policy(self, *, base: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # Ensure flags dict and typed booleans stay in sync before serialising.
        self._sync_flags_view()
        policy = dict(base or {})
        policy["mode"] = str(self.mode or "surgical")
        policy["origin"] = str(self.origin or "heuristic")
        if self.regime_state:
            policy["origin_regime_state"] = self.regime_state
        if self.verification_focus:
            policy["verification_focus"] = self.verification_focus
        if self.graph_target_kind:
            policy["graph_target_kind"] = self.graph_target_kind
        if self.graph_target_family:
            policy["graph_target_family"] = self.graph_target_family
        if self.allocator_arm:
            policy["allocator_arm"] = self.allocator_arm
        if self.file_paths:
            policy["action_file_paths"] = list(self.file_paths)
        if self.test_ids:
            policy.setdefault("graph_target_test_ids", list(self.test_ids))
            policy["action_test_ids"] = list(self.test_ids)
        if self.symbols:
            policy["action_symbols"] = list(self.symbols)
        if self.edit_spans:
            policy["edit_spans"] = [span.to_dict() for span in self.edit_spans]
        for name, value in dict(self.flags or {}).items():
            if str(name):
                policy[str(name)] = bool(value)
        # Mirror promoted flags into the search_policy top-level for
        # back-compat readers that still consult them by string key. Only emit
        # truthy values to avoid clobbering callers that intentionally omit
        # them (matches the previous behaviour in `_LEGACY_FLAG_KEYS`).
        for key in _LEGACY_FLAG_KEYS:
            if bool(getattr(self, key, False)):
                policy[key] = True
        policy["controller_action"] = self.to_dict()
        return policy

    @classmethod
    def from_search_policy(cls, policy: Optional[dict[str, Any]]) -> "ControllerAction":
        payload = dict(policy or {})
        nested = payload.get("controller_action")
        action = cls.from_dict(dict(nested or {})) if isinstance(nested, dict) else cls()
        if not action.kind:
            action.kind = "rollout_brief"
        action.mode = str(payload.get("mode") or action.mode or "surgical")
        action.origin = str(payload.get("origin") or action.origin or "heuristic")
        action.regime_state = str(payload.get("origin_regime_state") or action.regime_state or "")
        action.verification_focus = str(
            payload.get("verification_focus") or action.verification_focus or "targeted_validation"
        )
        action.graph_target_kind = str(
            payload.get("graph_target_kind") or action.graph_target_kind or ""
        )
        action.graph_target_family = str(
            payload.get("graph_target_family") or action.graph_target_family or ""
        )
        action.allocator_arm = str(payload.get("allocator_arm") or action.allocator_arm or "")
        if not action.file_paths:
            action.file_paths = _dedupe_strings(
                list(payload.get("action_file_paths") or [])
                + list(payload.get("graph_target_file_paths") or [])
            )
        if not action.test_ids:
            action.test_ids = _dedupe_strings(
                list(payload.get("action_test_ids") or [])
                + list(payload.get("graph_target_test_ids") or [])
            )
        if not action.symbols:
            action.symbols = _dedupe_strings(
                list(payload.get("action_symbols") or [])
                + list(payload.get("interface_symbols") or [])
            )
        if not action.edit_spans:
            action.edit_spans = _coerce_edit_spans(list(payload.get("edit_spans") or []))
        merged_flags = {
            str(name): bool(value) for name, value in dict(action.flags or {}).items() if str(name)
        }
        merged_flags.update(
            {
                str(name): bool(value)
                for name, value in dict(payload.get("flags") or {}).items()
                if str(name)
            }
        )
        # Promote any string-keyed legacy entries on the policy dict back into
        # the typed fields. We only track usage as `legacy` when the value is
        # truthy AND the typed field on the nested controller_action wasn't
        # already true (so round-trips of payloads we ourselves emitted, which
        # also mirror the truthy value at the top level, don't warn). False
        # entries are ignored so we don't pollute the flags dict with
        # negative defaults that callers never set.
        legacy_used = False
        for key in _LEGACY_FLAG_KEYS:
            if key not in payload:
                continue
            value = bool(payload.get(key))
            if not value:
                continue
            merged_flags[key] = True
            if not bool(getattr(action, key, False)):
                legacy_used = True
            setattr(action, key, True)
        if legacy_used:
            _warn_legacy_flag_usage("ControllerAction.from_search_policy")
        action.flags = merged_flags
        # Re-mirror typed -> flags so the dict view always reflects the
        # promoted truth even after the merge above.
        action._sync_flags_view()
        return action


def coerce_controller_action(
    payload: Any,
    *,
    fallback_policy: Optional[dict[str, Any]] = None,
    default_files: Optional[Iterable[Any]] = None,
    default_test_ids: Optional[Iterable[Any]] = None,
    default_symbols: Optional[Iterable[Any]] = None,
) -> ControllerAction:
    if isinstance(payload, ControllerAction):
        action = ControllerAction.from_dict(payload.to_dict())
    elif isinstance(payload, dict) and (
        "controller_action" in payload
        or "mode" in payload
        or "origin" in payload
        or "origin_regime_state" in payload
        or "verification_focus" in payload
    ):
        action = ControllerAction.from_search_policy(payload)
    elif isinstance(payload, dict):
        action = ControllerAction.from_dict(payload)
    else:
        action = ControllerAction.from_search_policy(fallback_policy or {})
    if not action.file_paths:
        action.file_paths = _dedupe_strings(list(default_files or []))
    if not action.test_ids:
        action.test_ids = _dedupe_strings(list(default_test_ids or []))
    if not action.symbols:
        action.symbols = _dedupe_strings(list(default_symbols or []))
    if not action.mode:
        action.mode = "surgical"
    if not action.kind:
        action.kind = "rollout_brief"
    if not action.origin:
        action.origin = "heuristic"
    if not action.verification_focus:
        action.verification_focus = "targeted_validation"
    return action


def sync_controller_action_payload(
    policy: Optional[dict[str, Any]],
    *,
    action: Any = None,
    default_files: Optional[Iterable[Any]] = None,
    default_test_ids: Optional[Iterable[Any]] = None,
    default_symbols: Optional[Iterable[Any]] = None,
) -> tuple[ControllerAction, dict[str, Any]]:
    normalized_policy = dict(policy or {})
    typed_action = coerce_controller_action(
        action if action is not None else normalized_policy,
        fallback_policy=normalized_policy,
        default_files=default_files,
        default_test_ids=default_test_ids,
        default_symbols=default_symbols,
    )
    return typed_action, typed_action.to_search_policy(base=normalized_policy)
