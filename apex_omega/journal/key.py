"""Canonical journal key / input-hash (plan §15.3, §02.5).

The cache-validity rule is the load-bearing invariant: a cache HIT *replays* the
recorded artifact, it does NOT re-derive.  Therefore the key must capture
*everything that determines the result* — prompt (volatile region stripped),
model, vendor, cli_version, and a hash of the scoped inputs *including the base
repo snapshot SHA*.  If the code under the worker changed, the snapshot SHA
changes → the key changes → the call correctly re-runs (the documented
stale-answer-vs-changed-code failure mode is thereby impossible).

A second invariant (plan §16): the SAME canonicalizer that strips the volatile
region for the journal key must also produce the stable prefix for provider-cache
breakpoints, so journal validity and provider cache-hit-rate never diverge.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping


# Patterns for the *volatile* region of a prompt — content that legitimately
# changes run-to-run without changing the semantic request.  Stripping these
# keeps the journal key (and the provider-cache stable prefix) stable.
_DEFAULT_VOLATILE_PATTERNS: tuple[str, ...] = (
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",  # ISO timestamps
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",  # UUIDs
    r"/tmp/[^\s'\"]+",                       # tmp paths
    r"/var/folders/[^\s'\"]+",               # macOS tmp paths
    r"\bsession[_-]?id[=:]\s*\S+",          # explicit session ids
)
_VOLATILE_RE = re.compile("|".join(_DEFAULT_VOLATILE_PATTERNS))

# Marker pair a caller may wrap around an explicitly-volatile span.
VOLATILE_OPEN = "<<<APEXΩ:VOLATILE>>>"
VOLATILE_CLOSE = "<<<APEXΩ:/VOLATILE>>>"
_VOLATILE_SPAN_RE = re.compile(
    re.escape(VOLATILE_OPEN) + r".*?" + re.escape(VOLATILE_CLOSE), re.DOTALL
)


def canonicalize_prompt(prompt: str, *, extra_patterns: Iterable[str] = ()) -> str:
    """Strip the declared volatile region from a prompt.  Deterministic and
    idempotent.  Used for BOTH the journal key and the provider-cache stable
    prefix (single canonicalizer, plan §16)."""
    if not prompt:
        return ""
    text = _VOLATILE_SPAN_RE.sub("«vol»", prompt)
    text = _VOLATILE_RE.sub("«vol»", text)
    for pat in extra_patterns:
        text = re.sub(pat, "«vol»", text)
    return text


def canonicalize(obj: Any) -> Any:
    """Recursively normalize a value into a JSON-canonical structure: dict keys
    sorted, sets → sorted lists, tuples → lists, dataclasses → dicts.  Raises
    ``TypeError`` on a genuinely non-serializable leaf (fail-loud: a silent
    ``default=str`` would let an object's memory-address repr poison the key and
    break determinism)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return canonicalize(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (set, frozenset)):
        return sorted((canonicalize(x) for x in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    raise TypeError(
        f"canonicalize: non-serializable value of type {type(obj).__name__!r}; "
        "journal/key inputs must be JSON-native to stay deterministic"
    )


def canonical_json(obj: Any) -> str:
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scoped_inputs_hash(scoped_inputs: Mapping[str, Any] | None) -> str:
    """Hash of the scoped inputs.  MUST include the base repo snapshot SHA if the
    caller wants changed-code-re-runs (callers put ``repo_snapshot_sha`` inside
    ``scoped_inputs``)."""
    return sha256_hex(canonical_json(scoped_inputs or {}))


def canonical_key(components: Mapping[str, Any]) -> str:
    """Compute the input-hash from the journal key components.

    Recognized components (plan §15.3 authoritative ∪ §02 engine variant):
      prompt | prompt_canonical, schema, model, vendor, cli_version, agentType,
      effort, scoped_inputs (or scoped_inputs_hash), item_id, stage,
      repo_snapshot_sha.
    Any extra component is folded in verbatim (so an over-specified caller is
    safe; an under-specified one risks a too-coarse key — that is the caller's
    responsibility per §15.3.2)."""
    c = dict(components)
    # Normalize the prompt into its canonical (volatile-stripped) form exactly once.
    if "prompt" in c and "prompt_canonical" not in c:
        c["prompt_canonical"] = canonicalize_prompt(str(c.pop("prompt")))
    elif "prompt" in c:
        c.pop("prompt")
    # Collapse scoped_inputs into a stable sub-hash so a huge scoped payload does
    # not bloat the key material (and matches the §15 frozen-dataclass field).
    if "scoped_inputs" in c and "scoped_inputs_hash" not in c:
        c["scoped_inputs_hash"] = scoped_inputs_hash(c.pop("scoped_inputs"))
    elif "scoped_inputs" in c:
        c.pop("scoped_inputs")
    return sha256_hex(canonical_json(c))
