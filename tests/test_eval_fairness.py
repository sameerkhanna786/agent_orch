"""Fairness guard (trust-audit change-list): the commit0 eval must provide the orchestrator/
agents ONLY (1) the original commit0 prompt and (2) the gold test suite — never a harness-derived
"design contract" that pre-digests the gold tests into API/enum/parametrization directives.

These tests fail if the design contract is ever re-wired into the evaluated path."""

from __future__ import annotations

from pathlib import Path

from apex_omega.autogen.architect import build_author_prompt

_ROOT = Path(__file__).resolve().parents[1]

# strings the gold-test-derived design contract used to inject
_CONTRACT_MARKERS = (
    "DESIGN CONTRACT", "repr_directive", "value_directive", "ALIAS-INCLUSIVE",
    "EXACT value strings", "Required source modules", "Parametrization keys for",
    "Required test counts per file", "PLAIN Enum",
)

# the contract-injecting entry points that must NOT be referenced by the evaluated path
_CONTRACT_FUNCS = (
    "safe_contract_text", "render_contract_prompt", "derive_design_contract",
    "_design_contract_enabled", "from .design_contract import",
    "from ..eval.design_contract import safe_contract",
)


def test_author_prompt_has_no_design_contract():
    rmap = {"modules": ["pkg"], "approach": "implement the parsers",
            "key_files": ["pkg/a.py"], "difficulty": "medium",
            # even if a stray design_contract key sneaks into the repo map, it must not surface:
            "design_contract": "DESIGN CONTRACT: define class Foo(enum.Enum) ..."}
    out = build_author_prompt(rmap)
    for m in _CONTRACT_MARKERS:
        assert m not in out, f"author prompt leaked contract marker: {m}"


def test_evaluated_path_does_not_wire_the_contract():
    for rel in ("apex_omega/eval/commit0_autogen.py", "apex_omega/autogen/architect.py"):
        src = (_ROOT / rel).read_text()
        for fn in _CONTRACT_FUNCS:
            assert fn not in src, f"{rel} still references contract entry point: {fn}"


def test_task_framing_is_rules_only_and_leak_safe():
    # the framing states the EVAL RULES (fair, symmetric) and leaks NO answer/API/enum/package hints.
    from apex.evaluation.commit0_benchmark import TASK_FRAMING_BLOCK
    t = TASK_FRAMING_BLOCK.lower()
    for kw in ("reimplementation", "out of scope", "do not", "exact match", "tests"):
        assert kw in t, f"framing missing rule keyword: {kw}"
    for banned in ("enum.enum", "values()", "repr_directive", "alias-inclusive", "locale",
                   "mimesis", "pydantic", "voluptuous", "jinja", "the fix is", "you need a"):
        assert banned not in t, f"framing leaked an answer/package hint: {banned}"


def test_author_and_scout_prompts_carry_framing_once():
    from apex.evaluation.commit0_benchmark import TASK_FRAMING_BLOCK
    from apex_omega.autogen.architect import build_author_prompt, build_scout_prompt
    rmap = {"modules": ["pkg"], "task_framing": TASK_FRAMING_BLOCK,
            "sample_source_files": [], "sample_test_files": [], "n_source_files": 1, "n_test_files": 1}
    ap = build_author_prompt(rmap)
    assert "TASK FRAMING" in ap and "out of scope" in ap.lower()
    assert ap.count("from-scratch reimplementation") == 1     # not double-rendered via repo-map dump
    sp = build_scout_prompt(rmap, {"repo": "x"}, 0)
    assert "out of scope" in sp.lower()                       # scouts plan within the rules


def test_design_contract_module_is_quarantined():
    # the module may remain on disk for reference, but must be banner-marked as quarantined.
    src = (_ROOT / "apex_omega/eval/design_contract.py").read_text()
    assert "QUARANTINED" in src and "NOT wired into the evaluated path" in src


# --- GOLD SCORING REQUIRED (commit0 trust guarantee) -----------------------------
# Every arm (v1 baselines + autogen) MUST score by exact gold expected-test-id match
# and can NEVER fall through to visible-suite (pytest_summary) acceptance.

def test_base_config_pins_gold_evaluation_contract():
    import json
    from apex.core.config import ApexConfig
    from apex.evaluation.commit0_benchmark import _commit0_expected_id_scoring_required
    base = json.loads((_ROOT / "configs/base_commit0_local.json").read_text())
    ec = base["benchmark"]["evaluation_contract"]
    assert ec["mode"] == "gold_suite_visible" and ec["scoring_universe"] == "expected_test_ids"
    assert _commit0_expected_id_scoring_required(ApexConfig.from_dict(base)) is True


def test_every_arm_funnel_requires_gold_scoring():
    import json
    from apex.core.config import ApexConfig
    from apex.evaluation.commit0_benchmark import _commit0_expected_id_scoring_required
    from apex_omega.ablation.arms import get_arm
    from apex_omega.eval.commit0_driver import build_arm_config_dict, pin_gold_scoring_contract
    from apex_omega.eval.commit0_autogen import _force_local_config_dict
    base = json.loads((_ROOT / "configs/base_commit0_local.json").read_text())
    # v1 baseline funnel + autogen arm via build_arm_config_dict
    for name in ("baseline", "B0_single_model", "B2_v1_full_cap16", "autogen_orchestrator"):
        cfg = build_arm_config_dict(base, get_arm(name), force_local=True)
        assert _commit0_expected_id_scoring_required(ApexConfig.from_dict(cfg)) is True, name
    # autogen (Mode C) funnel
    assert _commit0_expected_id_scoring_required(
        ApexConfig.from_dict(_force_local_config_dict(base))) is True
    # a stray non-gold override is OVERRIDDEN by the pin (gold merges last)
    poisoned = {**base, "benchmark": {**base["benchmark"],
                "evaluation_contract": {"mode": "custom", "scoring_universe": "runner_summary"}}}
    assert _commit0_expected_id_scoring_required(
        ApexConfig.from_dict(pin_gold_scoring_contract(poisoned))) is True


def test_framing_states_gold_expected_id_scoring_required():
    from apex.evaluation.commit0_benchmark import TASK_FRAMING_BLOCK
    t = TASK_FRAMING_BLOCK.lower()
    assert "required" in t and "expected" in t and "gold" in t and "exact match" in t


def test_nongold_accept_is_never_banked_as_solve():
    # the VerificationResult layer downgrades any accept whose scoring_source is NOT the gold
    # path ("commit0_test_ids") to indeterminate — last line of defense against a visible-suite
    # false positive ever counting as a solve.
    from apex_omega.eval.scoring import verification_from_commit0_evaluation as vfe

    class _Ev:
        passed = 851; failed = 0; errors = 0; total_tests = 851; pass_rate = 1.0
        scored_success = True; expected_test_coverage = {}; verification_taxonomy = ""
        evaluation_status = "solved"; scoring_source = "pytest_summary"

    vr = vfe(_Ev())
    assert vr.accepted is False and vr.indeterminate is True
    # and a genuine gold accept is preserved with the REAL gold count (total_tests, not total)
    class _Gold(_Ev):
        scoring_source = "commit0_test_ids"
    g = vfe(_Gold())
    assert g.accepted is True and g.total == 851 and g.passed == 851
