## 10. CTDG + Bidirectional Pruning

This section specifies the Code-Test Dependency Graph (CTDG) and the "bidirectional pruning" subsystem as a set of vendor-neutral workflow patterns layered on the APEX substrate (Section 2) and consumed by the speculative tree-search layer (Section 9), the verifier (Section 13), and the active controller (Section 14). It exists to answer one question cheaply and *safely*: **of the tests this repo can run, which ones should this worker run first, and which can it skip during fast iteration — without ever silently discarding a fault-revealing test before the final gate.**

The design is deliberately conservative because the adversarial review judged the headline ambition — "a static CTDG enables safe millisecond pruning in dynamic Python" — **unsound** ([PyCG ICSE'21](https://arxiv.org/abs/2103.00587): ~99.2% precision / ~69.9% recall; [Rothermel & Harrold](https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1015&context=csearticles) safety theorem). We therefore split "use the graph" from "prune as a gate" and keep execution evidence authoritative (the Cardinal Safety Contract, Section 13).

### 10.1 What this subsystem is — and is not

| Concern | Disposition | Why |
|---|---|---|
| Static import/call graph (tree-sitter / LSP) | **Prioritize + explain only** — reorder tests, never exclude | Reordering has zero false-negative risk ([ctdg synthesis](https://www.gauge.sh/blog/how-to-make-ci-fast-and-cheap-with-test-impact-analysis); Rothermel-Harrold). PyCG ~70% recall makes static *exclusion* unsafe. |
| Dynamic per-test coverage map (coverage.py contexts / testmon block-checksums) | **Actual prune gate during fast iteration** — advisory, never authoritative | "As safe as coverage.py" ([testmon.org](https://www.testmon.org/blog/determining-affected-tests/)); a false negative merely delays feedback inside the loop. |
| Full-suite stabilization backstop at final pre-accept state | **Mandatory under default safety mode** | Google TAP / Facebook PTS keep a periodic full run; selection is *never* the sole pre-merge gate. |
| Cheap pre-execution plan score | **Downgrade-only branch prioritizer** — never a kill | A pre-exec soft signal gating *exclusion* is the inverse-equivalent violation of the Cardinal Contract (adversarial verdict: `partially_sound`). |
| Static-AST CTDG as a test-pruning gate | **Rejected** | PyCG recall; reflection/monkeypatch/fixtures invisible; pytest set not statically enumerable. |
| AST / semantic-equivalence checks | **Reserved for patch validation (Section 13), not live navigation** | Equivalence is undecidable; useful only as a bounded overfitting/regression signal. |

The CTDG **feeds priors; the verifier decides.** It is never treated as an oracle.

### 10.2 Why static-only pruning is unsafe in dynamic Python (the load-bearing rationale)

A coding worker (Codex, Claude Code, or other) that trusts a static graph to *drop* tests will silently ship bad patches. The evidence is convergent:

- **Lossy recall.** PyCG, the SOTA static Python call graph, reports ~99.2% precision but only **~69.9% recall**, explicitly ignoring `eval`, `getattr`/`setattr` effects, built-in type-method effects, conditionals, and loops. ~30% of real call edges are absent; each missing edge is a candidate false negative — a pruned-but-fault-revealing test.
- **The test set is not statically enumerable.** pytest items are produced at *collection* time by `parametrize`, `pytest_generate_tests`, fixture graphs from arbitrary plugins, `conftest.py` tree effects, and `pytest_collection_modifyitems`. The only reliable enumeration is running `pytest --collect-only`. A static graph cannot even name what it would prune.
- **Reflection alone breaks static RTS.** [Shi et al. OOPSLA'19](https://lingming.cs.illinois.edu/publications/oopsla2019.pdf) (1173 versions / 24 Java projects) found reflection was the *only* cause of static-RTS unsafety, and reflection-aware safety pushed end-to-end cost from 69.1% to 85.8–91.2% of RetestAll. Python adds monkeypatch, dynamic imports, and ubiquitous `getattr` on top — strictly worse.
- **The safety/precision theorem.** Rothermel-Harrold: you cannot extract both maximal pruning *and* zero false negatives from imperfect dependency data. The residual gap must be closed by dynamic ground truth or a full-suite backstop — neither of which is "static."
- **Practitioner confirmation.** Agentic graph systems (ARISE, CodexGraph) deliberately *drop* dynamic-dispatch/eval/monkeypatch edges to avoid spurious edges — exactly the source of silent false negatives if such a graph gated tests. The marketing-vs-safety gap (Tach: "8x faster," zero correctness caveats) is the trap APEX must not fall into.

This directly collides with APEX v1's Cardinal Safety Contract: a static CTDG is a non-execution soft signal, and using it to *exclude* a candidate test is strictly stronger than the already-prohibited "promote an unverified candidate." Hence: **static graph reorders, dynamic coverage prunes (advisorily), full suite gates.**

### 10.3 Data structures

All artifacts live on the filesystem (filesystem-as-source-of-truth) under the run's repo-context cache, are content-addressed, and are journaled per `agent()` call (Section 15). Field types use Python-ish annotations; a coding agent may realize them as dataclasses, TS interfaces, or structs.

#### 10.3.1 Static layer — `CtdgStaticIndex`

Built once per repo snapshot, amortized like v1's `RepoContext`. It is an *extension* of v1's `RepoGraph` (which today emits `contains/imports/inherits/references/uses/rationale_for` edges and has **no** code→test edge — confirmed in the v1 ingest). We add a typed test-edge layer; we do not rebuild the graph.

```text
CtdgStaticIndex:
  repo_snapshot_id:   str            # git rev or tree hash of the snapshot the index was built on
  builder:            str            # "tree-sitter" | "lsp:<server>" | "regex-fallback"
  language:           str            # "python" | "js" | ... (MVP: python; others degrade to regex)
  symbol_nodes:       dict[SymbolId, SymbolNode]
  test_nodes:         dict[TestNodeId, TestNode]   # FILE/CLASS-level only at static layer (see note)
  code_to_test:       dict[SymbolId, list[TestEdge]]   # STATIC priors only; confidence-tagged
  confidence_default: float = 0.5    # static edges are priors, never authoritative
  built_at:           float
  notes:              list[str]      # e.g. "dynamic-dispatch edges dropped"

SymbolNode:    { id: SymbolId, kind: "func"|"method"|"class"|"module", file: str, span: (int,int) }
TestNode:      { id: TestNodeId, file: str, kind: "module"|"class", nodeids_known: bool }
TestEdge:      { test: TestNodeId, via: "import"|"call"|"inherit"|"name-mention",
                 confidence: float,        # EXTRACTED (>=0.8) vs INFERRED (<=0.6), mirrors v1 levels
                 source: "static" }
```

Note: static `test_nodes` are file/class granularity only. We never claim a static map to individual parametrized `nodeid`s — those are not statically knowable.

#### 10.3.2 Dynamic layer — `CoverageMap` (the real prune signal)

Borrows the highest-leverage idea from `pytest-testmon`: **block-level checksums, not file hashes** — a test re-runs only if a block it *actually executed* changed. Built from `coverage.py` dynamic contexts (`--cov-context=test` / `dynamic_context=test_function`) on the first full run, then incrementally maintained.

```text
CoverageMap:
  schema_version:   int
  selection_key:    SelectionKey        # invalidation key — see 10.3.3
  test_to_blocks:   dict[NodeId, list[BlockRef]]   # per real collected nodeid
  block_checksums:  dict[BlockRef, str]            # adler32/sha of normalized block source
  collected_set:    list[NodeId]         # from `pytest --collect-only` (NOT parsed from source)
  hierarchy_index:  dict[SymbolId, set[NodeId]]    # symbol -> covering tests (for over-select)
  tracer:           str                  # "coverage.py" | "ekstazi" | "build-dag" | ...
  built_at:         float

BlockRef:  { file: str, block_id: str }   # block = function/branch region per tracer
NodeId:    str                            # e.g. "tests/test_x.py::TestA::test_y[param-3]"
```

Safety boundary, stated explicitly in artifact metadata: coverage-derived selection is **only as safe as the tracer**. Dependencies on time, randomness, network, filesystem, env/global state, and C extensions are invisible to `coverage.py` and can cause a wrongly-deselected test. We mitigate by (a) over-selection on hierarchy changes, (b) hashing non-code inputs into the selection key, and (c) the mandatory backstop.

#### 10.3.3 The selection key (cache invalidation that closes false-negative holes)

A stale dependency DB is a documented false-negative source ([testmon issue #92](https://github.com/tarpas/pytest-testmon/issues/92)). The `SelectionKey` is hashed into every coverage decision; any change forces a full re-collect / full run.

```text
SelectionKey = sha256(concat(
  repo_snapshot_id,
  resolved_lockfile_hash,        # dependency bump -> full run
  python_version,                # interpreter change -> full run
  env_fingerprint,               # DJANGO_SETTINGS_MODULE, PYTHONHASHSEED, LANG, etc.
  test_seed,                     # PYTHONHASHSEED / pytest-randomly seed
  config_fingerprint,            # pytest.ini / pyproject [tool.pytest], conftest hashes
  coverage_schema_version,
  docker_image_digest            # pinned per RunManifest (Section 15)
))
```

This makes the selection deterministic and replayable (Section 15) and over-selects exactly where dynamic Python is riskiest (config/hierarchy churn). The container digest comes from the same pinning the RunManifest already enforces — keeping selection vendor-neutral and reproducible across hosts.

### 10.4 The two pruning channels

"Bidirectional" in APEX-Ω means **prioritize before, prune after — never exclude before evidence exists.** Concretely there are two channels, and only the *post-evidence* channel is allowed to drop work.

```text
                 ┌────────────────────────── worker turn / branch ──────────────────────────┐
  PRE  (priors)  │  static CTDG order + cheap plan score  ->  test ORDER + branch BUDGET     │
                 │  (downgrade-only; NEVER removes a test or a branch from the candidate set) │
  POST (gate)    │  dynamic coverage select  ->  run subset  ->  cheap-first cascade verdict   │
                 │  (advisory prune of UNCHANGED-block tests; full suite at final pre-accept) │
                 └───────────────────────────────────────────────────────────────────────────┘
```

#### 10.4.1 PRE channel — static prioritization (zero false-negative risk)

Used to order the candidate test set and to feed branch priors to the FrontierSearch / speculate() machinery (Sections 9, 14). It **never** removes a test.

```python
def order_tests(changed_symbols, static_index, collected_set):
    # 1. Score each collected nodeid by static proximity to the change.
    scored = []
    for nodeid in collected_set:                # collected_set comes from --collect-only
        test_node = test_node_of(nodeid)
        s = 0.0
        for sym in changed_symbols:
            for e in static_index.code_to_test.get(sym, []):
                if e.test == test_node.id:
                    s = max(s, e.confidence)     # EXTRACTED edges dominate INFERRED
        scored.append((nodeid, s))
    # 2. ENTIRE collected set is returned — reordered, never filtered.
    #    Unscored tests sort AFTER scored ones but are STILL INCLUDED.
    return [nid for nid, _ in sorted(scored, key=lambda x: -x[1])]
```

This is the safe win: it accelerates time-to-first-failure (matching RepoGraph / ARISE prioritization gains) at zero recall cost, and it satisfies the contract clause that soft signals may only re-rank.

#### 10.4.2 PRE channel — cheap pre-exec plan score (downgrade-only)

A cheap worker (any vendor model behind the Normalized Executor) may score a proposed plan/edit set to set **branch priority and budget share**, feeding FrontierSearch priors. The adversarial verdict (`partially_sound`) is honored exactly: it can lower a branch's priority but **can never remove it pre-execution**, and a *wildcard lane* always executes the lowest-scored unconventional branch so tail solutions are never silently killed (counters the Best@K ≪ Pass@K headroom and RLHF diversity collapse).

```python
def plan_prior(plan, ctdg, cheap_critic):           # cheap_critic: vendor-neutral, swappable
    # Prefer a generative/CoT critic (THINKPRM-style) over a scalar one for OOD robustness.
    score = cheap_critic.score(plan)                # in [0,1]; advisory metadata, journaled
    reach = ctdg_reachable(plan.edit_targets, ctdg) # static structural plausibility (a HINT)
    prior = clamp(0.5 + 0.3*(score-0.5) + 0.2*(reach-0.5), 0.05, 1.0)
    return BranchPrior(priority=prior, budget_share=prior, removable=False)  # NEVER a kill
```

Guardrails (all mandatory):
- **Downgrade-only.** A branch may be deprioritized to the minimum lane, never excluded. Mirror of v1's evidence-bound review (`accepted` flips True→False only).
- **No RL reward use.** This score must never become an RL/self-improvement reward; reward hacking on coding envs generalizes to sabotage ([Anthropic 2511.18397](https://arxiv.org/abs/2511.18397)).
- **Auto-degrade.** If only a weak critic is available, fall back to pure static-order scheduling (verifier strength is the binding constraint — [SWE-PRM](https://arxiv.org/html/2509.02360v1) shows weak critics can be net-negative).
- **Canary metric.** Emit a "pruned-but-would-have-passed" / "downgraded-but-won" canary so over-aggressive priors are detectable.

#### 10.4.3 POST channel — dynamic coverage prune (the only place work is dropped)

During fast iteration *inside* a worker's edit loop, after a patch touches code, select the at-risk subset via the `CoverageMap` and run only that subset under the existing cheap-first cascade (Section 13: rc==0→errors=1; rc==124→`regression_inconclusive`; AST→symbol-survival→targeted pytest). This *replaces and tightens* v1's heuristic ladder (`graph_target_test_ids[:4] > failing_test_ids[:8] > focus_test_files[:4]`) and narrows v1's `prune_by_regression` (baseline-passing tests in chunks of 50).

```python
def select_affected(changed_blocks, cov_map, static_index):
    if selection_key_changed(cov_map):          # lockfile/env/seed/config/digest change
        return FULL_SUITE                        # over-select: full run, rebuild map
    affected = set()
    for nid, blocks in cov_map.test_to_blocks.items():
        if any(b in changed_blocks for b in blocks):
            affected.add(nid)
    # OVER-SELECT on hierarchy changes: signature/class/module edits re-run everything
    # that touches the module (testmon bias toward false positives over false negatives).
    for sym in hierarchy_changed_symbols(changed_blocks):
        affected |= cov_map.hierarchy_index.get(sym, set())
    # NEW code not yet in the map -> cannot be covered statically -> include broadly.
    if introduces_new_symbols(changed_blocks):
        affected |= sibling_tests_of_changed_files(changed_blocks)
    return sorted(affected) or FULL_SUITE        # empty selection NEVER means "skip all"
```

Three invariants make this honest: (1) an empty selection means *run the full suite*, never "skip everything"; (2) any selection-key change forces a full run; (3) new/renamed symbols force broad inclusion. A false negative here only *delays* feedback because the backstop re-checks at the gate.

#### 10.4.4 POST channel — full-suite stabilization backstop (the brake)

At the **final pre-accept state** of any candidate that the cheap-first cascade has otherwise approved, run the complete suite (subject to the safety mode below). This is the Google/Facebook stabilization pattern and is what keeps fast selection honest. Selection accelerates feedback; the backstop is the merge gate. A candidate cannot reach SOLVED on selection evidence alone.

### 10.5 Per-repo safety mode (the explicit safety knob)

A single config flag governs how aggressively selection is trusted. The orchestrator **never silently gambles** — the mode is recorded in the RunManifest and journaled.

| Mode | PRE (order/priors) | POST (dynamic prune) | Backstop | When |
|---|---|---|---|---|
| `advisory` | on | reported only; full suite always run | always | unknown/high-dynamism repos; first runs |
| `prune-with-backstop` **(default)** | on | drops unchanged-block tests *inside loop* | **mandatory at final pre-accept** | normal operation |
| `prune-hard` | on | drops at the gate too | none | opt-in only; fast tracer-trusted repos |

```text
ctdg:
  enabled: true
  safety_mode: prune-with-backstop      # advisory | prune-with-backstop | prune-hard
  static_builder: tree-sitter           # tree-sitter | lsp | regex-fallback
  dynamic_tracer: coverage.py           # coverage.py | ekstazi | build-dag | none
  block_checksums: true
  over_select_on_hierarchy_change: true
  plan_prior:
    enabled: true
    critic: generative                  # generative | scalar | off
    downgrade_only: true                # HARD-LOCKED true; not user-overridable to false
    wildcard_lane: true
  selection_key_inputs: [lockfile, python_version, env, seed, config, image_digest]
```

`prune-hard` requires explicit per-repo opt-in and an accepted, *measured* non-100% catch rate; it is the only mode that may drop a fault-revealing test, and it is off by default precisely because the adversarial verdict forbids treating selection as a sole gate.

### 10.6 Vendor-neutrality: the tracer and graph are plugs

Both layers sit behind narrow interfaces so the engine is language- and vendor-agnostic (Section 3). Workers (Codex / Claude Code / mixed) consume the *outputs* (ordered test list, affected subset, branch priors) via the Normalized Executor; they do not care how the graph was built.

```text
StaticGraphProvider:                 CoverageTracer:
  build(repo_snapshot) -> CtdgStaticIndex     collect(cmd) -> CoverageMap
  changed_symbols(diff) -> set[SymbolId]      select(changed_blocks) -> set[NodeId]
                                              collected_set() -> list[NodeId]  # via --collect-only

# Plug table (graceful degradation per ACP-style capability negotiation, Section 8):
#   python  : StaticGraphProvider=tree-sitter/LSP ; CoverageTracer=coverage.py contexts
#   JVM     : StaticGraphProvider=STARTS          ; CoverageTracer=Ekstazi (class-level)
#   Bazel   : StaticGraphProvider=build-DAG       ; CoverageTracer=build-DAG reverse closure
#   no tracer: CoverageTracer=none -> safety_mode auto-forced to `advisory` (full suite always)
```

If a backend cannot supply a tracer, the engine degrades — it does not crash — by forcing `advisory` mode. Non-Python languages get regex-only static nodes (no use-edges), so their static layer contributes *less* ordering signal but never less safety.

### 10.7 Integration with the rest of the engine

- **Section 9 (speculative tree search):** static priors and the plan score feed FrontierSearch's existing ranking/budget machinery (`max_depth`, `max_frontier_branching`, `min_branch_reward`, virtual loss). The dynamic affected-set is the cheap check that prunes *speculate()* branches — but only on executed-coverage evidence, reusing v1's cheap-first ladder so branching stays affordable.
- **Section 13 (verifier):** the CTDG never overrides the verifier. Execution evidence is authoritative; the affected-subset run is execution evidence; the backstop is the final execution gate. AST/semantic-equivalence checks are used *there* for patch overfitting/regression-equivalence, not for live test selection.
- **Section 14 (active controller):** safety mode, plan-prior weight, and the over-select threshold are controller knobs (bandit → GEPA → RL staging). The controller may *re-weight* priors but inherits the hard lock that no soft signal excludes a candidate before execution.
- **Section 15 (determinism):** `SelectionKey`, the ordered test list, the affected subset, and every plan score are journaled per `agent()` call. Replay reproduces *which tests ran and the verdict*, not token streams (bit-reproducible output replay is rejected).

### 10.8 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Static graph drops a dynamic edge (false negative) | Static layer only *orders*; never excludes. |
| Coverage map stale after dep/env/seed change | `SelectionKey` invalidation forces full re-collect + full run. |
| Tracer blind to time/random/network/C-ext deps | Over-select on hierarchy change; default backstop; force `advisory` if no tracer. |
| New/renamed symbol not in map | Broad sibling inclusion; empty selection ⇒ full suite. |
| Cheap plan score kills a correct unconventional branch | Downgrade-only (hard-locked) + wildcard lane + canary metric; no RL reward use. |
| `prune-hard` ships a regression | Off by default; opt-in with measured catch rate; the only mode permitted to drop tests. |
| Flaky tests mis-attributed (Google: 84% of P→F flaky) | Treat selection as feedback accelerator; verifier handles flakiness; backstop re-confirms. |

The throughline: **the CTDG is a prior, the tracer is an advisory accelerator, and execution at the backstop is the brake.** Best-of-N over the full suite remains the floor we can never do worse than.
