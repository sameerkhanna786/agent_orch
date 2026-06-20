# Merge/Reduce Overhaul — research, diagnosis, fix

**Motivation (live evidence):** on the tightly-coupled repo BABEL the `converge` orchestration
reached only **925/5663** gold tests while the `ralph` baseline (one sequential single-worktree
lineage, no decomposition, no merge) reached **4458/5663**. The decompose → fan-out → REDUCE/MERGE
pipeline was *shedding* most of its parallel work. (Corroborated by `converge__minitorch` finishing
at only 189/230.)

## Research (web-grounded, workflow `wf_ccaa4cb5-681`, 10 agents)
- **Anthropic "Building Effective Agents":** there are two distinct patterns. PARALLELIZATION
  (sectioning/voting) is "aggregated programmatically" and is valid *only* for PRE-DEFINED INDEPENDENT
  subtasks. ORCHESTRATOR-WORKERS (which Anthropic names CODING as the canonical case) has a central
  LLM that *synthesizes* worker results. **Our `reduce_residuals` is a deterministic git-apply combine
  — i.e. we built the PARALLELIZATION aggregator for what is actually the ORCHESTRATOR-WORKERS case.**
  On a coupled repo the "independent sections" precondition fails, so the programmatic combine is the
  wrong tool and sheds work on every shared-file collision.
- **Field practice (MS "Swarm Diaries" etc.):** try the cheap deterministic merge FIRST; reserve the
  expensive integrator strictly for the hunks that ACTUALLY conflict. Synthesize from a WINNER and
  graft the rest onto it. And — the load-bearing warning — **LLM integrators silently destroy work
  while reporting success**, so the authority must be the deterministic re-score, never the merger's
  self-report (validates our Cardinal Contract).
- **Semantic merge (Spork/jdime/mergiraf, `git merge-file` histogram):** structured/AST or per-file
  3-way merge beats `git apply`'s all-or-nothing context match — but conflict markers in `.py` cause
  collection errors, and CRDT/OT "convergence ≠ correctness" must be avoided under the Cardinal
  Contract.

## Diagnosis (code-grounded) — why our reduce sheds work
`reduce_residuals` (apex_omega/autogen/context.py) was: acquire one merge worktree, `apply_diff`
(strict→3way) the carry then each module diff IN GIVEN ORDER; a diff that fails to apply is a CONFLICT
→ the **whole module diff is dropped** (re-queued) and a carry conflict falls back to **bare base**.
Two CRITICAL weaknesses produce the babel gap:
1. **All-or-nothing per module** — one conflicting hunk drops the module's entire 50–200-gold-id
   contribution (on a coupled repo most modules collide on shared files → most work shed).
2. **No regression floor** — a dropped foundational module or a bad merge can make the merged tree
   score BELOW the best single sub-result (even collection-break it), so converge can fall *below*
   ralph's coherent single lineage.

## Implemented (the adversarially-endorsed primary pair; deterministic, zero-LLM, replay-safe)
- **#1 Hunk-level partial apply** (`apply_diff_partial` in isolation/worktree.py; used by
  `reduce_residuals`, gated `APEX_OMEGA_MERGE_PARTIAL`, default ON): keep the cheap strict→3way clean
  path; only on failure use `git apply --reject` so a colliding module lands its non-conflicting
  hunks (the ~80–90% all-or-nothing shed) and only the rejected residue re-queues. `*.rej` files are
  deleted immediately so they never enter the scored artifact.
- **#2 No-silent-loss regression floor** (the true safety primary): after the merge is full-suite
  scored, if a VALID merge regresses below the prior frontier (`_best_gold_passed`), carry the BEST
  banked COHERENT candidate forward instead of the regression (`_best_coherent_candidate`; every
  banked candidate was full-suite scored, so it is a real tree, never a lone-module artifact). The
  merge stays banked for telemetry/select; we only refuse to make it the CARRY. This makes the merge
  MONOTONE — converge can never carry a worse tree than its strongest sub-result, killing the
  catastrophic-regression tail that let converge fall below ralph.

Both preserve the Cardinal Contract (only `ctx.select` on the full gold suite ACCEPTS; the merge can
only lower a score, never fake a pass) and are journal-replay-safe (the reduce stays zero-LLM). Tests:
`test_apply_diff_partial_lands_clean_hunks_drops_conflicting`,
`test_reduce_floor_reverts_regressing_merge_to_best_coherent` + existing converge/carry suite.

## Deferred (layer in after measuring #1+#2 against ralph)
- **#3 Per-file `git merge-file --diff-algorithm=histogram` 3-way** for files touched by >1 module
  (order-independent; recovers overlaps `--reject` can't). Risk: conflict markers → collection errors,
  so gate behind the #2 floor.
- **#4 LLM seam-reconciler** folded into the EXISTING journaled `repair_residual` (NOT into the
  zero-LLM reduce), driven by failing gold ids + diff3 context — last resort over the residue #1/#3
  can't reconcile. (Models resolve <60% of conflicts, so never the primary path.)
- **#5 Coupling-aware decomposition** — enforce file-ownership disjointness post-decompose so fewer
  conflicts are generated at all (attacks the root, multiplies #1–#4).

## Do-not-do (from the adversarial review)
No merge step may set `accepted`/self-certify; no LLM inside `reduce_residuals` (only in the journaled
`repair_residual`); no `git apply --union`/CRDT/OT auto-resolve (manufactures compiles-but-fails trees
that fake progress); never trust an LLM merger's "resolved successfully"; keep the cheap clean-apply
fast path.
