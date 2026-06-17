# APEX_NEXTGEN_PLAN.md — Validation Report

**Verdict: VALIDATED (fix-then-stands → fixed).** After adversarial validation and correction of all confirmed defects, the plan is internally consistent, free of detected hallucinations, and faithful to APEX v1's actual source code and the external literature it cites. I stand by it as a design document, subject to the residual caveats in §5.

Date: 2026-06-14. Validated artifact: `APEX_NEXTGEN_PLAN.md` (23 sections, ~82.7k words).

---

## 1. Methodology

A 50-agent validation workflow ran four phases:

1. **Extract** — 8 agents pulled **182 atomic, falsifiable factual claims** from the 23 sections, typed as `v1_fact` / `external_fact` / `redesign_fact` / `paradigm_fact` / `internal_ref`. Forward-looking design proposals were deliberately excluded (a proposal cannot be a hallucination).
2. **Audit** — 3 agents checked completeness, Fusion-Ledger consistency, and cross-reference/numeric self-consistency.
3. **Verify** — fan-out, routed by type and defaulting to skepticism: `v1_fact` claims checked against the **real v1 source** at `apex/apex/apex/` (ground truth, not the blueprint); `external_fact` claims checked against the **web** (primary sources; "unverified" unless confirmed); doc claims checked against `APEX_DESIGN.md` / the paradigm facts / the plan itself.
4. **Adjudicate** — verdicts aggregated into a severity-ranked defect list.

## 2. Results

| Metric | Value |
|---|---|
| Claims checked | 182 |
| Confirmed | 157 (86%) |
| Refuted | 5 |
| Misleading | 18 |
| Unverified | 2 |
| Fusion-Ledger consistency | **Clean** (no Rejected mechanism resurfaces as adopted; Cardinal Safety Contract upheld everywhere) |
| Completeness | No stubs/TODOs/truncation; all 23 sections coherent |

The hard-refute rate (~3%) is low. The Fusion Ledger — the plan's canonical in/out list — passed consistency cleanly. Most v1 behavioral constants (FrontierSearch params, `EpisodicMemoryBus`, escrow WAL, `FailureClass` taxonomy, deterministic ranking tuple, the 60/25/15 reuse split, the 31-entry ledger count) verified true against source.

## 3. Defects found and fixed (17 of 18; 1 false positive)

### Major (7)
1. **"16 is the default" cost-pathology framing (≈18 sites).** v1's default is `num_rollouts=5`; `max_rollouts=16` is the *cap*, reached only under adaptive allocation / escalation / portfolio floor. Verified directly: `_requested_rollout_budget` returns `num_rollouts` when adaptive is off (`core/config.py:536`, `planning/manager.py`). The §4.5 mechanistic claim (adaptive-OFF selects `max_rollouts`) was also wrong. **Fixed** globally to "non-adaptive fixed-K default (5), escalating to the 16 cap." The cost-pathology point is preserved (no down-scaling on easy tasks + escalation blow-up + caps off).
2. **Default backend misstated** as `claude-opus-4-8[1m]` @ `--effort max` for every stage. Actual default is `OPENAI_API`/`codex_cli:gpt-5.5` (first CLI preference; `claude_cli:opus` failover); `--effort max` is Claude-CLI-specific. **Fixed.**
3. **MAST (arXiv:2503.13657) misattribution** — the "0.370 mean / 17.2× error amplification / verifier as collective-delusion participant" figures are not in MAST (17.2× traces to a different DeepMind paper; the verifier line to a blog). **Fixed**: removed the figures, kept MAST for its actual 14-mode failure taxonomy, reframed "collective delusion" as a generic documented failure mode.
4. **HyperAgent "<13B models <5% on SWE-bench Verified"** — not in the paper and contradicted by it (its Llama-3-8B "Lite" variant scores ~16%). **Fixed** at all 5 sites; the (correct) Navigator/Editor ablation retained.
5. **Internal contradiction** — §9.10 `search.enabled=true` vs §21.3 "Default branching OFF." **Fixed**: reconciled to "engine on, branching gated by `activation_min_nodes=8`."
6. **R# namespace collision** — §18 uses R1–R7 for *rejected mechanisms*, §21 uses R1–R9 for *risks*. **Fixed**: disambiguation note added at §21.
7. **Overstated novelty premise** — "*Every* published learned orchestrator … one-hot identity" (AOrchestra is a counterexample). **Fixed**: hedged to "most (Puppeteer, AgentConductor) … none demonstrates open-pool cross-vendor generalization," preserving the genuine, narrower novelty gap.

### Minor (10, all fixed)
8. BAVT (2603.12634) does not test AB-MCTS-M → softened to "budget-agnostic schedule." 9. MEMOIR mis-cited as arXiv:2503.07826 (that's "Magnet") → corrected to 2605.17539 (×3). 10–11. Cross-ref homes for the localization-futility gate (→§16.6) and `pipeline()` (→§2/§16.3) corrected. 12. v1 parallelism "barrier waves only" → "continuous worker pool under wave-level planning." 13. Stale tier prices ($0.25/$15, "15–60×") → current ~5× lineup labeled. 15. TTFT "10–18% regression" softened to "variance." 16. `ReplayPlayer` is used by the `apex replay` CLI (only `ReplayRecorder` is unwired) and `resolve_available_llm_config` picks the *best-ranked* (not first) healthy candidate → both corrected.

### False positive (1)
14. The adjudicator flagged Snell "2–4×" and an "optimal-K<10 mis-sourced to Large Language Monkeys." Checked against the text: "Snell"/"2–4×" do not appear, and optimal-K<10 is already correctly sourced to arXiv:2411.17501 throughout. **No change** — the plan was already correct.

All edits were applied by exact-match replacement with per-edit match counts; post-edit residual greps for every bad phrase return clean. Document structure intact (1 real H1, 25 H2s = 23 sections + ToC + Exec Summary, 0 dead anchors).

## 4. Verdict

The corrected plan stands. Its thesis, architecture, Fusion Ledger, and execution-authoritative kernel were never in question; the defects were localized factual/citation errors, now corrected against ground truth. No section required regeneration.

## 5. Residual caveats (low severity — not blockers)

- **Citation traceability (D17):** the `-3.7pp` share-all figure is correctly cited to LTS (arXiv:2602.05965) where introduced (§6.1, §11), but a few summary-table references restate it without the inline citation. Cosmetic.
- **Unverified recent arXiv IDs (D18):** a handful of non-load-bearing 2026-dated IDs (e.g. ChromaFlow 2605.14102, InfoTree 2605.05262, Lemon 2605.14483, CODESKILL 2605.25430, SWE-RM 2512.21919) were not individually web-confirmed. All *load-bearing* recent citations (LTS, MEMOIR, EET, Don't-Break-the-Cache, AB-MCTS, etc.) were confirmed. Recommend a spot-audit of the remainder before the plan anchors a paper's related-work section.
- **General:** this is a *design synthesis*; its benchmark numbers and feasibility verdicts come from the cited sources and the research agents that read them. They are sound to the depth checked here but should still be independently reconfirmed before any number is published as a headline result.
