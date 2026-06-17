# APEX-Ω commit0 Experiments — Comprehensive Technical Report

**Audience:** technical stakeholder · **Scope:** runs 1–4 of the commit0 K-budget ladder · **Calibration note:** every cell is **n=1**, over **4 repos** per arm (B2 over 1). A single repo flip moves an arm's solve-rate by 25 points. Throughout, I separate *verified effects* (a mechanism confirmed in code + runs, or a lock-step multi-arm flip) from *variance* (any single-cell swing on a hard repo). All numbers below were re-verified against the run archives' `progress.jsonl`, `narration.jsonl`, per-eval `report.json`/`test_output.txt`, and `autogen_evidence.json`.

---

## 1. Executive summary

The four-run sequence produced **exactly one validated win and one validated regression**; everything else is noise at n=1.

- **The validated win (run-3):** A harness bug-fix (**P0.1 editable-shadow**) flipped jinja from `fail` to `SOLVE` for **both** orchestrated arms simultaneously. This moved orchestrated solve-rate **25% → 50%**. Crucially, this is a **measurement-correctness fix, not a new capability** — the orchestrators were already producing correct jinja code; the scorer was importing the base stub instead of the candidate's edits and reporting a **false zero**. (One caveat: not *every* run-1/2 jinja zero was a pure measurement artifact — at least one run-1 candidate had a genuine code error; see §5. The lock-step dual-arm run-3 flip remains the dominant, verified cause.)
- **The validated regression (run-4):** The "full design" layer (**repair lineages ON by default + autogen agent cap 8→16**) is a **net regression driven entirely by a time-budget blowout**. It produced **zero new solves**, converted the one verified autogen-jinja solve (run-3: 6 agents / 607s) into a **TIMEOUT** (16 agents / 4000s), and pushed three more cells into infrastructure **ERRORs**. Most strikingly, run-4 **BOTH orchestrated mimesis cells computed verified 6052/6052 full passes and the harness threw them away** when the outer wall-clock killed the cell before the winner was banked (autogen and template alike).
- **Honest scope of what the data can support:** Of 4 repos, **2 carry no discriminating signal** (voluptuous solved by everyone every run; pydantic solved by no one ever), **1 is a documented coin-flip** (mimesis — and several of its non-solves are clipped timeouts, not honest fails, so the coin has even fewer real observations than it appears), and **1 is the only real discriminator** (jinja — a single binary event per run). **No arm-vs-arm superiority claim is statistically supported.** In particular, **AUTOGEN_WON = 0**: there is no repo where the authored orchestrator solved something the fixed template missed. At every shared solve they tie, and at the one shared solve (voluptuous) autogen costs **4× the agents**.

**Immediate recommendation:** revert run-4's two knobs to the validated run-3 configuration — **autogen cap 16 → 8** and **repair OFF by default** (make repair opt-in). Both are one-line, provably-safe reverts that recover the only validated win-state at zero risk. Then, before any further capability work, land **acceptance-checkpointing** (bank a verified solve the instant it passes) and a **budget-aware eval timeout**, which together would have recovered the discarded mimesis 6052/6052 solves on **both** orchestrated arms.

---

## 2. What the current setup does (concise architecture)

A commit0 harness pits **five arms** against the same task: take a repo whose implementation is stripped to stubs and make its visible pytest suite pass. The unit of comparison is one **(arm, repo) cell**. **All arms reuse v1's identical repo-prep and the identical execution-authoritative scoring gate**, so the only thing that varies between arms is *how compute is spent*, not *what counts as solved*. The matrix is driven by `scripts/run_ladder.py` over 4 repos and is journaled/resume-safe via `apex_omega/eval/commit0_driver.py`.

**The five arms:**

| Ladder arm | Mode | What it does |
|---|---|---|
| `B0_codex_1shot` | A (v1 subprocess) | One codex rollout, single shot. Frontier counter-anchor. |
| `baseline_v1_k8` | A (v1 subprocess) | v1 best-of-8: 8 independent rollouts, v1 selector picks. |
| `omega_template_k8` | C (in-process) | Orchestrator path with **authoring OFF, scouting OFF** → runs the **fixed** `DEFAULT_ORCHESTRATION` template at an 8-agent budget. |
| `omega_autogen_k8` | C (in-process) | **Full pipeline:** scout fan-out (3 agents) → architect authors a tailored `orchestrate(ctx)` → freeze → sandboxed execute. Cap 16 in run-4. |
| `B2_v1_fullcap16` | A (v1 subprocess) | v1 with 16 forced rollouts. Cost-pathology witness; voluptuous-only sentinel. |

**Mode-C pipeline (template & autogen):** `prep` (one v1 `_prepare_repo`: clone → checkout base → history scrub → venv → editable install) → `scout` (parallel read-only fan-out returning `{difficulty, approach, key_files, risks}`, median-aggregated; difficulty maps to wave size via `easy:(1,8) / medium:(3,24) / hard:(8,64)`, with a **clamp** that prevents a scout from inflating a statically-easy repo) → `author` (LLM emits a tailored `orchestrate(ctx)`, linted, frozen; falls back to the template on failure) → **sandboxed orchestrate** (frozen script `exec`'d with only the curated `ctx` API — no imports/os/subprocess; a soundness/replay boundary, not a security jail) → **execution-authoritative select** (`ctx.select` returns the top *accepted* candidate or abstains).

**How "solved" is decided:** `score_fn` runs **real pytest in the candidate worktree** → v1's evaluation contract sets `accepted` (total>0, failed==0, errors==0, missing==0, pass_rate≥1.0) → `ctx.select` returns the top accepted candidate or abstains → the cell is solved iff `winner is not None and winner.accepted`. **The generated orchestrator cannot mark anything accepted itself** — the strategy decides where compute goes; the kernel decides what passes. There is **deliberately no fall-open-to-template rescue**: if the authored orchestration runs cleanly but abstains, that is autogen's real result, reported as failure.

**Anti-cheat / fairness (two layers):** (1) a *deterrent* anti-fetch prompt suffix (efficiency, not the boundary); (2) the *structural* boundary — an env sanitizer scrubs version literals/upstream URLs/CHANGELOG/dist-info/tags/reflog, and the candidate worktree **physically shadows** any site-packages install (cwd for flat-layout; `PYTHONPATH=<wt>/src` for src-layout). A fetched package cannot be imported over the candidate's edits, so a fetch-cheat can only *waste* an attempt — it can never produce a false-solve (0 observed).

**Run configs:** all runs C=6 concurrency except run-4 (C=4); cell-timeout 2400s in run-1/2, **3600s** in run-3/4; outer subprocess wall = `CELL_TIMEOUT + 600`; per-evaluation pytest cap fixed at **1800s** (`COMMIT0_OFFICIAL_EVALUATION_TIMEOUT_SECONDS`).

---

## 3. Results across run1–4

**Per-cell** (SOLVE / fail / **TMO**=cell-timeout / **ERR**=subprocess-timeout). Agent counts shown for orchestrated arms.

**TMO convention (applied consistently):** any cell whose `wall_s` reached the run's cell-timeout (2400s in run-1/2, 3600s in run-3/4) with a non-solve is labeled **TMO**, regardless of run. This relabels three run-1/2 cells that prior drafts called plain "fail" — they were clipped at exactly 2400s (corroborated by the code comment at `run_ladder.py:35` naming run-2 B0-mimesis and baseline-pydantic as 2400s clips). These cells are flagged with **‡**.

| Arm | Repo | run1 | run2 | run3 | run4 |
|---|---|---|---|---|---|
| B0_codex_1shot | voluptuous | SOLVE | SOLVE | SOLVE | SOLVE |
| B0_codex_1shot | jinja | fail | fail | fail | fail |
| B0_codex_1shot | mimesis | SOLVE | **TMO‡** | fail | SOLVE |
| B0_codex_1shot | pydantic | fail | fail | TMO | fail |
| baseline_v1_k8 | voluptuous | SOLVE | SOLVE | SOLVE | SOLVE |
| baseline_v1_k8 | jinja | SOLVE | SOLVE | SOLVE | SOLVE |
| baseline_v1_k8 | mimesis | **TMO‡** | SOLVE | fail | fail |
| baseline_v1_k8 | pydantic | fail | **TMO‡** | TMO | TMO\* |
| omega_template_k8 | voluptuous | SOLVE/1ag | SOLVE/1ag | SOLVE/1ag | SOLVE/3ag |
| omega_template_k8 | jinja | fail/8ag | fail/8ag | **SOLVE/8ag** | fail/8ag |
| omega_template_k8 | mimesis | fail/8ag | fail/8ag | fail/8ag | **ERR†** |
| omega_template_k8 | pydantic | fail/8ag | fail/8ag | fail/8ag | fail/8ag |
| omega_autogen_k8 | voluptuous | SOLVE/4ag | SOLVE/4ag | SOLVE/4ag | SOLVE/4ag |
| omega_autogen_k8 | jinja | fail/8ag | fail/8ag | **SOLVE/6ag** | **TMO/16ag (4000s)** |
| omega_autogen_k8 | mimesis | fail/8ag | fail/8ag | fail/8ag | **ERR†** |
| omega_autogen_k8 | pydantic | fail/8ag | fail/8ag | fail/8ag | **ERR** |
| B2_v1_fullcap16 | voluptuous | SOLVE | SOLVE | SOLVE | SOLVE |

\* baseline pydantic run-4 is a **cell-timeout (`wall_s:3600.3`, `dur_s:3.5`) with a failed-preflight result**, not data corruption: `baseline_result.json` is populated (`collected_test_count:0`, one failing-test entry, ImportError preflight) and is structurally identical to a normally-completed baseline cell that simply failed at import. It is an infra timeout and should be **excluded** from honest denominators, but the file itself is not "all-null/corrupted."
† **run-4 ERR cells with discarded verified solves:** *both* orchestrated mimesis cells computed full passes that were thrown away. `omega_autogen_k8`'s `autogen_mimesis_2/test_output.txt` = `6052 passed in 38.43s` (report.json `exitcode:0`, `summary.passed:6052/total:6052`); `omega_template_k8`'s `autogen_mimesis_3` (`6052 passed in 27.36s`) and `autogen_mimesis_7` (`6052 passed in 20.85s`) likewise both passed (`exitcode:0`). See §6.

**Status-field note (for anyone parsing `progress.jsonl`):** run-4 autogen-jinja is recorded `status:done` (it ran to wall 4000s and *abstained*), whereas the mimesis/pydantic ERR cells are genuine subprocess `TimeoutExpired`. "TIMEOUT" in the table above is reasonable shorthand for the jinja cell, but its underlying status differs from the true ERR cells — it produced a (zero) result rather than failing to return one.

**Per-arm solve-rate** (solved / cells-that-produced-an-honest-result; clipped-timeout cells now excluded per the ‡ relabeling):

| Arm | run1 | run2 | run3 | run4 |
|---|---|---|---|---|
| B0_codex_1shot | 2/4 (50%) | 1/3 (33%, +1 TMO) | 1/4 (25%) | 2/4 (50%) |
| baseline_v1_k8 | 2/3 (67%, +1 TMO) | 3/3 (100%, +1 TMO) | 2/4 (50%) | 2/4 (50%) |
| omega_template_k8 | 1/4 (25%) | 1/4 (25%) | **2/4 (50%)** | 1/3 (33%, +1 ERR) |
| omega_autogen_k8 | 1/4 (25%) | 1/4 (25%) | **2/4 (50%)** | 1/2 (+2 ERR) |
| B2_v1_fullcap16 | 1/1 | 1/1 | 1/1 | 1/1 |

**Note on run-4 autogen "1/2":** this matches the shipping tool — `ladder_report.json` reports `omega_autogen_k8` as `solve_rate 0.5, solved:1, total:2` (it drops the 2 ERR cells but keeps the jinja TMO as a counted zero-result). They are the same 1/2. By the report's own logic, however, the jinja TMO is itself an artifact of the same run-4 heaviness as the ERR cells; if it were excluded too, the only non-timeout autogen result in run-4 is voluptuous (1/1), making run-4 **even less comparable** than the headline 1/2 suggests. Either way, run-4's orchestrated denominators **cannot be used to rank arms.**

**Reading the trends honestly:** the orchestrated arms' **25% → 50%** jump in run-3 is the one real movement (jinja, both arms together). Every other per-arm wobble — B0's 50/33/25/50, baseline's run-2 spike — is the **mimesis coin** (partly on *clipped* data; see §5) plus the always-solved voluptuous. Run-4's orchestrated denominators (1/3, 1/2) are **not comparable** to prior runs: the ERR cells are missing data, not failures.

---

## 4. Conclusions about the CURRENT setup (what works, calibrated for tiny n)

**Per-arm:**
- **B0 (1-shot) and baseline_v1 are the efficiency frontier on this matrix.** baseline solves voluptuous + jinja reliably (it never hit the P0.1 bug — flat invocation, cwd-resolved), plus the mimesis coin; its true rate is ~50% on *honest* (non-clipped) cells. B0 solves voluptuous always + the mimesis coin. Neither carries scout/orchestration overhead.
- **The orchestrated arms (template, autogen) have not yet earned their cost.** Their entire defensible solve set across all runs is **voluptuous (always) + jinja (run-3 only)** — exactly the same set baseline already gets, and at the one shared solve (voluptuous) autogen pays **4 agents vs template's 1** for an identical diff. **AUTOGEN_WON = 0.**
- **B2cap16** is a 1-cell voluptuous sentinel — a "harness still works" canary with no discriminating signal.

**Per-repo:**
- **voluptuous** (149 tests, flat-layout): genuinely **easy**, 100% solve everywhere. The only finding is scout fan-out overhead (4× agents for zero benefit). It is a smoke test, not a benchmark item.
- **jinja** (851 tests, **src-layout**): **medium, and the run-1/2 failure was *predominantly* an artifact** (P0.1 false-zero), though at least one run-1 candidate had a genuine code error (§5). The only repo where arms actually diverge — and it's a single binary event per run.
- **mimesis** (in-workspace suite collects **6052 tests**; ~6159 is the upstream-PyPI count, not the measured suite): **medium-hard but in-workspace-solvable**, and a documented **coin-flip** — B0 solved it run-1+run-4, baseline solved it run-2, nobody run-3; template/autogen never *recorded* a solve (though run-4 *both* orchestrated arms computed one and lost it). Several mimesis non-solves (baseline run-1, B0 run-2) were **2400s timeout-clips, not honest in-workspace fails**, so the coin rests on even fewer real observations. n=1 cannot rank arms here.
- **pydantic** (~5091 tests / ~95–100 files): **hardest; genuinely unsolved by any arm in any run** (0/16). A from-scratch reimplementation out of the time/token budget. This 0/16 is a **real effect, not variance** — but it carries zero discriminating power and frequently hits the cell-timeout.

**Bottom line on the current setup:** the harness is *sound* (execution-authoritative, fetch-cheat-proof), and the bug-fixes made the instrument *honest*. But on this matrix the orchestration sophistication is currently **pure cost with zero solve-rate return** relative to plain best-of-N.

---

## 5. Attribution — what each layer changed (kept separate)

### Layer 1 — Bug-fixes (run1/2 → run3): one real, validated win; everything else noise.

**REAL (verified):** **jinja flipped `fail → SOLVE` in run-3 for BOTH orchestrated arms simultaneously** (template 8ag, autogen 6ag). A same-direction flip on the same repo across two independent arms at n=1 cannot be coincidence — it is a **shared cause**, verified in code as the **P0.1 editable-shadow fix**. The run-1/2 jinja "failures" were *predominantly* a **harness false-zero**: `score_fn` ran pytest in the candidate worktree but reused the BASE editable env, so jinja's src-layout imported the base stub instead of the candidate's edits (851 collection errors / 0 pass on *correct* code). Confirmed: run-1 both arms `solved:0`; run-3 autogen `solved:1/6ag/607s`, template `solved:1/8ag`. **Caveat:** not every run-1 jinja zero was a pure measurement artifact — `autogen_jinja_1/test_output.txt` shows a genuine `IndentationError` in the candidate's own `bccache.py` (broken candidate code, not the base-stub shadow). So P0.1 is the *dominant* verified cause (the lock-step dual-arm flip proves it), but at least one run-1 zero was a real code error. **This moved orchestrated arms 25% → 50% and is the headline result — but it is a measurement-correctness win, not a capability gain.**

**Correctness wins with zero rate effect (also real):**
- **P0.2 (memray):** the gate passed `--memray` but never loaded `-p pytest_memray` → rc=4 pre-collection false-zero. Fixed in run-3. Converted pydantic false-zeros into honest TMO/fail — **no solve-rate change** (pydantic is genuinely unsolvable in budget).
- **Timeout 2400→3600s:** run-1/2 had clipped 2400s artifacts (mimesis and pydantic cells reaching exactly 2400s with tiny `dur_s`); run-3 hard-repo cells ran to completion (mimesis 2974s genuine-fail, pydantic full 3600s). Improves trust in hard-repo numbers; no solve-rate effect. Note: this is why several run-1/2 mimesis/pydantic "fails" are now relabeled **TMO** in §3 — they were clipped, not honest fails, which further thins the real-n on the mimesis coin.

**NOISE inside Layer 1:** every mimesis flip (the coin, partly on clipped data); baseline's run-2 spike (= mimesis coin landing heads, on a run where the prior baseline-mimesis was itself a clip); B0's 50/33/25 wobble (= mimesis coin). Attribute all of it to variance.

### Layer 2 — New-capability (run3 → run4): net regression, fully attributed to a time-budget blowout.

**Confound to state upfront:** run-4 changed several things at once (C=6→4; repair-default + cap 8→16 + anti-fetch prompt + token ceiling + `--3way` + P0.4 + failing-test surfacing). The capability bundle is not isolated, **but the regression mechanism is unambiguous and verified.**

**REAL REGRESSION (verified in code + runs):**
- **autogen jinja: run-3 SOLVE (6ag/607s) → run-4 TIMEOUT (16ag/4000s).** Same repo, same arm, solving cleanly one run prior. `autogen_evidence.json` confirms run-4 jinja: 16 agents, 10 policy_violations + 10 infra_nonresults, class `strategy_sandbox_block`; narration seq 11 `agent budget: initial=8 soft_cap=16 (difficulty=hard, ceiling=1000)`, seq 28 a literal `score_fn evaluate_repo raised: TimeoutExpired` (the 1800s inner pytest cap), seq 29 `agent ceiling reached after wave`. The cell is recorded `status:done` having abstained at wall 4000s. **This is the single most load-bearing data point.**
- **autogen mimesis, autogen pydantic, template mimesis: ERR** (subprocess `TimeoutExpired`, verified in `progress.jsonl`). These are infra timeouts from the same heaviness — *not* strategy failures — which is why run-4's orchestrated denominators are uncomputable.
- **Repair lineages produced NO new solves in any cell in any run, and cost the verified orchestrated solves** (the run-3 autogen-jinja solve, and the discarded run-4 mimesis passes on both arms). Net value of the new-capability layer in this run: **negative.**

**NOISE inside Layer 2:** B0 25→50 (mimesis coin heads again, 1672s); baseline 50→50 (stable); template-voluptuous 1→3 agents (harness/scout-overhead artifact, still SOLVE); a separate mid-run-4 crash (`solve_and_repair(prompt=)` → TypeError) was caught and fixed *before* the final numbers. **Corroborating artifact:** the discarded `ladder_run4_aborted` archive shows autogen voluptuous and jinja both `solved:0` at `agents:3` (vs the healthy 4-agent voluptuous solve) — concrete evidence the crash truncated those cells and the archive was correctly discarded; it is **not** reflected in the final solve-rates.

**Net attribution:** the entire defensible signal across four runs is (1) **P0.1 is real and moved orchestrated arms 25%→50% on jinja** (a measurement bug, not new capability, with one run-1 candidate also genuinely broken), and (2) **the Layer-2 capability bundle is a net regression from a time-budget blowout** (zero new solves; the run-3 jinja solve lost; three cells errored; two verified mimesis passes discarded). Everything else involving mimesis or pydantic is variance at n=1.

---

## 6. What still fails and WHY

**jinja regression (run-4) — real, dual cause.** Two things happened, and they compound:
1. **Time-budget blowout (primary):** cap-16 forces a *second* 8-agent wave the doubling schedule never paid for in run-3; repair lineages add per-candidate work and more 1800s evals; and a *single* candidate's scoring tripped the flat **1800s** `evaluate_repo` cap (narration seq 28), consuming half the 3600s cell. The cell ran to wall 4000s and abstained.
2. **Genuine quality regression (secondary):** run-4 autogen-jinja's best candidates were weak — verified at **189/851** (`autogen_jinja_11`, exitcode 1) and **124/851** (`autogen_jinja_6`, exitcode 1) — nowhere near run-3's 851/851 — with 10/16 agents aborting as 0-token policy_violations. So even with infinite time, run-4 jinja was a weaker draw. **Both** the heaviness and the candidate quality regressed.

**mimesis variance — coin-flip, with a run-4 twist that overturns the "never solves" summary — on BOTH orchestrated arms.** Historically the failure mechanism is a **fetch-monoculture**: the architect bakes "restore official upstream mimesis" into the shared prefix of all attempts → every variant routes through the forbidden fetch, trips the workspace-jail, scores 0 tokens, never runs pytest. The env-sanitizer hid version/URLs but was **necessary-not-sufficient** — the model's prior ("mimesis is a real PyPI package") overrode the hidden version. **But run-4 is different and the ground truth is dispositive:** the repair/escalation phases fired (confirmed in narration), candidates produced real 120–150KB+ implementations, and **both orchestrated arms computed full in-workspace passes that the harness discarded:**
- `omega_autogen_k8`: `autogen_mimesis_2/test_output.txt` = `6052 passed in 38.43s`; its per-eval `report.json` is a **clean pass** (`exitcode:0`, `summary.passed:6052/total:6052`) — **not** all-null. What is missing is the *cell-level banked-winner record*: the **cell ERRORed (subprocess `TimeoutExpired`)** under the outer ~4200s wall before `ctx.select` banked a winner, so no finalized cell result was written. The per-eval report proves the solve was computed; the *bank* is what is absent.
- `omega_template_k8`: `autogen_mimesis_3` (`6052 passed in 27.36s`) and `autogen_mimesis_7` (`6052 passed in 20.85s`) — both per-eval `report.json` `exitcode:0`, 6052/6052. **Template-mimesis discarded TWO verified passes,** for the same cell-level reason.

(Note: I do not assert a specific completed-agent count for run-4 mimesis — only 5 eval directories exist per orchestrated arm (`autogen_mimesis_1..5` / `..7` for template), `autogen_evidence.json` mimesis fields are null because the cell errored, and the evals are named `_N`, not `repair0/repair1`. The narration confirms the repair and escalation-wave phases *fired*; it does not substantiate a precise agent tally.) **So in run-4 the binding constraint on mimesis was time, not strategy: the repair layer *worked* and the harness threw the solves away** — on both arms, which makes the acceptance-checkpointing fix more general than an autogen-only patch.

**pydantic genuine hardness — not an artifact.** 0/16 across all runs and arms. It is a near-complete library reimplementation (~95–100 files / ~5091 tests). B0/baseline hit the cell-timeout without finishing one solver pass; template ran the full 8-agent fan and genuinely failed ("no accepted candidate" through wave 3); autogen run-4 burned tens of millions of tokens across multiple waves with 400–553KB diffs and **no winning candidate hiding in the evals** (unlike mimesis, I checked — the near-misses don't exist for pydantic). This is the one repo where the constraint is **not** a harness artifact: it is a genuine from-scratch reimplementation that exceeds the per-cell time *and* token budget.

---

## 7. Fairness / measurement validity (real vs artifact)

**What is measured honestly (real, defensible):**
- **Acceptance is execution-grounded, not self-reported.** The generated orchestrator has no path to set `accepted`; the winner comes only from real pytest → v1's evaluation contract. Strategy decides where compute goes; the kernel decides what passes. This split is real in code.
- **A fetch-cheat cannot produce a false-solve — by construction, 0 observed.** The candidate worktree shadows site-packages (cwd / `PYTHONPATH=<wt>/src`); if the package resolves outside the worktree the result is `indeterminate` (excluded), never a false-accept and never a false-zero.
- **The env sanitizer removes the cheat *surface*** (versions/URLs/CHANGELOG/dist-info/tags/reflog) and is conservative (keeps `__version__` when a visible test pins it).
- **P0.1/P0.2 are legitimate instrument corrections, not score inflation.** The run-3 jinja flip is the single most trustworthy positive result in the dataset.
- **Cells are isolated and resume-safe;** ERR/TMO cells are correctly *not* counted as solves.

**What is an artifact (do not interpret as signal):**
- **Run-4 ERR cells are infrastructure timeouts, not capability results.** The "1/3" and "1/2" denominators are 3–4 *missing* cells; and both run-4 orchestrated-mimesis ERRs sit on top of verified, discarded solves.
- **The run-4 autogen-jinja TIMEOUT is caused by a fixed, un-scaled 1800s inner pytest cap** interacting with the 16-agent fan — a **config artifact, fully reversible**, not a capability regression in isolation. (Its status is `done`/abstain, distinct from the true subprocess-`TimeoutExpired` ERR cells, but it is just as much a budget artifact.)
- **The run-4 NET REGRESSION is a budget-allocation artifact.** It shows that under a fixed cell budget, doubling the cap + adding repair over-spends per cell; it is **not** evidence about the repair *mechanism's* merit (that question is confounded with the timeout).
- **`baseline_v1_k8 / pydantic` (run-4) is a cell-timeout with a failed-preflight result** (`wall_s:3600.3`, `dur_s:3.5`; populated `baseline_result.json` with `collected_test_count:0` + ImportError) — an infra non-result to be **excluded**, not "corruption" and not an honest fail.
- **The same timeout-clip pathology exists in run-1/2** (baseline-mimesis r1 @2400.2/8.4s, B0-mimesis r2 @2400.1/3.8s, baseline-pydantic r2 @2400.2/4.1s — code comment `run_ladder.py:35` names the latter two). These were prior-draft "fails" but are clipped non-results; §3 relabels them TMO. This means **no per-arm solve-rate in any run is currently clean** once timeout-clips are excluded — strengthening, not weakening, the case for the Tier-1.3 fix.
- **Scout difficulty labels are noisy across identical runs** (voluptuous rated easy/medium/hard in different runs) — another reason per-cell agent counts aren't yet comparable.

**Variance vs real (the honesty test):** of 4 repos, **2 carry no signal** (voluptuous always / pydantic never), **1 is a coin-flip** (mimesis — partly on clipped data), and **1 is the only discriminator** (jinja — one binary event/run). Every per-arm solve-rate is k/4 (k/2–k/3 after ERR + clip attrition); the gap between 1/4, 2/4, 3/4 is inside binomial noise at n=4. **The current matrix cannot support any claim of arm superiority.** The only statements the data supports: (a) the bug-fixes are real instrument corrections; (b) the run-4 budget regression is real and mechanism-attributable; (c) pydantic is uniformly unsolved.

---

## 8. Prioritized improvement roadmap

Effort in eng-days for one principal engineer. Value = how much it moves "can we publish an apples-to-apples arm comparison" or "does the orchestration earn its cost."

### Immediate recommendation (do first)

> **Revert run-4's two knobs to the validated run-3 config — repair OFF by default, autogen cap 16 → 8.** This is one line each, provably safe, and recovers the only validated win-state. Run-3 (cap-8, repair-off) is the only config that ever produced the orchestrated solve set (voluptuous + jinja), and repair has **never** caused a solve in any run while costing several verified solves. Do this *before* anything else so every subsequent change is A/B'd against a clean baseline, not the regression.

| # | Change | File:line | Edit | Value | Effort |
|---|---|---|---|---|---|
| **P0a** | autogen cap 16→8 | `scripts/run_ladder.py:94-95` | `--autogen-max-agents 8` | very high | trivial |
| **P0b** | repair off by default | `apex_omega/autogen/templates.py:33` | `make_repairing_attempt` → `make_attempt` (== `solve_attempt`, documented `context.py:251-252`) | very high | trivial |

These two alone eliminate the run-4 regression (jinja SOLVE→TIMEOUT and the three ERR cells). Keep the P0.1/P0.2/P0.3 fixes and the scout-difficulty clamp — they are the only things that ever added a solve.

### Tier 1 — Stop the instrument from discarding verified solves (high value / low effort)

| # | Change | Where | Value | Effort |
|---|---|---|---|---|
| **1.1** | **Acceptance-checkpointing.** Write a `winner.json`/`ACCEPTED.json` (candidate_id, content_sha, diff path, pass_rate) the *instant* `ctx.select` returns an accepted candidate; have `run_autogen_cell` and `run_ladder.py`'s outer-timeout `except` prefer that checkpoint over emitting ERR. Recovers the run-4 mimesis discards on **both** orchestrated arms. | `apex_omega/autogen/context.py:153-161,226-235,294-296`; `apex_omega/eval/commit0_autogen.py`; `scripts/run_ladder.py:169-174` | **very high** | ~1d |
| **1.2** | **Budget-aware per-eval timeout.** `evaluate_repo` already accepts `timeout_seconds` (never passed today). Pass `min(1800, remaining_cell_budget*0.4)`, threading a cell-start deadline. One candidate can no longer eat 1800s of a 3600s cell. A scoring timeout already maps to `indeterminate`, so capping is safe. | `apex_omega/eval/commit0_autogen.py:308-317` | **very high** | ~0.5d |
| **1.3** | **Never score a timeout/infra-kill as `solved:0`.** Any cell with `wall_s≥CELL_TIMEOUT` or a null/partial result JSON → `status:"timeout"`/infra_nonresult, **excluded from denominators**. Add a `nonresult_reason` field. This *retroactively* corrects not just run-4 baseline-pydantic but the run-1/2 timeout-clips (baseline-mimesis r1, B0-mimesis r2, baseline-pydantic r2) — confirming no per-arm rate is currently clean. | `apex_omega/eval/commit0_driver.py:214`; `scripts/run_ladder.py` parse_result | high | ~0.5d |
| **1.4** | **Per-repo eval-timeout override for hard repos.** Add `mimesis: {evaluation_timeout_seconds: 2700}` via the existing override mechanism (config-only, no code). | `configs/commit0_task_overrides.json` | high | trivial |

**Experiment for Tier 1:** re-run *both* run-4 orchestrated mimesis cells (autogen *and* template) and assert each 6052/6052 candidate is banked as `solved:1` despite the wall. This recovers two known-lost solves and proves the fix is arm-general.

### Tier 2 — Make the numbers statistically real (high value / medium effort)

| # | Change | Where | Value | Effort |
|---|---|---|---|---|
| **2.1** | **n≥3–5 seeds per (arm, repo) with pass@k + bootstrap CIs.** Greenfield (no seed plumbing today): add `--seeds N` to `run_ladder.py cells()`, expand cells to `..._s{seed}`, add `"seed"` to the journal key so each seed is independently resumable, offset the strategy/vendor cycle by seed. Add `scripts/aggregate_seeds.py` (Wilson CIs, McNemar across discriminating repos only). **Start with n=5 on jinja + mimesis; skip voluptuous (always) and pydantic (never).** | `scripts/run_ladder.py`; `commit0_driver.py:130` | **very high** | ~2-3d |
| **2.2** | **Equalize cross-arm budget OR report solve-rate-vs-budget curves.** Run-4 broke apples-to-apples three ways (C=6→4, cap asymmetric, repair autogen-only). Pin total budget (agents AND tokens AND wall) across compared arms, or emit efficiency-frontier curves. Never change concurrency between runs you compare. | `scripts/run_ladder.py`; `commit0_driver.py:109` | high | ~1-2d |

### Tier 3 — Make repair able to help without losing solves (medium-high value)

Only after Tiers 1–2 land. Repair must become **time-aware, difficulty/delta-gated, and cap-neutral**:

- **Time accountant (P1):** give `OrchestrationContext` a monotonic deadline from the already-plumbed `timeout_seconds`; add `time_left()` / `can_start_within(est)` / `p95_eval_cost()` (record eval elapsed at `context.py:153,226`). Gate the repair loop and the wave-escalation loop on the deadline, and expose the time API to authored orchestrators (`architect.py:43-75`). The doubling schedule (`plan_waves`) is agent-count-aware but **time-blind** — that blindness is the proximate cause of the TIMEOUT.
- **Difficulty + delta gate:** only repair when `difficulty in {easy, medium}` AND parent `pass_rate ∈ [0.3, 1.0)` AND parent diff is small (not a from-scratch rewrite). Kills repair on pydantic and heavy-reimpl jinja where it can never converge.
- **Cap-neutral:** let repair *replace* an attempt (reserve ≤2 of 8 slots), not double the cap. Re-enable behind `--autogen-repair-iters N` (default 0), validated with n≥5 on mimesis.

### Tier 4 — Efficiency & anti-cheat hardening (do opportunistically)

- **Gate scout fan-out on easy/clamped repos** (run template-shaped 1-agent best-of-N) — the 4× agent cost on voluptuous is the cleanest AUTOGEN_WON-blocker. Medium value, low effort.
- **Prompt-level fetch-strip + forced in-workspace attempt** for the monoculture (env scrub proven insufficient): guarantee ≥1 name-neutral "implement from stubs + tests" attempt per fan; convert a 0-token policy_violation into one hardened retry. Medium value, low-medium effort.
- **Flat-layout `find_spec`-origin negative control:** extend the src-layout import-origin assertion to flat-layout and log `accepted_solve_import_origin` on every accepted solve, so you can *state as a measured guarantee* "N/N accepted solves resolved inside the worktree." Highest value-per-effort in the anti-cheat space (~0.5d). **Defer** the heavy jail-detector/per-attempt-venv work — the worktree shadow already makes fetch-false-solves impossible, so it changes zero solve-rate numbers; build it only if a reviewer demands an active network-deny proof.
- **Quarantine pydantic** as `tier:"stretch"`, excluded from every per-arm denominator and reported separately as "0/N, out-of-budget, diagnostic." This removes ~25 points of denominator noise from every arm. Treat a from-scratch attempt (decomposition + far larger budget) as a separate research bet, not a tuning pass.

### Suggested sequence

1. **P0a/P0b** revert → reproduce run-3 baseline (clean control).
2. **Tier 1** (1.1–1.4) → re-run same cap-8/repair-off config; confirm no ERR cells and that the computed mimesis passes now bank on both orchestrated arms.
3. **Tier 2** (seeds + CIs, equal budget) → first publishable hard-repo solve-rates with intervals.
4. **Tier 3** gated repair behind a flag → test whether time-aware, delta-gated repair adds a solve (n≥5 on mimesis/jinja).

**The single highest-leverage next fix is acceptance-checkpointing + budget-aware eval timeout (Tier 1):** run-4 proved the harness can compute verified solves (mimesis 6052/6052 on *both* orchestrated arms) and throw them away on a wall-clock technicality. Recovering those discarded solves, then adding seeds for statistical power, is what converts this program from anecdote to result.

---

**Key verified artifacts:** `runs/archive/ladder_run4_fulldesign_20260616-034913/{progress.jsonl (autogen jinja status:done/4000.2s/16ag; baseline pydantic 3600.3s/3.5s), omega_autogen_k8__jinja/narration.jsonl (seq 11 budget initial=8 soft_cap=16; seq 28 evaluate_repo TimeoutExpired; seq 29 agent ceiling reached), omega_autogen_k8__jinja/.../evals/autogen_jinja_{6,11}/report.json (124/851, 189/851), omega_autogen_k8__mimesis/.../evals/autogen_mimesis_2/{test_output.txt 6052 passed, report.json exitcode:0 6052/6052}, omega_template_k8__mimesis/.../evals/autogen_mimesis_{3,7}/{test_output.txt, report.json exitcode:0 6052/6052}, baseline_v1_k8__pydantic/.../baseline_result.json (collected_test_count:0, populated), ladder_report.json (omega_autogen_k8 solve_rate 0.5 solved:1 total:2)}`; `runs/archive/ladder_run4_aborted_20260616-011122/progress.jsonl` (autogen voluptuous/jinja solved:0 @agents:3 — crash evidence); `runs/archive/ladder_run3_bugfixes_20260616-003704/{progress.jsonl (jinja autogen SOLVE/6ag/607s), ladder_report.json (omega_autogen_k8 0.5)}`; run-1/2 clip cells in `runs/archive/ladder_run2base_.../progress.jsonl` (baseline-mimesis 2400.2/8.4) and `ladder_run2_.../progress.jsonl` (B0-mimesis 2400.1/3.8, baseline-pydantic 2400.2/4.1). **Key code paths:** `apex_omega/eval/commit0_autogen.py` (score_fn / P0.1), `apex_omega/autogen/{context.py, templates.py, architect.py}` (orchestration + repair), `scripts/run_ladder.py:35` (CELL_TIMEOUT comment naming the 2400s clips), `scripts/run_ladder.py:36,94-95,171` (cell-timeout, cap-16, outer wall), `apex/evaluation/commit0_benchmark.py:181` (fixed 1800s eval cap).
