export const meta = {
  name: 'apex-comprehensive-report',
  description: 'Cross-run comprehensive analysis of the APEX-Omega commit0 experiments (run-1..run-4): conclusions about the current setup + what to improve + how',
  phases: [
    { title: 'Analyze' },
    { title: 'Improve' },
    { title: 'Synthesize' },
    { title: 'Verify' },
  ],
}

const ROOT = '/Users/sameertkhanna/Documents/agent_orch'

// Deterministically-extracted cross-run data (parsed inline from each run's progress.jsonl).
const DATA = [
  'CROSS-RUN RESULTS (commit0; K-budget orchestrated arms; execution-authoritative pytest gate).',
  'Runs (all C=6 except run-4 C=4; run-4 cell-timeout 3600s, autogen cap raised 8->16, repair lineages ON):',
  '  run1 = first clean run (original code, post strip-fix)   archive: ladder_run2base_20260615-221141',
  '  run2 = repeat of run1 (n=2 variance, same code)          archive: ladder_run2_prefix_20260615-230551',
  '  run3 = VERIFIED BUG-FIXES (P0.1 editable-shadow, P0.2 memray, P0.3 reflog-scrub, scout-difficulty clamp, 3600 timeout)  archive: ladder_run3_bugfixes_20260616-003704',
  '  run4 = FULL DESIGN (run3 + repair lineages default + autogen cap 16 + anti-fetch prompt + token ceiling + --3way + P0.4 + failing-test surfacing), C=4  archive: ladder_run4_fulldesign_20260616-034913 (also live runs/ladder)',
  '',
  'PER-CELL (SOLVE / fail / TMO=cell-timeout / ERR=subprocess-timeout):                run1   run2   run3   run4',
  '  B0_codex_1shot     voluptuous   SOLVE SOLVE SOLVE SOLVE',
  '  B0_codex_1shot     jinja        fail  fail  fail  fail',
  '  B0_codex_1shot     mimesis      SOLVE fail  fail  SOLVE',
  '  B0_codex_1shot     pydantic     fail  fail  TMO   fail',
  '  baseline_v1_k8     voluptuous   SOLVE SOLVE SOLVE SOLVE',
  '  baseline_v1_k8     jinja        SOLVE SOLVE SOLVE SOLVE',
  '  baseline_v1_k8     mimesis      fail  SOLVE fail  fail',
  '  baseline_v1_k8     pydantic     fail  fail  TMO   TMO',
  '  omega_template_k8  voluptuous   SOLVE/1ag SOLVE/1ag SOLVE/1ag SOLVE/3ag',
  '  omega_template_k8  jinja        fail/8ag  fail/8ag  SOLVE/8ag fail/8ag',
  '  omega_template_k8  mimesis      fail/8ag  fail/8ag  fail/8ag  ERR',
  '  omega_template_k8  pydantic     fail/8ag  fail/8ag  fail/8ag  fail/8ag',
  '  omega_autogen_k8   voluptuous   SOLVE/4ag SOLVE/4ag SOLVE/4ag SOLVE/4ag',
  '  omega_autogen_k8   jinja        fail/8ag  fail/8ag  SOLVE/6ag TMO/16ag(4000s)',
  '  omega_autogen_k8   mimesis      fail/8ag  fail/8ag  fail/8ag  ERR',
  '  omega_autogen_k8   pydantic     fail/8ag  fail/8ag  fail/8ag  ERR',
  '  B2_v1_fullcap16    voluptuous   SOLVE SOLVE SOLVE SOLVE',
  '',
  'PER-ARM SOLVE-RATE (solved / cells-that-produced-a-result):',
  '  B0_codex_1shot     2/4  1/4  1/4  2/4',
  '  baseline_v1_k8     2/4  3/4  2/4  2/4',
  '  omega_template_k8  1/4  1/4  2/4  1/3 (+1 ERR)',
  '  omega_autogen_k8   1/4  1/4  2/4  1/2 (+2 ERR)',
  '  B2_v1_fullcap16    1/1  1/1  1/1  1/1',
  '',
  'KEY VERIFIED FACTS (ground-truth in code + runs):',
  ' - jinja autogen/template "failures" in run1/2 were a HARNESS FALSE-ZERO (P0.1): score_fn ran pytest in the',
  '   candidate worktree but reused the BASE editable env, so src-layout (jinja2=src/jinja2) imported the base STUB.',
  '   Fixing P0.1 FLIPPED jinja to SOLVE for both arms in run3 (the validated win: orchestrated arms 25%->50%).',
  ' - pydantic P0.2: gate passed --memray but never -p pytest_memray -> rc=4 pre-collection false-zero (fixed run3).',
  ' - mimesis is in-workspace-solvable but a COIN-FLIP across runs (B0 solved run1+run4, baseline solved run2, none run3);',
  '   template/autogen NEVER solve it. n=1/cell cannot rank arms on hard repos.',
  ' - pydantic: NEVER solved by any arm; a from-scratch reimpl (~95-100 files / 5091 tests) within the time/token budget.',
  ' - RUN-4 NET REGRESSION: repair lineages (default ON) + autogen cap 8->16 made cells MUCH heavier -> they blew the',
  '   time budget. autogen jinja: run3 SOLVE(6ag/607s) -> run4 TIMEOUT(16ag/4000s). autogen mimesis+pydantic and',
  '   template mimesis ERRORED (subprocess timeout). Repair produced NO new solves within budget and LOST solves to timeout.',
  ' - A mid-run-4 regression was caught+fixed first: the authored orchestrator called solve_and_repair(prompt=) which the',
  '   new method rejected -> TypeError crashed every autogen cell (3 agents/abstain); fixed to accept prompt + tolerate kwargs.',
  ' - Fairness/anti-cheat: env sanitizer scrubs upstream version/URLs; the candidate worktree SHADOWS any site-packages',
  '   install (cwd flat / P0.1 PYTHONPATH src-layout) so a fetched package physically cannot be imported over the edits',
  '   -> fetch-cheat cannot produce a false-solve (0 observed). Acceptance is execution-authoritative (real pytest).',
  '',
  'KEY CODE/ARTIFACT PATHS to read for depth:',
  ' - design + verdict: APEX_AUTOGEN_NEXTGEN.md, APEX_AUTOGEN_NEXTGEN_IMPL.md',
  ' - signals: runs/ladder/SIGNALS_LEDGER.md, runs/ladder/autogen_evidence.json',
  ' - orchestration: apex_omega/autogen/{context.py (solve_attempt/repair_attempt/solve_and_repair), templates.py, architect.py}',
  ' - eval: apex_omega/eval/{commit0_autogen.py (score_fn P0.1), commit0_driver.py (token ceiling), scoring.py}',
  ' - runner: scripts/run_ladder.py (CELL_TIMEOUT, ARMS, concurrency)',
  ' - per-run data: runs/archive/ladder_*/progress.jsonl and .../omega_autogen_k8__*/narration.jsonl (run-4 has the timeouts)',
].join('\n')

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    supported_claims: { type: 'array', items: { type: 'string' }, description: 'report claims well-supported by the data' },
    unsupported_or_overstated: { type: 'array', items: { type: 'string' }, description: 'claims NOT supported / overstated / conflate variance with effect' },
    missing_points: { type: 'array', items: { type: 'string' }, description: 'important conclusions or caveats the report omits' },
    required_corrections: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string', enum: ['sound', 'minor_fixes', 'major_fixes'] },
  },
  required: ['supported_claims', 'unsupported_or_overstated', 'missing_points', 'required_corrections', 'verdict'],
}

// ===================== ANALYZE (parallel) =====================
phase('Analyze')
log('Comprehensive report: parallel cross-run analysis')

const analyses = [
  { key: 'attribution', prompt: 'Produce a CROSS-RUN ATTRIBUTION analysis: for each arm, the solve-rate trend run1->run2->run3->run4, and attribute every change to a specific cause (variance vs a code change). Separate the BUG-FIX layer effect (run1/2 -> run3) from the NEW-CAPABILITY layer effect (run3 -> run4). State precisely what improved, what regressed, and what is noise. Use the data table; read run dirs/narration if needed.' },
  { key: 'per-repo', prompt: 'Per-repo DEEP DIVE (voluptuous, jinja, mimesis, pydantic): for each, what is the true difficulty, which arms solve it and why, the variance across runs, and the binding constraint (false-zero? coinflip? time budget? genuine from-scratch reimpl?). Read the run-4 narration for the mimesis/pydantic timeouts. Conclude what each repo NEEDS to be solved reliably.' },
  { key: 'run4-regression', prompt: 'ROOT-CAUSE the run-4 regression. The repair lineages (default) + autogen cap 8->16 made autogen cells blow the time budget (jinja SOLVE 607s -> TIMEOUT 4000s; mimesis/pydantic subprocess-timeout ERRORs). Read apex_omega/autogen/{context.py solve_and_repair, templates.py} + scripts/run_ladder.py CELL_TIMEOUT. Was repair EVER going to help within budget, or is it net-negative as configured? Quantify the time/agent cost. Decide: should repair-as-default + cap-16 be reverted/gated, and what is the minimal time-budget-aware redesign that would let repair help without losing solves to timeout?' },
  { key: 'efficiency', prompt: 'EFFICIENCY analysis: agents/solve and (where available) tokens/solve per arm. Quantify the scout overhead (autogen voluptuous = 4 agents = 3 scouts + 1 solver vs template 1). Did the scout-difficulty clamp (run3+) help? Did cap-16/repair (run4) buy anything for its cost? Read runs/ladder + archives. Conclude where compute is wasted and the cheapest config that preserves the solves.' },
  { key: 'setup-architecture', prompt: 'Describe WHAT THE CURRENT SETUP ACTUALLY DOES, end-to-end, for a reader who has not seen it: the arms (B0/baseline_v1/omega_template/omega_autogen/B2), the scout->architect->author->sandboxed-orchestrate->execution-authoritative-select pipeline, the repair lineage capability, the worktree isolation, the env sanitizer, and how "solved" is decided. Read apex_omega/autogen/* + apex_omega/eval/* + the design docs. Be precise and concrete (files/functions).' },
  { key: 'fairness-anticheat', prompt: 'Assess FAIRNESS + ANTI-CHEAT + MEASUREMENT VALIDITY: the env sanitizer, the worktree-shadow that makes fetch-cheat unable to false-solve, execution-authoritative acceptance, and the timeout/ERR artifacts that corrupt run-4 hard-repo cells. What is measured honestly vs what is an artifact? What is the minimum needed (e.g. per-attempt venvs, scaled timeouts, n>=3 seeds) for a publishable, apples-to-apples result?' },
]
const analysisResults = (await parallel(analyses.map(a => () =>
  agent('You are a rigorous ML-systems analyst. ' + a.prompt + '\n\nGround every claim in the data; flag anything that is variance (n is tiny) vs a real effect. Repo root: ' + ROOT + '.\n\n=== DATA ===\n' + DATA,
    { label: 'analyze:' + a.key, phase: 'Analyze' }).then(r => ({ key: a.key, text: r }))
))).filter(Boolean)
const ANALYSIS = analysisResults.map(r => '## [' + r.key + ']\n' + r.text).join('\n\n')
log('Analyze complete: ' + analysisResults.length + ' analyses')

// ===================== IMPROVE (parallel) =====================
phase('Improve')
const improves = [
  { key: 'orchestration-time-budget', prompt: 'Propose concrete improvements to the ORCHESTRATION + TIME BUDGET so the repair capability helps instead of timing out: time-aware repair (stop before the cell deadline), per-attempt budget decoupled from the cell cap, scaled cell timeouts vs agent budget, and whether to keep repair opt-in/gated. Give exact knobs/files.' },
  { key: 'hard-repos', prompt: 'Propose how to actually solve the HARD repos: mimesis (in-workspace-solvable coin-flip -> how to make it reliable: variance reduction, n>=3, decompose, stronger localization) and pydantic (from-scratch reimpl -> is it feasible at all in-budget; if not, how to measure honestly). Ground in the per-repo evidence.' },
  { key: 'measurement-anticheat', prompt: 'Propose improvements to MEASUREMENT + ANTI-CHEAT: per-attempt venvs (so dist-info provenance + true fetch-deny become safe), n>=3 seeds with pass@k reporting, eliminating timeout/ERR artifacts, and whether the deferred jail-detector/provenance are worth building given the worktree-shadow already blocks fetch-false-solves. Prioritize by value/effort.' },
]
const improveResults = (await parallel(improves.map(a => () =>
  agent('You are a principal engineer proposing PRIORITIZED, concrete improvements (value/effort each). ' + a.prompt + '\n\nBuild on the analysis below; be specific (files/knobs/experiments). Repo root: ' + ROOT + '.\n\n=== DATA ===\n' + DATA + '\n\n=== ANALYSIS ===\n' + ANALYSIS,
    { label: 'improve:' + a.key, phase: 'Improve' }).then(r => ({ key: a.key, text: r }))
))).filter(Boolean)
const IMPROVE = improveResults.map(r => '## [' + r.key + ']\n' + r.text).join('\n\n')
log('Improve complete: ' + improveResults.length + ' proposals')

// ===================== SYNTHESIZE =====================
phase('Synthesize')
const draft = await agent(
  'You are the lead author. Write a COMPREHENSIVE, calibrated report on the APEX-Omega commit0 experiments for a technical stakeholder. Sections: (1) Executive summary; (2) What the current setup does (concise architecture); (3) Results across run1-4 with the per-cell table and per-arm solve-rates; (4) Conclusions about the CURRENT SETUP — what works, per-arm and per-repo, honestly calibrated for tiny n; (5) Attribution — what each layer changed (bug-fixes = the validated win; new-capability = net regression via timeouts), kept separate; (6) What still fails and WHY (jinja regression cause, mimesis variance, pydantic genuine hardness); (7) Fairness/measurement validity (what is real vs artifact); (8) PRIORITIZED improvement roadmap (what to improve + how, value/effort, concrete files/knobs/experiments) including the immediate recommendation on repair-as-default + cap-16. Be honest, do not overclaim, distinguish variance from effect. Output GitHub-flavored markdown.\n\n=== DATA ===\n' + DATA + '\n\n=== ANALYSIS ===\n' + ANALYSIS + '\n\n=== IMPROVEMENT PROPOSALS ===\n' + IMPROVE,
  { label: 'synthesize-report', phase: 'Synthesize' }
)

// ===================== VERIFY (adversarial) + finalize =====================
phase('Verify')
const verdicts = (await parallel([1, 2].map(i => () =>
  agent('Adversarially fact-check this draft report against the DATA. Lens ' + i + ' of 2: ' +
    (i === 1 ? 'numeric accuracy + variance-vs-effect (does it overclaim any solve-rate change that is actually n=1 noise? are the per-cell outcomes and attributions correct?).'
             : 'completeness + honesty (does it omit a key caveat, the timeout/ERR artifacts, or overstate the repair capability? is the roadmap grounded?).') +
    '\n\n=== DATA ===\n' + DATA + '\n\n=== DRAFT REPORT ===\n' + draft,
    { label: 'verify:' + i, phase: 'Verify', schema: VERIFY_SCHEMA }).then(v => Object.assign({ lens: i }, v))
))).filter(Boolean)
const VERIFY = verdicts.map(v => '[lens ' + v.lens + '] verdict=' + v.verdict + '\n  unsupported/overstated: ' + JSON.stringify(v.unsupported_or_overstated) + '\n  missing: ' + JSON.stringify(v.missing_points) + '\n  corrections: ' + JSON.stringify(v.required_corrections)).join('\n')

const finalReport = await agent(
  'Produce the FINAL report by applying the adversarial fact-check corrections to the draft. Fix every overstatement, add every missing caveat, keep it calibrated and honest (tiny n; variance vs effect explicit). Keep all 8 sections. Output ONLY the final GitHub-flavored markdown report (no preamble).\n\n=== DRAFT ===\n' + draft + '\n\n=== FACT-CHECK FINDINGS ===\n' + VERIFY,
  { label: 'finalize-report', phase: 'Verify' }
)

return { report: finalReport, verify: VERIFY }
