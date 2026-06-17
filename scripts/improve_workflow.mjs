export const meta = {
  name: 'apex-gold-test-guided-design',
  description: 'Design + validated plan to improve BOTH the template and autogen arms via gold-test-suite-guided design (derive API/enum/parametrization contract from the visible tests; no answer-leak)',
  phases: [
    { title: 'Map' },
    { title: 'Ground' },
    { title: 'Research' },
    { title: 'Design' },
    { title: 'Validate' },
    { title: 'Plan' },
  ],
}

const ROOT = '/Users/sameertkhanna/Documents/agent_orch'

const CONTEXT = [
  'PROJECT: APEX-Omega, an agent-orchestration system on the commit0 benchmark (implement a Python package whose',
  'source is stripped to stubs so its VISIBLE pytest suite passes; the agent IS GIVEN the visible test files as the',
  'spec; scoring is execution-authoritative with EXACT expected-test-ID matching — accepted iff total>0, failed==0,',
  'errors==0, missing==0, pass_rate>=1.0). Repo root: ' + ROOT + '.',
  '',
  'TWO ARMS TO IMPROVE:',
  ' - omega_template: a FIXED lean DEFAULT_ORCHESTRATION best-of-N (authoring/scouting OFF).',
  ' - omega_autogen: scout fan-out -> architect authors a tailored orchestrate(ctx) -> freeze -> sandboxed execute.',
  ' Both run coding agents (Codex CLI / Claude Code; vendor-agnostic) in isolated git worktrees, scored by real pytest.',
  '',
  'WHY THIS WORK (verified results across 4 runs):',
  ' - run-3 bug-fixes flipped jinja fail->SOLVE for BOTH arms (25%->50%): P0.1 editable-shadow was a SCORING bug; the',
  '   agents ALREADY wrote correct jinja code. Takeaway: the orchestrators can produce correct code; the wins came from',
  '   making the instrument honest, not from new capability.',
  ' - run-4 "full design" (repair lineages default + autogen cap 8->16) was a NET REGRESSION via TIME-BUDGET BLOWOUT:',
  '   heavier cells blew the cell wall-clock (jinja SOLVE 607s -> TIMEOUT 4000s; mimesis/pydantic ERR). LESSON: any',
  '   added work (repair/escalation/extra waves) MUST be time-budget-aware. Repair is now opt-in (default OFF).',
  ' - AUTOGEN_WON=0: autogen never beat the lean template; at the one shared solve (voluptuous) it costs 4x agents',
  '   (3 scouts + 1 solver). So improvements must EARN their cost and not regress the cheap wins.',
  '',
  'THE CENTRAL FINDING (mimesis ID-fidelity root cause, investigated):',
  ' - mimesis expected_test_ids=6159; orchestrated candidates collect 6052 and PASS all of them, but only ~2306 match',
  '   the expected set -> missing=3853, extra=3746 -> contract correctly REJECTS (missing!=0). It is NOT a capability',
  '   gap and NOT a scoring bug: the candidate covers the same locales/providers but is not ID-FAITHFUL.',
  ' - 55.7% of the gap is PURELY the Locale enum string-repr: candidate built Locale(Enum) so pytest ids render',
  '   "Locale.DE_AT"; the reference is class Locale(str, Enum) value "de-at" so ids render "de-at". Remaining ~44% is',
  '   finer id-rendering (pytest en0/en1 duplicate-param indexing from the reference locale list/order; some files',
  '   render [Locale.RU] vs others [cs] depending on whether the test stringifies the enum).',
  ' - B0/baseline solved mimesis (missing=0) by DRAWING a faithful Locale impl -> mimesis solve is a FIDELITY GAMBLE',
  '   (= the documented coin-flip). The goal: make that fidelity a DETERMINISTIC TARGET via gold-test guidance.',
  '',
  'THE GOAL (design for BOTH arms): GOLD-TEST-SUITE-GUIDED DESIGN. A mechanism that derives a DESIGN CONTRACT from the',
  'visible tests + the gold expected-ID inventory — required modules, class/function signatures, enum members + their',
  'required STRING semantics (e.g. str(Locale.DE_AT)=="de-at"), parametrization DOMAINS (locale codes, provider names),',
  'and required per-file/per-test parametrization counts/ids — and feeds it into the SCOUT/ARCHITECT plans (autogen)',
  'and the SOLVER prompts (template), so the agent targets the exact required surface + parametrization by DESIGN.',
  '',
  'HARD FAIRNESS BOUNDARY (must be a first-class design constraint + a validation lens):',
  ' - LEGITIMATE: deriving STRUCTURE from the visible tests + the expected-ID inventory (the agent already has the test',
  '   files; commit0 = "implement so the visible tests pass"). Parametrization keys (locale codes), enum members +',
  '   required str semantics, API signatures, required test counts are all spec-level structure.',
  ' - ILLEGITIMATE (answer-leak): some parametrized test IDs EMBED THE EXPECTED OUTPUT, e.g.',
  '   test_romanize_cyrillic_string[Locale.RU-<cyrillic input>-<romanized expected output>]. Handing the model the raw',
  '   value-bearing IDs leaks the answer. The design MUST extract parametrization STRUCTURE/keys while STRIPPING value',
  '   payloads, and must never hand the solver the expected outputs. Keep STRICT exact-ID acceptance unchanged.',
  '',
  'KEY FILES:',
  ' - prompts/issue: apex_omega/eval/commit0_autogen.py (prompt_builder; how expected ids/count reach the solver),',
  '   apex/evaluation/commit0_benchmark.py::build_issue_description (what the solver is currently told about the tests),',
  '   _load_expected_test_ids (the gold inventory loader).',
  ' - orchestration: apex_omega/autogen/{architect.py (scout/agent_scout/build_repo_map/author), context.py',
  '   (solve_attempt/repair_attempt/solve_and_repair/prompt usage), templates.py (DEFAULT_ORCHESTRATION), sandbox.py}.',
  ' - scoring/contract: apex_omega/eval/scoring.py, apex_omega/kernel/{verify.py, select.py}; v1 contracts (exact-id).',
  ' - budget/runner: apex_omega/engine/budget.py, scripts/run_ladder.py.',
  ' - results/evidence: APEX_COMMIT0_REPORT.md, runs/validation_checkpoint/* (the mimesis 6052-vs-6159 evidence),',
  '   runs/archive/ladder_run3_bugfixes_* (the jinja flip), runs/archive/ladder_run4_fulldesign_* (the timeout regression).',
  '',
  'OUTPUT: a unified, validated DESIGN + FILE-LEVEL IMPLEMENTATION PLAN for both arms, calibrated and honest, that keeps',
  'strict exact-ID acceptance, honors the fairness boundary, is time-budget-aware (run-4 lesson), vendor-agnostic, and',
  'does not regress the cheap wins (voluptuous 1-agent, jinja).',
].join('\n')

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    scores: {
      type: 'object', additionalProperties: false,
      properties: {
        fixes_id_fidelity_gap: { type: 'integer', description: '1-10: would actually drive mimesis missing->0 (and similar fidelity repos)' },
        fair_no_answer_leak: { type: 'integer', description: '1-10: derives structure without leaking value-bearing IDs/outputs' },
        improves_both_arms: { type: 'integer', description: '1-10: genuinely helps template AND autogen' },
        time_budget_safe: { type: 'integer', description: '1-10: respects the run-4 timeout lesson' },
        no_regression_cheap_wins: { type: 'integer', description: '1-10: keeps voluptuous-1-agent / jinja' },
        implementable_vendor_agnostic: { type: 'integer', description: '1-10: maps cleanly to the code; Codex+Claude' },
      },
      required: ['fixes_id_fidelity_gap', 'fair_no_answer_leak', 'improves_both_arms', 'time_budget_safe', 'no_regression_cheap_wins', 'implementable_vendor_agnostic'],
    },
    total: { type: 'number' },
    strengths: { type: 'array', items: { type: 'string' } },
    weaknesses: { type: 'array', items: { type: 'string' } },
    best_ideas_to_graft: { type: 'array', items: { type: 'string' } },
  },
  required: ['scores', 'total', 'strengths', 'weaknesses', 'best_ideas_to_graft'],
}

const VALIDATION_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    concern: { type: 'string' },
    confidence_pct: { type: 'integer', description: '0-100 calibrated' },
    verdict: { type: 'string', enum: ['sound', 'likely', 'uncertain', 'unsound'] },
    gaps: { type: 'array', items: { type: 'string' } },
    required_changes: { type: 'array', items: { type: 'string' } },
    reasoning: { type: 'string' },
  },
  required: ['concern', 'confidence_pct', 'verdict', 'gaps', 'reasoning'],
}

const FINAL_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    executive_summary: { type: 'string' },
    design_doc: { type: 'string', description: 'FULL markdown design: the gold-test-guided-design mechanism for both arms, every component, how it fixes the fidelity gap, the fairness boundary, time-budget-awareness, what to keep/cut' },
    implementation_plan: { type: 'string', description: 'FILE-LEVEL markdown plan: exact files/functions to change/add, drafted code for the critical pieces (test-contract extractor; value-stripped coverage oracle; prompt/scout/architect wiring), test plan, and the empirical validation-run command' },
    fairness_analysis: { type: 'string', description: 'explicit analysis of the answer-leak boundary: what is derived/surfaced vs what is withheld, and why it is legitimate under commit0' },
    per_concern_confidence: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { concern: { type: 'string' }, confidence_pct: { type: 'integer' }, note: { type: 'string' } },
        required: ['concern', 'confidence_pct', 'note'],
      },
    },
    overall_confidence_pct: { type: 'integer' },
    overall_assessment: { type: 'string', description: 'honest/calibrated; state what empirical validation is still required' },
    residual_risks: { type: 'array', items: { type: 'string' } },
    validation_plan: { type: 'string', description: 'how to empirically test the design (which cells/repos, what to measure, how to confirm no leak)' },
  },
  required: ['executive_summary', 'design_doc', 'implementation_plan', 'fairness_analysis', 'per_concern_confidence', 'overall_confidence_pct', 'overall_assessment', 'residual_risks', 'validation_plan'],
}

const RESEARCH_GUIDE = [
  'Use external web search to find CURRENT (2022-2025) SOTA and CITE sources (title + url). Load the approved search',
  'tool via ToolSearch: query "select:mcp__plugin_meta_mux__three_pai_external_web_search" then call it; you may also',
  'load WebFetch for specific pages. If web tools are unavailable, use training knowledge (cutoff Jan 2026) and mark',
  'such claims [UNVERIFIED]. For each technique: the mechanism in 2-4 sentences, evidence if known, and SPECIFICALLY',
  'how it maps onto the APEX-Omega arms (which files/primitives) to make agents test-suite-guided WITHOUT leaking answers.',
].join('\n')

// ===================== STAGE A: Map + Ground + Research (one fan-out) =====================
phase('Map')
log('Improvement design: mapping current mechanisms + grounding the finding + SOTA research, in parallel')

const mapTasks = [
  { key: 'what-solver-sees', prompt: 'Determine EXACTLY what the solver agent currently receives about the tests. Read apex_omega/eval/commit0_autogen.py prompt_builder and apex/evaluation/commit0_benchmark.py::build_issue_description (and _load_expected_test_ids). Does the solver prompt already include the expected test IDs, just the count, or neither? What about the visible test FILES (are they on disk in the worktree for the agent to read)? Quote the exact strings/fields. This decides whether the fix is SURFACING gold info vs USING it better.' },
  { key: 'scout-architect', prompt: 'Map the autogen scout->architect pipeline: apex_omega/autogen/architect.py (build_repo_map, build_scout_prompt, agent_scout, build_author_prompt, author_orchestration, API_REFERENCE/INVARIANTS). What do the scout and architect currently analyze (difficulty proxy, approach) and what do they NOT analyze (the test suite structure / expected parametrization)? Where would a test-contract analysis hook in?' },
  { key: 'template-prompt-path', prompt: 'Map the template arm + the solver prompt path: apex_omega/autogen/templates.py (DEFAULT_ORCHESTRATION) and apex_omega/autogen/context.py (solve_attempt/repair_attempt: how prompt_builder is called, where ANTI_FETCH_POLICY is appended). Where exactly would a test-derived DESIGN CONTRACT be injected into the solver prompt for the template arm (and the repair prompt)?' },
  { key: 'expected-ids-and-contract', prompt: 'Map how the gold expected IDs + the acceptance contract work: _load_expected_test_ids, apex_omega/eval/scoring.py (verification_from_commit0_evaluation, missing_expected), the exact-id matching. What structure is available in the gold ID inventory (parametrization keys, value payloads)? Identify which parts of an expected ID are SAFE structure (e.g. locale code) vs value-bearing answer (e.g. romanize output). Propose a value-stripping rule.' },
  { key: 'repair-budget', prompt: 'Map the repair + escalation + budget mechanisms after the run-4 fix: context.py (solve_and_repair, repair_iters ceiling, make_repairing_attempt), the budget-aware eval timeout (commit0_autogen.py eval_cap), Budget (engine/budget.py), plan_waves. How would a gold-test COVERAGE-FEEDBACK repair loop (diff collected-vs-expected parametrization keys -> targeted fix) be made time-budget-aware here?' },
  { key: 'v1-baseline-why-wins', prompt: 'Read how B0/baseline (v1 subprocess path) get the task + tests (apex/evaluation/commit0_benchmark.py issue/prompt construction for the solve agent). They solved mimesis (missing=0) sometimes. What do they tell their solver that the orchestrated arms do not, that makes a faithful Locale impl more likely? Is there a prompt/spec difference to learn from?' },
]

const groundTasks = [
  { key: 'mimesis-fidelity-fix', prompt: 'Ground the fix target precisely: using runs/validation_checkpoint/* evidence (the 6052-vs-6159, Locale.DE_AT vs de-at, the en0/en1 indexing, the [Locale.RU] vs [cs] split, the romanize value-bearing IDs), specify EXACTLY what a design contract would need to convey to make a mimesis candidate ID-faithful (missing->0): the Locale(str,Enum) requirement + required str semantics, the full locale-code domain, provider list, and the parametrization-id rendering rules — WITHOUT leaking romanize/value outputs. Be concrete about the safe-structure-vs-answer line on real examples.' },
  { key: 'jinja-voluptuous-keep', prompt: 'Ground what must NOT regress: jinja (run-3 SOLVE, agents already write correct code) and voluptuous (1-agent template solve). Read runs/archive/ladder_run3_bugfixes_*/ for jinja and the voluptuous cells. What about these makes them easy/solved, and how could a heavier test-contract step accidentally hurt them (latency, over-specification, the scout overhead)? Conclude the guardrails (e.g. gate the contract step by difficulty/size).' },
  { key: 'run4-timeout-constraint', prompt: 'Ground the time-budget constraint from run-4: read runs/archive/ladder_run4_fulldesign_*/ (autogen jinja TIMEOUT 4000s; mimesis/pydantic ERR) + scripts/run_ladder.py CELL_TIMEOUT + the 1800s inner eval cap. Quantify the budget any new gold-test-guided step (analysis + coverage-feedback repair) may consume, and the rules that keep it from blowing the cell wall-clock.' },
]

const researchTasks = [
  { key: 'test-guided-synthesis', prompt: 'Research TEST-DRIVEN / SPEC-GUIDED code synthesis: deriving implementations from a test suite, test-as-spec, using failing-test signals to steer generation, oracle/contract extraction from tests. What works for repo-scale library reimplementation? ' + RESEARCH_GUIDE },
  { key: 'api-contract-extraction', prompt: 'Research deriving an API/DESIGN CONTRACT from tests statically: AST analysis of test files to extract required imports, class/function signatures, attributes, enum members, and parametrization domains; pytest --collect-only to get the expected node-id inventory. Tools/techniques and reliability. ' + RESEARCH_GUIDE },
  { key: 'swe-agent-test-usage', prompt: 'Research how SOTA SWE/code agents USE the provided tests to guide design (Agentless, SWE-agent, AutoCodeRover, spec-driven agents). Do they parse tests into a plan? localize edits from tests? How do top performers convert tests into structured guidance? ' + RESEARCH_GUIDE },
  { key: 'parametrization-fidelity', prompt: 'Research reproducing exact pytest PARAMETRIZATION / node-ids: how parametrize ids are rendered (enum __str__, str-Enum vs Enum, ids= callables, duplicate-id indexing en0/en1, Python 3.11 enum __str__ change), and how a generator can be steered to reproduce an exact expected id set. ' + RESEARCH_GUIDE },
  { key: 'fair-test-use-anti-leak', prompt: 'Research the boundary between LEGITIMATE test-as-spec guidance and ANSWER-LEAKAGE in code-gen benchmarks (e.g. test IDs/strings that embed expected outputs; oracle leakage; train/eval contamination). Techniques to use tests for STRUCTURE while withholding expected output values. ' + RESEARCH_GUIDE },
]

const aThunks = []
for (const t of mapTasks) aThunks.push(() => agent(t.prompt + '\n\n=== CONTEXT ===\n' + CONTEXT, { label: 'map:' + t.key, phase: 'Map' }).then(r => ({ cat: 'map', key: t.key, text: r })))
for (const t of groundTasks) aThunks.push(() => agent(t.prompt + '\n\n=== CONTEXT ===\n' + CONTEXT, { label: 'ground:' + t.key, phase: 'Ground' }).then(r => ({ cat: 'ground', key: t.key, text: r })))
for (const t of researchTasks) aThunks.push(() => agent(t.prompt + '\n\n=== CONTEXT ===\n' + CONTEXT, { label: 'research:' + t.key, phase: 'Research' }).then(r => ({ cat: 'research', key: t.key, text: r })))

const aResults = (await parallel(aThunks)).filter(Boolean)
const byCat = (c) => aResults.filter(r => r.cat === c)
const fmt = (arr) => arr.map(r => '### [' + r.cat + ':' + r.key + ']\n' + r.text).join('\n\n')
const EVIDENCE = '# CURRENT MECHANISMS (MAP)\n' + fmt(byCat('map')) + '\n\n# GROUNDING\n' + fmt(byCat('ground')) + '\n\n# RESEARCH (SOTA)\n' + fmt(byCat('research'))
log('Stage A complete: ' + byCat('map').length + ' map + ' + byCat('ground').length + ' ground + ' + byCat('research').length + ' research')

// ===================== STAGE B: Design panel -> judge -> synthesize =====================
phase('Design')
const angles = [
  { key: 'static-test-contract', prompt: 'Design angle: a deterministic TEST-SUITE ANALYZER that extracts a DESIGN CONTRACT from the visible tests (AST + pytest --collect-only): required import surface, class/function signatures, enum members + REQUIRED string semantics, parametrization DOMAINS (locale/provider lists), and per-file expected test counts/ids — VALUE-STRIPPED (no answer payloads). Inject the contract into BOTH the solver prompt (template) and the scout/architect (autogen). Specify the extractor, the value-stripping rule, and the injection points.' },
  { key: 'gold-coverage-oracle', prompt: 'Design angle: a GOLD-COVERAGE FEEDBACK loop. After an attempt, diff the candidate-collected parametrization KEYS vs the expected-ID inventory (value-stripped) and tell the agent EXACTLY which parametrizations/locales/ids it is missing or rendering wrong (e.g. "you emit Locale.DE_AT; expected id is de-at"), driving a targeted, TIME-BUDGET-AWARE repair. No expected outputs leaked. Specify the oracle, the value-stripping, and the budget gating.' },
  { key: 'architect-as-test-analyst', prompt: 'Design angle: make the autogen ARCHITECT a TEST ANALYST — its primary job is to read the test suite, derive the contract, and author an orchestrate(ctx) whose decomposition + solver prompts target the exact required surface + parametrization. Scouts do test-topology analysis instead of generic difficulty. Specify the new scout/architect prompts + the plan shape.' },
  { key: 'lean-contract-preamble', prompt: 'Design angle: MINIMAL change — keep the lean template, but prepend a compact, value-stripped test-derived DESIGN-CONTRACT preamble to the solver prompt (and a coverage hint on repair). No heavy architecture; gate the contract step by repo size/difficulty so easy repos (voluptuous) stay 1-agent. Specify the compact contract format + the gating.' },
]
const proposals = (await parallel(angles.map(a => () =>
  agent('You are a principal AI-systems architect. Produce a concrete design proposal from this angle:\n\n' + a.prompt + '\n\nREQUIREMENTS: make BOTH arms gold-test-guided; FIX the mimesis ID-fidelity gap (missing->0) deterministically; HONOR the fairness boundary (structure yes, value-bearing outputs no); keep STRICT exact-id acceptance; be TIME-BUDGET-AWARE (run-4 lesson); do NOT regress voluptuous-1-agent/jinja; vendor-agnostic; map to specific files/functions. Ground in the evidence.\n\n' + EVIDENCE + '\n\n=== CONTEXT ===\n' + CONTEXT,
    { label: 'design:' + a.key, phase: 'Design' }).then(r => ({ key: a.key, text: r }))
))).filter(Boolean)

const judged = (await parallel(proposals.map(p => () =>
  agent('Critically score this design proposal for gold-test-guided improvement of the APEX-Omega arms. Tough, calibrated judge.\n\nPROPOSAL [' + p.key + ']:\n' + p.text + '\n\nScore each dimension 1-10 and name the best ideas to graft. Pay special attention to the FAIRNESS/answer-leak dimension. Evidence:\n' + EVIDENCE,
    { label: 'judge:' + p.key, phase: 'Design', schema: JUDGE_SCHEMA }).then(v => Object.assign({ key: p.key }, v))
))).filter(Boolean)
const scoreboard = judged.map(j => '[' + j.key + '] total=' + j.total + ' scores=' + JSON.stringify(j.scores) + ' graft=' + JSON.stringify(j.best_ideas_to_graft) + ' weaknesses=' + JSON.stringify(j.weaknesses)).join('\n')
const proposalsText = proposals.map(p => '## PROPOSAL [' + p.key + ']\n' + p.text).join('\n\n')

const design = await agent(
  'You are the lead architect. Synthesize ONE unified, implementable design that makes BOTH the template and autogen arms GOLD-TEST-SUITE-GUIDED. Take the highest-scoring proposal as the spine and graft the best ideas from the others. The design MUST: (1) derive a value-stripped DESIGN CONTRACT from the visible tests + expected-ID inventory (API surface, enum/str semantics, parametrization domains, required ids) and feed it into the solver prompt (template) AND the scout/architect (autogen); (2) deterministically fix the mimesis ID-fidelity gap (missing->0) — show it on the Locale.DE_AT->de-at + en0/en1 + [Locale.RU]/[cs] examples; (3) HONOR the fairness boundary (no value-bearing/answer leakage; keep strict exact-id acceptance) and state the exact value-stripping rule; (4) include a TIME-BUDGET-AWARE gold-coverage feedback repair loop; (5) NOT regress voluptuous-1-agent/jinja (gate by size/difficulty); (6) be vendor-agnostic. Give the end-to-end flow and map each component to specific files/functions.\n\nPROPOSALS:\n' + proposalsText + '\n\nJUDGE SCOREBOARD:\n' + scoreboard + '\n\nEVIDENCE:\n' + EVIDENCE + '\n\n=== CONTEXT ===\n' + CONTEXT,
  { label: 'synthesize-design', phase: 'Design' }
)
log('Stage B complete: unified design synthesized')

// ===================== STAGE C: adversarial validation by concern =====================
phase('Validate')
const concerns = [
  { key: 'fixes-mimesis-fidelity', ask: 'Will this design ACTUALLY drive mimesis missing->0? Walk it through the Locale(str,Enum) requirement, the locale-code domain, the en0/en1 indexing, and the [Locale.RU] vs [cs] split. Where could it still leave residual missing ids?' },
  { key: 'fairness-no-answer-leak', ask: 'Adversarially test the fairness boundary: can the proposed contract/coverage-oracle leak expected OUTPUT values (e.g. romanize mappings embedded in test IDs, or asserted return values)? Default skeptical — find any path that hands the solver an answer. Is it within commit0 "tests-as-spec"?' },
  { key: 'no-regression-cheap-wins', ask: 'Does the added test-analysis/contract step regress the cheap wins (voluptuous 1-agent, jinja)? Latency, over-specification, scout overhead, complexity. Is the size/difficulty gating sufficient?' },
  { key: 'time-budget-safe', ask: 'Does the design respect the run-4 time-budget lesson? Quantify the added cost (analysis + coverage-feedback repair) and confirm it cannot blow the cell wall-clock. Is the budget gating concrete?' },
  { key: 'both-arms-real', ask: 'Does this GENUINELY improve BOTH arms (template AND autogen), or mostly one? Is the template change real (not just autogen)? Are the injection points correct?' },
  { key: 'pydantic-and-generalization', ask: 'Does the contract approach help the genuinely-huge repo (pydantic, ~5091 tests) or is it still out of budget? Does the mechanism generalize beyond mimesis-style parametrization fidelity to other repos?' },
  { key: 'implementable-vendor-agnostic', ask: 'Is it implementable against the actual code (files/functions named correctly)? Vendor-agnostic (Codex + Claude)? Any place it assumes a vendor or a non-existent hook?' },
]
const validations = (await parallel(concerns.map(cc => () =>
  agent('Adversarially validate the proposed design on this concern: ' + cc.ask + '\n\nBe concrete and skeptical; ground in the evidence. Give a calibrated 0-100 confidence, a verdict, gaps, and required changes.\n\nDESIGN:\n' + design + '\n\nEVIDENCE:\n' + EVIDENCE + '\n\n=== CONTEXT ===\n' + CONTEXT,
    { label: 'val:' + cc.key, phase: 'Validate', schema: VALIDATION_SCHEMA }).then(v => Object.assign({}, v, { concern: cc.key }))
))).filter(Boolean)
const validationsText = validations.map(v => '[' + v.concern + '] verdict=' + v.verdict + ' conf=' + v.confidence_pct + '% gaps=' + JSON.stringify(v.gaps) + ' needed=' + JSON.stringify(v.required_changes || []) + ' :: ' + v.reasoning).join('\n')
log('Stage C complete: ' + validations.length + ' adversarial validations')

// ===================== STAGE D: implementation plan + critic + final =====================
phase('Plan')
const planAndCritique = await parallel([
  () => agent('Produce a FILE-LEVEL implementation plan for the design, ready to execute against ' + ROOT + '. Include: ordered steps; EXACT files/functions to change + NEW modules; DRAFTED code for the critical pieces — (a) the test-suite design-contract EXTRACTOR (AST + collect-only, value-stripped), (b) the value-stripping rule that keeps parametrization keys but removes answer payloads, (c) the gold-coverage feedback oracle + its TIME-BUDGET-AWARE repair wiring, (d) the injection into the solver prompt (template) and scout/architect (autogen); the unit/integration test plan (pytest under tests/, currently 92 green); and the exact empirical validation-run command (scripts/run_ladder.py; mimesis is the key cell; confirm missing->0 + no leak). Gate the contract step by repo size/difficulty to protect the cheap wins.\n\nDESIGN:\n' + design + '\n\nMAP/EVIDENCE:\n' + EVIDENCE + '\n\nVALIDATION FINDINGS (close these):\n' + validationsText + '\n\n=== CONTEXT ===\n' + CONTEXT,
    { label: 'impl-plan', phase: 'Plan' }),
  () => agent('Completeness critic. Given the design + validations, what is MISSING/underspecified? Especially: any answer-leak path not closed; any residual mimesis-missing-id class unhandled; any cheap-win regression; any unbudgeted cost; any place the template arm is under-served vs autogen; any unverified claim + how to verify. Be specific and harsh.\n\nDESIGN:\n' + design + '\n\nVALIDATIONS:\n' + validationsText + '\n\nEVIDENCE:\n' + EVIDENCE,
    { label: 'completeness-critic', phase: 'Plan' }),
])
const plan = planAndCritique[0] || ''
const critique = planAndCritique[1] || ''
log('Stage D: producing calibrated final deliverables')

const final = await agent(
  'You are the lead architect delivering the FINAL package. Integrate the design, the file-level plan, the adversarial validations, and the completeness critique; revise to close every gap (especially any answer-leak path and any residual mimesis-missing class). Be CALIBRATED AND HONEST — do not overclaim; give per-concern confidence and state what empirical validation (a real run) is required. The design MUST: gold-test-guide BOTH arms, fix mimesis ID-fidelity deterministically, honor the fairness boundary (strict exact-id acceptance kept), be time-budget-aware, not regress the cheap wins, and be vendor-agnostic.\n\nUNIFIED DESIGN:\n' + design + '\n\nFILE-LEVEL PLAN:\n' + plan + '\n\nADVERSARIAL VALIDATIONS:\n' + validationsText + '\n\nCOMPLETENESS CRITIQUE:\n' + critique + '\n\n=== CONTEXT ===\n' + CONTEXT,
  { label: 'final-synthesis', phase: 'Plan', schema: FINAL_SCHEMA }
)

return final
