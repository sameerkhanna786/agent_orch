export const meta = {
  name: 'apex-autogen-backbone',
  description: 'Map + design + file-level plan for re-aligning the autogen BACKBONE to the dynamic-workflow/ultracode model (agents run-to-completion, budget+ceiling-bounded not timer-killed, resumable/checkpointed, architect authors workflows with rich primitives+patterns; features layer on top)',
  phases: [
    { title: 'Map' },
    { title: 'Research' },
    { title: 'Design' },
    { title: 'Validate' },
    { title: 'Plan' },
  ],
}

const ROOT = '/Users/sameertkhanna/Documents/agent_orch'

const CONTEXT = [
  'PROJECT: APEX-Omega, an agent orchestrator evaluated on commit0 (implement a stubbed Python package so its visible',
  'pytest suite passes; execution-authoritative scoring with EXACT expected-test-id matching). Repo root: ' + ROOT + '.',
  '',
  'THE VISION (this design): re-align the AUTOGEN BACKBONE so it faithfully mirrors how the harness\' OWN dynamic',
  'workflows + ultracode work, then layer features/novelty on top. The architect ALREADY authors a deterministic',
  'orchestrate(ctx) script (like an .mjs workflow); we want the runtime guarantees + primitive vocabulary to match the',
  'workflow model. THE PRINCIPLE: agents run to COMPLETION; the DEFAULT IS UNBOUNDED (NO token/cost budget by default —',
  '"never optimize for cost"; matches BOTH the Budget primitive\'s documented "DEFAULTS UNBOUNDED" invariant AND the',
  'workflow model where the token budget is OPTIONAL/off-by-default). The ONLY always-on guards are the 1000-agent',
  'RUNAWAY BACKSTOP and the resumable/anytime-checkpoint journal (a long run is never DISCARDED). A token/agent budget',
  'is strictly OPT-IN, never the default. NEVER kill by a wall-clock that discards a verified result.',
  '',
  'THE REFERENCE MODEL — how the dynamic-workflow tool (the target) actually behaves (documented semantics):',
  ' - agent(prompt,{schema,...}) runs a subagent to COMPLETION; returns its result, or null only if skipped or a',
  '   TERMINAL api error after retries. No per-agent wall-clock kill.',
  ' - The workflow runs in the BACKGROUND to completion; no fixed run-duration cap.',
  ' - Bounds are BUDGET/CEILING, not time: concurrency cap min(16,cores-2)~=12 simultaneous; a 1000-agent LIFETIME',
  '   backstop (runaway guard); <=4096 items per parallel/pipeline call; an OPTIONAL token budget (hard ceiling only',
  '   if set, else infinite).',
  ' - RESUME: relaunch from a runId -> the longest unchanged prefix of agent() calls returns CACHED results instantly;',
  '   first edited/new call onward re-runs. Same script+args => 100% cache hit. (Determinism: no Date.now/random.)',
  ' - parallel([...]) = BARRIER (await all; a thrown thunk -> null). pipeline(items,...stages) = per-item streaming, NO',
  '   inter-stage barrier. log()/phase() narrate.',
  ' - QUALITY PATTERNS the planner composes: adversarial-verify (N skeptics per finding, kill on majority-refute),',
  '   perspective-diverse verify, judge-panel (generate N attempts -> score -> synthesize), loop-until-dry (spawn',
  '   finders until K dry rounds), completeness-critic, multi-modal sweep, no-silent-caps.',
  '',
  'CURRENT AUTOGEN ~= WORKFLOW MODEL (already near 1:1; this design closes the gaps):',
  '   agent()<->engine.agent()/ctx.solve_attempt; parallel()/pipeline()<->ctx.parallel/ctx.pipeline; concurrency 12',
  '   <->engine.max_concurrent=min(16,cpu-2); 1000 backstop<->engine.max_total_agents (agent_ceiling); token budget',
  '   <->Budget(total); phase/log<->ctx.phase/ctx.log; schema<->scout JSON schema; I-author-.mjs<->architect authors',
  '   orchestrate(ctx) (frozen+lint+journaled); execution-authoritative result<->ctx.select (real pytest).',
  '',
  'THE KEY DIVERGENCE (the bug to fix) + WHAT IS ALREADY DONE:',
  ' - The EVAL CELL has an OUTER WALL-CLOCK GUILLOTINE: scripts/run_ladder.py subprocess.run(timeout=CELL_TIMEOUT+600)',
  '   KILLS the whole cell mid-flight. run-4 lost VERIFIED mimesis 6052-passing work this way. That timer is a HARNESS',
  '   artifact, NOT part of the workflow paradigm.',
  ' - DONE so far (move toward the model): acceptance-checkpointing (context.py writes accepted_checkpoint.json the',
  '   instant a candidate is accepted; run_ladder._recover_checkpoint recovers it on a timeout-kill); budget-aware',
  '   per-eval timeout (commit0_autogen eval_cap); opt-in repair via a repair_iters ceiling (default 0); a Budget.total',
  '   token ceiling currently set to 40M in commit0_driver. CONSTRAINT (revise per this design): the DEFAULT MUST BE',
  '   UNBOUNDED — make that 40M ceiling OPT-IN (env/flag), not the default; the always-on guard stays the 1000-agent',
  '   backstop + resumability, not a default cost bound. STILL DIVERGENT: per-AGENT timeout is ScopedTask(timeout_seconds=',
  '   cell_timeout) so one agent can eat the whole wall (the true run-4 root cause); no full mid-run RESUME of a',
  '   killed cell (the journal caches agent() calls but the cell wrapper re-runs from scratch); the wall-clock still',
  '   exists as a guillotine rather than a pause+resume/budget bound.',
  '',
  'EXISTING ASSETS TO BUILD ON (do NOT reinvent): the JOURNAL (apex_omega/journal/{wal.py,resume.py,key.py};',
  'resume_or_run_json content-keyed replay); the SANDBOX (apex_omega/autogen/sandbox.py: AST lint + freeze content-',
  'hash + restricted builtins -> deterministic replay); BUDGET (apex_omega/engine/budget.py); the ENGINE concurrency/',
  'ceiling/journaling (apex_omega/engine/runtime.py); the CARDINAL CONTRACT (apex_omega/kernel/select.py exec-',
  'authoritative, no self-accept, monotone downgrade-only refute).',
  '',
  'INVARIANTS TO PRESERVE: execution-authoritative acceptance (only real pytest sets accepted; the plan cannot self-',
  'accept; soft signals only downgrade); sandbox determinism/replayability; vendor-agnostic (Codex CLI + Claude Code,',
  'vendor is a WorkerSpec field); fewest-agents-first + a guaranteed best-of-N FLOOR (never worse than the template);',
  'no answer-leak when features are layered (see the design-contract fairness firewall).',
  '',
  'FEATURES THAT MUST LAYER ON TOP (NOT be baked into the backbone) — keep them composable:',
  ' - design_contract.py (JUST BUILT, apex_omega/eval/design_contract.py): gold-test-guided design contract (value-',
  '   stripped; fairness firewall; enum dual-regime). It enriches solver/architect prompts.',
  ' - test-driven REPAIR lineages (context.py solve_and_repair/repair_attempt, opt-in repair_iters).',
  ' - verifier-guided selection; and the workflow QUALITY PATTERNS (adversarial-verify/judge-panel/loop-until-dry/',
  '   completeness-critic) exposed so the architect can COMPOSE them.',
  '',
  'KEY FILES: apex_omega/engine/{runtime.py,pipeline.py,budget.py}; apex_omega/autogen/{context.py,architect.py,',
  'templates.py,sandbox.py}; apex_omega/eval/{commit0_autogen.py,commit0_driver.py,scoring.py,design_contract.py};',
  'apex_omega/journal/{wal.py,resume.py,key.py}; apex_omega/kernel/{select.py,verify.py}; scripts/run_ladder.py.',
  'Evidence/history: APEX_COMMIT0_REPORT.md, APEX_GOLD_TEST_GUIDED_DESIGN.md, runs/archive/ladder_run4_fulldesign_*',
  '(the timeout regression), runs/validation_checkpoint/* (the mimesis fidelity finding).',
  '',
  'OUTPUT: a unified BACKBONE design + FILE-LEVEL implementation plan: (1) run-to-completion / budget+ceiling-bounded /',
  'resumable-checkpointed runtime (remove the guillotine; per-agent timeout decoupling; full mid-run resume reusing the',
  'journal); (2) the architect\'s expanded PRIMITIVE + PATTERN vocabulary (compose adversarial-verify/judge/loop-until-',
  'dry like the workflow tool) with the floor + budget-awareness preserved; (3) how the existing features re-layer on',
  'it. Calibrated, honest, preserves all invariants, vendor-agnostic.',
].join('\n')

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    scores: {
      type: 'object', additionalProperties: false,
      properties: {
        faithful_to_workflow_model: { type: 'integer', description: '1-10: run-to-completion, budget/ceiling-bounded, resumable' },
        no_discard_of_verified_work: { type: 'integer', description: '1-10: a verified result is never lost to a timer' },
        preserves_invariants: { type: 'integer', description: '1-10: exec-authoritative, sandbox determinism, floor, no-leak' },
        architect_primitive_richness: { type: 'integer', description: '1-10: composable patterns w/o blowing budget' },
        features_layer_cleanly: { type: 'integer', description: '1-10: design-contract/repair/verify compose on top' },
        implementable_low_risk: { type: 'integer', description: '1-10: maps to code; reuses journal/sandbox/budget; vendor-agnostic' },
      },
      required: ['faithful_to_workflow_model', 'no_discard_of_verified_work', 'preserves_invariants', 'architect_primitive_richness', 'features_layer_cleanly', 'implementable_low_risk'],
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
    confidence_pct: { type: 'integer' },
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
    design_doc: { type: 'string', description: 'FULL markdown: the backbone runtime model (run-to-completion/budget/ceiling/resume), the architect primitive+pattern vocabulary, how features re-layer, every component mapped to files, invariants preserved' },
    implementation_plan: { type: 'string', description: 'FILE-LEVEL phased plan with drafted code for the critical pieces (budget/ceiling governor + anytime-checkpoint + full mid-run resume; per-agent timeout decoupling; the new ctx pattern-primitives; guillotine removal/pause-resume), test plan, validation-run plan' },
    primitives_api: { type: 'string', description: 'the proposed expanded ctx/architect API surface (signatures + 1-line semantics) for the composable patterns' },
    per_concern_confidence: { type: 'array', items: { type: 'object', additionalProperties: false, properties: { concern: { type: 'string' }, confidence_pct: { type: 'integer' }, note: { type: 'string' } }, required: ['concern', 'confidence_pct', 'note'] } },
    overall_confidence_pct: { type: 'integer' },
    overall_assessment: { type: 'string' },
    residual_risks: { type: 'array', items: { type: 'string' } },
    phasing: { type: 'string', description: 'ordered implementation phases (backbone first, then re-layer features) with the go/no-go check per phase' },
  },
  required: ['executive_summary', 'design_doc', 'implementation_plan', 'primitives_api', 'per_concern_confidence', 'overall_confidence_pct', 'overall_assessment', 'residual_risks', 'phasing'],
}

const RESEARCH_GUIDE = [
  'Use external web search to find CURRENT SOTA and CITE sources (title+url). Load via ToolSearch: query',
  '"select:mcp__plugin_meta_mux__three_pai_external_web_search" then call it; or WebFetch a page. If web tools are',
  'unavailable, use training knowledge (cutoff Jan 2026) and mark claims [UNVERIFIED]. For each idea: the mechanism in',
  '2-4 sentences, evidence if known, and SPECIFICALLY how it maps onto the APEX-Omega engine (files/primitives).',
].join('\n')

// ===================== STAGE A: Map + Research =====================
phase('Map')
log('Backbone design: mapping current engine vs the workflow model + SOTA research, in parallel')

const mapTasks = [
  { key: 'engine-vs-workflow-gap', prompt: 'Read apex_omega/engine/{runtime.py,pipeline.py,budget.py} and apex_omega/autogen/context.py. Produce an EXACT mapping of the current engine primitives to the dynamic-workflow model (agent/parallel/pipeline/concurrency/ceiling/budget) and list every DIVERGENCE from "agents run to completion, bounded by budget+ceiling, resumable". Pinpoint where (if anywhere) work is bounded by TIME rather than budget.' },
  { key: 'resume-journal', prompt: 'Read apex_omega/journal/{wal.py,resume.py,key.py} and how engine.agent + resume_or_run_json journal/replay calls (+ the materialize/diff cache-hit path in context.py solve_attempt). Determine: can a KILLED/paused autogen cell RESUME without re-running completed agents (like the workflow tool resumes from a runId, caching agent() by content)? What EXACTLY is cached/keyed, and what is missing for full mid-run resume of a cell? Be concrete about the journal key + the replay mechanics.' },
  { key: 'architect-authoring', prompt: 'Read apex_omega/autogen/{architect.py (author_orchestration, _freeze/load_frozen, API_REFERENCE, INVARIANTS, build_author_prompt), templates.py, sandbox.py (lint_source, run_orchestration, SAFE_BUILTINS)}. Explain how the architect authors+freezes+runs a deterministic orchestrate(ctx) today, and the EXACT ctx primitive vocabulary it can call. Compare to the workflow tool patterns (adversarial-verify, judge-panel, loop-until-dry, completeness-critic). What primitives/patterns are MISSING that we would add so the architect composes workflows the way the tool does?' },
  { key: 'timeout-kill-paths', prompt: 'Map EVERY wall-clock / kill / discard path end-to-end: scripts/run_ladder.py (CELL_TIMEOUT, subprocess.run timeout, _recover_checkpoint, _strip_checkout), apex_omega/eval/commit0_autogen.py (cell_timeout_seconds, eval_cap, autosolve timeout_seconds), apex_omega/autogen/context.py (ScopedTask timeout_seconds, _cell_deadline if any, solve_and_repair guards). For each: does it DISCARD work or checkpoint/resume? What must change to make the cell budget-bounded + resumable instead of guillotined? (The acceptance-checkpoint + recovery are already added — assess their completeness.)' },
  { key: 'invariants-safety', prompt: 'Read apex_omega/kernel/{select.py,verify.py}, apex_omega/eval/scoring.py, apex_omega/autogen/sandbox.py. Enumerate the invariants the backbone MUST preserve (execution-authoritative acceptance; no plan-self-accept; monotone downgrade-only refute; sandbox determinism/replay; best-of-N floor; vendor-agnostic; no answer-leak). For each, note how a richer/run-to-completion backbone could threaten it and the guardrail.' },
  { key: 'workflow-model-reference', prompt: 'Synthesize a PRECISE reference spec of the dynamic-workflow/ultracode model the backbone should mirror (from the CONTEXT reference-model section + your understanding): run-to-completion agents; budget+ceiling bounds (concurrency cap, 1000 lifetime backstop, optional token budget); background + resume-by-prefix-cache; parallel barrier vs pipeline streaming; the composable quality patterns. Produce the TARGET contract the autogen backbone is being aligned to, as a checklist the design must satisfy.' },
]

const researchTasks = [
  { key: 'durable-execution', prompt: 'Research DURABLE / RESUMABLE execution + checkpointing for long agent runs: durable-execution engines (Temporal/workflows-as-code), event-sourcing/WAL, idempotent replay, anytime/checkpointed results so a kill never loses verified work. How to make a long agent cell resume from a journal without re-running or double-applying. ' + RESEARCH_GUIDE },
  { key: 'agents-as-code', prompt: 'Research PLANNER-AUTHORS-PROGRAM / agents-as-code orchestration (LangGraph, OpenAI Swarm/Agents SDK, Anthropic Claude Agent SDK, AutoGen, DSPy programs, code-as-policy). Patterns where a planner emits an executable orchestration over sub-agents, the runtime guarantees (retries/resume/budget), and the tradeoffs vs a fixed harness. Map to the architect authoring orchestrate(ctx). ' + RESEARCH_GUIDE },
  { key: 'compute-budget-scaling', prompt: 'Research TEST-TIME COMPUTE scaling + BUDGET ALLOCATION (not wall-clock) for agent ensembles: bounding by tokens/agents with anytime results, adaptive allocation (spend more only while improving), early-stop/plateau, verifier-guided budget. How to govern "escalate until verified or budget" without a guillotine. ' + RESEARCH_GUIDE },
  { key: 'composable-verify-patterns', prompt: 'Research COMPOSABLE orchestration patterns as reusable primitives a planner can invoke: adversarial verification / debate / N-skeptic refutation, judge/jury panels, self-consistency, reflexion loops, loop-until-converged, completeness critics. How to expose these as a small composable API (like parallel/pipeline) rather than bespoke code. ' + RESEARCH_GUIDE },
]

const aThunks = []
for (const t of mapTasks) aThunks.push(() => agent(t.prompt + '\n\n=== CONTEXT ===\n' + CONTEXT, { label: 'map:' + t.key, phase: 'Map' }).then(r => ({ cat: 'map', key: t.key, text: r })))
for (const t of researchTasks) aThunks.push(() => agent(t.prompt + '\n\n=== CONTEXT ===\n' + CONTEXT, { label: 'research:' + t.key, phase: 'Research' }).then(r => ({ cat: 'research', key: t.key, text: r })))

const aResults = (await parallel(aThunks)).filter(Boolean)
const byCat = (c) => aResults.filter(r => r.cat === c)
const fmt = (arr) => arr.map(r => '### [' + r.cat + ':' + r.key + ']\n' + r.text).join('\n\n')
const EVIDENCE = '# CURRENT ENGINE / GAP MAP\n' + fmt(byCat('map')) + '\n\n# RESEARCH (SOTA)\n' + fmt(byCat('research'))
log('Stage A complete: ' + byCat('map').length + ' map + ' + byCat('research').length + ' research')

// ===================== STAGE B: Design panel -> judge -> synthesize =====================
phase('Design')
const angles = [
  { key: 'durable-resumable', prompt: 'Design angle: DURABLE/RESUMABLE backbone. Make a cell run-to-completion and fully RESUMABLE from the journal (cache agent() results by content-key so a re-launched/paused cell skips completed agents, like the workflow runId resume). Replace the guillotine with pause+resume; anytime acceptance-checkpoint so a verified result is never lost. Specify the journal-key + replay + resume entrypoint changes.' },
  { key: 'budget-governor', prompt: 'Design angle: BUDGET-GOVERNOR (not timer) with DEFAULT UNBOUNDED. By DEFAULT there is NO token/cost budget (never optimize for cost — matches Budget\'s documented invariant); the always-on guard is the 1000-agent runaway backstop + per-agent timeout decoupling + resumability, NOT a default cost bound. A token budget is OPT-IN (env/flag); WHEN set, adaptive escalation (spend only while improving, plateau-stop). Remove all wall-clock DISCARD; anytime checkpoint. Specify the governor, the opt-in budget surface, the per-agent vs cell timeout split, and the escalation/stop rules (and what bounds a run when NO budget is set).' },
  { key: 'rich-architect-primitives', prompt: 'Design angle: RICH ARCHITECT PRIMITIVES. Expand the ctx vocabulary so the architect composes the workflow quality patterns: ctx.verify/refute (adversarial), ctx.judge_panel (generate->score->synthesize), ctx.loop_until_dry, ctx.critic (completeness), perspective-diverse verify — as a small composable API atop solve_attempt/parallel/pipeline, all execution-authoritative + budget-aware + floor-preserving. Specify the API + how the architect is taught to use it (API_REFERENCE/exemplars).' },
  { key: 'minimal-faithful-port', prompt: 'Design angle: MINIMAL FAITHFUL PORT. The smallest set of changes that makes autogen behave EXACTLY like the workflow tool (agent run-to-completion; concurrency+1000 ceiling; optional token budget; resume; no guillotine), deferring rich patterns. Maximize fidelity-per-change; identify what is already there vs the few real gaps.' },
]
const proposals = (await parallel(angles.map(a => () =>
  agent('You are a principal AI-systems architect. Produce a concrete backbone design from this angle:\n\n' + a.prompt + '\n\nREQUIREMENTS: faithful to the run-to-completion/budget+ceiling/resumable workflow model; a verified result is NEVER discarded by a timer; PRESERVE all invariants (exec-authoritative, sandbox determinism, best-of-N floor, vendor-agnostic, no answer-leak); features (design-contract/repair/verify) must LAYER cleanly on top, not be baked in; reuse the existing journal/sandbox/budget; map every component to specific files/functions. Ground in the evidence.\n\n' + EVIDENCE + '\n\n=== CONTEXT ===\n' + CONTEXT,
    { label: 'design:' + a.key, phase: 'Design' }).then(r => ({ key: a.key, text: r }))
))).filter(Boolean)

const judged = (await parallel(proposals.map(p => () =>
  agent('Critically score this autogen-backbone design. Tough, calibrated judge. Weight fidelity-to-workflow-model + never-discard-verified-work + invariant-preservation highest.\n\nPROPOSAL [' + p.key + ']:\n' + p.text + '\n\nScore each dimension 1-10 and name the best ideas to graft. Evidence:\n' + EVIDENCE,
    { label: 'judge:' + p.key, phase: 'Design', schema: JUDGE_SCHEMA }).then(v => Object.assign({ key: p.key }, v))
))).filter(Boolean)
const scoreboard = judged.map(j => '[' + j.key + '] total=' + j.total + ' scores=' + JSON.stringify(j.scores) + ' graft=' + JSON.stringify(j.best_ideas_to_graft) + ' weaknesses=' + JSON.stringify(j.weaknesses)).join('\n')
const proposalsText = proposals.map(p => '## PROPOSAL [' + p.key + ']\n' + p.text).join('\n\n')

const design = await agent(
  'You are the lead architect. Synthesize ONE unified BACKBONE design that re-aligns autogen to the dynamic-workflow/ultracode model. Take the highest-scoring proposal as the spine and graft the best ideas. It MUST: (1) make a cell RUN-TO-COMPLETION with the DEFAULT UNBOUNDED (NO token/cost budget by default — never optimize for cost; matches Budget\'s "defaults unbounded" invariant); the always-on guards are the 1000-agent RUNAWAY BACKSTOP + per-agent timeout decoupling + resumability, and a token budget is strictly OPT-IN (env/flag) — NOT a wall-clock that discards work and NOT a default cost bound; (2) be fully RESUMABLE/checkpointed (reuse the journal so a re-launched/paused cell skips completed agents; anytime acceptance-checkpoint so a verified result is never lost); (3) give the architect a RICH composable PRIMITIVE/PATTERN vocabulary (adversarial-verify, judge-panel, loop-until-dry, completeness-critic) atop solve_attempt/parallel/pipeline, all execution-authoritative + budget-aware + floor-preserving; (4) keep the design-contract/repair/verifier FEATURES layering cleanly ON TOP; (5) PRESERVE every invariant; (6) be vendor-agnostic. Give the end-to-end runtime model, the API surface, and map each component to specific files/functions.\n\nPROPOSALS:\n' + proposalsText + '\n\nJUDGE SCOREBOARD:\n' + scoreboard + '\n\nEVIDENCE:\n' + EVIDENCE + '\n\n=== CONTEXT ===\n' + CONTEXT,
  { label: 'synthesize-design', phase: 'Design' }
)
log('Stage B complete: unified backbone design synthesized')

// ===================== STAGE C: adversarial validation by concern =====================
phase('Validate')
const concerns = [
  { key: 'unbounded-default-runaway-guard', ask: 'The DEFAULT IS UNBOUNDED (no token/cost budget by default, by design). With NO budget set, is the 1000-agent runaway BACKSTOP + per-agent timeout decoupling + resumability + plateau-stop a SOUND guard against a pathological/hanging authored plan, WITHOUT imposing a default cost bound? Where is the anytime/runaway guarantee weakest when unbounded? Confirm the opt-in budget path does not accidentally become a default.' },
  { key: 'resume-correctness', ask: 'Is the resume/replay correct? Does a re-launched cell reproduce results WITHOUT re-running completed agents AND without double-applying diffs (the journal/materialize interaction)? Is the journal key deterministic enough for a 100% cache hit on an unchanged plan?' },
  { key: 'no-discard-verified', ask: 'Does the design GUARANTEE a verified-accepted result is never discarded (the run-4 failure)? Walk a mid-flight kill + resume through the acceptance-checkpoint + journal. Any window where a verified solve is still lost?' },
  { key: 'invariants-preserved', ask: 'Are exec-authoritative acceptance, no-self-accept, monotone refute, sandbox determinism, the best-of-N floor, and no-answer-leak all preserved under the richer/run-to-completion backbone? Find any erosion (esp. the new ctx pattern-primitives bypassing the gate, or richer plans defeating determinism).' },
  { key: 'no-regression-and-complexity', ask: 'Does the richer backbone regress the cheap wins or re-introduce the run-4 complexity blowup? Is "fewest-agents-first + floor + budget-aware" still enforced so a fancy authored plan cannot be worse than the lean template?' },
  { key: 'features-layer-cleanly', ask: 'Do the design-contract, repair lineages, and verifier-guided selection actually compose ON TOP of this backbone without modifying it? Is the boundary backbone-vs-feature clean and stated?' },
  { key: 'implementable-vendor-agnostic', ask: 'Is it implementable against the real code (files/functions named right; reuses journal/sandbox/budget)? Vendor-agnostic (Codex + Claude)? Any assumed hook that does not exist?' },
]
const validations = (await parallel(concerns.map(cc => () =>
  agent('Adversarially validate the proposed backbone design on this concern: ' + cc.ask + '\n\nBe concrete and skeptical; ground in the evidence. Give a calibrated 0-100 confidence, a verdict, gaps, and required changes.\n\nDESIGN:\n' + design + '\n\nEVIDENCE:\n' + EVIDENCE + '\n\n=== CONTEXT ===\n' + CONTEXT,
    { label: 'val:' + cc.key, phase: 'Validate', schema: VALIDATION_SCHEMA }).then(v => Object.assign({}, v, { concern: cc.key }))
))).filter(Boolean)
const validationsText = validations.map(v => '[' + v.concern + '] verdict=' + v.verdict + ' conf=' + v.confidence_pct + '% gaps=' + JSON.stringify(v.gaps) + ' needed=' + JSON.stringify(v.required_changes || []) + ' :: ' + v.reasoning).join('\n')
log('Stage C complete: ' + validations.length + ' adversarial validations')

// ===================== STAGE D: implementation plan + critic + final =====================
phase('Plan')
const planAndCritique = await parallel([
  () => agent('Produce a FILE-LEVEL, PHASED implementation plan for the backbone, ready to execute against ' + ROOT + '. Phase it: BACKBONE FIRST (run-to-completion/budget+ceiling governor; per-agent timeout decoupling; anytime checkpoint + full mid-run RESUME reusing the journal; remove/neutralize the wall-clock guillotine), THEN re-layer features. Include: ordered steps; EXACT files/functions to change + NEW modules; DRAFTED code for the critical pieces (the budget/ceiling governor + plateau-stop; per-agent vs cell timeout split; the resume entrypoint + journal-cache replay; the new composable ctx primitives — verify/judge_panel/loop_until_dry/critic — with the floor + exec-authoritative gate intact); the unit/integration test plan (pytest under tests/, currently 92 green); and the empirical validation-run plan (mimesis + jinja + voluptuous; confirm no discard, resume works, no regression). Each phase has a go/no-go check.\n\nDESIGN:\n' + design + '\n\nMAP/EVIDENCE:\n' + EVIDENCE + '\n\nVALIDATION FINDINGS (close these):\n' + validationsText + '\n\n=== CONTEXT ===\n' + CONTEXT,
    { label: 'impl-plan', phase: 'Plan' }),
  () => agent('Completeness critic. Given the backbone design + validations, what is MISSING/underspecified? Especially: any path that still DISCARDS verified work; any resume double-apply / non-deterministic-key bug; any invariant erosion by the new primitives; any way the richer architect re-introduces the run-4 blowup; any unverified claim + how to verify. Be specific and harsh.\n\nDESIGN:\n' + design + '\n\nVALIDATIONS:\n' + validationsText + '\n\nEVIDENCE:\n' + EVIDENCE,
    { label: 'completeness-critic', phase: 'Plan' }),
])
const plan = planAndCritique[0] || ''
const critique = planAndCritique[1] || ''
log('Stage D: producing calibrated final deliverables')

const final = await agent(
  'You are the lead architect delivering the FINAL backbone package. Integrate the design, the phased file-level plan, the adversarial validations, and the completeness critique; revise to close every gap (especially any verified-work-discard path, any resume correctness bug, any invariant erosion). Be CALIBRATED AND HONEST — give per-concern confidence and state what empirical validation is required. The backbone MUST: run-to-completion, be budget+ceiling-bounded (never timer-discarded), be resumable/checkpointed, give the architect rich composable patterns, keep features layering on top, preserve all invariants, and be vendor-agnostic. Provide the primitives_api surface and the phasing (backbone first, then features) with go/no-go gates.\n\nUNIFIED DESIGN:\n' + design + '\n\nFILE-LEVEL PLAN:\n' + plan + '\n\nADVERSARIAL VALIDATIONS:\n' + validationsText + '\n\nCOMPLETENESS CRITIQUE:\n' + critique + '\n\n=== CONTEXT ===\n' + CONTEXT,
  { label: 'final-synthesis', phase: 'Plan', schema: FINAL_SCHEMA }
)

return final
