export const meta = {
  name: 'autogen-nextgen-design',
  description: 'Research-grounded, adversarially-validated redesign of the APEX-Omega autogen orchestrator to reliably solve all commit0 repos (with plan-driven work beyond flat rollouts)',
  phases: [
    { title: 'Map' },
    { title: 'Forensics' },
    { title: 'Research' },
    { title: 'Design' },
    { title: 'Validate' },
    { title: 'Plan' },
  ],
}

const ROOT = '/Users/sameertkhanna/Documents/agent_orch'

const CONTEXT = [
  'PROJECT: APEX-Omega — an agent-orchestration system evaluated on the commit0 benchmark (implement a Python',
  "package's stub functions from its hidden test suite; scoring is execution-authoritative — the REAL pytest gate",
  'decides "solved", never a soft score). Repo root: ' + ROOT + '.',
  '',
  'THE ARM UNDER REDESIGN ("autogen" / generated-code orchestration): a pipeline that SCOUTS difficulty, has an',
  'ARCHITECT author a custom orchestration plan (Python code using ctx.agent()/ctx.parallel() primitives), runs it',
  'in a sandbox, and selects a winner under a "Cardinal Contract". Compare against: "template" arm (a FIXED lean',
  'DEFAULT_ORCHESTRATION best-of-N, no authoring); "baseline_v1" (the v1 commit0 harness as worker); "B0" (single-model 1-shot).',
  '',
  'CLEAN RUN RESULTS (just completed; K=8 agents/cell, sanitized container). Repos: voluptuous (medium), jinja/mimesis/pydantic (hard):',
  ' - solve-rate: B0 50% (2/4), baseline_v1 50% (2/4, 1 load-timeout), template 25% (1/4), autogen 25% (1/4), B2cap16 1/1.',
  ' - AUTOGEN_WON = 0 (no repo where autogen solved & template did not). On the only shared solve (voluptuous) autogen',
  '   used 4 agents vs template 1 (4x cost, same result).',
  ' - Everyone solves voluptuous; nobody RELIABLY solves the hard repos (scattered single solves: baseline got jinja, B0 got mimesis).',
  '',
  'VERIFIED FAILURE MODES:',
  ' 1. FETCH-UPSTREAM SHORTCUT persists despite container sanitization. mimesis autogen: 8 agents -> 3 genuine',
  '    attempts + 5 policy_violation (tried to download/clone/restore the OFFICIAL upstream package into the',
  '    workspace -> workspace-jail blocked it -> no result). Violation text saturated: download x33, restore x31,',
  '    upstream x27, clone x20, official x16. Env-layer scrub (hiding version literals/URLs) is NECESSARY BUT NOT',
  "    SUFFICIENT: the agent's prior 'this is a real PyPI package, go get it' overrides a hidden version. The architect",
  '    sometimes bakes "restore official upstream" into the SHARED prefix of all attempts -> a MONOCULTURE where every',
  '    variant routes through the same forbidden fetch with no in-workspace fallback.',
  ' 2. HIGH VARIANCE. The SAME mimesis autogen cell SOLVED in one draw (535s, 5 agents) and FAILED in another',
  '    (875s, 8 agents, 5/8 fetch). The architect is stochastic; n=1/cell cannot rank arms on hard repos.',
  ' 3. COMPLEXITY NOT PAYING OFF. The lean template matches autogen solve-rate at 1/4 the agents; scout->architect->author',
  '    adds failure surface (monoculture, overhead) without yet adding solve power.',
  '',
  'HARD DESIGN CONSTRAINTS (must hold):',
  ' - Execution-authoritative scoring stays (real pytest gate). Keep the env sanitizer.',
  ' - Vendor-agnostic: must work with Codex CLI AND Claude Code as the underlying agent.',
  ' - Orchestration plans MUST be allowed to do ADDITIONAL WORK ON TOP OF ROLLOUTS when the plan indicates it is',
  '   needed/desired/planned-for: test-driven repair loops (implement -> run tests -> read failures -> fix -> repeat),',
  '   follow-up/debug agents conditioned on test output, escalation. Today best-of-N is a FLAT fan of K one-shot rollouts;',
  '   commit0_autogen caps via autogen-max-agents (8) with scout waves 4->6->8 and a hard agent_ceiling backstop (1000).',
  '   The new design should let a plan legitimately schedule more agents/iterations UP TO the ceiling when justified by',
  '   test signal — not be locked to K one-shots.',
  '',
  'KEY FILES:',
  ' - autogen pipeline: apex_omega/autogen/{architect.py, context.py, templates.py, sandbox.py, __init__.py}',
  ' - best-of-N + selection: apex_omega/workflows/best_of_n.py',
  ' - cell execution + caps: apex_omega/eval/commit0_autogen.py, commit0_driver.py',
  ' - engine primitives: apex_omega/engine/{runtime.py, pipeline.py, budget.py}',
  ' - scoring + sanitize + jail: apex_omega/eval/{scoring.py, repo_sanitize.py, registry.py}; apex/evaluation/commit0_benchmark.py (17k lines — GREP, do not read whole)',
  ' - arms/knobs: apex_omega/ablation/{arms.py, safety_modes.py}, apex_omega/cli.py',
  ' - existing design corpus: .apex_plan_sections/*.md, APEX_NEXTGEN_PLAN.md, APEX_DESIGN.md',
  ' - run artifacts: runs/ladder/SIGNALS_LEDGER.md, runs/ladder/autogen_evidence.json,',
  '   runs/ladder/omega_autogen_k8__{repo}/{journal/calls_wal.jsonl, journal/diffs/, narration.jsonl, orchestrator/, cells/},',
  '   runs/ladder/omega_template_k8__{repo}/ and baseline_v1_k8__{repo}/ for comparison.',
].join('\n')

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    scores: {
      type: 'object', additionalProperties: false,
      properties: {
        addresses_all_failure_modes: { type: 'integer', description: '1-10: fixes fetch-shortcut, variance, complexity-not-paying-off' },
        supports_additional_work_beyond_rollouts: { type: 'integer', description: '1-10' },
        vendor_agnostic: { type: 'integer', description: '1-10' },
        implementation_feasibility: { type: 'integer', description: '1-10: maps cleanly onto existing code, low risk' },
        expected_solverate_lift: { type: 'integer', description: '1-10' },
        variance_reduction: { type: 'integer', description: '1-10' },
      },
      required: ['addresses_all_failure_modes', 'supports_additional_work_beyond_rollouts', 'vendor_agnostic', 'implementation_feasibility', 'expected_solverate_lift', 'variance_reduction'],
    },
    total: { type: 'number' },
    strengths: { type: 'array', items: { type: 'string' } },
    weaknesses: { type: 'array', items: { type: 'string' } },
    best_ideas_to_graft: { type: 'array', items: { type: 'string' }, description: 'specific ideas worth merging into the winner' },
  },
  required: ['scores', 'total', 'strengths', 'weaknesses', 'best_ideas_to_graft'],
}

const VALIDATION_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    repo: { type: 'string' },
    lens: { type: 'string' },
    will_solve_confidence_pct: { type: 'integer', description: '0-100 calibrated probability the design solves THIS repo' },
    verdict: { type: 'string', enum: ['will_solve', 'likely', 'uncertain', 'will_not_solve'] },
    remaining_gaps: { type: 'array', items: { type: 'string' } },
    required_design_changes: { type: 'array', items: { type: 'string' }, description: 'concrete changes needed to raise confidence' },
    reasoning: { type: 'string' },
  },
  required: ['repo', 'lens', 'will_solve_confidence_pct', 'verdict', 'remaining_gaps', 'reasoning'],
}

const FINAL_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    executive_summary: { type: 'string', description: 'tight summary of the diagnosis + the design + the confidence verdict' },
    design_doc: { type: 'string', description: 'FULL master design doc in markdown' },
    implementation_plan: { type: 'string', description: 'FILE-LEVEL implementation plan in markdown with drafted code for the critical pieces' },
    per_repo_confidence: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          repo: { type: 'string' },
          confidence_pct: { type: 'integer' },
          justification: { type: 'string' },
          residual_gaps: { type: 'array', items: { type: 'string' } },
        },
        required: ['repo', 'confidence_pct', 'justification', 'residual_gaps'],
      },
    },
    overall_confidence_pct: { type: 'integer', description: 'calibrated, honest overall confidence it solves ALL runnable repos' },
    overall_assessment: { type: 'string', description: 'honest, calibrated — state what empirical validation is still required' },
    residual_risks: { type: 'array', items: { type: 'string' } },
    validation_run_command: { type: 'string' },
  },
  required: ['executive_summary', 'design_doc', 'implementation_plan', 'per_repo_confidence', 'overall_confidence_pct', 'overall_assessment', 'residual_risks', 'validation_run_command'],
}

const RESEARCH_GUIDE = [
  'Use external web search to find CURRENT (2023-2025) SOTA and CITE sources (title + url). Load the approved search',
  'tool via ToolSearch: query "select:mcp__plugin_meta_mux__three_pai_external_web_search" then call it; you may also',
  'load WebFetch to read specific pages. If web tools are unavailable, draw on your training knowledge (cutoff Jan 2026)',
  'and clearly mark such claims [UNVERIFIED]. For each technique report: what problem it solves, the mechanism in 2-4',
  'sentences, measured results if known, and SPECIFICALLY how it integrates into the APEX-Omega autogen pipeline (which',
  'files/primitives) to fix our verified failure modes.',
].join('\n')

// ===================== STAGE A: Map + Forensics + Research =====================
phase('Map')
log('Stage A: mapping codebase + forensics on the run + SOTA research, in parallel')

const mapTasks = [
  { key: 'autogen-pipeline', prompt: 'Read apex_omega/autogen/{architect.py, context.py, templates.py, sandbox.py, __init__.py}. Explain precisely: how an orchestration plan is scouted/authored/generated; the scout->architect->author flow; the DEFAULT_ORCHESTRATION template; how generated plan code calls ctx.agent()/ctx.parallel(); the sandbox (SAFE_BUILTINS) and what it forbids; how candidate solutions are produced/returned. CRUCIALLY: identify the exact mechanism bounding a plan amount of work, and whether a generated plan can currently do iterative/additional work (loops, follow-ups) or only one-shot fan-out.' },
  { key: 'best-of-n-selection', prompt: 'Read apex_omega/workflows/best_of_n.py. Explain the best-of-N flow, worker_specs cycling across K rollouts, the Cardinal Contract selection/acceptance logic, exactly how a winning rollout is chosen, and whether/where any work BEYOND the flat K one-shot rollouts is possible today. Quote key functions.' },
  { key: 'cell-execution-caps', prompt: 'Read apex_omega/eval/commit0_autogen.py FULLY and relevant parts of commit0_driver.py. Explain run_autogen_cell: roles of autogen-max-agents (8), agent_ceiling (1000), autogen-scout-agents (3); the wave structure (4->6->8); how agents are counted/capped; how acceptance/scoring is gated to the REAL pytest; where workspace-jail / policy-violation enforcement is invoked. MOST IMPORTANT: pinpoint the EXACT lines/conditions where additional work beyond rollouts is capped today, and state the minimal changes to let a plan schedule more agents/iterations (repair loops, follow-up/debug agents) UP TO the ceiling when justified by test signal.' },
  { key: 'engine-primitives', prompt: 'Read apex_omega/engine/{runtime.py, pipeline.py, budget.py}. Explain the agent() primitive, concurrency gate (max_concurrent), journaling/resume, max_total_agents backstop, budget accounting. Then: through which primitives can a generated plan LEGITIMATELY schedule additional agents/iterations within the ceiling, and what would a clean API for a test-driven repair loop look like on top of these primitives?' },
  { key: 'scoring-sanitize-jail', prompt: 'Read apex_omega/eval/{scoring.py, repo_sanitize.py, registry.py}. Then GREP apex/evaluation/commit0_benchmark.py for workspace-jail / policy-violation enforcement and the acceptance/solve decision (search: policy_violation, workspace, jail, /tmp, fetch, allowlist, finalization_status, solved). Explain how solved is decided; how fetch-upstream (policy_violation) is detected/blocked; and WHY env-layer sanitization fails to stop the fetch-upstream shortcut. Propose where a prompt-level / acceptance-level guard would best live.' },
  { key: 'arms-knobs', prompt: 'Read apex_omega/ablation/{arms.py, safety_modes.py} and apex_omega/cli.py. Document EVERY autogen-relevant knob/flag (autogen-max-agents, autogen-author, autogen-scout-agents, rollouts, cell-timeout, agent-mode) and how arms are configured. Distinguish configurable vs hard-coded.' },
  { key: 'design-corpus', prompt: 'Read .apex_plan_sections/{06_sota.md, 09_search.md, 13_verify.md, 14_controller.md, 16_efficiency.md, 17_selfimprove.md, 18_fusion.md} and skim APEX_NEXTGEN_PLAN.md + APEX_DESIGN.md. Summarize what the project ALREADY intends for orchestration/search/verification/efficiency/self-improvement so the new design builds ON it. Flag any already-planned mechanism addressing our failure modes.' },
]

const forensicsTasks = [
  { key: 'jinja', prompt: 'Forensically analyze runs/ladder/omega_autogen_k8__jinja/ (journal/calls_wal.jsonl, journal/diffs/*, narration.jsonl, orchestrator/, cells/). What did the 8 agents do? Count policy_violations vs genuine attempts. For genuine attempts: what diffs, and WHY not accepted (test failures? wrong files? incomplete)? Then read runs/ladder/baseline_v1_k8__jinja/ which SOLVED jinja — what did baseline do that autogen did not? Give the concrete root cause + the fix.' },
  { key: 'mimesis', prompt: 'Forensically analyze runs/ladder/omega_autogen_k8__mimesis/. Characterize the 5/8 fetch-upstream policy_violations and the 3 genuine attempts. Was fetch-upstream baked into the SHARED plan prefix by the architect (monoculture)? Read the chosen strategy. Explain the VARIANCE (solved earlier 535s/5 agents, failed here 875s/8 agents). Give root cause + the precise mechanism to prevent the monoculture and guarantee >=1 in-workspace attempt.' },
  { key: 'pydantic', prompt: 'Forensically analyze runs/ladder/omega_autogen_k8__pydantic/ (Rust-backed pydantic-core). What did the 8 attempts do — did builds/tests run? Is failure a genuine capability gap, an environment/build issue, or strategy? Compare with runs/ladder/{B0_codex_1shot,baseline_v1_k8,omega_template_k8}__pydantic (all failed). State concretely what an orchestration plan must DO to solve pydantic.' },
  { key: 'behavioral-diff', prompt: 'Why does the LEAN template (1 agent solved voluptuous; 25% overall) MATCH the COMPLEX autogen (4 agents; 25% overall) at 1/4 cost? Read apex_omega/autogen/templates.py and compare runs/ladder/omega_template_k8__{repo} vs omega_autogen_k8__{repo} across repos (narration + diffs + agents). What does scout->architect->author ADD that is not paying off, and what genuinely helps? Conclude with a keep/cut list.' },
  { key: 'acceptance-efficiency', prompt: 'Across ALL autogen cells (runs/ladder/omega_autogen_k8__*): when an attempt produced a diff, why accepted or rejected? Trace the Cardinal Contract acceptance path (best_of_n.py + scoring.py) against the WALs. Quantify scout/architect overhead (voluptuous: 4 agents for what template did in 1). Confirm the baseline_v1 mimesis 2400s timeout was a load artifact. Output: acceptance failure modes + efficiency leaks the new design must close.' },
]

const researchTasks = [
  { key: 'swebench-sota', prompt: 'Research SOTA agent architectures for repo-level code fixing / SWE-bench / commit0 (Agentless, SWE-agent, AutoCodeRover, Moatless, OpenHands/CodeAct, SWE-bench leaders). What drives high solve rates (localization, test-driven iteration, structured tool use, scaffolding vs free agent)? ' + RESEARCH_GUIDE },
  { key: 'iterative-repair', prompt: 'Research TEST-DRIVEN ITERATIVE REPAIR / self-debugging loops (Reflexion, Self-Debugging, LDB, AgentCoder, self-repair with execution feedback). This is our additional-work-beyond-rollouts requirement: implement -> run tests -> read failures -> fix -> repeat. Mechanisms, stopping criteria, how many iterations help, diminishing returns. ' + RESEARCH_GUIDE },
  { key: 'search-planning', prompt: 'Research search/planning over candidate solutions for code: LATS, MCTS for code, Tree-of-Thoughts, best-first/beam search guided by test signal, reflection-augmented search. How they beat flat best-of-N and the compute/quality tradeoff. ' + RESEARCH_GUIDE },
  { key: 'multiagent-orchestration', prompt: 'Research multi-agent orchestration / dynamic-workflow patterns: planner-executor, generator-critic, role specialization, debate, blackboard, orchestrator-worker, agentic workflow frameworks. Which reliably improve coding correctness vs add overhead? ' + RESEARCH_GUIDE },
  { key: 'localization-context', prompt: 'Research repo-level fault/edit LOCALIZATION and context retrieval for code agents (which files/functions to edit; embedding/AST/grep retrieval; test-based localization). Strong localization is often the biggest lever on solve rate. ' + RESEARCH_GUIDE },
  { key: 'anti-reward-hacking', prompt: 'Research preventing reward-hacking / spec-gaming / shortcut-taking by code agents, and verification-grounded acceptance. Our instance: agents fetch the OFFICIAL upstream package instead of implementing from tests. Techniques: constraining tool/action space, prompt-level prohibitions, sandbox/jail design, acceptance checks detecting copied/fetched solutions, requiring in-workspace derivation. ' + RESEARCH_GUIDE },
  { key: 'claude-agent-orchestration', prompt: 'Research Anthropic Claude Agent SDK and dynamic-workflow / agentic orchestration patterns (orchestrator-subagent, ensemble + verification, structured outputs, programmatic agent loops, deterministic harnesses). How would these inform a vendor-agnostic orchestrator driving Codex CLI OR Claude Code as worker? ' + RESEARCH_GUIDE },
  { key: 'variance-ensembling', prompt: 'Research variance reduction and test-time compute scaling for code agents: self-consistency, verifier-guided selection, ensembling, reranking by test pass-rate, voting on patches, converting lucky single solves into reliable solves. ' + RESEARCH_GUIDE },
]

const aThunks = []
for (const t of mapTasks) aThunks.push(() => agent(t.prompt + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT, { label: 'map:' + t.key, phase: 'Map' }).then(r => ({ cat: 'map', key: t.key, text: r })))
for (const t of forensicsTasks) aThunks.push(() => agent(t.prompt + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT, { label: 'forensics:' + t.key, phase: 'Forensics' }).then(r => ({ cat: 'forensics', key: t.key, text: r })))
for (const t of researchTasks) aThunks.push(() => agent(t.prompt + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT, { label: 'research:' + t.key, phase: 'Research' }).then(r => ({ cat: 'research', key: t.key, text: r })))

const aResults = (await parallel(aThunks)).filter(Boolean)
const byCat = (c) => aResults.filter(r => r.cat === c)
const fmt = (arr) => arr.map(r => '### [' + r.cat + ':' + r.key + ']\n' + r.text).join('\n\n')
const MAP = fmt(byCat('map'))
const FOR = fmt(byCat('forensics'))
const RES = fmt(byCat('research'))
const EVIDENCE = '# === CODEBASE MAP ===\n' + MAP + '\n\n# === RUN FORENSICS ===\n' + FOR + '\n\n# === EXTERNAL RESEARCH (SOTA) ===\n' + RES
log('Stage A complete: ' + byCat('map').length + ' map + ' + byCat('forensics').length + ' forensics + ' + byCat('research').length + ' research')

// ===================== STAGE B: Design panel -> judge -> synthesize =====================
phase('Design')
const angles = [
  { key: 'tdd-repair-loop', prompt: 'Design angle: TEST-DRIVEN REPAIR LOOPS. Make each unit of work a LOOP (implement -> run real tests -> read failing output -> targeted fix -> repeat until pass or per-attempt budget), not a one-shot best-of-N rollout. Directly realizes additional-work-beyond-rollouts. Specify the loop, stopping criteria, how it uses the real pytest gate as feedback, and how multiple loops diversify + select.' },
  { key: 'verifier-guided-search', prompt: 'Design angle: VERIFIER-GUIDED / BEST-FIRST SEARCH over candidate patches. Expand the most promising partial solutions (ranked by real test pass-rate), reflect on failures, branch. Beyond flat best-of-N. Specify search state, expansion/selection by test signal, budget control, and mapping to ctx.agent()/parallel primitives.' },
  { key: 'anti-shortcut-grounded', prompt: 'Design angle: ANTI-SHORTCUT + GROUNDED ACCEPTANCE (minimal, bulletproof). Center on eliminating the fetch-upstream monoculture: (a) prompt-level strip/prohibition of fetch|clone|download|restore-upstream in generated solver prompts; (b) FORCE >=1 generic in-workspace DEFAULT_ORCHESTRATION-shaped attempt in every fan; (c) acceptance requiring in-workspace derivation (detect/reject fetched solutions). Keep everything else lean.' },
  { key: 'lean-plus-escalation', prompt: 'Design angle: LEAN-FIRST WITH ADAPTIVE ESCALATION. Start from what WORKS (the lean template). Only escalate — fan out, author a custom plan, or enter repair loops — when a cheap default attempt FAILS its tests AND scout difficulty is high. Adaptive test-time compute. Specify the escalation ladder and gates.' },
]
const proposals = (await parallel(angles.map(a => () =>
  agent('You are a principal AI-systems architect. Produce a concrete design proposal from this angle:\n\n' + a.prompt + '\n\nREQUIREMENTS: address EVERY verified failure mode (fetch-shortcut, variance, complexity-not-paying-off); SUPPORT additional work beyond rollouts; be VENDOR-AGNOSTIC (Codex CLI + Claude Code); keep execution-authoritative scoring; map concretely onto existing code (name files/functions to change or add). Ground every choice in the evidence.\n\n' + EVIDENCE + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT,
    { label: 'design:' + a.key, phase: 'Design' }).then(r => ({ key: a.key, text: r }))
))).filter(Boolean)

const judged = (await parallel(proposals.map(p => () =>
  agent('Critically score this design proposal for the APEX-Omega autogen redesign. Be a tough, calibrated judge.\n\nPROPOSAL [' + p.key + ']:\n' + p.text + '\n\nScore each dimension 1-10 and identify the best ideas worth grafting into a unified design. Evidence:\n' + EVIDENCE,
    { label: 'judge:' + p.key, phase: 'Design', schema: JUDGE_SCHEMA }).then(v => Object.assign({ key: p.key }, v))
))).filter(Boolean)

const scoreboard = judged.map(j => '[' + j.key + '] total=' + j.total + ' scores=' + JSON.stringify(j.scores) + ' strengths=' + JSON.stringify(j.strengths) + ' weaknesses=' + JSON.stringify(j.weaknesses) + ' graft=' + JSON.stringify(j.best_ideas_to_graft)).join('\n')
const proposalsText = proposals.map(p => '## PROPOSAL [' + p.key + ']\n' + p.text).join('\n\n')

const design = await agent(
  'You are the lead architect. Synthesize ONE unified, implementable design for the APEX-Omega autogen orchestrator that will RELIABLY solve all runnable commit0 repos (voluptuous, jinja, mimesis, pydantic) and generalize toward the full set. Take the highest-scoring proposal as the spine and GRAFT the best ideas from the others. The design MUST: (1) eliminate the fetch-upstream monoculture (prompt strip + forced in-workspace attempt + grounded acceptance); (2) reduce variance into reliable solves (diversity + verifier-guided selection + repair loops); (3) cut complexity that does not pay off while keeping what helps; (4) FULLY support additional work beyond flat rollouts — test-driven repair loops and plan-driven escalation up to the agent ceiling, with concrete stopping criteria; (5) stay vendor-agnostic and execution-authoritative. Describe the end-to-end flow, every mechanism, and map each to specific files/functions to change or add.\n\nPROPOSALS:\n' + proposalsText + '\n\nJUDGE SCOREBOARD:\n' + scoreboard + '\n\nEVIDENCE:\n' + EVIDENCE + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT,
  { label: 'synthesize-design', phase: 'Design' }
)
log('Stage B complete: unified design synthesized')

// ===================== STAGE C: per-repo adversarial validation =====================
phase('Validate')
const repos = ['voluptuous', 'jinja', 'mimesis', 'pydantic']
const lenses = [
  { key: 'will-it-solve', ask: 'Will this design ACTUALLY solve this repo end-to-end? Walk the design through this repo step by step using the forensic evidence. Where could it still fail?' },
  { key: 'what-breaks', ask: 'Adversarially attack the design for this repo: what specific failure mode (fetch-shortcut recurrence, build/env issue, localization miss, acceptance gap, variance) could make it fail here? Default skeptical.' },
  { key: 'regression-cheap-wins', ask: 'Does this design REGRESS the cheap wins (voluptuous solved in 1 agent)? Does added machinery risk breaking what already works or blow the agent budget for this repo?' },
]
const validations = (await parallel(repos.flatMap(repo => lenses.map(l => () =>
  agent('Adversarially validate the proposed design for repo="' + repo + '" through the lens: ' + l.ask + '\n\nBe concrete and skeptical; ground in the forensics. Give a calibrated 0-100 confidence this design solves ' + repo + ', a verdict, remaining gaps, and concrete design changes that would raise confidence.\n\nDESIGN:\n' + design + '\n\nFORENSICS:\n' + FOR + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT,
    { label: 'val:' + repo + ':' + l.key, phase: 'Validate', schema: VALIDATION_SCHEMA }).then(v => Object.assign({}, v, { repo: repo, lens: l.key }))
)))).filter(Boolean)
const validationsText = validations.map(v => '[' + v.repo + ' / ' + v.lens + '] verdict=' + v.verdict + ' conf=' + v.will_solve_confidence_pct + '% gaps=' + JSON.stringify(v.remaining_gaps) + ' needed=' + JSON.stringify(v.required_design_changes || []) + ' :: ' + v.reasoning).join('\n')
log('Stage C complete: ' + validations.length + ' adversarial validations')

// ===================== STAGE D: implementation plan + critic + final =====================
phase('Plan')
const planAndCritique = await parallel([
  () => agent('Produce a FILE-LEVEL implementation plan for the design below, ready for an engineer to execute against the codebase at ' + ROOT + '. Include: ordered steps; the EXACT files/functions to change and NEW modules to add (use the codebase map for accuracy); DRAFTED code for the critical pieces — (a) anti-fetch-upstream guard + forced in-workspace fallback, (b) the test-driven repair loop / additional-work scheduler that lets a plan do work beyond flat rollouts up to the ceiling, (c) verifier-guided candidate selection; how it stays vendor-agnostic; the unit/integration test plan (pytest under tests/, currently 76 passing); and the exact resumable validation-run command (scripts/run_ladder.py, env APEX_OMEGA_PYTHON + LADDER_CONCURRENCY).\n\nDESIGN:\n' + design + '\n\nCODEBASE MAP:\n' + MAP + '\n\nVALIDATION FINDINGS (close these gaps):\n' + validationsText + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT,
    { label: 'impl-plan', phase: 'Plan' }),
  () => agent('Completeness critic. Given the design and the adversarial validations, what is MISSING or UNDERSPECIFIED? Identify: any verified failure mode left unaddressed; any repo with low confidence and why; any hand-wavy mechanism needing concretizing; any unverified claim and how to verify it; any way the additional-work-beyond-rollouts requirement is not fully honored. Be specific and harsh.\n\nDESIGN:\n' + design + '\n\nVALIDATIONS:\n' + validationsText + '\n\nEVIDENCE:\n' + EVIDENCE,
    { label: 'completeness-critic', phase: 'Plan' }),
])
const plan = planAndCritique[0] || ''
const critique = planAndCritique[1] || ''
log('Stage D: producing calibrated final deliverables')

const final = await agent(
  'You are the lead architect delivering the FINAL package. Integrate the design, the file-level plan, the adversarial validations, and the completeness critique into the final deliverables. Revise the design/plan to close the gaps raised. Be CALIBRATED AND HONEST about confidence — do not overclaim; give per-repo confidence with justification and state exactly what empirical validation (a real run) is still required. The design MUST honor: anti-fetch-shortcut, variance->reliability, lean-where-possible, FULL support for additional work beyond rollouts (test-driven repair loops + plan-driven escalation up to the ceiling), vendor-agnostic, execution-authoritative scoring.\n\nUNIFIED DESIGN:\n' + design + '\n\nFILE-LEVEL PLAN:\n' + plan + '\n\nADVERSARIAL VALIDATIONS:\n' + validationsText + '\n\nCOMPLETENESS CRITIQUE:\n' + critique + '\n\n=== PROJECT CONTEXT ===\n' + CONTEXT,
  { label: 'final-synthesis', phase: 'Plan', schema: FINAL_SCHEMA }
)

return final
