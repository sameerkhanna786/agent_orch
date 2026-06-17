# Anthropic Dynamic Workflows: Expanded Authoritative Guide & Build Template

**Author:** Manus AI
**Date:** June 17, 2026

This document is the definitive, expanded guide to Anthropic's dynamic workflow orchestration in Claude Code (v2.1.154+). It integrates the complete conceptual context, precise runtime mechanics, UI/UX controls, the agent-teams comparison, and a comprehensive build-ready template derived from practitioner experience [1] [2] [3] [4].

---

## Part 1: Full Context & Architecture

### 1.1 The Orchestration Philosophy
Anthropic introduced dynamic workflows in May 2026 to solve the limitations of a single context window. When a single agent attempts a massive task, it encounters three structural failure modes [3] [5]:
1.  **Agentic Laziness:** Stopping after partial progress (e.g., fixing 35 of 50 files).
2.  **Self-Preferential Bias:** Grading its own work too generously.
3.  **Goal Drift:** Losing edge-case constraints over long, compacted contexts.

Dynamic workflows solve this by moving orchestration out of the model's context window and into a deterministic JavaScript script executed by a background `node:vm` runtime [2] [5]. Claude writes the script, and the runtime executes it, spawning up to 1,000 ephemeral subagents. Intermediate results live in script variables, keeping the main session's context perfectly clean [2]. 

This creates a **zero-token orchestration** layer. In a benchmark 100-interview synthesis run, 113 agents consumed 1.95M tokens to do the reasoning, but the JavaScript code that routed, scored, and merged them spent exactly zero model tokens [3].

### 1.2 "Who Holds the Plan?" Comparison
Understanding Anthropic's orchestration primitives requires asking "who holds the plan" [1] [6] [7]:

| Primitive | Who holds the plan? | Communication | Best for |
| :--- | :--- | :--- | :--- |
| **Subagents** | Main Claude (orchestrator) | Report results back to main agent only | Focused tasks where only result matters |
| **Agent Teams** | The peers, between them | Teammates message each other directly | Complex work requiring discussion |
| **Dynamic Workflows** | A JavaScript program | Agents work in script variables; final result returns | Dozens to hundreds of agents per run |

**Static vs. Dynamic Harnesses:** You can build a *static* harness yourself using the Agent SDK for embedded agents you ship in your app. A *dynamic* harness is the reverse: Claude (powered by Opus 4.8) writes it in the moment for workspace tasks, tailor-made and disposable, thrown away when done unless explicitly saved [4].

### 1.3 Triggers and the `ultracode` Setting
Workflows are triggered via several paths [1] [5] [8]:
1.  **Keyword Trigger:** Including `ultracode` in a prompt forces Claude to write a workflow for that specific task. The UI highlights this; dismiss it with `Option+W` / `Alt+W`.
2.  **`/effort ultracode`:** A session-scoped setting combining `xhigh` reasoning with automatic workflow orchestration. Claude autonomously plans workflows for substantive tasks. It resets on a new session.
3.  **Bundled Workflows:** Using built-in commands like `/deep-research <question>`.
4.  **Saved Commands:** Successful runs can be saved (press `s` in the `/workflows` view) as custom slash commands (e.g., `/<name>`), stored in `.claude/workflows/` (project scope) or `~/.claude/workflows/` (user scope) [9].

*(Note: The `ultrathink` keyword is distinct; it sets the thinking token budget to 31,999 for deeper reasoning on a single turn, but does not trigger a workflow [10].)*

---

## Part 2: Runtime Mechanics & Script API

### 2.1 The Script API Surface
The runtime injects core primitives and helpers into the script environment [2]:

*   **`agent(prompt, opts?)`**: Spawns a single subagent. Returns the final text string, or a validated object if a schema is provided [2].
    *   `schema`: A JSON Schema object. Under the hood, this uses a structured-output tool with `terminate: true`. Validation happens at the tool-call layer, with up to 2 automatic nudges on failure before throwing [22].
    *   `isolation: "worktree"`: Runs the agent in an isolated git worktree to prevent parallel write conflicts. Worktrees are created at `.claude/worktrees/<value>/` on a branch named `worktree-<value>`. You can copy untracked files automatically by creating a `.worktreeinclude` file, or configure `worktree.baseRef: "head"` to branch from unpushed commits [23].
    *   `model`: Overrides the session model (e.g., `'opus'`, `'fable'`, or routing cheap triage to Haiku) [2].
    *   `phase`: Assigns the agent to a UI progress group to avoid racing on the global `phase()` state [2].
    *   `label`: The display name shown in the UI [2].
    *   `agentType`: Uses a custom subagent type instead of the default workflow agent [2].
*   **`pipeline(items, ...stages)`**: Streams items through stages independently with **no barrier**. Item A can enter stage 3 while item B is in stage 1. Each stage callback receives `(prevResult, originalItem, index) => Promise<any>`. If a stage returns `null` or an array, that exact value becomes the `prevResult` for the next stage [2] [24].
*   **`parallel(thunks)`**: Executes tasks concurrently with a **barrier**, waiting for all to complete. You must pass an array of thunks (`() => agent(...)`), not promises, so the runtime can control concurrency. Failed thunks resolve to `null` rather than rejecting the call, so results should always be filtered via `.filter(Boolean)` [2] [25].
*   **`workflow(nameOrRef, args?)`**: Composes workflows by running another inline (limited to one level deep). Accepts either a string name (`"deep-research"`) or a file reference (`{ scriptPath: "./path/to/script.js" }`). The sub-workflow inherits the parent's concurrency cap, agent counter, and token budget [2] [26].
*   **Helpers:**
    *   `phase(title)`: Starts a progress group in the UI.
    *   `log(msg)`: Emits a narrator line to the UI.
    *   `args`: Carries the JSON arguments passed when launching a saved workflow (e.g., `Run /triage on issues 1024 and 1025`).
    *   `budget`: Exposes the token target (`budget.total`). It is `null` if launched without a target, so guard any loop-until-budget on `budget.total` or it will run to the agent cap.

### 2.2 Determinism, Caching, and Error Recovery
To guarantee resumability, the runtime journals every `agent()` call. When a script crashes, is paused, or is edited and rerun, the runtime replays the **longest unchanged prefix** from the cache [2] [11]. Editing a script invalidates the cache only for the modified sections and subsequent calls [27]. Consequently, **non-deterministic functions throw errors** inside workflows (e.g., `Date.now()`, `Math.random()`, argless `new Date()`, and network APIs) [2] [22]. If you need a timestamp, pass it through `args`.

**Bounded Repair Loops:** A critical failure mode in orchestration is the unbounded loop. To mitigate this, dynamic workflows employ bounded repair loops with explicit round caps (e.g., 100 rounds for sharding, 80 for compilation). When an agent encounters a blocking issue it cannot resolve within the cap, it utilizes an explicit IOU mechanism, writing `todo!("blocked_on: X::Y")`. This sentinel defers resolution to a downstream phase, ensuring loops terminate [28].

**Exact Limits:**
*   **Concurrency:** Capped at `min(16, os.cpus().length - 2)` per workflow. Excess agents queue automatically [2].
*   **Total Agents:** Hard cap of 1,000 agents per run [2] [5].
*   **Context Compaction Errors:** Extremely long `ultracode` runs can hit context limits and fail to auto-compact, throwing an error at `/$bunfs/root/src/entrypoints/cli.js` [17].
*   **Nesting:** `workflow()` calls are limited to one level deep [2].

### 2.3 UI and Control (`/workflows`)
When a workflow is requested, the launch prompt offers options including "1. Yes, run it" and "3. View raw script" to inspect the JS before execution [12].

Running `/workflows` opens the progress view [1]:
*   `Enter` / `→`: Drill into a phase or agent to read prompts and tool calls.
*   `p`: Pause or resume the run.
*   `x`: Stop the selected agent or the whole workflow.
*   `r`: Restart a running agent.
*   `s`: Save the script.

---

## Part 3: Security, Cost, and Failure Modes

### 3.1 Security and Permissions
Subagents spawned by a workflow always run in `acceptEdits` mode (file edits auto-approve) and inherit the user's tool allowlist [5]. Tools not on the allowlist will pause the run for approval. 

**The Quarantine Pattern:** To defend against prompt injection when processing untrusted content, bar agents reading that content from taking high-privilege actions (like Bash or Write). *Warning: GitHub Issue #63762 notes that dynamic workflows currently ignore a subagent's declared `tools:` allowlist and always grant Write/Edit. Use global PreToolUse hooks as a stopgap [13].*

**Subagent `memory` and Process Hangs:** Custom subagents (defined via YAML) support a `memory` field (`project`, `user`, or `local`) that creates a persistent directory and auto-injects the first 200 lines of `MEMORY.md` into the system prompt [18]. Be aware that subagents chaining local shell processes (e.g., PowerShell) can hang, spiking host RAM and potentially crashing the machine [19].

### 3.2 Cost Compounding and Mitigation
Workflows pay the base prompt and context cost for every agent spawned. In a fan-out pattern, this compounds linearly [14]. Furthermore, mistakes do not stop the clock: a workflow that hits a snag may spend 5x more tokens recovering, and parallel exploratory branches that are discarded are already paid for [15].

**Mitigation:** Route simpler tasks to Claude 3.5 Haiku using the `model` option in `agent()`. Aggressively prune the context window passed to each agent (e.g., pass only AST nodes, not the whole file) [5] [14]. *Note: Fable 5 with `ultracode` has been reported to spawn excessive parallel agents (e.g., 7 for a single task), consuming massive tokens compared to Opus 4.8 [20].*

### 3.3 Five Common Practitioner Mistakes
Practitioners consistently highlight five anti-patterns when deploying dynamic workflows [15]:
1.  **Using workflows for repetitive known-structure tasks:** Build a static pipeline instead.
2.  **Not setting token budgets:** A dynamic workflow without step or tool-call circuit breakers can run indefinitely on an edge case.
3.  **Trusting the model to know when it is done:** Add explicit completion conditions or a separate verification call.
4.  **Skipping the minimal footprint principle:** Autonomous workflows accumulate permissions; prefer reversible actions.
5.  **Building dynamic before static:** Always validate a static version of the task first before handing orchestration to the model.

---

## Part 4: Build-Ready Templates and Patterns

To build robust workflows that scale without accruing "slop debt," orchestration must follow the **Fan Out $\rightarrow$ Reduce $\rightarrow$ Synthesize** shape, incorporating verification anchored on deterministic ground truth [2] [16].

### 4.1 The `pipeline` vs `parallel` Decision Rule
This distinction is critical for performance [2]:
*   **Default to `pipeline()`** for multi-stage work so fast items don't wait for slow ones.
*   **Reach for `parallel()` ONLY when a downstream stage needs *all* prior results at once** (e.g., global deduplication, early-exit on totals, or cross-item comparison). If a transform stage has no cross-item dependency, it should be a pipeline.

### 4.2 Quality Patterns
The leverage of dynamic workflows comes from repeatable quality patterns [2]:
*   **Adversarial verify:** For each finding, spawn N independent skeptics prompted to *refute* it. Kill it unless a majority survive.
*   **Perspective-diverse verify:** Give each verifier a distinct lens (correctness, security, performance) instead of N identical ones.
*   **Judge panel:** Generate N attempts from different angles, score with parallel judges, and synthesize from the winner.
*   **Loop-until-dry:** For unknown-size discovery, keep spawning finders until K consecutive rounds surface nothing new. *Critical detail: dedupe against everything SEEN, not just confirmed results, or rejected findings reappear and the loop never converges.* In the Bun rewrite, repair loops were explicitly bounded (e.g., 100 rounds for crate sharding) and used `todo!("blocked_on: X::Y")` for unresolved items to ensure the loop terminated [21].

### 4.3 A Shipped Example: `/deep-research`
The built-in `/deep-research` command is a production example of these patterns. Its decoded script reveals it was ported from Anthropic's internal "bughunter" architecture, replacing git/grep with WebSearch/WebFetch [12]. It runs in five phases [1] [2] [12]:
1.  **Scope:** One agent decomposes the question into 3-6 distinct search angles.
2.  **Search:** Web searches run in parallel (fan-out) via `pipeline()`.
3.  **Fetch:** Dedupe URLs across angles, fetch top sources (max 15), extract claims (plain JS reduce).
4.  **Verify:** Each claim gets an adversarial three-vote check (`VOTES_PER_CLAIM = 3`). Skeptics try to refute it; a claim survives only if `refuted < 2`.
5.  **Synthesize:** One final agent writes the cited report from claims that held up.

### 4.4 The Blueprint Scripts

#### Pattern A: Deep Research Pipeline (Decoded)
This is the structural blueprint of the bundled `/deep-research` command, demonstrating how to mix `pipeline` and `parallel`, use strict schemas, and manage a fetch budget [12] [24].

```javascript
// 1. SCHEMAS (Enforce exact phase boundaries)
const SCOPE_SCHEMA = {
  type: "object",
  required: ["angles"],
  properties: {
    angles: {
      type: "array", minItems: 3, maxItems: 6,
      items: {
        type: "object", required: ["label", "query"],
        properties: { label: { type: "string" }, query: { type: "string" } }
      }
    }
  }
};

const VERDICT_SCHEMA = {
  type: "object", required: ["refuted", "evidence", "confidence"],
  properties: {
    refuted: { type: "boolean" },
    evidence: { type: "string" },
    confidence: { enum: ["high", "medium", "low"] }
  }
};

// 2. BUDGETING & DEDUP HELPERS
const normURL = (u) => {
  try { const p = new URL(u); return (p.hostname.replace(/^www\./, "") + p.pathname.replace(/\/$/, "")).toLowerCase(); }
  catch { return u.toLowerCase(); }
};
const seen = new Map();
let fetchSlots = 15;

// 3. THE WORKFLOW
phase("Scope");
const scope = await agent(`Decompose this question: ${args.question}`, { schema: SCOPE_SCHEMA });

// Pipeline: Search -> URL Dedup -> Fetch & Extract (streams items independently)
const allClaims = await pipeline(
  scope.angles,
  async (angle) => await agent(`Search: ${angle.query}`, { phase: "Search" }),
  async (searchResult) => {
    // Plain JS dedup step inside the pipeline
    const novel = searchResult.urls.filter(u => {
      const n = normURL(u);
      if (seen.has(n) || fetchSlots <= 0) return false;
      seen.set(n, true);
      fetchSlots--;
      return true;
    });
    
    // Nested parallel fan-out for the novel URLs
    return await parallel(novel.map(url => async () => 
      await agent(`Fetch and extract claims from ${url}`, { phase: "Fetch" })
    ));
  }
);

// 4. BARRIER & VERIFY (Wait for all claims, rank, then verify)
phase("Verify");
const rankedClaims = allClaims.flat().filter(Boolean).slice(0, 25);

const verified = await parallel(rankedClaims.map(claim => async () => {
  // 3-vote adversarial check
  const verdicts = await parallel([1,2,3].map(() => async () => 
    await agent(`Refute this claim: ${claim.text}`, { schema: VERDICT_SCHEMA, phase: "Verify" })
  ));
  
  const valid = verdicts.filter(Boolean);
  const refutedCount = valid.filter(v => v.refuted).length;
  // Survival rule: needs enough votes, and < 2 refutations
  if (valid.length >= 2 && refutedCount < 2) return claim;
  return null;
}));

phase("Synthesize");
return await agent(`Write report from these verified claims: ${JSON.stringify(verified.filter(Boolean))}`);
```

#### Pattern B: Codebase Audit (Fan-out to Reduce)
A complete, annotated template for a codebase-wide audit that isolates writers using `worktree`.

```javascript
// 1. THE META BLOCK (Must be a pure literal)
export const meta = {
  name: "robust-codebase-audit",
  description: "Fan out across files, adversarially verify findings, and synthesize.",
  whenToUse: "Run an audit on the codebase. Pass args {targetDir} to scope it.",
  phases: [
    { title: "Discovery", detail: "Parallel sweep across files" },
    { title: "Verification", detail: "Adversarial challenge of findings" },
    { title: "Synthesis", detail: "Merge confirmed findings" }
  ]
};

// Handle optional args
const targetDir = (args && args.targetDir) || "src/";

// 2. DISCOVERY PHASE (Fan Out)
phase('Discovery');
const files = await agent(`List all relevant target files in ${targetDir} as a JSON array`, {
  schema: { type: 'array', items: { type: 'string' } },
  label: 'file-discovery'
});

// Use pipeline for streaming execution without barriers
const rawFindings = await pipeline(
  files,
  async (file) => {
    return await agent(`Audit ${file} for issues based on the rubric.`, {
      schema: { 
        type: 'object', 
        properties: { 
          hasIssue: { type: 'boolean' },
          details: { type: 'string' }
        }
      },
      phase: 'Discovery',
      label: `audit-${file}`
    });
  }
);

// 3. REDUCE (Plain JavaScript, no agents)
const candidateFindings = rawFindings
  .filter(f => f && f.hasIssue)
  .map(f => f.details);

// 4. VERIFICATION PHASE (Adversarial Verification)
phase('Verification');
// Use parallel here because the next step (Synthesis) needs all verified results
const verifiedFindings = await parallel(
  candidateFindings.map(finding => async () => {
    const verification = await agent(
      `ADVERSARIAL REVIEW: Attempt to refute this finding. Is it a false positive? \n\nFinding: ${finding}`,
      {
        schema: {
          type: 'object',
          properties: {
            isFalsePositive: { type: 'boolean' },
            reasoning: { type: 'string' }
          }
        },
        phase: 'Verification',
        label: 'adversarial-verifier'
      }
    );
    
    // Only return findings that survived the adversarial challenge
    if (verification && !verification.isFalsePositive) {
      return finding;
    }
    return null;
  })
);

// Filter out nulls from failed thunks or false positives
const confirmedFindings = verifiedFindings.filter(Boolean);

// 5. SYNTHESIS PHASE
phase('Synthesis');
const finalReport = await agent(
  `Synthesize these confirmed findings into a final markdown report: \n\n${JSON.stringify(confirmedFindings)}`,
  {
    phase: 'Synthesis',
    label: 'report-writer'
  }
);

return finalReport;
```

***

### References

[1] Anthropic. "Orchestrate subagents at scale with dynamic workflows." Claude Code Documentation. https://code.claude.com/docs/en/workflows
[2] AlexOp. "Claude Code Workflows: Deterministic Multi-Agent Orchestration." May 28, 2026. https://alexop.dev/posts/claude-code-workflows-deterministic-orchestration/
[3] Huryn, Paweł. "Claude Dynamic Workflows for PMs: The Ultimate Guide." Product Compass. June 7, 2026. https://www.productcompass.pm/p/claude-code-dynamic-workflows
[4] Vu Minh, Chien. "A Harness for Every Task: Putting a Team of Claudes on One Job." Towards Data Science. June 12, 2026. https://towardsdatascience.com/a-harness-for-every-task-putting-a-team-of-claudes-on-one-job/
[5] Lushbinary Team. "A Harness for Every Task: Claude Code Dynamic Workflows Explained." June 3, 2026. https://lushbinary.com/blog/claude-code-harness-every-task-dynamic-workflows-guide/
[6] Anthropic. "Create custom subagents." Claude Code Documentation. https://code.claude.com/docs/en/sub-agents
[7] Anthropic. "Orchestrate teams of Claude Code sessions." Claude Code Documentation. https://code.claude.com/docs/en/agent-teams
[8] Claude Fast. "Ultracode in Claude Code: What It Actually Does." June 2026. https://claudefa.st/blog/guide/development/ultracode
[9] Anthropic. "Explore the .claude directory." Claude Code Documentation. https://code.claude.com/docs/en/claude-directory
[10] Hacker News. "Ultrathink is a Claude Code magic word." March 2026. https://news.ycombinator.com/item?id=43739997
[11] Claude Fast. "Dynamic Workflows in Claude Code: How the Harness Actually Works." June 9, 2026. https://claudefa.st/blog/guide/development/dynamic-workflows
[12] Azukiazusa. "Trying Dynamic Workflow in Claude Code." May 29, 2026. https://azukiazusa.dev/en/blog/claude-code-dynamic-workflow
[13] GitHub. "Dynamic workflows ignore the subagent tools: allowlist (always grant Write/Edit) #63762." https://github.com/anthropics/claude-code/issues/63762
[14] Awesome Claude. "Claude Code Dynamic Workflows — Guide + 24 Copy-Paste Scripts." June 2026. https://awesomeclaude.ai/claude-code-workflows
[15] MindStudio Team. "Anthropic Dynamic Workflows: What Everyone Gets Wrong About When to Use Them." June 3, 2026. https://www.mindstudio.ai/blog/anthropic-dynamic-workflows-when-to-use-them
[16] Hacker News. "Dynamic Workflows in Claude Code." May 27, 2026. https://news.ycombinator.com/item?id=48311705
[17] GitHub. "Ultracode context compaction error #63848." https://github.com/anthropics/claude-code/issues/63848
[18] Vectorize. "Claude Code Subagents Shared Memory." May 6, 2026. https://hindsight.vectorize.io/blog/2026/05/06/claude-code-subagents-shared-memory
[19] GitHub. "Subagents chaining local shell processes hang #63661." https://github.com/anthropics/claude-code/issues/63661
[20] GitHub. "Fable 5 Ultracode excessive token consumption #66867." https://github.com/anthropics/claude-code/issues/66867
[21] GitHub. "Bun PR #30412 Rewrite Methodology." https://gist.github.com/michaellady/7d552137fb1e37ab9bf637e450016c25
[22] Michael Liv. "pi-dynamic-workflows." GitHub. https://github.com/michaelliv/pi-dynamic-workflows
[23] Anthropic. "Worktrees." Claude Code Documentation. https://code.claude.com/docs/en/worktrees
[24] Claude Fast. "Dynamic Workflows in Claude Code: How the Harness Actually Works." https://claudefa.st/blog/guide/development/dynamic-workflows
[25] Piebald AI. "tool-description-workflow.md." GitHub. https://github.com/Piebald-AI/claude-code-system-prompts/blob/main/system-prompts/tool-description-workflow.md
[26] Build This Now. "Claude Code Dynamic Workflows." https://www.buildthisnow.com/blog/guide/development/claude-code-dynamic-workflows
[27] Agdal Tech. "Opus 4.8 Ships Dynamic Workflows." https://agdal.tech/article.php?slug=opus-48-ships-dynamic-workflows-hundreds-of-parallel-subagents-per-session-read-this-before-you-wire-it-into-prod
[28] Michael Lady. "Bun Zig-to-Rust Rewrite Methodology." GitHub Gist. https://gist.github.com/michaellady/7e63223d5d72d9ad18a03efa1f376aae
