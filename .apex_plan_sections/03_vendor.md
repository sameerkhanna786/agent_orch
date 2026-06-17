## 3. Vendor-Agnostic Execution: Codex, Claude Code, or Both

APEX-Ω runs with Codex (`codex_cli`), Claude Code (`claude_cli`), or **both in one run** — and ideally any agent CLI/API (`gemini_cli`, `opencode_cli`, `openai_api`). This is a hard requirement, not a nicety. The dynamic-workflow paradigm (see Section 2) is a *concept* — orchestration-as-code spawning subagent workers — and the Claude Code Workflow tool is only one implementation of it. APEX-Ω owns a vendor-neutral orchestration engine; vendors are merely the **leaf workers** that the engine fans out, pipelines, and refutes against each other. This section specifies the normalized Executor interface, the per-vendor adapters with concrete flags, the ACP-style capability-negotiation handshake, the heterogeneous-fleet routing model, the cost-arbitrage cascade, and the run-manifest pinning that makes cross-vendor runs replayable.

The load-bearing claim — and the reason this is feasible today — is that **the filesystem (and the git diff over it) is the source of truth.** APEX v1 already verifies acceptance on the resulting diff regardless of which vendor produced it (`LLMBackend` already spans `claude_cli/codex_cli/gemini_cli/opencode_cli/metacode_cli/openai_api`). The adversarial verdict on this claim is `sound_with_caveats` (high confidence): vendor neutrality is genuinely feasible, but "without losing the paradigm benefits" is an overclaim. The honest framing is **vendor-neutral with bounded, declared degradation**, never lossless portability. We build to that truth.

### 3.1 The Core Invariant: Filesystem/Git-Diff as Contract, JSON Events as Telemetry

Every vendor's JSON event vocabulary differs — Codex emits `item.completed` with item types (agent message, reasoning, command exec, file change, MCP tool call, web search, plan update); Claude emits `tool_use` inside `stream-json` NDJSON; Gemini emits a `stats` object (per-model tokens, per-tool `totalCalls/Success/Fail`, files +/-). But **all of them mutate the working tree**, and APEX verifies on the resulting git diff. This is independently confirmed by both the v1 paradigm ingest ("filesystem/git diff as the source of truth — vendor-neutrality enabler") and the vendor SOTA research ([FILESYSTEM-AS-TRUTH is the real enabler... Treat JSON streams as telemetry, not as the contract](https://developers.openai.com/codex/noninteractive)).

The consequences are precise and non-negotiable:

- **Executor parsing stays best-effort / observational.** Correctness *never* depends on trusting a vendor's self-reported output. This directly honors the pitfall "Do not trust vendor self-reported JSON as the correctness contract." A vendor that reports `success: true` but produced a diff that fails verification is a failure, full stop.
- **Structured output (when present) is a convenience, re-validated downstream.** Claude's native `--json-schema` → `structured_output` and Codex's `--output-schema` are accepted, but APEX re-parses and re-validates the returned object at the engine layer (schema validation at the tool layer, model retries on mismatch — exactly v1's `run_structured_prompt` contract). For vendors with no native schema (Gemini, opencode), APEX embeds the schema in the prompt and post-parses (§3.4).
- **Replay reproduces artifacts, not token streams.** Because temperature-0 is not bitwise reproducible across hosted APIs ([batch non-invariance, Thinking Machines, Sep 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): 80 distinct completions per 1000 identical temp-0 prompts), APEX replays recorded diffs and re-runs verification (§3.7). This is already aligned with v1's diff-verification; the accepted mechanism "Bit-reproducible agent OUTPUT replay" is `reject`, and we honor that.

This invariant is what makes the harness-dominates-model threat survivable. [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0) shows 30–50pt same-model swings across harnesses; a normalization-leaky executor can erase the entire cross-vendor diversity gain. Because correctness lives in the diff, a parsing bug degrades *telemetry*, not *verdicts* — but flag/sandbox/schema mapping (§3.3) is still load-bearing and gets a conformance test (§3.8).

### 3.2 The Normalized Executor Interface

The Executor is the canonical generalization of v1's `CLIModelClient.run_structured_prompt` (cli_backend.py) — the existing multi-vendor `agent()` primitive that already returns a normalized `CLIModelResult` and **never raises to the caller** (every abnormal exit becomes a typed result). The accepted mechanism "Normalized Executor + ACP-style capability negotiation" is `adopt`: consolidate v1's scattered per-vendor fragments into one interface; degrade-not-crash.

```python
# Data structures (engine-internal; field types are normative)

@dataclass(frozen=True)
class CapabilityProfile:
    vendor: str                       # "codex_cli" | "claude_cli" | "gemini_cli" | "opencode_cli" | "openai_api"
    model: str                        # human alias, e.g. "opus" | "gpt-5.5" — resolved to launcher id at command-build time
    cli_version: str                  # npm-resolved exact version, e.g. "@openai/codex@0.140.2"
    internet: bool                    # web-search/internet mode available
    native_schema: bool               # native structured-output (Claude --json-schema, Codex --output-schema)
    sandbox_levels: tuple[str, ...]   # e.g. ("read-only","workspace-write","danger-full-access") | ("yolo",)
    thinking: str                     # "effort:low..max" | "extended" | "none"
    bidirectional_stream: bool        # only Claude --input-format stream-json documented
    tool_interception: str            # "pre-tool-hook" | "none" — and known_interception_gaps
    mcp: bool                         # accepts an injected MCP server set

@dataclass(frozen=True)
class ScopedTask:
    prompt: str
    schema: dict | None               # JSON schema for structured_output, if any
    allowed_tools: tuple[str, ...]    # restricted tool allowlist for this worker
    sandbox: str                      # requested sandbox level (normalized to APEX floor on degradation)
    effort: str                       # "low".."max"
    mcp_servers: tuple[McpServerRef, ...]
    cwd: str                          # the worktree path (isolation is APEX-owned; see §3.4)
    label: str; phase: str            # narration (maps to phase()/log())

@dataclass(frozen=True)
class ExecResult:
    final_message: str
    structured_output: dict | None    # re-validated by engine, NOT trusted as correctness
    usage: Usage                      # input/cached_input/output/reasoning tokens, normalized cross-vendor
    session_id: str | None            # for vendor-native resume (codex exec resume; claude session_id)
    raw_events: list[dict]            # best-effort parsed NDJSON; telemetry only
    finalization_status: str          # "completed"|"timeout"|"policy_violation"|"output_limit"|"progress_abort"|"isolation_error"
    fs_diff: GitDiff                  # observe(): the authoritative artifact

class Executor(Protocol):
    def negotiate(self) -> CapabilityProfile: ...        # ACP-style initialize; §3.5
    def spawn(self, cwd: str) -> "Session": ...          # bind to a worktree
class Session(Protocol):
    def run(self, task: ScopedTask) -> ExecResult: ...   # == agent(); returns structured result + observes diff
    def observe(self) -> GitDiff: ...                    # git diff over the worktree (the contract)
    def resume(self, session_id: str) -> "Session": ...  # vendor-native resume where available; else replay (§3.7)
```

The Executor lifecycle is exactly: `spawn(worktree_cwd) -> session.run(ScopedTask) -> {final_message, structured_output?, usage, session_id, raw_events} + observe(git diff)`. This is `agent()` in the paradigm sense, now first-class multi-vendor. Three of four vendors expose NDJSON event streams (Codex `--json`, Claude `--output-format stream-json`, Gemini `--output-format stream-json`), so **one NDJSON reader with per-vendor event-type maps covers three of four**; opencode normalizes via its OpenAPI server or `serve acp`.

Reuse mandate (accepted mechanism "Reuse v1's cli_backend.py/llm_routing.py/backend_portfolio.py/cli_turn_parser.py"): the Executor wraps, not replaces, v1's machinery. `cli_turn_parser.py` (`CLITurnParser`) remains the NDJSON/turn splitter feeding `raw_events` and the `turn_observer` mid-flight steering channel. The S1–S7 progress watchdog (cli_backend.py) governs liveness — **progress-based, never wall-clock** (Section 15) — so a long legitimate agentic turn on any vendor is not false-killed.

### 3.3 Per-Vendor Adapters (Concrete Flags)

Each adapter maps the common `ScopedTask` to vendor-native argv. These flags are the literal contract a coding agent builds against; pin and record the resolved CLI version (§3.7) because npm-distributed CLIs drift fast (e.g., Codex profile semantics broke at 0.134.0; `--full-auto` deprecated).

| Capability | Codex (`codex exec`) | Claude Code (`claude -p`) | Gemini CLI (`gemini -p`) | opencode |
|---|---|---|---|---|
| Headless entry | `codex exec` (alias `e`) | `-p`/`--print` | `-p`/`--prompt` (or non-TTY) | `opencode run` / `serve acp` |
| Event stream | `--json` (JSONL: thread/turn/item/error) | `--output-format stream-json` (NDJSON: system/init, api_retry, stream_event) | `--output-format stream-json` | OpenAPI 3.1 server `/doc`; `serve acp` NDJSON over stdio |
| JSON result | (via `--json` + `-o`) | `--output-format json` (result, session_id, total_cost_usd) | `--output-format json` (response + stats + error) | server response body |
| Structured output | `--output-schema <file>` | `--json-schema` → `structured_output` | none native → embed in prompt + post-parse | none native → embed + post-parse |
| Sandbox | `--sandbox {read-only\|workspace-write\|danger-full-access}` (read-only default) | `--permission-mode {acceptEdits\|dontAsk\|...}` + `--allowedTools` | `--yolo` (all-or-nothing) | server perms |
| Tool allowlist | (config/required-MCP) | `--allowedTools` | (limited) | server config |
| MCP | required-MCP (config) | `--mcp-config` | built-in/config | ACP passes MCP at session start |
| Final message file | `-o`/`--output-last-message` | (in JSON result) | `--session-summary <file>` | response |
| Git-repo bypass | `--skip-git-repo-check` | (n/a) | (n/a) | (n/a) |
| Reproducibility flags | `--ephemeral`, `--ignore-user-config`, `--ignore-rules` | `--bare` (skip hooks/skills/MCP/CLAUDE.md auto-discovery; becoming `-p` default) | (limited) | `--attach http://host:port` (avoid MCP cold-start) |
| Resume | `codex exec resume --last\|<SESSION_ID>` | `session_id` replay (+ `--replay-user-messages`) | (limited) | server session |
| Auth | CLI login or inline `CODEX_API_KEY` (exec-only) | `ANTHROPIC_API_KEY`/`apiKeyHelper` (with `--bare`) | provider auth | `OPENCODE_SERVER_PASSWORD`/basic-auth |

Canonical launch templates (the adapter emits these, modulo negotiated degradation):

- **Codex:** `codex exec --json --sandbox workspace-write --skip-git-repo-check --output-schema <f> -m <model>`
- **Claude:** `claude -p --bare --output-format stream-json --json-schema <f> --allowedTools <…> --permission-mode acceptEdits --mcp-config <f> --model <model>`
- **Gemini:** `gemini -p --output-format stream-json --yolo --session-summary <f>` (schema embedded in prompt)
- **opencode:** `opencode run --attach http://host:port "<prompt>"` or `opencode serve acp` (ACP over stdio)
- **openai_api:** in-process adapter over v1's `LLMClient` + `AgentLoop` fallback path (OpenAI-compatible chat.completions), kept for completeness; not the primary path.

Note on `--bare` and reproducible CI: from 2026-06-15, Claude subscription `-p`/Agent-SDK usage draws a **separate monthly Agent SDK credit pool**. This breaks naive cross-vendor cost accounting (§3.6) and must be modeled before any savings number is published. Codex's inline `CODEX_API_KEY` is exec-only and unsafe as a job-level env var on repo-controlled code; the adapter passes it per-invocation, never exports it.

### 3.4 Graceful Degradation to APEX's Own Floor

When a vendor lacks a capability, the Executor **degrades to APEX's own primitives** rather than crashing. The two load-bearing degradations:

1. **No native schema (Gemini, opencode) → embed schema in prompt + post-parse.** This is exactly v1's `_augment_prompt_for_backend` behavior (Codex additionally normalizes `additionalProperties=False`, required=all keys via `_normalize_schema_for_codex`). Prompt-embedded schema is *weaker* than native validation — this is a declared, bounded loss, not a hidden one. The engine re-validates and retries (up to v1's `max_attempts=4` for claude/codex; 1 otherwise).
2. **No read-only sandbox (Gemini `--yolo` is all-or-nothing) → wrap in APEX worktree + `fcntl` isolation.** APEX's per-rollout git-worktree isolation + advisory `fcntl` lock (the accepted, kept-verbatim mechanism; CAID ablation 63.3 vs 57.2) is the *floor*. Even a vendor running "full access" is confined to its own worktree, so concurrent same-file edits cannot corrupt siblings. The 4-tier degradation ladder (seed_clone → worktree → snapshot → synthetic) is preserved from v1.

Other declared per-vendor losses (the abstraction degrades, it does not preserve losslessly):

- **Tool-call interception is non-uniform.** v1's independent-CLI tool-call review wires the vendor's native pre-tool hook (claude `PreToolUse` / gemini `BeforeTool` / opencode `tool.execute.before`) to an external reviewer, but has documented `known_interception_gaps` and the reviewer **fails open** (malformed → allow). So the finer-grained verify-and-refute benefit (Section 13) is only *partially* preserved across vendors. The engine **records when an interception gap means a tool call went un-reviewed**, so the degradation is visible, not silent.
- **Bidirectional streaming** is only documented for Claude (`--input-format stream-json`); other vendors get single-shot scoped tasks.
- **Internet/web-search** differs (Codex `web_search` config, Gemini built-in, Claude via WebSearch tool/MCP); negotiation (§3.5) records which is active per-vendor.

### 3.5 ACP-Style Capability Negotiation

The Executor performs an `initialize`-style handshake modeled on the [Agent Client Protocol](https://agentclientprotocol.com/get-started/introduction) (`protocolVersion=1`, JSON-RPC 2.0 over stdio, capability negotiation in `initialize`, adopted by 25+ agents incl. Gemini CLI and Copilot CLI). This consolidates v1's scattered fragments (`_internet_launcher_args`, schema delivery, `CLIToolHookSupport`, `_CLIBackendSandboxSpec`, `--effort low..max`) into **one** negotiation layer rather than per-call special-casing.

```python
def negotiate(vendor, model, cli_version) -> CapabilityProfile:
    probe = run_capability_probe(vendor)        # vendor self-report (advisory only)
    declared = STATIC_CAPABILITY_TABLE[vendor]  # APEX's hardcoded, version-keyed truth
    # APEX's declared table WINS on conflict — vendor self-report is advisory telemetry.
    profile = merge(declared, probe, prefer=declared)
    profile.cli_version = resolve_npm_version(vendor)   # recorded for manifest + drift detection
    assert_conformance(profile, cli_version)    # §3.8: does the mapped sandbox/schema actually take effect?
    return profile
```

APEX can either (a) speak ACP as a client to ACP-capable workers (opencode `serve acp`; Gemini was first external integration) or (b) borrow ACP's `initialize`/capability schema for its own Executor handshake over the existing subprocess path. We adopt **(b) as the default** (it covers all five vendors uniformly via the subprocess adapters) and **(a) opportunistically** for opencode where it avoids MCP cold-start. The borrowed [A2A Agent Card](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) idea — a small, signed, discoverable manifest of `{skills, auth, transport, model}` — is the template for the per-backend `CapabilityProfile`/run-manifest schema (§3.7), so fleets are self-describing and replayable. **Caveat (honored pitfall):** "ACP" overloads three protocols and remote (HTTP/WebSocket) support is WIP; A2A/Agents-SDK target agent-to-agent/in-process, not fleet leaf-execution. We borrow the *handshake pattern*, not bet the executor transport on a single emerging standard.

### 3.6 Mixed in One Run: Heterogeneous-Fleet Routing

Cross-vendor diversity is a **first-class diversity/search axis** (accepted mechanism, `adopt`), placed alongside v1's strategy-axis/brief-family/effort/seed axes (CLI backends ignore temperature, so `(vendor, model)` is a *stronger* diversity lever — different model families fail differently, decorrelating hallucinations). The controller (Section 14) routes a `(vendor, model)` per decision node.

The evidence is direct: [Dissecting the SWE-Bench Leaderboards](https://arxiv.org/html/2506.17208v2) shows Devlo (70.2% SWE-bench Verified) generated candidates with three distinct models (Claude 3.7 Sonnet + o3 + Gemini 2.5 Pro); TRAE (70.4%) generated with Claude 3.7 Sonnet + Gemini 2.5 Pro + o4-mini and **selected with o1**; AgentScope used Qwen2.5 to select among Claude 3.5 trials. Heterogeneous-fleet generation + an execution-grounded selector beats single-vendor best-of-N.

The critical, non-negotiable condition (adversarial verdict): **this win materializes ONLY with an execution-grounded selector.** Without verification, diverse-but-wrong candidates add noise ([HeuriGym: higher diversity lowers yield via invalid outputs]). APEX's Cardinal Safety Contract (Section 13; execution-evidence-authoritative; soft signals re-rank-within-tier or downgrade only, never promote) is exactly that selector. **Rule: never ship cross-vendor diversity without the Cardinal-Safety verifier gate.** That is the difference between coverage and noise.

Reviewer-independence caveat (honored pitfall "Do not fold metacode into the opencode family when claiming cross-vendor reviewer independence"): v1's family-disjoint reviewer check (actor family ≠ reviewer family) currently folds `metacode` into the `opencode` family, so an `opencode + metacode` pair gets **no** independence gain. For true decorrelated cross-vendor review we either (a) stop folding metacode into the opencode family, or (b) explicitly document that same-family pairs yield no independence gain and exclude them from independence accounting. The roadmap chooses (a).

Resilience substrate (accepted mechanism "Two-tier failure memory + self-evicting BackendPortfolio", `adopt`, kept verbatim): a 429/stall on one vendor must not poison a healthy one. v1's distinction between **call-failover** (current-stage reroute only — 429/529, stall, connection reset) and **backend-level global reroute** (auth/401, missing binary, SDK breakage) is preserved, with the self-evicting `BackendPortfolio` (`run_backend_portfolio.json`) honoring `retry_after_seconds`. A per-vendor retry adapter keys off native signals (Claude `system/api_retry` with categories rate_limit/overloaded/server_error; Codex/Gemini exit codes) to drive unified backoff + cross-vendor failover, avoiding thundering-herd into 429s.

### 3.7 Cost Arbitrage: A Verification-Gated Cascade, Not Blind Routing

Cost arbitrage is **demoted from "net advantage" to "opt-in, verification-gated cascade"** (adversarial verdict; the weakest leg of the bundled claim). The accepted mechanism "Model economy as sub-role, verification-gated cascade" is `adopt-modified`. Do **not** do static up-front routing: [xRouter (2510.08439)](https://arxiv.org/html/2510.08439v1) shows hand-crafted "expensive-for-hard/cheap-for-easy" trees are brittle and do not transfer across providers, and the **almost-right trap** means cheap executors needing 3–4 retries + human review cost *more* than one frontier pass. **Measure token YIELD (cost per verified-resolved task), not invoice.**

The cascade (fits APEX's existing cheap-first verify-on-diff loop perfectly):

```
1. Frontier PLANNER (one vendor) decomposes + writes scoped contracts.   [keep frontier here]
2. Cheap cross-vendor EXECUTOR satisfies a narrow, well-specified step.   [cheapen ONLY here]
3. APEX cheap-first verification on the diff (AST → symbol survival → targeted pytest).
4. On verification FAILURE → escalate to frontier executor (rewrite-cycle cap).
5. Frontier REVIEWER owns the final quality gate.                        [keep frontier here]
```

Which roles are safe to cheapen is settled by the [HyperAgent ablation](https://arxiv.org/html/2409.16299v1): weakening the **Navigator (codebase exploration)** or **Editor (multi-file editing)** roles causes the *worst* resolve-rate drops because they need sustained long-context environment interaction; the **run/verify Executor** is the substitutable role. `<13B` models score `<5%` on SWE-bench Verified. Therefore: cheap is safe for run/verify and narrow single-tool calls; **risky** for navigation and multi-file edits on hard repo SWE. The accepted mechanism "Heavy-orchestrator + thin executor as the default execution shape" is `reject` — that default regresses toward the cheap-model baseline. Cascade-with-verification (à la [FrugalGPT](https://arxiv.org/abs/2305.05176), up to 98% cost cut at GPT-4 quality with a cheap scorer; [Aider architect/editor](https://aider.chat/2024/09/26/architect.html), Pareto-improving when the editor is competent) is the safe form.

This stays consistent with v1's "never optimize for cost" directive via the `budget{}` primitive (Section 2): `budget {total, spent(), remaining()}` is first-class but **defaulted unbounded**; cost arbitrage is opt-in. Realizing it requires cross-vendor token/cost *normalization* (different tokenizers, tiers, and the Claude Agent-SDK credit pool) that v1 deliberately does not have yet — so we build the accounting layer before publishing any savings figure.

#### Run-Manifest Pinning & Artifact Replay

`RunManifest` extends v1's existing manifest (which already pins `apex_git_sha`, python/platform, model_versions, docker_images digest-pinned, harness versions) to pin **per rollout**: `{vendor, model, resolved cli_version (npm), session_id, sandbox_mode, capability_profile, prompt_hash}`. Replay reproduces **artifacts (diffs) and re-runs verification**, not token streams — because temp-0 is not bitwise reproducible across hosted APIs (Thinking Machines; the accepted "Bit-reproducible agent OUTPUT replay" is `reject`). This satisfies the mandate's "pin vendor+model+version for replay" and is the substrate the durable journaled resume (Section 15) needs. Version pinning (`npm i -g @openai/codex@X.Y.Z`, `@anthropic-ai/claude-code@X.Y.Z`) defends against fast-moving CLI breaking changes; the resolved version is captured, not just requested.

### 3.8 Uniform MCP Tool Plane & Conformance Testing

To make tool capability identical regardless of leaf vendor, APEX **injects the same MCP server set into every backend** (Codex required-MCP, Claude `--mcp-config`, ACP passes MCP endpoints+credentials at session start, opencode server config). This means a branch's tool capabilities do not depend on which vendor executes it — a precondition for fair cross-vendor diversity and for the controller to route freely.

Because harness-dominates-model (30–50pt swings), the Executor is treated as **part of the harness** and gets a per-vendor **conformance test** asserting that the mapped sandbox/schema/tool-allowlist actually take effect (e.g., a read-only request truly blocks writes; an injected MCP server is truly reachable; an embedded schema truly yields a parseable object). This surfaces version drift *loudly* at run start rather than silently eroding the diversity gain mid-run, honoring "pin CLI versions against drift."

### 3.9 Summary of Dispositions

| Mechanism | Disposition | Net |
|---|---|---|
| Filesystem/git-diff as contract; JSON events as telemetry | adopt (kept verbatim) | The enabler; correctness never trusts vendor self-report |
| Normalized Executor + ACP-style negotiation, graceful degradation | adopt | Consolidate v1 fragments; degrade-not-crash to APEX floor |
| `(vendor, model)` as first-class diversity axis | adopt | Decorrelated cross-family errors widen coverage — **only** with the Cardinal-Safety selector |
| Two-tier failure memory + self-evicting BackendPortfolio | adopt (verbatim) | One vendor's 429 cannot poison a healthy fleet |
| RunManifest pins vendor+model+cli_version+profile; artifact replay | adopt (verbatim) | Reproduce diffs, not token streams (temp-0 not bitwise reproducible) |
| Cost arbitrage as verification-gated cascade | adopt-modified | Cascade-not-route; measure token yield; latent until opt-in |
| Heavy-orchestrator + thin executor as the default shape | reject | Cheapening navigation/multi-file editing regresses to cheap-model baseline |
| Trusting vendor self-reported JSON as correctness contract | reject | Verify on diff |
| Bit-reproducible agent OUTPUT replay | reject | Impossible across hosted APIs |
| Folding metacode into opencode for reviewer independence | reject | Same-family pair → no decorrelation gain |

The net verdict to carry forward: **feasibility is sound; "no benefit loss" is an overclaim (bounded, declared degradation is the truth); cross-vendor diversity is a sound advantage given APEX's execution-grounded selector; cost arbitrage is a conditional, opt-in cascade.** Cross-references: the engine primitives this Executor plugs into are Section 2; verify-and-refute and the Cardinal Safety Contract are Section 13; the model economy cascade detail is Section 12; isolation/determinism/durable resume are Section 15; the active controller that routes `(vendor, model)` is Section 14.
