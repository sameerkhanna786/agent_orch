## 15. Isolation, Determinism & Durable Resumable Runs

This section specifies the **hardened substrate** on which every other APEX-Ω mechanism stands: per-rollout filesystem isolation, best-effort determinism, and a durable, restart-survivable journal that makes the engine's `agent()`/`parallel()`/`pipeline()` primitives (see Section 2) resumable and the speculative tree-search of Section 9 replayable. The design lifts APEX v1's load-bearing invariants verbatim where they are sound (worktree isolation, `fcntl` locks, deterministic snapshot SHAs, the escrow WAL) and *generalizes* two of them — the narrow one-candidate escrow WAL and the unwired `ReplayRecorder` — into a single per-`agent()`-call journaled resume that beats the reference Claude Code implementation's session-scoped resume. Throughout, three claims are held precisely and never overstated: **(a)** determinism is *best-effort around irreducibly-stochastic workers*, never bit-for-bit agent output; **(b)** what is reproduced is **artifacts** (diffs + re-run verification + selected winner), not token streams; **(c)** durable resume is *unbuilt in v1 as a general mechanism* (the v1 ingest is explicit: `ReplayRecorder` has no production callsite and the escrow WAL is a one-candidate backstop), so this section is a **build spec**, not a description of a preserved guarantee.

The unifying principle is **per-unit scoping**: locks, kills, registries, secret boundaries, and journals are all scoped to a single rollout/run, never machine-wide. There is no global mutex anywhere in this design.

### 15.1 Per-Rollout Isolation: Worktrees, Locks, and Three-Tier Degradation

Parallel and speculative workers that edit the same file conflict; the documented fix in the source paradigm and in v1 is **git worktrees — an isolated checkout per rollout**. APEX-Ω keeps v1's hardened version of this verbatim and exposes it as the `isolation:"worktree"` option on the `agent()` primitive (Section 2) and as the implicit default for any file-editing worker spawned by `parallel()`/`pipeline()`.

#### 15.1.1 The lock-before-touch invariant

Every rollout acquires a **per-rollout advisory lock before touching the workspace path**, and releases it on *every* exit path (success or failure). This is the v1 invariant (Section 8.2 of the v1 blueprint) and it is non-negotiable: it is the primitive that makes any parallelism or branching safe (the v1 ingest attributes a CAID improvement of 63.3 vs 57.2 to this isolation discipline).

```
LOCK_PATH = workspace_dir / ".locks" / f"rollout_{rollout_id}.lock"

acquire_rollout_workspace(rollout_id, workspace_dir):
    lock = open(LOCK_PATH, mode="w")            # create lock file FIRST
    try:
        fcntl.flock(lock.fd, LOCK_EX | LOCK_NB) # POSIX; non-blocking
    except BlockingIOError:
        raise ConcurrentWorktreeError(rollout_id)   # never silently share
    # ---- ONLY NOW may we create/mutate the worktree for this rollout ----
    workspace = provision_isolation_tier(rollout_id, workspace_dir)
    return WorkspaceHandle(lock=lock, workspace=workspace, tier=workspace.tier)

# Windows fallback (no fcntl): atomic PID-marker file via O_CREAT|O_EXCL;
#   stale-marker reclaim only if the recorded PID is provably dead.
release_rollout_workspace(handle):  # in a finally: block, always
    fcntl.flock(handle.lock.fd, LOCK_UN); handle.lock.close()
```

The lock is **scoped to the rollout id, not the machine**. Two concurrent solves sharing a `workspace_dir` cannot destroy each other's worktrees, and reaping one stalled rollout never touches its siblings (the per-rollout `RolloutCLIRegistry` binds each worker pid to one `rollout_id`; `SIGTERM->SIGKILL` escalation is scoped to that rollout's pids only). **Pitfall honored:** do **not** take a global/machine-wide mutex — that would serialize the whole fleet and defeat the scaling unlock.

#### 15.1.2 Three-tier degradation (degrade, never crash)

Isolation degrades downward on any failure so the engine stays runnable on dirty or non-git repos. This is the ACP-style "graceful degradation" mandate applied to the filesystem (Section 3):

| Tier | Used when | Diff source of truth | Determinism property |
|------|-----------|----------------------|----------------------|
| `seed_clone` (override) | caller supplies a pre-built clone | clone's git diff | inherits seed's SHA |
| `worktree` (preferred) | clean git repo | `git worktree add` checkout | bit-identical diff vs base SHA |
| `snapshot` | dirty tree / worktree-add fails | deterministic synthetic commit | **deterministic snapshot SHA** |
| `synthetic` | non-git repo | content-hash overlay | content-hash identity |

**Deterministic snapshot SHA (kept verbatim from v1):** the snapshot tier commits with a *fixed* author/committer (`APEX Snapshot <apex@local>`), a *fixed* date (`2026-01-01T00:00:00+0000`), and a commit message containing `source_head8 + dirty_hash8` (sha256 over sorted dirty rel-paths + bytes). Two identical source states therefore produce **bit-identical commit SHAs and bit-identical diff text** — the property that lets verification and replay compare diffs across runs. Note the date is a *constant*, not `Date.now()`; per the no-timestamps-in-orchestration pitfall, the orchestration layer never reads the wall clock.

A `WorktreePool` recycles worktrees (~10x cheaper than create+warmup, per v1) and is the natural home for the warm-pool / CoW-clone optimizations of Section 16; the pool is a *cost* lever and never weakens isolation — every checked-out worktree still takes its own per-rollout lock.

### 15.2 Best-Effort Determinism (What We Pin, What We Cannot)

Determinism is the prerequisite for both sound replay and a learned controller's off-policy credit assignment (Section 14). APEX-Ω pins everything pinnable *around* the stochastic worker and is explicit about the boundary.

**Honesty boundary (adversarial verdict honored).** The adversarial review of speculative search is explicit: "bit-for-bit determinism" overstates what v1 guarantees. We therefore state precisely what determinism means here:

- **Reproducible:** orchestration control flow, dispatch ordering, candidate ordering, cluster reduction order, selection (the deterministic ranking tuple terminates in a content sha1 / `-cluster_id`, *never insertion order*), snapshot SHAs, diff text from identical source states, and atomic artifact bytes.
- **NOT reproducible from scratch:** the worker's token stream (temperature-0 hosted APIs are batch-non-invariant), and therefore the *exact shape of a re-run speculative tree*. These are reproduced only via **replay from the journal** (Section 15.3), never re-derived.

We **reject** the "bit-reproducible agent OUTPUT replay" mechanism as impossible across hosted APIs and instead reproduce **artifacts**: diffs, re-run verification results, and the selected winner.

Concrete determinism rules the engine enforces:

| Source of nondeterminism | Rule | Rationale / SOTA anchor |
|--------------------------|------|-------------------------|
| Worker sampling | `temperature=0.0` default (CLI backends ignore temperature anyway; diversity comes from `(vendor, model)` + prompts, not temp — Section 13) | v1 invariant |
| Candidate / dispatch ordering | sort by `(rollout_id, content_hash)`; reduce clusters in `cluster_id` order | eliminates slot-0 bias |
| Local mutation / mutant scoring | `Random(0)` seeded | v1 |
| Search-control sampling (Thompson, AB-MCTS wider-vs-deeper) | **seed the RNG from a content hash** of the node's accepted-evidence set, not `Math.random()` | adversarial verdict: AB-MCTS Thompson sampling must be content-hash-seeded to remain replayable ([AB-MCTS, arXiv:2503.04412](https://arxiv.org/abs/2503.04412)) |
| Hedge / cancel / preempt decisions | **journal the decision** (Section 15.3); replay reproduces it from the log, never re-derives from timing | Dean & Barroso hedging is wall-clock-triggered and inherently nondeterministic ([The Tail at Scale, CACM 2013](https://cacm.acm.org/research/the-tail-at-scale/)) |
| Wall clock / RNG / network in orchestration | **forbidden in orchestration (workflow) code**; confined to journaled worker "activities" | Temporal: "non-determinism is fatal to replay" ([Temporal docs](https://docs.temporal.io/workflow-execution)) |
| File writes | atomic `mkstemp -> write -> flush -> os.fsync -> os.replace` (readers see old-or-new, never torn) | v1; durable-execution practice |

**Seeding search-control nondeterminism (required for Section 9/14).** Any place the controller samples — Thompson sampling for wider-vs-deeper, an epsilon-greedy bandit arm choice, a tie-break among equal-priority branches — derives its seed as `seed = int.from_bytes(sha256(canonical_evidence_bytes)[:8])`. Because the evidence bytes are themselves journaled and content-addressed, the same evidence yields the same sample on replay. This is the mechanism that lets a *stochastic* search controller remain deterministically replayable.

### 15.3 Durable Input-Hash Journaled Resume

This is APEX-Ω's headline differentiator over the reference implementation, whose resume is session-scoped and "starts the workflow fresh" after a full restart ([Claude Code workflows docs](https://code.claude.com/docs/en/workflows)). The mandate is explicit: **unchanged `agent()` calls return cached results; only edited/new calls re-run — and this survives a full process restart.** The design is the durable-execution template (Temporal event-history + deterministic replay; DBOS Postgres checkpoints; Inngest memoized `step.run`) applied at the granularity of a single worker call.

#### 15.3.1 The per-call journal (generalize escrow WAL/CCEDF to per-node)

We **generalize** v1's escrow WAL (CCEDF: fsync-durable, flock-guarded, monotonic-seq, idempotent exactly-once) from a one-candidate confirmed-pass backstop into a **per-`agent()`-call write-ahead log**. The escrow WAL already proved the hard part — making concurrent, nondeterministically-*timed* operations replay-stable by ordering on a **monotonic seq (not wall-clock)** with idempotent keys so duplicate appends are harmless. We reuse that exact pattern per node.

**Journal record (`JournalEntry`):**

```python
@dataclass(frozen=True)
class JournalEntry:
    seq: int                 # monotonic, max(existing)+1 under flock; ORDER KEY (not wall-clock)
    input_hash: str          # sha256 over the canonical key below — the cache key
    kind: str                # "agent" | "hedge_decision" | "cancel" | "node_expand" | "node_prune"
    # ---- the input-hash key components (all that determine an agent() result) ----
    prompt_canonical: str    # prompt with volatile spans normalized out (see 15.3.2)
    model_id: str            # pinned launcher id, e.g. "claude-opus-4-8[1m]"
    vendor: str              # "claude_cli" | "codex_cli" | "gemini_cli" | ...
    cli_version: str         # vendor CLI version string (from RunManifest, 15.4)
    scoped_inputs_hash: str  # sha256 of the worker's scoped inputs: base snapshot SHA,
                             #   discovery_scope (file_paths/symbols/test_ids), schema, effort
    # ---- the recorded outcome (what replay returns for a cache hit) ----
    result_status: str       # "ok" | "infra_nonresult" | "abstain"
    structured_result: dict  # validated schema object OR final text
    fs_diff_ref: str         # content-addressed blob ref for the produced git diff
    usage: dict              # tokens in/out, cache_read/cache_creation (Section 16 SLO)
    idempotency_key: str     # f"{run_id}::{node_id}::{attempt}"
```

The journal lives at `<run_dir>/journal/calls_wal.jsonl`, appended with the CCEDF discipline. To avoid the O(n)-per-append seq scan (a v1 cost note that "matters only if record volume grew" — and it now does, since every call is journaled), the engine keeps an in-memory `next_seq` counter recovered once on startup by reading the tail, then appends are O(1); the flock still guards cross-process appends.

#### 15.3.2 Resume algorithm: replay cached, re-run edited/new

```
resume_or_run(call_request):
    key = canonical_key(call_request)        # prompt_canonical, model, vendor, cli_version,
                                             #   scoped_inputs_hash  -> sha256
    hit = journal.lookup(key)                # index built from WAL on startup; latest-wins by seq
    if hit and hit.result_status != "infra_nonresult":
        materialize_fs_diff(hit.fs_diff_ref) # restore the produced diff into the worktree
        log("replay", key); return hit.structured_result   # CACHE HIT: no worker spawned
    # MISS (new call) or EDITED call (key changed because prompt/scope/model changed):
    result = executor.spawn(call_request)    # the only place a worker runs
    journal.append(JournalEntry(... key ..., result ...))   # durable BEFORE returning
    return result
```

**Cache-validity semantics (the subtle part the adversarial verdict flagged).** A "cached result" means *replaying the recorded output*, not re-deriving it. The cache key therefore must capture **everything that determines the result**: prompt, model, vendor, CLI version, and the scoped inputs (crucially the **base snapshot SHA** — if the code under the worker changed, the snapshot SHA changes, the key changes, and the call correctly re-runs). This is exactly the "unchanged -> cached / edited -> re-run" contract. Getting the key wrong in either direction is the failure mode: too-coarse a key replays a stale answer against changed code; too-fine a key never gets a cache hit and defeats resume.

**`prompt_canonical` normalization.** The prompt is canonicalized before hashing by stripping a declared *volatile region* (timestamps, session ids, run-local paths) from the *stable prefix* — the same prefix-stability discipline Section 16 needs for provider caching. This serves two ends at once: a stable journal key and maximal provider-cache hit-rate. (Per the cache pitfall, a single byte of drift in the stable region both breaks provider caching *and* spuriously invalidates a journal entry, so the canonicalizer is shared between the journal and the prompt-assembly layer.)

#### 15.3.3 Journaling hedge / cancel / speculative decisions

Speculative search (Section 9) and hedging (Section 16) inject timing- and sampling-nondeterminism into orchestration, which would break replay if left implicit. The fix (adversarial verdict, recommended adaptation B) is to make **every branch expand, prune, hedge-fire, and cancel a journaled decision** keyed by seq:

- A `hedge_decision` entry records *that* a duplicate was fired and *which* branch's result was taken; on replay the engine reads this and reproduces the same winner **without** re-running the deadline timer.
- A `node_expand` / `node_prune` entry records the controller's choice (and the content-hash seed that drove any sampling); replay reconstructs the tree in **seq order** regardless of original timing.

Because ordering is by monotonic seq and keys are idempotent, **duplicate appends are harmless and replay is order-stable** — the exact CCEDF property, now serving the whole tree. This is what makes the adversarial verdict's "speculative search trees can preserve replay" hold *in v1's actual sense* (orchestration/selection determinism + replay-from-journal), and only in that sense.

#### 15.3.4 Idempotency for external side effects (at-least-once reality)

Worker calls that mutate external state (a repo push, a network write) are **at-least-once**, not exactly-once, because a crash can occur after the side effect but before the journal append (Temporal/DBOS/Step Functions all enforce this distinction). Every such activity therefore carries `idempotency_key = run_id::node_id::attempt` so a re-run on resume is a no-op or a safe overwrite. **Pitfall:** never speculate or auto-resume an *irreversible, externally-visible* effect (deleting records, sending mail); the effect taxonomy of Section 9 gates which actions are eligible, and the journal records the attempt so resume can detect a possible duplicate. Storage backend: a local fsync'd JSONL WAL is the portable default (no infra dependency); a Postgres-backed journal (DBOS-style, transactional exactly-once when a step writes the same Postgres) is an **optional** drop-in for high-fan-out deployments, with the documented caveat that Postgres status-row contention bites under hundreds of concurrent appends — partition or batch in that regime.

### 15.4 RunManifest Pinning for Replay

Replay across vendors and hosts is only sound if the environment is pinned. The `RunManifest` (kept verbatim from v1, satisfying the vendor-agnostic mandate's "pin vendor+model+version for replay") is captured side-effect-free and never raises:

```python
@dataclass(frozen=True)
class RunManifest:
    apex_git_sha: str; apex_dirty: bool
    python_version: str; platform: str
    seed: int
    redacted_env: dict[str, str]          # APEX_* only, secrets stripped
    model_versions: dict[str, str]        # alias -> pinned launcher id
    vendor_cli_versions: dict[str, str]   # claude_cli/codex_cli/... -> version (NEW emphasis)
    docker_images: dict[str, str]         # tag -> repo@sha256:digest
    upstream_harness_versions: dict[str, str]
```

**Docker digest pinning** uses v1's `resolve_image` precedence — `@sha256:` short-circuit (return as-is, avoiding the double-pin bug) -> prepinned registry file -> `docker_inspect` (30s) -> bare tag — and records which path won. Image drift silently changing scores is the failure this prevents. For a **heterogeneous fleet**, `vendor_cli_versions` is load-bearing: the journal's `cli_version` component (Section 15.3.1) is read from here, so a replay that finds a different CLI version correctly treats the call as edited and re-runs it rather than replaying an output the new binary could not have produced.

### 15.5 The CI Acceptance Test: Kill Mid-Run -> Identical Resume

Durable resume is only a guarantee if it is *gated by a test*. Borrowing the durable-execution community's standard validation ("test by killing mid-run and confirming identical resumed output"), APEX-Ω adds a **mandatory CI acceptance test** before the durability claim may be asserted:

```
test_kill_midrun_identical_resume:
  1. Start a fixed-seed run on a pinned task (RunManifest captured).
  2. At a deterministic checkpoint (e.g. after node_expand seq==K), SIGKILL the engine.
  3. Restart from the same run_dir.
  4. ASSERT: every pre-kill agent() call is a journal REPLAY (zero workers re-spawned for them).
  5. ASSERT: the resumed trajectory is identical in seq order (same node_expand/prune/hedge log).
  6. ASSERT: the SELECTED WINNER is byte-identical (same content sha1) AND its re-run
            verification reproduces the same accept decision and the same diff text.
  7. ASSERT (negative): editing one node's prompt before restart causes EXACTLY that node
            (and its dependents) to re-run, and nothing else.
```

This test asserts on **artifacts and trajectory**, not token streams — consistent with the honesty boundary of Section 15.2. It is the same property Temporal's replay tests check and is the concrete evidence that "we did better than the reference impl." Until it is green, the plan treats durable resume as *unbuilt* (per the v1 ingest's note that `ReplayRecorder` has no production callsite).

### 15.6 Interaction with the Cardinal Safety Contract and Speculation

A perfectly deterministic, perfectly resumable run can still be **reproducibly wrong** if it prunes a correct candidate before execution evidence exists. The adversarial verdict is explicit here: a deterministic-but-unsafe pruner is *worse* than a nondeterministic one because it is reliably wrong, matching the documented "premature pruning prunes correct paths / PRM reliability decreases with distance from terminal states" failure mode ([Limits of PRM-Guided Tree Search, arXiv:2510.20272](https://arxiv.org/html/2510.20272)). Therefore this section's machinery is subordinate to the Cardinal Safety Contract (Section 13):

- Speculative pruning (Section 9) may only **re-order exploration** or **downweight** branches via journaled, content-seeded decisions; it may **never** terminate a branch on a soft/learned/LLM signal in a way that prevents a would-be execution-verified candidate from reaching the verifier.
- The journal records *which* branches were expanded vs pruned, but selection still runs the full execution-authoritative cascade over whatever survivors reached it; the deterministic ranking tuple places every soft key strictly below every execution key (Section 13).

In short: isolation and determinism make search **safe to run and cheap to replay**; the Cardinal Contract keeps it **honest**. Neither substitutes for the other.

### 15.7 Build Notes for a Vendor-Neutral Implementer (Codex or Claude Code)

A coding agent building this on either backend must observe:

1. **No clock/RNG/network in orchestration code.** All three live only inside journaled worker activities. Lint for `time.*`, `random.*` (except explicitly seeded `Random(content_hash)`), and socket calls in the orchestration module; a violation silently breaks replay (Section 15.2).
2. **Lock first, touch second, release in `finally`.** The `acquire -> provision -> ... -> release` ordering of Section 15.1.1 is the single most common place to introduce a corruption bug; the release must be unconditional.
3. **The journal append must be durable before the result is returned to the caller** (write-ahead), or a crash in the return path loses a completed call's record and forces a needless re-run.
4. **Share one canonicalizer** between the journal key (Section 15.3.2) and the prompt-assembly stable-prefix contract (Section 16) so journal validity and provider-cache hit-rate never diverge.
5. **Wire the journal into `agent()` itself** (the `resume_or_run` wrapper of Section 15.3.2), not into per-stage callers — this is the v1 "promote `ReplayRecorder` into a production per-call journal" seam, and centralizing it is what makes *every* primitive (`parallel`, `pipeline`, speculative trees) resumable for free.
6. **Heterogeneous fleets:** the journal key includes `vendor` and `cli_version`, so a run can replay a Claude-orchestrated, Codex-executed trajectory exactly, and a mid-run vendor swap correctly re-runs only the affected calls.

#### Config keys (defaults chosen to be safe, not cheap)

| Key | Default | Meaning |
|-----|---------|---------|
| `isolation.tier_preference` | `["seed_clone","worktree","snapshot","synthetic"]` | degradation order |
| `isolation.worktree_pool_size` | `cores-2` | recycled worktrees (cost lever, Section 16) |
| `determinism.temperature` | `0.0` | worker sampling |
| `determinism.search_seed_source` | `"content_hash"` | how Thompson/bandit RNG is seeded |
| `journal.enabled` | `true` | durable per-call journaling ON by default (the differentiator) |
| `journal.backend` | `"jsonl_wal"` | or `"postgres"` (DBOS-style, opt-in) |
| `journal.materialize_diffs` | `true` | restore fs diffs on cache hit |
| `resume.require_manifest_match` | `true` | refuse cross-host replay on RunManifest mismatch unless `--force` |
| `resume.on_cli_version_mismatch` | `"rerun"` | `rerun` (safe) \| `replay` (force, unsafe) |

### 15.8 Honest Limitations

- **Not bit-reproducible worker output.** Hosted-API temperature-0 batch non-invariance makes this impossible; we reproduce artifacts, not token streams. Any claim otherwise is rejected.
- **At-least-once external side effects.** Idempotency keys mitigate but do not eliminate the window between a side effect and its journal append; irreversible effects must never be auto-resumed or speculated.
- **Journal validity hinges on the cache key.** A missing key component (e.g. forgetting an effort-level flag) replays a stale answer; the shared canonicalizer and the negative arm of the CI test (Section 15.5, step 7) are the guards.
- **Postgres-journal contention at extreme fan-out** is real (DBOS caveat); the JSONL WAL default avoids it, and the Postgres backend is opt-in with partitioning guidance.
- **Process/memory state is not snapshotted.** v1's isolation is filesystem/git-only by design (the fast path); a worker's live dev-server or language-server state is lost on rollback. Process-memory checkpoint/restore (DeltaBox/Crab-class) is a Section 16 optimization, deliberately out of scope here, and is the one place the filesystem-as-source-of-truth invariant trades capability for cost/simplicity.

These limitations are stated so the engine never *fakes* a guarantee it cannot keep — the fail-loud-never-fake invariant applied to the substrate itself.
