# Ask: Avocado-5.13 access + eval-rate capacity for an orchestration-research harness

**Where to post:** MetaCode group (fb.workplace.com/groups/839543739176528 — the "Metacode++ Part 1"
thread [g3zr]) and/or the Avocado dogfooding group (`acdogfooding`). Tag the Coding Acceleration
V-Team / model owners.

**Suggested message:**

> Hi team — I'm running a vendor-neutral coding-agent **orchestration** research harness (APEX-Ω) that
> drives MetaCode as the per-attempt agent on the commit0 benchmark (implement a library from its
> hidden gold pytest suite). Two asks:
>
> **1. Access to `avocado-5.13`.** Per "Metacode++ Part 1" it's the strongest checkpoint, but I get
> `Error: Model not found: meta/avocado-5.13 ... not loaded for the meta provider` when I run
> `metacode run --model meta/avocado-5.13`. A system `META_EXPERIMENT_MODEL_OVERRIDE_ID=avocado-5.13`
> is set but blind-routing is skipped because the model isn't provisioned for me. Could I be added to
> the allowlist? (`meta/avocado-code-latest` and `meta/avocado-code-internal-0604` also intermittently
> disappear from `metacode models` — the list drops from 8 model-api entries to 5 bundled-only — so the
> only stable models I can pin are the bundled ones, newest of which is `avocado-code-internal-0529`.)
>
> **2. Eval-rate capacity.** My harness fans out concurrent `metacode run` calls (best-of-N + repair
> waves). At ~4 concurrent cells I measured **46% `infra_nonresult`** (model-not-found / timeouts);
> isolated at QPS=1 it succeeds 3/3 but with **15–106s latency for a trivial one-line task**. This looks
> like Plugboard/gateway capacity collapse under concurrency. Is there a higher eval rate-limit tier, a
> dedicated/BYOC capacity path, or a recommended max-concurrency for an automated eval workload? I can
> share trajectories / request IDs.
>
> Goal: a clean apples-to-apples comparison of our orchestrator across models (Codex anchor + Avocado
> challenger). Happy to feed results back into your evals flywheel. Thanks!

**Internal context (do not necessarily post):**
- Evidence files: `/tmp/omega_mc_eval` (the 46%-infra run), `orchestration_research/EVAL_BACKEND_SWITCH.md`.
- Harness already wires MetaCode as a first-class backend (`LLMBackend.METACODE_CLI`,
  `metacode run --format json --yolo --model <m>`); `5.13` is pre-registered in config — a one-line flip
  to use it once access is granted.
- `--pure` and `--no-init` both break `metacode run` for us (provider/model not loaded; lightweight-cmd
  only) — not viable workarounds.
