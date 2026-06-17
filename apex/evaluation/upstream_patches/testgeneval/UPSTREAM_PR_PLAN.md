# Upstream PR Plan for `kjain14/testgeneval`

This document describes the plan for upstreaming the changes that APEX
currently carries as out-of-tree patches against
[`kjain14/testgeneval`](https://github.com/kjain14/testgeneval). Every
divergence APEX maintains against the upstream harness is a fairness
risk; we publish this plan so the divergence is documented and so a
reviewer can audit the delta.

The patches that this plan corresponds to live in:

* `mutation_timeout_argument.patch` — adds a `--mutation_timeout` flag,
  fixes a macOS bind-mount bug, and threads the new flag through the
  `swebench_docker` evaluator.
* `baseline_covs_keyerror.patch` — defensively handles dataset rows that
  do not contain a `baseline_covs` key, preventing the
  `generate_report.py` aggregator from crashing on partial datasets.

---

## PR 1 — `--mutation_timeout` flag plumbing

### Motivation

Upstream hard-codes the mutation phase timeout to one hour
(`swebench_docker/evaluate_instance.py:378`). For long-running mutation
suites (e.g. SymPy / Django) this is insufficient and the harness kills
the mutmut process mid-run, producing a `mutation_score=-1` row that
silently looks like an APEX miss in published numbers.

### Scope

1. Add `--mutation_timeout` CLI argument to `run_evaluation.py`.
2. Thread `mutation_timeout: int = 3600` through:
   * `run_evaluation.main()`
   * `swebench_docker/run_docker.py:run_docker_evaluation()`
   * `swebench_docker/evaluate_instance.py:main()` (replacing the
     hard-coded `mutation_timeout=3600` literal at line 378)
3. Pass `MUTATION_TIMEOUT=<value>` into the container via `-e` flags
   (the in-container `evaluate_instance.py` reads it back via
   `os.getenv("MUTATION_TIMEOUT")`).
4. Document the new flag in `README.md`'s evaluation section.

### Compatibility note

Default value remains `3600` so existing callers see no behaviour change.

### Filing checklist

* [ ] Open issue describing the SymPy timeout symptom (with a
      reproducer log tail) before opening the PR.
* [ ] PR title: `Add --mutation_timeout flag to evaluation pipeline`
* [ ] Reference the issue and link to the macOS reproducer.
* [ ] Include a unit test that asserts the env var round-trips.

---

## PR 2 — macOS bind-mount fix for `tempfile.mktemp`

### Motivation

`swebench_docker/run_docker.py` currently uses
`tempfile.mktemp(suffix=".json")` to write the per-task instance JSON
that the container reads. On macOS, the system tempdir
(`/var/folders/...`) is **not** shared with Docker Desktop by default,
and the bind-mount lands as a *directory* inside the container instead
of a file. The container then fails to read the JSON and the task
silently reports `error: ENOENT`.

### Scope

Replace the `tempfile.mktemp` + `open()` pair with a
`tempfile.NamedTemporaryFile(dir=log_dir, ...)` so the temp file lives
inside `log_dir` (which is already in the user's mounted-paths whitelist
per the upstream README's macOS instructions).

### Filing checklist

* [ ] Open issue with the macOS reproducer (`docker info | grep
      "OSType"` → `linux` confirms Docker Desktop, then run the harness
      against any task and observe the missing JSON).
* [ ] PR title: `Use log_dir for per-task instance JSON tempfile (fixes macOS bind-mount)`
* [ ] Note that the original behaviour relied on `tempfile.mktemp`
      which is *deprecated* upstream as of Python 3.12 anyway — this
      gives the maintainers a second motivation to merge.

---

## PR 3 — defensive `baseline_covs` handling in `swebench_utils.get_eval_report`

### Motivation

`swebench_docker/swebench_utils.py:319` does
`swe_bench_instances[instance_id]["baseline_covs"]` with a bare key
lookup. The published `kjain14/testgenevallite` dataset has rows that
were uploaded **without** `baseline_covs` populated (those rows are the
"lite" subset where the publisher did not run the baseline-coverage
sweep). The current code raises `KeyError('baseline_covs')` and the
entire `generate_report.py` invocation aborts — meaning a single
ill-formed row renders the whole pass@1 / mutation / coverage report
unreadable.

### Scope

1. Replace `swe_bench_instances[instance_id]["baseline_covs"]` with
   `swe_bench_instances[instance_id].get("baseline_covs", {})` at every
   call site (currently lines 199-201 and 319).
2. Downstream: when `baseline_cov_info` is empty, treat it as "no
   baseline coverage available" — emit a `coverage_imp_baseline = 0.0`
   sentinel and a `coverage_imp_baseline_missing = True` diagnostic in
   the per-instance row instead of aborting the whole report.
3. Add a regression test that constructs a `swe_bench_instances` dict
   with one row missing `baseline_covs` and asserts the report still
   emits.

### Filing checklist

* [ ] Open issue listing the affected `kjain14/testgenevallite` rows.
* [ ] PR title: `Defensively handle missing baseline_covs in get_eval_report`
* [ ] Include the regression test in the PR.

---

## Rollout

Once each of the above lands upstream and a release tag is cut:

1. Bump the pinned `testgeneval` version in
   `apex/apex/evaluation/runners/_preflight.py` to the new tag.
2. Delete the corresponding patch file from
   `apex/apex/evaluation/upstream_patches/testgeneval/`.
3. Set the default of
   `BenchmarkConfig.testgeneval_apply_upstream_patches` to `False` and
   emit a deprecation warning when callers explicitly enable it.
4. After one full release cycle with no regression reports, remove the
   config flag entirely.

The Phase 1 fairness audit (`FairnessAuditMode.PARALLEL`) emits the
delta between the patched-harness scoring and the unpatched-harness
scoring per task, so we can monitor the impact of removing each patch
quantitatively before merging.
