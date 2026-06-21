"""Diff-blob banking & relinking invariants (O1/NEW-I5).

A scored candidate diff can end up with an empty ``fs_diff_ref`` so downstream
select/carry skips it and the best frontier is lost. These tests pin:

  * ``commit()`` stores the diff blob BEFORE updating ``_index[input_hash]``, so a
    committed/indexed entry NEVER references an un-retrievable blob.
  * a looked-up committed entry always returns its diff via ``load_diff``.
  * ``ensure_diff_linked`` is idempotent, relinks a cleared ref, and does NOT
    append a new WAL record (replay determinism preserved).
  * ``commit`` persists candidate scoring meta into ``structured_result``.
"""

from __future__ import annotations

import tempfile

from apex_omega.journal.wal import Journal, RESULT_OK
from apex_omega.journal.key import sha256_hex


def _mk(d=None):
    return Journal(d or tempfile.mkdtemp(), run_id="t")


def _commit(j, h, payload, **kw):
    return j.commit(
        input_hash=h, kind="agent", prompt_canonical="p", model_id="m", vendor="v",
        cli_version="cv", scoped_inputs_hash="s", result_status=RESULT_OK,
        structured_result={"value": payload}, fs_diff_text="diff:" + payload, usage={},
        **kw,
    )


# -- commit: blob before index, lookup always retrievable ----------------------

def test_committed_entry_always_returns_its_diff():
    j = _mk()
    _commit(j, "H1", "alpha")
    hit = j.lookup("H1")
    assert hit is not None
    assert hit.fs_diff_ref  # linked
    assert j.load_diff(hit.fs_diff_ref) == "diff:alpha"


def test_commit_stores_blob_before_index():
    """When the entry lands in _index its fs_diff_ref blob must already be on disk."""
    d = tempfile.mkdtemp()
    j = _mk(d)
    entry = _commit(j, "H2", "beta")
    # The blob the indexed entry points at must physically exist *now*.
    indexed = j.lookup("H2")
    assert indexed.fs_diff_ref == entry.fs_diff_ref
    blob = j.diffs_dir / f"{indexed.fs_diff_ref}.diff"
    assert blob.exists(), "indexed entry references a blob that was never written"
    assert blob.read_text() == "diff:beta"


def test_commit_blob_present_after_restart():
    d = tempfile.mkdtemp()
    j = _mk(d)
    _commit(j, "H3", "gamma")
    j2 = _mk(d)
    hit = j2.lookup("H3")
    assert hit is not None and j2.load_diff(hit.fs_diff_ref) == "diff:gamma"


# -- ensure_diff_linked --------------------------------------------------------

def test_ensure_diff_linked_relinks_cleared_ref():
    j = _mk()
    _commit(j, "H4", "delta")
    # Simulate a candidate whose committed entry lost its fs_diff_ref.
    from dataclasses import replace
    cleared = replace(j._index["H4"], fs_diff_ref="")
    j._index["H4"] = cleared
    assert j.lookup("H4").fs_diff_ref == ""  # frontier would be skipped

    relinked = j.ensure_diff_linked("H4", "diff:delta")
    assert relinked is not None
    assert relinked.fs_diff_ref == sha256_hex("diff:delta")
    assert j.lookup("H4").fs_diff_ref == relinked.fs_diff_ref
    assert j.load_diff(relinked.fs_diff_ref) == "diff:delta"


def test_ensure_diff_linked_idempotent():
    j = _mk()
    _commit(j, "H5", "epsilon")
    r1 = j.ensure_diff_linked("H5", "diff:epsilon")
    r2 = j.ensure_diff_linked("H5", "diff:epsilon")
    assert r1.fs_diff_ref == r2.fs_diff_ref == sha256_hex("diff:epsilon")
    # idempotent: ref already correct from the original commit, no change in identity.
    assert j.lookup("H5").fs_diff_ref == r1.fs_diff_ref


def test_ensure_diff_linked_does_not_append_wal_record():
    """Relink is an in-memory _index repair; replay determinism must be preserved."""
    j = _mk()
    _commit(j, "H6", "zeta")
    seq_before = j._next_seq
    n_lines_before = j.wal_path.read_text().count("\n")

    from dataclasses import replace
    j._index["H6"] = replace(j._index["H6"], fs_diff_ref="")
    j.ensure_diff_linked("H6", "diff:zeta")

    assert j._next_seq == seq_before, "ensure_diff_linked must NOT allocate a new seq"
    assert j.wal_path.read_text().count("\n") == n_lines_before, \
        "ensure_diff_linked must NOT append a WAL record"


def test_ensure_diff_linked_relink_lost_after_restart():
    """Confirm the relink is purely in-memory: a fresh Journal over the same dir
    rebuilds the ORIGINAL recorded ref (deterministic replay), not the relink."""
    d = tempfile.mkdtemp()
    j = _mk(d)
    _commit(j, "H7", "eta")
    orig_ref = j._index["H7"].fs_diff_ref
    from dataclasses import replace
    j._index["H7"] = replace(j._index["H7"], fs_diff_ref="")
    j.ensure_diff_linked("H7", "diff:eta")
    # Replay from WAL: index reflects only recorded records.
    j2 = _mk(d)
    assert j2.lookup("H7").fs_diff_ref == orig_ref


def test_ensure_diff_linked_no_entry_returns_none():
    j = _mk()
    assert j.ensure_diff_linked("NOPE", "diff:x") is None


def test_ensure_diff_linked_empty_text_is_noop():
    j = _mk()
    _commit(j, "H8", "theta")
    before = j.lookup("H8").fs_diff_ref
    res = j.ensure_diff_linked("H8", "")
    assert res is j._index["H8"]
    assert j.lookup("H8").fs_diff_ref == before


# -- candidate scoring meta persistence ----------------------------------------

def test_commit_persists_candidate_meta():
    j = _mk()
    e = _commit(j, "H9", "iota", gold_passed=True, pass_rate=0.875,
                indeterminate=False, content_sha="abc123")
    sr = j.lookup("H9").structured_result
    assert sr["gold_passed"] is True
    assert sr["pass_rate"] == 0.875
    assert sr["indeterminate"] is False
    assert sr["content_sha"] == "abc123"
    # meta also present on the returned entry
    assert e.structured_result["pass_rate"] == 0.875


def test_commit_meta_absent_when_not_provided():
    j = _mk()
    _commit(j, "H10", "kappa")
    sr = j.lookup("H10").structured_result
    for k in ("gold_passed", "pass_rate", "indeterminate", "content_sha"):
        assert k not in sr


def test_commit_meta_does_not_clobber_existing_keys():
    j = _mk()
    j.commit(input_hash="H11", kind="agent", prompt_canonical="p", model_id="m",
             vendor="v", cli_version="cv", scoped_inputs_hash="s", result_status=RESULT_OK,
             structured_result={"pass_rate": 0.5}, fs_diff_text="d", usage={},
             pass_rate=0.99)
    assert j.lookup("H11").structured_result["pass_rate"] == 0.5
