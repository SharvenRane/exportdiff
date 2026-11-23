import numpy as np
import pytest

from exportdiff.compare import diff_runs, flip_vector, noise_floor, probe_indices, summarise_target, worst_images
from exportdiff.match import Detections
from exportdiff.stats import bootstrap_rate_ci, clopper_pearson, mcnemar_exact, rate_summary


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(5, 100)
    # reference values from the exact beta quantiles
    assert lo == pytest.approx(0.016431, abs=1e-5)
    assert hi == pytest.approx(0.112836, abs=1e-5)


def test_clopper_pearson_edges():
    assert clopper_pearson(0, 50)[0] == 0.0
    assert clopper_pearson(0, 50)[1] == pytest.approx(1 - 0.025 ** (1 / 50))
    assert clopper_pearson(50, 50)[1] == 1.0
    assert clopper_pearson(50, 50)[0] == pytest.approx(0.025 ** (1 / 50))
    with pytest.raises(ValueError):
        clopper_pearson(0, 0)
    with pytest.raises(ValueError):
        clopper_pearson(6, 5)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(0, 10) == mcnemar_exact(10, 0)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_bootstrap_is_reproducible_and_brackets_rate():
    x = np.zeros(400, dtype=bool)
    x[:40] = True
    a = bootstrap_rate_ci(x, seed=3)
    assert a == bootstrap_rate_ci(x, seed=3)
    assert a[0] < 0.1 < a[1]


def test_rate_summary():
    s = rate_summary(np.array([True, False, False, False]))
    assert s["k"] == 1 and s["n"] == 4 and s["rate"] == 0.25


def _run(ids, shift=None):
    out = {}
    for i in ids:
        b = np.array([[0, 0, 10, 10]], float)
        if shift and i in shift:
            b = b + shift[i]
        out[i] = Detections(b, np.array([0.9]), np.array([1]))
    return out


def test_noise_floor_takes_worst_self_run():
    ids = list(range(10))
    ref = _run(ids)
    same = _run(ids)
    moved = _run(ids, {3: 20.0})
    runs = {"rerun": flip_vector(diff_runs(ref, same, ids), ids),
            "batch": flip_vector(diff_runs(ref, moved, ids), ids)}
    nf = noise_floor(runs)
    assert nf["runs"]["rerun"]["k"] == 0
    assert nf["floor"] == pytest.approx(0.1) and nf["floor_run"] == "batch"
    assert nf["union_flags"].sum() == 1
    with pytest.raises(ValueError):
        noise_floor({})


def test_identical_rerun_has_zero_floor():
    ids = list(range(5))
    nf = noise_floor({"rerun": flip_vector(diff_runs(_run(ids), _run(ids), ids), ids)})
    assert nf["floor"] == 0.0


def test_missing_image_counts_as_empty_answer():
    ids = [1, 2]
    ref = _run(ids)
    d = diff_runs(ref, {1: ref[1]}, ids)
    assert not d[1].flipped and d[2].vanished == [0]


def test_threshold_applied_inside_diff():
    ids = [1]
    ref = {1: Detections(np.array([[0, 0, 10, 10]], float), np.array([0.26]), np.array([0]))}
    new = {1: Detections(np.array([[0, 0, 10, 10]], float), np.array([0.24]), np.array([0]))}
    assert diff_runs(ref, new, ids, threshold=0.25)[1].vanished == [0]
    assert not diff_runs(ref, new, ids, threshold=0.2)[1].flipped


def test_probe_and_summary():
    ids = list(range(20))
    margins = np.linspace(0.0, 1.0, 20)
    margins[5] = np.inf
    probe = probe_indices(margins, 4)
    assert list(probe) == [0, 1, 2, 3]
    assert len(probe_indices(margins, 100)) == 20
    ref = _run(ids)
    new = _run(ids, {0: 50.0, 1: 50.0, 15: 50.0})
    diffs = diff_runs(ref, new, ids)
    nf = noise_floor({"rerun": np.zeros(20, bool)})
    s = summarise_target("x", diffs, ids, nf, margins, probe)
    assert s["flip"]["k"] == 3 and s["probe"]["flips"] == 2
    assert s["mcnemar_vs_noise"]["export_only"] == 3
    assert s["boxes"]["vanished"] == 3 and s["boxes"]["appeared"] == 3
    assert worst_images(diffs, ids, 2) == [0, 1]


def test_mcnemar_large_counts_do_not_overflow():
    # 2.0 ** n overflows a float past n = 1023; exact p must still come out
    assert mcnemar_exact(1500, 0) == pytest.approx(0.0, abs=1e-300)
    assert mcnemar_exact(800, 800) == 1.0
    p = mcnemar_exact(600, 500)
    assert 0.0 < p < 0.01
    # small case agrees with the closed form
    assert mcnemar_exact(7, 2) == pytest.approx(2 * sum(__import__("math").comb(9, i) for i in range(3)) / 2**9)
