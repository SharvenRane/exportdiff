import numpy as np
import pytest

from exportdiff.analyze import CocoPerImage, bootstrap_counts, paired_interval, sensitivity
from exportdiff.data import COCO80_TO_91
from exportdiff.match import Detections


def dets(boxes, classes, scores):
    return Detections(np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float), classes)


def test_sensitivity_margin_and_threshold():
    ref = {1: dets([[0, 0, 10, 10]], [0], [0.28]), 2: dets([[0, 0, 10, 10]], [0], [0.90])}
    new = {1: Detections.empty(), 2: dets([[0, 0, 10, 10]], [0], [0.90])}
    s = sensitivity(ref, {"t": new}, [1, 2], [0.25, 0.30], [0.02, 0.05])
    at25 = s["t"][0.25]
    assert at25["flip"]["k"] == 1
    assert at25["hard"][0.02]["k"] == 1  # 0.28 sits 0.03 above 0.25
    assert at25["hard"][0.05]["k"] == 0  # but not 0.05 above
    assert s["t"][0.30]["flip"]["k"] == 0  # at 0.30 the box was never deployed
    assert s["_n"] == 2


def test_hard_count_never_grows_with_margin():
    rng = np.random.default_rng(1)
    ref, new = {}, {}
    for i in range(50):
        ref[i] = dets([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 1], rng.uniform(0.2, 0.6, 2))
        new[i] = dets([[0, 0, 10, 10]], [0], rng.uniform(0.2, 0.6, 1))
    s = sensitivity(ref, {"t": new}, list(range(50)), [0.25], [0.02, 0.05, 0.10])["t"][0.25]
    ks = [s["hard"][m]["k"] for m in (0.02, 0.05, 0.10)]
    assert ks == sorted(ks, reverse=True) and ks[0] <= s["flip"]["k"]


def _coco(images, anns):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO

    c = COCO()
    c.dataset = {"images": [{"id": i, "width": 100, "height": 100} for i in images],
                 "categories": [{"id": COCO80_TO_91[k], "name": str(k)} for k in range(3)],
                 "annotations": [{"id": n + 1, "image_id": i, "category_id": COCO80_TO_91[k],
                                  "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}
                                 for n, (i, k, x, y, w, h) in enumerate(anns)]}
    c.createIndex()
    return c


ANNS = [(1, 0, 10, 10, 30, 30), (1, 1, 50, 50, 20, 20), (2, 0, 5, 5, 40, 40), (2, 2, 60, 10, 25, 30),
        (3, 1, 20, 20, 30, 30), (3, 0, 70, 70, 20, 20)]


def _run():
    return {
        1: dets([[11, 9, 41, 40], [52, 50, 70, 72], [0, 0, 20, 20]], [0, 1, 2], [0.91, 0.62, 0.33]),
        2: dets([[5, 6, 44, 45], [61, 12, 85, 40], [60, 60, 90, 90]], [0, 2, 0], [0.88, 0.47, 0.29]),
        3: dets([[22, 20, 50, 49], [70, 72, 89, 90], [20, 20, 50, 50]], [1, 0, 2], [0.77, 0.41, 0.52]),
    }


def test_per_image_map_matches_pycocotools():
    from exportdiff.cocoeval import coco_map

    gt = _coco([1, 2, 3], ANNS)
    run = _run()
    want = coco_map(run, [1, 2, 3], None, gt)
    got = CocoPerImage(run, [1, 2, 3], gt).map()
    assert got[0] == pytest.approx(want["map50_95"], abs=1e-12)
    assert got[1] == pytest.approx(want["map50"], abs=1e-12)


def test_counts_equal_duplicated_dataset():
    from exportdiff.cocoeval import coco_map

    gt = _coco([1, 2, 3], ANNS)
    run = _run()
    got = CocoPerImage(run, [1, 2, 3], gt).map(np.array([2, 1, 0]))
    # the same multiset built explicitly: image 1 twice (as id 101), image 2 once, image 3 dropped
    anns2 = [a for a in ANNS if a[0] in (1, 2)] + [(101,) + a[1:] for a in ANNS if a[0] == 1]
    gt2 = _coco([1, 2, 101], anns2)
    run2 = {1: run[1], 2: run[2], 101: run[1]}
    want = coco_map(run2, [1, 2, 101], None, gt2)
    assert got[0] == pytest.approx(want["map50_95"], abs=1e-12)
    assert got[1] == pytest.approx(want["map50"], abs=1e-12)


def test_bootstrap_counts_reproducible():
    a = list(bootstrap_counts(10, 3, seed=4))
    b = list(bootstrap_counts(10, 3, seed=4))
    assert all((x == y).all() and x.sum() == 10 for x, y in zip(a, b))


def test_paired_interval_of_identical_series_is_zero():
    s = np.random.default_rng(0).uniform(0.3, 0.4, size=(200, 2))
    iv = paired_interval(s, s)
    assert iv["map50_95"] == (0.0, 0.0) and iv["b"] == 200


def test_run_without_detections_scores_zero_like_coco_map():
    from exportdiff.cocoeval import coco_map

    gt = _coco([1, 2, 3], ANNS)
    run = {i: Detections.empty() for i in (1, 2, 3)}
    want = coco_map(run, [1, 2, 3], None, gt)
    assert CocoPerImage(run, [1, 2, 3], gt).map(np.array([3, 0, 0])) == (want["map50_95"], want["map50"]) == (0.0, 0.0)
