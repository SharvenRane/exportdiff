import numpy as np
import pytest

from exportdiff.match import Detections, diff_answers, image_margin, iou_matrix


def dets(boxes, classes, scores=None, second=None):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    scores = np.full(len(boxes), 0.9) if scores is None else scores
    return Detections(boxes, scores, classes, second)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 0, 0]])
    got = iou_matrix(a, b)[0]
    assert got == pytest.approx([1.0, 50 / 150, 0.0, 0.0])


def test_iou_empty_shapes():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_identical_answers_do_not_flip():
    a = dets([[0, 0, 10, 10], [20, 20, 40, 40]], [1, 2])
    d = diff_answers(a, a)
    assert not d.flipped and d.changes == 0 and len(d.pairs) == 2


def test_both_empty_is_not_a_flip():
    d = diff_answers(Detections.empty(), Detections.empty())
    assert not d.flipped


def test_vanished_and_appeared():
    ref = dets([[0, 0, 10, 10], [20, 20, 40, 40]], [1, 2])
    new = dets([[0, 0, 10, 10], [100, 100, 120, 120]], [1, 3])
    d = diff_answers(ref, new)
    assert d.vanished == [1] and d.appeared == [1] and d.relabelled == []
    assert d.flipped and d.changes == 2


def test_class_change_on_matched_box():
    ref = dets([[0, 0, 10, 10]], [1])
    new = dets([[0, 0, 10, 11]], [7])
    d = diff_answers(ref, new)
    assert d.relabelled == [(0, 0)] and not d.vanished and not d.appeared


def test_match_iou_boundary():
    ref = dets([[0, 0, 10, 10]], [1])
    exactly_half = dets([[0, 0, 10, 5]], [1])  # IoU 0.5 exactly, kept
    below = dets([[0, 0, 10, 4.9]], [1])
    assert not diff_answers(ref, exactly_half, 0.5).flipped
    d = diff_answers(ref, below, 0.5)
    assert d.vanished == [0] and d.appeared == [0]


def test_hungarian_beats_greedy():
    # greedy on the best IoU first would pair ref0 with new1 and leave ref1 unmatched
    # IoU: r0n0 0.818, r0n1 0.6, r1n0 0.6, r1n1 0.25; greedy takes r0n0 and strands r1
    ref = dets([[0, 0, 10, 10], [3.5, 0, 13.5, 10]], [1, 1])
    new = dets([[1, 0, 11, 10], [-2.5, 0, 7.5, 10]], [1, 1])
    d = diff_answers(ref, new, 0.5)
    assert not d.flipped
    assert sorted((r, c) for r, c, _ in d.pairs) == [(0, 1), (1, 0)]


def test_one_side_empty():
    ref = dets([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 2])
    d = diff_answers(ref, Detections.empty())
    assert d.vanished == [0, 1] and d.appeared == []
    d = diff_answers(Detections.empty(), ref)
    assert d.appeared == [0, 1]


def test_above_threshold_is_inclusive():
    a = dets([[0, 0, 1, 1]] * 3, [0, 0, 0], scores=np.array([0.2499, 0.25, 0.9]))
    assert len(a.above(0.25)) == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        Detections(np.zeros((2, 4)), np.zeros(1), np.zeros(2))


def test_image_margin():
    a = dets([[0, 0, 1, 1]] * 2, [0, 1], scores=np.array([0.30, 0.10]), second=np.array([0.29, 0.0]))
    # threshold distances 0.05 and 0.15, class gap of the box above threshold 0.01
    assert image_margin(a, 0.25) == pytest.approx(0.01)
    assert image_margin(Detections.empty(), 0.25) == float("inf")


def test_hard_change_split():
    ref = dets([[0, 0, 10, 10]], [1], scores=np.array([0.27]))
    d = diff_answers(ref, Detections.empty())
    assert d.vanished_scores == [pytest.approx(0.27)]
    assert d.flipped and not d.hard(0.25, 0.05)  # grazing the threshold
    ref_strong = dets([[0, 0, 10, 10]], [1], scores=np.array([0.80]))
    assert diff_answers(ref_strong, Detections.empty()).hard(0.25, 0.05)
    relabel = diff_answers(dets([[0, 0, 10, 10]], [1], scores=np.array([0.26])),
                           dets([[0, 0, 10, 10]], [2], scores=np.array([0.26])))
    assert relabel.hard(0.25, 0.05)
    assert not diff_answers(ref, ref).hard(0.25, 0.05)


def test_assignment_prefers_all_valid_pairs_over_more_total_iou():
    # 1D boxes of height 10. Raw total IoU prefers A-Y (1.0) + B-X (0.25, invalid) = 1.25 over the
    # all valid A-X (0.538) + B-Y (0.538) = 1.077. Every reference box has a valid partner, so the
    # answer did not change; the old matcher reported a vanished and an appeared box here.
    ref = dets([[0, 0, 10, 10], [-3, 0, 7, 10]], [1, 1])
    new = dets([[3, 0, 13, 10], [0, 0, 10, 10]], [1, 1])
    d = diff_answers(ref, new, 0.5)
    assert not d.flipped
    assert sorted((r, c) for r, c, _ in d.pairs) == [(0, 0), (1, 1)]


def test_assignment_still_maximises_iou_among_valid_pairs():
    ref = dets([[0, 0, 10, 10]], [1])
    new = dets([[1, 0, 11, 10], [0, 0, 10, 10]], [1, 1])
    d = diff_answers(ref, new, 0.5)
    assert d.pairs[0][:2] == (0, 1) and d.appeared == [0]
