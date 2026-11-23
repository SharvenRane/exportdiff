"""Per image detection answers and how two answers differ.

Definitions
-----------
answer      the set of (class, box) a deployed detector reports for one image: every box whose score
            is at or above the deployment confidence threshold, after the shared decoding and NMS.
match       class agnostic one to one assignment between the reference answer and the export answer
            (Hungarian) that maximises the number of pairs with IoU >= the match IoU (0.5), then their
            total IoU; only those pairs count as matched.
flip        an image whose answer changed: at least one box appeared (export only), vanished
            (reference only) or changed class (matched pair with different class labels).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MATCH_IOU = 0.5


@dataclass
class Detections:
    """Detections for one image in original pixel coordinates, xyxy boxes."""

    boxes: np.ndarray  # (k, 4) float
    scores: np.ndarray  # (k,) float
    classes: np.ndarray  # (k,) int
    second: np.ndarray | None = None  # (k,) score of the runner up class for each box, optional

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if self.second is not None:
            self.second = np.asarray(self.second, dtype=np.float64).reshape(-1)
        k = len(self.boxes)
        if len(self.scores) != k or len(self.classes) != k or (self.second is not None and len(self.second) != k):
            raise ValueError("boxes, scores, classes (and second) must have the same length")

    def __len__(self) -> int:
        return len(self.boxes)

    def above(self, threshold: float) -> "Detections":
        keep = self.scores >= threshold
        return Detections(self.boxes[keep], self.scores[keep], self.classes[keep],
                          None if self.second is None else self.second[keep])

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a (m, 4) and b (n, 4). Degenerate boxes give IoU 0."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


@dataclass
class AnswerDiff:
    """How the export answer differs from the reference answer on one image."""

    n_ref: int
    n_new: int
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (ref idx, new idx, IoU)
    vanished: list[int] = field(default_factory=list)  # ref indices with no partner
    appeared: list[int] = field(default_factory=list)  # new indices with no partner
    relabelled: list[tuple[int, int]] = field(default_factory=list)  # (ref idx, new idx) with class change
    vanished_scores: list[float] = field(default_factory=list)  # reference scores of vanished boxes
    appeared_scores: list[float] = field(default_factory=list)  # export scores of appeared boxes

    def hard(self, threshold: float, margin: float) -> bool:
        """A change that is not just a box grazing the threshold: a class change, or a box that
        vanished or appeared while its score sat at least `margin` above the threshold."""
        bar = threshold + margin
        return bool(self.relabelled) or any(s >= bar for s in self.vanished_scores + self.appeared_scores)

    @property
    def changes(self) -> int:
        return len(self.vanished) + len(self.appeared) + len(self.relabelled)

    @property
    def flipped(self) -> bool:
        return self.changes > 0

    @property
    def min_pair_iou(self) -> float:
        return min((p[2] for p in self.pairs), default=1.0)


def diff_answers(ref: Detections, new: Detections, match_iou: float = DEFAULT_MATCH_IOU) -> AnswerDiff:
    """Hungarian match (class agnostic, most valid pairs, then most IoU) and classify every unmatched or relabelled box.

    Both inputs must already be the deployed answers (filtered at the confidence threshold).
    """
    from scipy.optimize import linear_sum_assignment

    m, n = len(ref), len(new)
    out = AnswerDiff(n_ref=m, n_new=n)
    if m == 0 or n == 0:
        out.vanished = list(range(m))
        out.appeared = list(range(n))
        out.vanished_scores = [float(s) for s in ref.scores]
        out.appeared_scores = [float(s) for s in new.scores]
        return out
    iou = iou_matrix(ref.boxes, new.boxes)
    # Maximise the number of valid pairs (IoU >= match_iou) first and their total IoU second.
    # Maximising raw total IoU alone can prefer an assignment that contains an invalid pair over
    # one where every pair is valid, which reports a flip on an image whose answer did not change.
    # Each valid pair is worth more than any sum of IoUs (at most min(m, n)), invalid pairs nothing.
    valid = iou >= match_iou
    weight = np.where(valid, min(m, n) + 1.0 + iou, 0.0)
    rows, cols = linear_sum_assignment(-weight)
    matched_r, matched_c = set(), set()
    for r, c in zip(rows.tolist(), cols.tolist()):
        if iou[r, c] >= match_iou:
            matched_r.add(r)
            matched_c.add(c)
            out.pairs.append((r, c, float(iou[r, c])))
            if ref.classes[r] != new.classes[c]:
                out.relabelled.append((r, c))
    out.vanished = [i for i in range(m) if i not in matched_r]
    out.appeared = [j for j in range(n) if j not in matched_c]
    out.vanished_scores = [float(ref.scores[i]) for i in out.vanished]
    out.appeared_scores = [float(new.scores[j]) for j in out.appeared]
    return out


def image_margin(ref_all: Detections, threshold: float) -> float:
    """How close the reference answer on this image is to changing.

    The smaller of (a) the distance of any reference box score to the confidence threshold and
    (b) the gap between the best and runner up class score of any box above the threshold.
    Images with no boxes at all return +inf. Small margin = fragile answer.
    """
    if len(ref_all) == 0:
        return float("inf")
    m = float(np.min(np.abs(ref_all.scores - threshold)))
    if ref_all.second is not None:
        above = ref_all.scores >= threshold
        if above.any():
            m = min(m, float(np.min(ref_all.scores[above] - ref_all.second[above])))
    return m
