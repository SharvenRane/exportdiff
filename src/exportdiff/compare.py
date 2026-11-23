"""Compare a run of per image detections against the reference run.

A run is a dict image_id -> Detections holding every decoded box (low confidence included, so the
same run also feeds mAP). The deployed answer is taken at `threshold` inside this module.

Noise floor
-----------
The reference model compared with itself under changes that should not matter (a rerun, a
different batch size, a shuffled image order). An export is only flagged when the exact lower
bound of its flip rate clears the noise floor, the largest flip rate seen among those self runs.

Probe set
---------
The `probe_size` images whose reference answer has the smallest margin (see match.image_margin).
The question is whether checking only those images would have caught the export.
"""
from __future__ import annotations

import numpy as np

from .match import DEFAULT_MATCH_IOU, AnswerDiff, Detections, diff_answers, image_margin
from .stats import auroc, clopper_pearson, mcnemar_exact, rate_summary

DEFAULT_THRESHOLD = 0.25
DEFAULT_PROBE = 200
DEFAULT_HARD_MARGIN = 0.05


def diff_runs(ref: dict, new: dict, ids: list, threshold: float = DEFAULT_THRESHOLD,
              match_iou: float = DEFAULT_MATCH_IOU) -> dict:
    """Per image AnswerDiff for every id (missing ids in `new` count as empty answers)."""
    out = {}
    for i in ids:
        r = ref.get(i, Detections.empty()).above(threshold)
        n = new.get(i, Detections.empty()).above(threshold)
        out[i] = diff_answers(r, n, match_iou)
    return out


def flip_vector(diffs: dict, ids: list) -> np.ndarray:
    return np.array([diffs[i].flipped for i in ids], dtype=bool)


def hard_flip_vector(diffs: dict, ids: list, threshold: float = DEFAULT_THRESHOLD,
                     margin: float = DEFAULT_HARD_MARGIN) -> np.ndarray:
    """Images whose answer changed by more than a box grazing the threshold (see AnswerDiff.hard)."""
    return np.array([diffs[i].hard(threshold, margin) for i in ids], dtype=bool)


def noise_floor(self_runs: dict[str, np.ndarray], level: float = 0.95) -> dict:
    """Summarise reference against itself runs. `self_runs` maps a run name to its flip vector."""
    if not self_runs:
        raise ValueError("need at least one self run for a noise floor")
    per = {name: rate_summary(v, level) for name, v in self_runs.items()}
    worst = max(per, key=lambda k: per[k]["rate"])
    any_flip = np.logical_or.reduce([np.asarray(v, dtype=bool) for v in self_runs.values()])
    return {
        "runs": per,
        "floor": per[worst]["rate"],
        "floor_run": worst,
        "floor_upper_exact": per[worst]["ci_exact"][1],
        "union_flags": any_flip,
    }


def margins_for(ref: dict, ids: list, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return np.array([image_margin(ref.get(i, Detections.empty()), threshold) for i in ids], dtype=np.float64)


def probe_indices(margins: np.ndarray, size: int = DEFAULT_PROBE) -> np.ndarray:
    """Positions of the `size` smallest margins (stable, ties broken by position)."""
    size = min(size, len(margins))
    return np.argsort(margins, kind="stable")[:size]


def summarise_target(name: str, diffs: dict, ids: list, noise: dict, margins: np.ndarray,
                     probe: np.ndarray, level: float = 0.95, threshold: float = DEFAULT_THRESHOLD,
                     hard_margin: float = DEFAULT_HARD_MARGIN, hard_floor: float | None = None) -> dict:
    flips = flip_vector(diffs, ids)
    s = rate_summary(flips, level)
    floor = noise["floor"]
    union = noise["union_flags"]
    # paired test: does this export flip images that the reference noise does not
    b = int((flips & ~union).sum())
    c = int((~flips & union).sum())
    pf = flips[probe]
    k_probe = int(pf.sum())
    probe_ci = clopper_pearson(k_probe, len(probe), level) if len(probe) else (float("nan"), float("nan"))
    finite = np.isfinite(margins)
    out = {
        "target": name,
        "flip": s,
        "above_noise": bool(s["k"] > 0 and s["ci_exact"][0] > floor),
        "excess_over_noise": s["rate"] - floor,
        "mcnemar_vs_noise": {"export_only": b, "noise_only": c, "p": mcnemar_exact(b, c)},
        "boxes": {
            "vanished": int(sum(len(diffs[i].vanished) for i in ids)),
            "appeared": int(sum(len(diffs[i].appeared) for i in ids)),
            "relabelled": int(sum(len(diffs[i].relabelled) for i in ids)),
            "ref_boxes": int(sum(diffs[i].n_ref for i in ids)),
            "new_boxes": int(sum(diffs[i].n_new for i in ids)),
            "images_vanish": int(sum(bool(diffs[i].vanished) for i in ids)),
            "images_appear": int(sum(bool(diffs[i].appeared) for i in ids)),
            "images_relabel": int(sum(bool(diffs[i].relabelled) for i in ids)),
        },
        "hard_flip": _hard(diffs, ids, threshold, hard_margin, hard_floor, level),
        "probe": {
            "size": int(len(probe)),
            "flips": k_probe,
            "rate": k_probe / len(probe) if len(probe) else float("nan"),
            "ci_exact": probe_ci,
            "caught": bool(k_probe > 0 and probe_ci[0] > floor),
            "share_of_all_flips": (k_probe / s["k"]) if s["k"] else float("nan"),
            "margin_auroc": auroc(-np.where(finite, margins, 1e9), flips) if 0 < s["k"] < len(ids) else float("nan"),
        },
    }
    return out


def worst_images(diffs: dict, ids: list, k: int = 12) -> list:
    """Ids with the most box changes, ties broken by the lowest matched IoU then by id."""
    flipped = [i for i in ids if diffs[i].flipped]
    flipped.sort(key=lambda i: (-diffs[i].changes, diffs[i].min_pair_iou, str(i)))
    return flipped[:k]


__all__ = [
    "AnswerDiff",
    "diff_runs",
    "flip_vector",
    "noise_floor",
    "margins_for",
    "probe_indices",
    "summarise_target",
    "worst_images",
]


def _hard(diffs: dict, ids: list, threshold: float, margin: float, floor: float | None, level: float) -> dict:
    s = rate_summary(hard_flip_vector(diffs, ids, threshold, margin), level)
    s["margin"] = margin
    s["noise_floor"] = floor
    s["above_noise"] = None if floor is None else bool(s["k"] > 0 and s["ci_exact"][0] > floor)
    return s
