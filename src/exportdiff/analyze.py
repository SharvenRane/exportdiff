"""Second look at a finished run, from the cached per target detections.

Two questions the main run leaves open:

sensitivity   how much the hard flip rate depends on the hard margin and the deployment threshold.
              For every target and every (threshold, margin) pair this recomputes the plain flip
              count and the hard flip count with exact intervals, plus the same for the reference
              self runs so the noise floor is visible at every setting.

map interval  a paired bootstrap interval on the mAP change of each target. Images are resampled
              with replacement and the same resample is applied to the reference and the export,
              so the interval reflects how much the change depends on which images were drawn.
              pycocotools matching is done once per run (COCOeval.evaluate); each replicate only
              redoes the accumulate step with per image counts, which is exact and fast.
"""
from __future__ import annotations

import contextlib
import io
import pickle
from pathlib import Path

import numpy as np

from .compare import diff_runs, flip_vector, hard_flip_vector
from .data import COCO80_TO_91
from .match import DEFAULT_MATCH_IOU
from .stats import clopper_pearson

SELF_RUNS = ("torch_rerun", "torch_batch8", "torch_batch8_shuffled")


def sensitivity(ref: dict, runs: dict, ids: list, thresholds, margins, match_iou: float = DEFAULT_MATCH_IOU,
                level: float = 0.95) -> dict:
    """run name -> threshold -> {"flip": {...}, "hard": {margin: {...}}} with counts, rates, exact CIs."""
    out = {}
    n = len(ids)
    for name, run in runs.items():
        per_t = {}
        for t in thresholds:
            diffs = diff_runs(ref, run, ids, t, match_iou)
            f = flip_vector(diffs, ids)
            row = {"flip": _rate(f, level), "hard": {}}
            for m in margins:
                row["hard"][m] = _rate(hard_flip_vector(diffs, ids, t, m), level)
            per_t[t] = row
        out[name] = per_t
    out["_n"] = n
    return out


def _rate(flags: np.ndarray, level: float) -> dict:
    k, n = int(flags.sum()), int(flags.size)
    return {"k": k, "n": n, "rate": k / n, "ci_exact": clopper_pearson(k, n, level)}


class CocoPerImage:
    """pycocotools bbox evaluation split per image, so mAP can be recomputed for any image multiset.

    `map(counts)` gives (mAP50 95, mAP50) for the dataset in which image ids[i] appears counts[i]
    times. With all counts 1 it reproduces COCOeval.stats[0] and stats[1] (area all, 100 detections).
    """

    def __init__(self, run: dict, ids: list, gt, max_det: int = 100):
        from pycocotools.cocoeval import COCOeval

        self.ids = list(ids)
        pos = {i: j for j, i in enumerate(self.ids)}
        res = []
        for i in self.ids:
            d = run.get(i)
            if d is None:
                continue
            for b, s, c in zip(d.boxes, d.scores, d.classes):
                res.append({"image_id": i, "category_id": COCO80_TO_91[int(c)],
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                            "score": float(s)})
        self.cats = []
        self.empty = not res  # no detections at all: precision is 0 wherever ground truth exists
        if self.empty:
            return
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(res), "bbox")
        ev.params.imgIds = self.ids
        ev.params.maxDets = [max_det]
        ev.params.areaRng = [ev.params.areaRng[0]]
        ev.params.areaRngLbl = ["all"]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
        p = ev.params
        self.rec_thrs = np.asarray(p.recThrs)
        self.n_iou = len(p.iouThrs)
        n_img = len(p.imgIds)
        for k in range(len(p.catIds)):
            block = ev.evalImgs[k * n_img:(k + 1) * n_img]
            es = [e for e in block if e is not None]
            if not es:
                self.cats.append(None)
                continue
            npig = np.zeros(len(self.ids), dtype=np.int64)
            scores, tps, fps, owner = [], [], [], []
            for e in es:
                j = pos[e["image_id"]]
                npig[j] += int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
                sc = np.asarray(e["dtScores"][:max_det], dtype=np.float64)
                if sc.size == 0:
                    continue
                dtm = np.asarray(e["dtMatches"])[:, :max_det]
                dig = np.asarray(e["dtIgnore"])[:, :max_det].astype(bool)
                scores.append(sc)
                tps.append(np.logical_and(dtm != 0, ~dig))
                fps.append(np.logical_and(dtm == 0, ~dig))
                owner.append(np.full(sc.size, j, dtype=np.int64))
            if scores:
                sc = np.concatenate(scores)
                order = np.argsort(-sc, kind="mergesort")  # same stable order as COCOeval.accumulate
                cat = {"tp": np.concatenate(tps, axis=1)[:, order], "fp": np.concatenate(fps, axis=1)[:, order],
                       "owner": np.concatenate(owner)[order], "npig": npig}
            else:
                cat = {"tp": np.zeros((self.n_iou, 0), bool), "fp": np.zeros((self.n_iou, 0), bool),
                       "owner": np.zeros(0, np.int64), "npig": npig}
            self.cats.append(cat)

    def precision(self, counts: np.ndarray | None = None) -> np.ndarray:
        """(T, R, K) interpolated precision, -1 where a category has no ground truth in the sample."""
        c = np.ones(len(self.ids), dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64)
        n_r = len(self.rec_thrs)
        prec = -np.ones((self.n_iou, n_r, len(self.cats)))
        eps = np.spacing(1)
        for k, cat in enumerate(self.cats):
            if cat is None:
                continue
            npig = int((cat["npig"] * c).sum())
            if npig == 0:
                continue
            rep = c[cat["owner"]]
            sel = np.repeat(np.arange(len(rep)), rep)
            tp_sum = np.cumsum(cat["tp"][:, sel], axis=1, dtype=np.float64)
            fp_sum = np.cumsum(cat["fp"][:, sel], axis=1, dtype=np.float64)
            nd = tp_sum.shape[1]
            for t in range(self.n_iou):
                q = np.zeros(n_r)
                if nd:
                    tp, fp = tp_sum[t], fp_sum[t]
                    rc = tp / npig
                    pr = tp / (fp + tp + eps)
                    pr = np.maximum.accumulate(pr[::-1])[::-1]
                    inds = np.searchsorted(rc, self.rec_thrs, side="left")
                    ok = inds < nd
                    q[ok] = pr[inds[ok]]
                prec[t, :, k] = q
        return prec

    def map(self, counts: np.ndarray | None = None) -> tuple[float, float]:
        if self.empty:
            return 0.0, 0.0
        p = self.precision(counts)
        allv = p[p > -1]
        p50 = p[0][p[0] > -1]
        return (float(allv.mean()) if allv.size else -1.0, float(p50.mean()) if p50.size else -1.0)


def bootstrap_counts(n: int, b: int, seed: int = 0):
    """Yield b reproducible resample count vectors of length n (each sums to n)."""
    rng = np.random.default_rng(seed)
    for _ in range(b):
        yield np.bincount(rng.integers(0, n, size=n), minlength=n)


def map_series(ev: CocoPerImage, b: int, seed: int = 0) -> np.ndarray:
    """(b, 2) array of (mAP50 95, mAP50) over the bootstrap resamples of `bootstrap_counts`."""
    return np.array([ev.map(c) for c in bootstrap_counts(len(ev.ids), b, seed)])


def paired_interval(ref_series: np.ndarray, new_series: np.ndarray, level: float = 0.95) -> dict:
    """Percentile interval of the per replicate change new minus ref (same resamples on both sides)."""
    d = np.asarray(new_series) - np.asarray(ref_series)
    a = (1.0 - level) / 2.0
    lo = np.percentile(d, 100 * a, axis=0)
    hi = np.percentile(d, 100 * (1 - a), axis=0)
    return {"map50_95": (float(lo[0]), float(hi[0])), "map50": (float(lo[1]), float(hi[1])),
            "b": int(d.shape[0]), "level": level}


def load_runs(runs_dir: str | Path, names) -> dict:
    out = {}
    for n in names:
        with open(Path(runs_dir) / f"{n}.pkl", "rb") as fh:
            out[n] = pickle.load(fh)["run"]
    return out


def _series_job(args):
    runs_dir, name, ids, ann, b, seed = args
    from .cocoeval import load_coco

    run = load_runs(runs_dir, [name])[name]
    ev = CocoPerImage(run, ids, load_coco(ann))
    return name, ev.map(), map_series(ev, b, seed)


def analyze(runs_dir: str | Path, targets: list[str], ann: str | None, thresholds=(0.25, 0.30),
            margins=(0.02, 0.05, 0.10), b: int = 1000, seed: int = 0, jobs: int = 1,
            match_iou: float = DEFAULT_MATCH_IOU) -> dict:
    """Sensitivity table and paired bootstrap mAP intervals for every target in a cached run folder."""
    present = [t for t in targets if (Path(runs_dir) / f"{t}.pkl").exists()]
    selfs = [s for s in SELF_RUNS if (Path(runs_dir) / f"{s}.pkl").exists()]
    runs = load_runs(runs_dir, ["torch"] + selfs + present)
    ref = runs.pop("torch")
    ids = list(ref.keys())
    out = {"n_images": len(ids), "targets": present, "self_runs": selfs,
           "sensitivity": sensitivity(ref, runs, ids, thresholds, margins, match_iou)}
    if ann:
        names = ["torch"] + present
        work = [(str(runs_dir), n, ids, ann, b, seed) for n in names]
        if jobs > 1:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(min(jobs, len(work))) as pool:
                got = pool.map(_series_job, work)
        else:
            got = [_series_job(w) for w in work]
        point = {n: m for n, m, _ in got}
        series = {n: s for n, _, s in got}
        out["map"] = {n: {"map50_95": point[n][0], "map50": point[n][1]} for n in names}
        out["map_change"] = {}
        for t in present:
            iv = paired_interval(series["torch"], series[t])
            out["map_change"][t] = {"map50_95": point[t][0] - point["torch"][0],
                                    "map50": point[t][1] - point["torch"][1], "ci": iv}
    return out
