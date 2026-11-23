"""End to end run: reference passes, noise floor, every target, mAP, answer diff, report."""
from __future__ import annotations

import pickle
import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .compare import (DEFAULT_PROBE, DEFAULT_THRESHOLD, diff_runs, flip_vector, hard_flip_vector, margins_for,
                      noise_floor, probe_indices, summarise_target, worst_images)
from .data import image_id, list_images, preprocess
from .decode import DEFAULT_CONF, DEFAULT_MAX_DET, DEFAULT_NMS_IOU, decode
from .match import DEFAULT_MATCH_IOU
from .stats import rate_summary
from .runtimes import TorchReference, make


@dataclass
class Config:
    model: str
    data: str
    targets: list[str]
    ann: str | None = None
    calib: str | None = None
    calib_n: int = 128
    limit: int | None = None
    out: str = "results"
    work: str = "work"
    imgsz: int = 640
    threshold: float = DEFAULT_THRESHOLD
    match_iou: float = DEFAULT_MATCH_IOU
    nms_iou: float = DEFAULT_NMS_IOU
    conf: float = DEFAULT_CONF
    max_det: int = DEFAULT_MAX_DET
    probe: int = DEFAULT_PROBE
    worst: int = 12
    seed: int = 0
    n_images: int = 0
    extra: dict = field(default_factory=dict)


def log(msg: str) -> None:
    print(f"[exportdiff {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_pass(rt, files: list[Path], cfg: Config, order: np.ndarray | None = None) -> tuple[dict, float]:
    """Run one runtime over the files in `order` with its batch size; returns id -> Detections."""
    order = np.arange(len(files)) if order is None else order
    out = {}
    t0 = time.perf_counter()
    bs = rt.batch_size
    for s in range(0, len(order), bs):
        chunk = [files[j] for j in order[s:s + bs]]
        pre = [preprocess(f, cfg.imgsz) for f in chunk]
        raw = rt.infer(np.stack([p[0] for p in pre]))
        for f, (_, lb), r in zip(chunk, pre, raw):
            out[image_id(f)] = decode(r, lb, cfg.conf, cfg.nms_iou, cfg.max_det)
        if (s // bs) % 500 == 0:
            log(f"  {rt.name}: {min(s + bs, len(order))} / {len(order)}")
    return out, time.perf_counter() - t0


def _cached(path: Path, fn):
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    val = fn()
    with open(path, "wb") as fh:
        pickle.dump(val, fh)
    return val


def run(cfg: Config) -> dict:
    out = Path(cfg.out)
    work = Path(cfg.work)
    runs_dir = work / "runs" / f"{Path(cfg.model).stem}_{cfg.limit or 'all'}"
    out.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    files = list_images(cfg.data, cfg.limit)
    ids = [image_id(f) for f in files]
    cfg.n_images = len(files)
    calib = list_images(cfg.calib)[: cfg.calib_n] if cfg.calib else []
    weights = Path(cfg.model)
    log(f"{len(files)} images, {len(calib)} calibration images, targets {cfg.targets}")

    infos, timings, runs, errors = {}, {}, {}, {}

    def ref_pass(key: str, bs: int, order=None):
        def go():
            rt = TorchReference(batch_size=bs)
            rt.prepare(weights, work, calib, cfg.imgsz)
            infos[key] = rt.info()
            r, dt = run_pass(rt, files, cfg, order)
            return {"run": r, "seconds": dt, "info": rt.info(), "names": rt.names}
        log(f"reference pass {key} (batch {bs})")
        d = _cached(runs_dir / f"{key}.pkl", go)
        runs[key], timings[key], infos[key] = d["run"], d["seconds"], d["info"]
        return d

    ref_d = ref_pass("torch", 1)
    names = ref_d["names"]
    ref_pass("torch_rerun", 1)
    ref_pass("torch_batch8", 8)
    shuffled = np.random.default_rng(cfg.seed).permutation(len(files))
    ref_pass("torch_batch8_shuffled", 8, shuffled)

    for t in cfg.targets:
        def go(t=t):
            rt = make(t)
            rt.prepare(weights, work, calib, cfg.imgsz)
            r, dt = run_pass(rt, files, cfg)
            return {"run": r, "seconds": dt, "info": rt.info()}
        log(f"target {t}")
        try:
            d = _cached(runs_dir / f"{t}.pkl", go)
            runs[t], timings[t], infos[t] = d["run"], d["seconds"], d["info"]
        except Exception as e:  # one broken target must not sink the others
            errors[t] = f"{type(e).__name__}: {e}"
            log(f"target {t} failed: {errors[t]}")
            traceback.print_exc()

    ref = runs["torch"]
    self_names = ["torch_rerun", "torch_batch8", "torch_batch8_shuffled"]
    self_diffs = {k: diff_runs(ref, runs[k], ids, cfg.threshold, cfg.match_iou) for k in self_names}
    noise = noise_floor({k: flip_vector(v, ids) for k, v in self_diffs.items()})
    hard_noise = {k: rate_summary(hard_flip_vector(v, ids, cfg.threshold), 0.95) for k, v in self_diffs.items()}
    hard_floor = max(v["rate"] for v in hard_noise.values())
    margins = margins_for(ref, ids, cfg.threshold)
    probe = probe_indices(margins, cfg.probe)

    summary, diffs = {}, {}
    for t in cfg.targets:
        if t not in runs:
            continue
        diffs[t] = diff_runs(ref, runs[t], ids, cfg.threshold, cfg.match_iou)
        summary[t] = summarise_target(t, diffs[t], ids, noise, margins, probe,
                                      threshold=cfg.threshold, hard_floor=hard_floor)
        f = summary[t]["flip"]
        log(f"{t}: {f['k']}/{f['n']} images changed ({100 * f['rate']:.2f}%), above noise {summary[t]['above_noise']}")

    maps = {}
    if cfg.ann:
        from .cocoeval import coco_map, load_coco

        gt = load_coco(cfg.ann)
        for k in ["torch"] + self_names + [t for t in cfg.targets if t in runs]:
            maps[k] = coco_map(runs[k], ids, cfg.ann, gt)
            log(f"mAP {k}: {100 * maps[k]['map50_95']:.2f} (mAP50 {100 * maps[k]['map50']:.2f})")

    from .report import draw_pair, write_html, write_json

    gal_dir = out / "gallery"
    gal_dir.mkdir(exist_ok=True)
    galleries = {}
    by_id = dict(zip(ids, files))
    for t, dd in diffs.items():
        items = []
        for i in worst_images(dd, ids, cfg.worst):
            name = f"{t}_{i}.jpg"
            draw_pair(by_id[i], ref[i].above(cfg.threshold), runs[t][i].above(cfg.threshold), dd[i], names, t,
                      gal_dir / name)
            items.append((i, f"gallery/{name}", dd[i].changes))
        galleries[t] = items

    result = {
        "exportdiff": __version__,
        "config": {k: v for k, v in vars(cfg).items()},
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "processor": platform.processor()},
        "runtimes": infos,
        "seconds": timings,
        "targets": cfg.targets,
        "errors": errors,
        "noise": noise,
        "hard_noise": {"runs": hard_noise, "floor": hard_floor},
        "map": maps,
        "summary": summary,
        "probe_ids": [ids[j] for j in probe],
        "probe_max_margin": float(margins[probe].max()) if len(probe) else None,
        "flipped_ids": {t: [i for i in ids if dd[i].flipped] for t, dd in diffs.items()},
        "noise_flipped_ids": {k: [i for i in ids if v[i].flipped] for k, v in self_diffs.items()},
        "worst": {t: [g[0] for g in v] for t, v in galleries.items()},
    }
    write_json(result, out / "result.json")
    write_html(result, galleries, out / "report.html")
    log(f"wrote {out / 'result.json'} and {out / 'report.html'}")
    return result
