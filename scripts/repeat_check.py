"""Repeat check: is the flip rate a property of the export, or of one lucky run?

Runs, in a fresh process, (1) the PyTorch reference again, (2) OpenVINO INT8 on the already cached IR,
and (3) OpenVINO INT8 exported again from scratch (new NNCF calibration) into an empty work folder.
Each is compared with the cached runs of the first measurement.

python scripts/repeat_check.py --model yolo11n.pt --data coco/val2017 --calib coco128/images/train2017 \
    --runs work/runs/yolo11n_all --cache work --fresh work_repeat --out results/repeat_yolo11n.json
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from exportdiff.compare import diff_runs, flip_vector, hard_flip_vector
from exportdiff.data import image_id, list_images
from exportdiff.pipeline import Config, log, run_pass
from exportdiff.report import write_json
from exportdiff.runtimes import OpenVINOInt8, TorchReference
from exportdiff.stats import clopper_pearson


def counts(a: dict, b: dict, ids: list, t: float = 0.25, m: float = 0.05) -> dict:
    d = diff_runs(a, b, ids, t)
    f, h = flip_vector(d, ids), hard_flip_vector(d, ids, t, m)
    n = len(ids)
    return {"flips": int(f.sum()), "hard": int(h.sum()), "n": n, "rate": float(f.mean()),
            "ci_exact": clopper_pearson(int(f.sum()), n), "hard_rate": float(h.mean()),
            "hard_ci_exact": clopper_pearson(int(h.sum()), n), "flagged": f}


def identical(a: dict, b: dict, ids: list) -> bool:
    return all(np.array_equal(a[i].boxes, b[i].boxes) and np.array_equal(a[i].scores, b[i].scores)
               and np.array_equal(a[i].classes, b[i].classes) for i in ids)


def bin_md5(folder: Path) -> str:
    bins = sorted(folder.rglob("*.bin"))
    return hashlib.md5(b"".join(p.read_bytes() for p in bins)).hexdigest() if bins else ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--runs", required=True, help="cached runs of the first measurement")
    p.add_argument("--cache", required=True, help="work folder of the first measurement (holds its IR)")
    p.add_argument("--fresh", required=True, help="empty work folder for the new export")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = Config(model=a.model, data=a.data, targets=[])
    files = list_images(a.data)
    ids = [image_id(f) for f in files]
    calib = list_images(a.calib)[:128]
    old = {}
    for k in ("torch", "openvino-int8"):
        with open(Path(a.runs) / f"{k}.pkl", "rb") as fh:
            old[k] = pickle.load(fh)["run"]
    weights = Path(a.model)
    cache, fresh = Path(a.cache), Path(a.fresh)
    fresh.mkdir(parents=True, exist_ok=True)

    log("reference, new process")
    rt = TorchReference(batch_size=1)
    rt.prepare(weights, cache, calib, cfg.imgsz)
    torch2, _ = run_pass(rt, files, cfg)

    log("openvino-int8, cached IR, new process")
    ov = OpenVINOInt8()
    ov.prepare(weights, cache, calib, cfg.imgsz)
    ov_same, _ = run_pass(ov, files, cfg)

    log("openvino-int8, exported again from scratch")
    ov2 = OpenVINOInt8()
    ov2.prepare(weights, fresh, calib, cfg.imgsz)
    ov_new, _ = run_pass(ov2, files, cfg)

    stem = weights.stem
    first = counts(old["torch"], old["openvino-int8"], ids)
    second = counts(torch2, ov_new, ids)
    both = first.pop("flagged") & second.pop("flagged")
    res = {
        "n_images": len(ids),
        "reference_repeat_vs_first": counts(old["torch"], torch2, ids),
        "reference_repeat_bitwise_identical": identical(old["torch"], torch2, ids),
        "int8_same_ir_vs_first": counts(old["openvino-int8"], ov_same, ids),
        "int8_same_ir_bitwise_identical": identical(old["openvino-int8"], ov_same, ids),
        "int8_reexport_vs_first_int8": counts(old["openvino-int8"], ov_new, ids),
        "int8_weights_identical": bin_md5(cache / f"{stem}_int8_openvino_model")
        == bin_md5(fresh / f"{stem}_int8_openvino_model"),
        "first_measurement_vs_reference": first,
        "repeat_vs_repeat_reference": second,
        "images_flipped_in_both": int(both.sum()),
    }
    for v in res.values():
        if isinstance(v, dict):
            v.pop("flagged", None)
    write_json(res, Path(a.out))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
