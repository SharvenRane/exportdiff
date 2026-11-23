"""Command line: exportdiff run --model yolo11n.pt --data coco/val2017 --targets onnx,onnx-int8"""
from __future__ import annotations

import argparse
import sys

from .compare import DEFAULT_PROBE, DEFAULT_THRESHOLD
from .runtimes import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="exportdiff", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compare exports of a detector against its PyTorch reference")
    r.add_argument("--model", required=True, help="Ultralytics detection weights, for example yolo11n.pt")
    r.add_argument("--data", required=True, help="directory of images, for example coco/val2017")
    r.add_argument("--ann", help="COCO instances json for mAP (optional)")
    r.add_argument("--targets", required=True, help=f"comma separated, from: {', '.join(REGISTRY)}")
    r.add_argument("--calib", help="directory of calibration images for INT8 targets")
    r.add_argument("--calib-n", type=int, default=128)
    r.add_argument("--limit", type=int, help="use only the first N images (sorted by name)")
    r.add_argument("--out", default="results")
    r.add_argument("--work", default="work", help="cache for exports and per target detections")
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="deployment confidence threshold")
    r.add_argument("--probe", type=int, default=DEFAULT_PROBE, help="probe set size")
    r.add_argument("--worst", type=int, default=12, help="images drawn per target in the report")
    r.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze", help="hard margin sensitivity and paired bootstrap mAP intervals from a cached run")
    a.add_argument("--runs", required=True, help="cached detections folder, for example work/runs/yolo11n_all")
    a.add_argument("--targets", required=True, help="comma separated target names to analyse")
    a.add_argument("--ann", help="COCO instances json; without it only the sensitivity table is computed")
    a.add_argument("--thresholds", default="0.25,0.30", help="deployment confidence thresholds")
    a.add_argument("--margins", default="0.02,0.05,0.10", help="hard flip margins above the threshold")
    a.add_argument("--boot", type=int, default=1000, help="bootstrap resamples of the images")
    a.add_argument("--jobs", type=int, default=1, help="worker processes for the bootstrap")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True, help="output json")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "run":
        from .pipeline import Config, run

        targets = [t.strip() for t in a.targets.split(",") if t.strip()]
        unknown = [t for t in targets if t not in REGISTRY]
        if unknown:
            print(f"unknown targets: {unknown}; known: {list(REGISTRY)}", file=sys.stderr)
            return 2
        cfg = Config(model=a.model, data=a.data, ann=a.ann, targets=targets, calib=a.calib, calib_n=a.calib_n,
                     limit=a.limit, out=a.out, work=a.work, imgsz=a.imgsz, threshold=a.threshold, probe=a.probe,
                     worst=a.worst, seed=a.seed)
        res = run(cfg)
        return 1 if res["errors"] else 0
    if a.cmd == "analyze":
        from .analyze import analyze
        from .report import write_json

        floats = lambda s: [float(x) for x in s.split(",") if x.strip()]
        res = analyze(a.runs, [t.strip() for t in a.targets.split(",") if t.strip()], a.ann,
                      floats(a.thresholds), floats(a.margins), a.boot, a.seed, a.jobs)
        from pathlib import Path

        write_json(res, Path(a.out))
        print(f"wrote {a.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
