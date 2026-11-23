"""COCO mAP on the same decoded boxes the answer diff uses."""
from __future__ import annotations

import contextlib
import io

from .data import COCO80_TO_91


def coco_map(run: dict, ids: list, ann_file: str, coco=None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = coco or COCO(ann_file)
    res = []
    for i in ids:
        d = run.get(i)
        if d is None:
            continue
        for b, s, c in zip(d.boxes, d.scores, d.classes):
            res.append({"image_id": i, "category_id": COCO80_TO_91[int(c)],
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s)})
    if not res:
        return {"map50_95": 0.0, "map50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"map50_95": float(ev.stats[0]), "map50": float(ev.stats[1])}


def load_coco(ann_file: str):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(ann_file)
