"""Shared decoding and NMS, run once in common code for every runtime.

Input is the raw head output of an Ultralytics style detector with NMS left out of the graph:
shape (4 + num_classes, anchors), rows 0..3 = box center x, center y, width, height in letterbox
pixels, remaining rows = per class scores already passed through the sigmoid.
"""
from __future__ import annotations

import numpy as np

from .data import Letterbox
from .match import Detections

DEFAULT_CONF = 0.001  # low, so the same decoded boxes also feed mAP
DEFAULT_NMS_IOU = 0.7
DEFAULT_MAX_DET = 300
MAX_NMS = 30000


def decode(raw: np.ndarray, lb: Letterbox, conf: float = DEFAULT_CONF, iou: float = DEFAULT_NMS_IOU,
           max_det: int = DEFAULT_MAX_DET, max_nms: int = MAX_NMS) -> Detections:
    import torch
    from torchvision.ops import batched_nms

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] < 6:
        raise ValueError(f"expected raw head output (4 + classes, anchors), got {raw.shape}")
    box = raw[:4].T
    cls = raw[4:].T
    best = cls.argmax(axis=1)
    s1 = cls[np.arange(len(cls)), best]
    keep = s1 >= conf
    if not keep.any():
        return Detections.empty()
    box, cls, best, s1 = box[keep], cls[keep], best[keep], s1[keep]
    s2 = np.partition(cls, -2, axis=1)[:, -2]
    if len(s1) > max_nms:
        top = np.argsort(-s1, kind="stable")[:max_nms]
        box, best, s1, s2 = box[top], best[top], s1[top], s2[top]
    xyxy = np.empty_like(box)
    xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
    xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
    xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
    xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
    k = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(s1), torch.from_numpy(best), iou)
    k = k[:max_det].numpy()
    xyxy, best, s1, s2 = xyxy[k].astype(np.float64), best[k], s1[k], s2[k]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - lb.pad_x) / lb.gain
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - lb.pad_y) / lb.gain
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, lb.width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, lb.height)
    return Detections(xyxy, s1, best, s2)
