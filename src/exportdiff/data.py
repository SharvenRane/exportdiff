"""Image loading and the one shared preprocessing every runtime receives.

Every runtime is fed the exact same float32 tensor for a given image, so any difference in the
answer comes from the runtime or the export, never from preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# COCO 80 class index (Ultralytics order) to the 91 id COCO category ids.
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


@dataclass
class Letterbox:
    gain: float
    pad_x: float
    pad_y: float
    width: int  # original image width
    height: int


def list_images(root: str | Path, limit: int | None = None) -> list[Path]:
    """Images in `root`, sorted by file name (COCO ids sort the same way), optionally the first `limit`."""
    files = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images found in {root}")
    return files[:limit] if limit else files


def image_id(path: Path):
    """COCO style integer id from the file stem when it is numeric, else the stem."""
    s = path.stem
    return int(s) if s.isdigit() else s


def letterbox(img_rgb: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to size x size, same rounding as the Ultralytics LetterBox."""
    import cv2

    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, Letterbox(gain=r, pad_x=left, pad_y=top, width=w, height=h)


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(path: Path, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    """CHW float32 in [0, 1] plus the letterbox needed to map boxes back."""
    img, lb = letterbox(read_rgb(path), size)
    x = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return x, lb
