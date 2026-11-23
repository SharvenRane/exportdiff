# exportdiff

Your exported detector kept its mAP and still changed the boxes on some images. `exportdiff` runs the
PyTorch reference and each export on the same images, with the same preprocessing and the same
decoding plus NMS, and reports which images changed their answer, whether that is beyond the
reference's own noise, and whether a cheap 200 image probe set would have caught it.

Status: early prototype, private. First numbers are in [FIRST_RESULTS.md](FIRST_RESULTS.md).

## Method

* **Answer** per image: the set of (class, box) at or above the deployment confidence threshold
  (default 0.25) after shared decoding and class aware NMS (IoU 0.7).
* **Match**: class agnostic Hungarian assignment maximising IoU, pairs kept at IoU 0.5 or more.
* **Flip**: an image where any box appeared, vanished or changed class.
* **Hard flip**: a flip that is more than a box grazing the threshold: a class change, or a box that
  vanished or appeared with a score at least 0.05 above the threshold.
* **Noise floor**: the reference against itself under a rerun, batch size 8, and batch size 8 with a
  shuffled order. An export is flagged only when the exact lower bound of its flip rate clears the
  largest self run flip rate.
* **Probe set**: the 200 images whose reference answer has the smallest margin (distance of any box
  score to the threshold, or gap between best and runner up class).
* **Statistics**: exact Clopper Pearson intervals, bootstrap intervals, and an exact McNemar test on
  images flipped by the export but not by noise versus the reverse.
* **mAP**: pycocotools on the same decoded boxes (confidence 0.001, 300 detections).

## Targets

| target | what |
|---|---|
| `onnx` | ONNX Runtime CPU, FP32, Ultralytics ONNX export |
| `onnx-int8` | ONNX Runtime static INT8 (QDQ, per channel, MinMax), box decoding tail kept in float |
| `onnx-int8-naive` | same quantizer with nothing excluded |
| `openvino` | OpenVINO CPU, FP32 IR, inference precision forced to f32 |
| `openvino-default` | OpenVINO CPU, FP32 IR, plugin default precision (bf16 on CPUs that support it) |
| `openvino-int8` | OpenVINO INT8 IR from the Ultralytics export (NNCF) |
| `tensorrt-fp16`, `tensorrt-int8` | stubs, need an NVIDIA GPU |

## Usage

```
pip install -e ".[run,openvino,test]"
exportdiff run --model yolo11n.pt --data coco/val2017 --ann coco/annotations/instances_val2017.json \
    --calib coco128/images/train2017 --targets onnx,onnx-int8,openvino-int8 --limit 1000 --out results/run1
```

Writes `result.json` and `report.html` (with the worst 12 images per target drawn side by side) to
`--out`, and caches exports and per target detections in `--work`.

Tests: `pytest` (no model download).

## License

MIT
