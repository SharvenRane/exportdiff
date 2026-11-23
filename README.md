# exportdiff

Status: work in progress. The method and the code run end to end, but the results below are
preliminary and cover a narrow setting. Nothing here is a release.

An exported or quantized detector can keep its mAP and still give a different answer on many
individual images. `exportdiff` measures that directly. It runs the PyTorch reference and each export
on the same images, with the same preprocessing and the same decoding plus NMS, and reports:

* which images changed their answer, with exact intervals;
* whether that is more than the reference disagreeing with itself (the noise floor);
* how many of those changes are more than a box grazing the confidence threshold (hard flips);
* whether a cheap probe set of the 200 most fragile images would have caught the export;
* the mAP change, with a paired bootstrap interval.

## The one command

```
exportdiff run --model yolo11n.pt --data coco/val2017 --ann coco/annotations/instances_val2017.json \
    --calib coco128/images/train2017 --targets onnx,openvino-default,openvino-int8 --out results/run1
```

This writes `result.json` and `report.html` (the worst images per target drawn side by side) to
`--out` and caches exports and per target detections in `--work`. A second command reads that cache
and adds the threshold and margin sensitivity table and the mAP intervals:

```
exportdiff analyze --runs work/runs/yolo11n_all --targets onnx,openvino-default,openvino-int8 \
    --ann coco/annotations/instances_val2017.json --jobs 8 --out results/analysis_yolo11n.json
```

## Preliminary results

Preliminary, CPU only, two models, COCO val2017 (5,000 images). YOLO11n and YOLO26n from
Ultralytics 8.4, one AMD Ryzen 7 9700X box, torch 2.15 CPU, ONNX Runtime 1.32, OpenVINO 2026.4,
NNCF 3.4. Deployment threshold 0.25, hard margin 0.05 unless stated. Intervals are 95%: exact
binomial over images for flip rates, paired image bootstrap (1,000 resamples) for the mAP change.

| model | target | mAP50 95 | mAP change (95% CI) | images whose answer changed (95% CI) | hard flips (95% CI) |
|---|---|---|---|---|---|
| YOLO11n | PyTorch reference | 38.73 | | | |
| YOLO11n | ONNX Runtime FP32 | 38.73 | 0.00 (0.00 to 0.00) | 0.00% (0.00 to 0.07) | 0.00% (0.00 to 0.07) |
| YOLO11n | OpenVINO FP32, f32 forced | 38.73 | 0.00 (0.00 to 0.00) | 0.00% (0.00 to 0.07) | 0.00% (0.00 to 0.07) |
| YOLO11n | OpenVINO FP32 IR, default precision (bf16) | 38.74 | +0.01 (-0.07 to +0.07) | 12.62% (11.71 to 13.57) | 1.76% (1.41 to 2.16) |
| YOLO11n | OpenVINO INT8 (Ultralytics export, NNCF) | 38.55 | -0.18 (-0.33 to -0.04) | 36.46% (35.12 to 37.81) | 14.64% (13.67 to 15.65) |
| YOLO11n | ONNX Runtime INT8, head tail kept float | 37.50 | -1.24 (-1.62 to -0.98) | 55.10% (53.71 to 56.49) | 38.54% (37.19 to 39.91) |
| YOLO11n | ONNX Runtime INT8, nothing excluded | 0.00 | -38.73 (-39.86 to -38.20) | 98.18% (97.77 to 98.53) | 97.78% (97.33 to 98.17) |
| YOLO26n | PyTorch reference | 40.33 | | | |
| YOLO26n | ONNX Runtime FP32 | 40.33 | 0.00 (0.00 to 0.00) | 0.00% (0.00 to 0.07) | 0.00% (0.00 to 0.07) |
| YOLO26n | OpenVINO FP32, f32 forced | 40.33 | 0.00 (0.00 to 0.00) | 0.00% (0.00 to 0.07) | 0.00% (0.00 to 0.07) |
| YOLO26n | OpenVINO FP32 IR, default precision (bf16) | 40.33 | 0.00 (-0.06 to +0.08) | 10.88% (10.03 to 11.78) | 1.04% (0.78 to 1.36) |
| YOLO26n | OpenVINO INT8 (Ultralytics export, NNCF) | 39.71 | -0.62 (-0.80 to -0.45) | 35.96% (34.63 to 37.31) | 15.38% (14.39 to 16.41) |
| YOLO26n | ONNX Runtime INT8, head tail kept float | 38.29 | -2.04 (-2.32 to -1.77) | 50.50% (49.10 to 51.90) | 36.84% (35.50 to 38.19) |
| YOLO26n | ONNX Runtime INT8, nothing excluded | 0.00 | -40.33 (-41.48 to -39.77) | 98.34% (97.95 to 98.68) | 97.86% (97.42 to 98.24) |

The reference against itself (a rerun, batch 8, and batch 8 in shuffled order) changed the answer
on 0 of 5,000 images for YOLO11n and on 1 of 5,000 for YOLO26n (batch 8), and never produced a hard
flip at any threshold or margin below.

### How much the hard flip rate depends on the margin and the threshold

Hard flip rate (percent of images) for each deployment threshold and hard margin. A hard flip is a
class change, or a box that appeared or vanished with a score at least the margin above the threshold.

| model | target | 0.25, margin 0.02 | 0.25, margin 0.05 | 0.25, margin 0.10 | 0.30, margin 0.02 | 0.30, margin 0.05 | 0.30, margin 0.10 |
|---|---|---|---|---|---|---|---|
| YOLO11n | ONNX Runtime FP32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| YOLO11n | OpenVINO FP32, f32 forced | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| YOLO11n | OpenVINO FP32 IR, default precision (bf16) | 3.12 | 1.76 | 1.20 | 2.26 | 1.38 | 0.82 |
| YOLO11n | OpenVINO INT8 (Ultralytics export, NNCF) | 25.68 | 14.64 | 7.42 | 22.90 | 13.28 | 5.62 |
| YOLO11n | ONNX Runtime INT8, head tail kept float | 48.72 | 38.54 | 25.76 | 44.76 | 34.98 | 22.08 |
| YOLO11n | ONNX Runtime INT8, nothing excluded | 98.06 | 97.78 | 97.42 | 97.66 | 97.42 | 96.66 |
| YOLO26n | ONNX Runtime FP32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| YOLO26n | OpenVINO FP32, f32 forced | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| YOLO26n | OpenVINO FP32 IR, default precision (bf16) | 2.04 | 1.04 | 0.66 | 1.44 | 0.70 | 0.32 |
| YOLO26n | OpenVINO INT8 (Ultralytics export, NNCF) | 25.46 | 15.38 | 7.68 | 22.66 | 12.98 | 6.52 |
| YOLO26n | ONNX Runtime INT8, head tail kept float | 44.08 | 36.84 | 21.28 | 36.60 | 27.54 | 15.80 |
| YOLO26n | ONNX Runtime INT8, nothing excluded | 98.16 | 97.86 | 97.16 | 97.64 | 97.16 | 96.52 |

The hard rate depends strongly on the margin: for OpenVINO INT8 it roughly halves from 0.02 to 0.05
and halves again from 0.05 to 0.10, and it moves less with the threshold. It never reaches zero for
any reduced precision target, and it stays at zero for the reference against itself and for both
FP32 controls in every cell. So the claim that holds across the whole range is the ordering and the
gap to the noise floor, not any single hard rate.

### Repeat run

A second, independent run in a new process: the PyTorch reference again, OpenVINO INT8 again on the
same IR, and OpenVINO INT8 exported again from scratch (a new NNCF calibration on the same 128 images).

| check | YOLO11n | YOLO26n |
|---|---|---|
| reference, new process, against the first reference | 0 images changed, bitwise identical | 0 images changed, bitwise identical |
| OpenVINO INT8, same IR, new process, against the first INT8 run | 0 images changed, bitwise identical | 0 images changed, bitwise identical |
| first INT8 export against the reference | 36.46% changed, 14.64% hard | 35.96% changed, 15.38% hard |
| second INT8 export against the reference | 36.12% changed, 14.76% hard | 35.78% changed, 15.06% hard |
| first INT8 export against the second | 19.48% changed, 3.02% hard | 20.02% changed, 3.62% hard |
| images changed by both exports | 1,585 | 1,531 |

Inference is deterministic on this machine. INT8 calibration is not: the second export has different
weights and disagrees with the first on about a fifth of the images. The flip rate against the
reference reproduced to within 0.4 points for both models, so the rate is a property of the export
recipe, while the exact set of images that flip is not.

### What these numbers do and do not show

* FP32 exports through ONNX Runtime and OpenVINO with f32 forced gave the same answer as PyTorch on
  every image, for both models. That is the control.
* OpenVINO's FP32 IR runs in bf16 by default on CPUs that support it. mAP did not move (its interval
  straddles zero), while 11% to 13% of images changed their answer. Most of those are boxes crossing the threshold; the hard
  rate is about 1% to 2%.
* OpenVINO INT8 from the stock Ultralytics export lost a fraction of a point of mAP while about a
  third of the images changed their answer.
* The ONNX Runtime INT8 recipe is mine and untuned. Quantizing everything, including the box and score
  tail of the head, destroys the output; that row shows a known pitfall, not a finding.
* Absolute mAP is below the Ultralytics model cards because exportdiff uses square 640 letterboxing
  and one class per box. It is identical for every target, so the changes are comparable; the
  absolute values are not.
* YOLO26n ran through its one to many head with external NMS, not its NMS free end to end head.
* Flip intervals treat the 5,000 images as independent draws. One machine, one CPU type: the bf16
  row depends on the CPU supporting bf16.

Full write up of the first measurement, including the probe set results: [FIRST_RESULTS.md](FIRST_RESULTS.md).
Raw numbers: `results/`.

## Method

* **Answer** per image: the set of (class, box) at or above the deployment confidence threshold
  (default 0.25) after shared decoding and class aware NMS (IoU 0.7).
* **Match**: class agnostic Hungarian assignment that keeps as many pairs at IoU 0.5 or more as
  possible, then maximises their total IoU.
* **Flip**: an image where any box appeared, vanished or changed class.
* **Hard flip**: a class change, or a box that appeared or vanished with a score at least the hard
  margin (default 0.05) above the threshold.
* **Noise floor**: the reference against itself under a rerun, batch size 8, and batch size 8 with a
  shuffled order. An export is flagged only when the exact lower bound of its flip rate clears the
  largest self run flip rate.
* **Probe set**: the 200 images whose reference answer has the smallest margin (distance of any box
  score to the threshold, or gap between best and runner up class).
* **Statistics**: exact Clopper Pearson intervals, bootstrap intervals, and an exact McNemar test on
  images flipped by the export but not by noise against the reverse.
* **mAP**: pycocotools on the same decoded boxes (confidence 0.001, 100 detections per image for
  the COCO metric). The bootstrap redoes only the accumulate step of pycocotools with per image
  resample counts, and reproduces pycocotools exactly when every count is one.

## Targets

| target | what |
|---|---|
| `onnx` | ONNX Runtime CPU, FP32, Ultralytics ONNX export |
| `onnx-int8` | ONNX Runtime static INT8 (QDQ, per channel, MinMax), box decoding tail kept in float |
| `onnx-int8-naive` | same quantizer with nothing excluded |
| `openvino` | OpenVINO CPU, FP32 IR, inference precision forced to f32 |
| `openvino-default` | OpenVINO CPU, FP32 IR, plugin default precision (bf16 on CPUs that support it) |
| `openvino-int8` | OpenVINO INT8 IR from the Ultralytics export (NNCF) |
| `tensorrt-fp16`, `tensorrt-int8` | stubs, not implemented yet |

## Reproduce

```
pip install -e ".[run,openvino,test]"
pytest
```

The tests need no model download. For the numbers above, get COCO val2017 with its instance
annotations and Ultralytics coco128 (calibration only, drawn from train2017, so it does not overlap
the evaluation images), then run the two commands above once per model with all six targets.
`results/logs/run.sh` is the exact Docker invocation used (set `EXPORTDIFF_WORKDIR` first), and
`scripts/repeat_check.py` is the repeat run. A full run of six targets on 5,000 images takes
about 20 minutes per model on 8 CPU threads.

## Not done yet

* TensorRT FP16 and INT8 on an NVIDIA GPU (adapter stub only). This is the target most people ship
  to, and the one where the probe set has to catch a small flip rate.
* A target with a flip rate near 1%, to test the probe set where it matters.
* Segmentation and pose matchers; only boxes are compared today.
* RF-DETR as a second model family, and the YOLO26n end to end head as its own row.
* A CI gate (`--fail-above` exit code) so a pipeline can refuse an export.
* GPU references, other CPUs, and mobile runtimes.

## License

MIT, see [LICENSE](LICENSE).

## Citation

No paper yet. If you use exportdiff before there is one, please cite the repository:

```
@misc{rane_exportdiff,
  author = {Rane, Sharven},
  title  = {exportdiff: per image answer changes after exporting a detector},
  year   = {2026},
  url    = {https://github.com/SharvenRane/exportdiff},
  note   = {work in progress}
}
```
