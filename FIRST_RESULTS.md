# First results (2026-11-22)

First real measurement of `exportdiff` on the full COCO val2017 set (5,000 images), two official
Ultralytics detectors (YOLO11n and YOLO26n, weights downloaded by Ultralytics 8.4.216 from its
v8.4.0 asset release), CPU only, on one Linux box (AMD Ryzen 7 9700X, 8 threads, Docker).

Software: torch 2.15.0 CPU, onnxruntime 1.32.0, openvino 2026.4.1, nncf 3.4.0, pycocotools 2.0.11.
Raw outputs: `results/y11full/` and `results/y26full/` (`result.json`, `report.html`, `gallery/`),
logs and the exact run script in `results/logs/`.

## Setup in one paragraph

Every runtime gets the same letterboxed 640 x 640 tensor, and its raw head output goes through the
same decoding plus class aware NMS (IoU 0.7). The deployed answer of an image is every box with score
0.25 or more. Answers are matched class agnostically by Hungarian assignment at IoU 0.5; an image
flips when any box appears, vanishes or changes class. A **hard** flip excludes boxes that merely
graze the threshold: it needs a class change or an appearing or vanishing box scoring at least 0.30.
INT8 calibration used the 128 images of Ultralytics coco128 (drawn from train2017, so no overlap
with the evaluation set). mAP is pycocotools on the same decoded boxes at confidence 0.001.

## Noise floor (PyTorch reference against itself)

| model | rerun | batch 8 | batch 8, shuffled order |
|---|---|---|---|
| YOLO11n | 0 / 5000 | 0 / 5000 | 0 / 5000 |
| YOLO26n | 0 / 5000 | 1 / 5000 (image 89556) | 0 / 5000 |

So the noise floor is 0% for YOLO11n and 0.02% for YOLO26n (exact 95% upper bound about 0.11%).
No self run produced a hard flip.

## YOLO11n (reference mAP50 95 = 38.73, mAP50 = 54.24)

| target | mAP50 95 | change | images whose answer changed (95% exact CI) | hard flips (95% CI) | probe flips of 200 |
|---|---|---|---|---|---|
| ONNX Runtime FP32 | 38.73 | 0.00 | 0 (0.00% to 0.07%) | 0 | 0 |
| OpenVINO FP32 (f32 forced) | 38.73 | 0.00 | 0 (0.00% to 0.07%) | 0 | 0 |
| OpenVINO FP32 IR, plugin default (bf16) | 38.74 | +0.01 | 631 = 12.62% (11.71% to 13.57%) | 88 = 1.76% (1.41% to 2.16%) | 122 |
| OpenVINO INT8 (Ultralytics export, NNCF) | 38.55 | -0.18 | 1825 = 36.50% (35.16% to 37.85%) | 734 = 14.68% (13.71% to 15.69%) | 167 |
| ONNX Runtime INT8, head tail kept float | 37.50 | -1.23 | 2756 = 55.12% (53.73% to 56.51%) | 1931 = 38.62% (37.27% to 39.99%) | 190 |
| ONNX Runtime INT8, nothing excluded | 0.00 | -38.73 | 4909 = 98.18% | 4889 = 97.78% | 200 |

## YOLO26n (reference mAP50 95 = 40.33, mAP50 = 56.03)

| target | mAP50 95 | change | images whose answer changed (95% exact CI) | hard flips (95% CI) | probe flips of 200 |
|---|---|---|---|---|---|
| ONNX Runtime FP32 | 40.33 | 0.00 | 0 (0.00% to 0.07%) | 0 | 0 |
| OpenVINO FP32 (f32 forced) | 40.33 | 0.00 | 0 (0.00% to 0.07%) | 0 | 0 |
| OpenVINO FP32 IR, plugin default (bf16) | 40.33 | 0.00 | 544 = 10.88% (10.03% to 11.78%) | 52 = 1.04% (0.78% to 1.36%) | 113 |
| OpenVINO INT8 (Ultralytics export, NNCF) | 39.71 | -0.62 | 1800 = 36.00% (34.67% to 37.35%) | 772 = 15.44% (14.45% to 16.47%) | 153 |
| ONNX Runtime INT8, head tail kept float | 38.29 | -2.04 | 2525 = 50.50% (49.10% to 51.90%) | 1844 = 36.88% (35.54% to 38.24%) | 172 |
| ONNX Runtime INT8, nothing excluded | 0.00 | -40.33 | 4917 = 98.34% | 4893 = 97.86% | 198 |

Every non FP32 row is far above the noise floor: exact McNemar p below 1e-160 for every one of
them (export only flips versus noise only flips, for example 631 versus 0 and 544 versus 1 for bf16).

## Does the probe set catch it?

Yes, in every case where there was something to catch, and it stayed silent on both FP32 rows. The
200 lowest margin images flip at 56% to 95% (naive rows aside), against 11% to 55% on the full set, and the reference
margin ranks flipped images well (AUROC 0.84 to 0.90). The probe holds only 7% to 21% of all flipped
images, so it is a detector of "this export changes answers", not a full inventory.
Caveat: the rates here are so large that almost any 200 images would catch them. The probe's real
test is a target with a flip rate near 1%, which this run did not include.

## Does the headline hold?

"Accuracy barely moved while X% of images changed a box":

* **Holds for OpenVINO INT8**, the path a user gets from `yolo export format=openvino int8=True`:
  mAP down 0.18 (YOLO11n) and 0.62 (YOLO26n) while 36% of images changed their answer and about
  15% changed it in a way that is not a box grazing the threshold.
* **Holds, with a softer number, for OpenVINO's default precision.** The FP32 IR on this CPU runs
  in bf16 unless asked otherwise. mAP is unchanged to two decimals while 11% to 13% of images change
  answer; only 1% to 2% are hard flips, the rest are boxes crossing 0.25. A user who never asked for
  reduced precision gets it.
* **Does not hold for ONNX Runtime static INT8 with my recipe**: mAP fell 1.2 to 2.0 points, which is
  not "barely", although half the images changing is still the bigger story.
* **FP32 exports are clean**: ONNX Runtime and OpenVINO f32 produced the identical answer on all
  5,000 images for both models, even though decoded scores differ by up to about 0.002 and a handful of
  images differ in low confidence boxes. That is the control the headline needs.

Defensible headline from this run: *"OpenVINO INT8 YOLO26n lost 0.6 mAP while 36% of COCO images
changed their answer (15% beyond threshold jitter); the reference never disagreed with itself on
more than 1 image in 5,000."*

## Honest caveats

* **Threshold jitter dominates the plain flip rate.** An image flips if any single box crosses 0.25,
  and the average image has about 5 deployed boxes. Lead with the hard flip rate or report both;
  the 0.05 hard margin is my choice and should get a sensitivity sweep.
* **mAP is below Ultralytics' published numbers** (YOLO11n is listed at 39.5) because exportdiff
  uses square 640 letterboxing and one class per box instead of rectangular batches and multi label
  NMS. It is the same for every target, so the deltas are valid; the absolute values are not
  comparable to the model cards.
* **YOLO26n ran through its one to many head with external NMS** (Ultralytics' default export,
  `nms=None`), not its NMS free one to one head. The end to end head is a separate row to add.
* **INT8 recipes**: the ORT INT8 recipe (QDQ, per channel, MinMax, 128 images) is mine and untuned;
  the OpenVINO INT8 row is the stock Ultralytics export. The naive ORT row shows why the head tail
  must stay float (box pixels and 0..1 scores in one INT8 tensor wipe out every score), which is a
  known pitfall, not a finding.
* **One machine, CPU only.** The bf16 row depends on the CPU having bf16 support (Zen 5 here). No
  GPU, no TensorRT, no mobile runtime yet.
* **Single run, no interval on the mAP change**; the flip intervals are exact binomial intervals over
  images, which treats images as independent draws from COCO.
* The default noise floor is the largest self run rate; with zero or one self flip the floor is
  effectively zero, so "above noise" is easy to clear here. A GPU reference will be noisier.

## Found along the way

* `mcnemar_exact` overflowed (`2.0 ** n` past 1023 discordant pairs) and crashed the first full run.
  Fixed here with the binomial CDF plus a regression test. The same function in `coreml-flipcheck`
  (`src/coreml_flipcheck/stats.py`) has the same bug and needs the same fix.
* Ultralytics builds export output paths by string replacing the weights file name in the full
  path, so a work folder named `yolo11n.pt-cache` turned into `yolo11n_openvino_model/-cache/...`.
  Worked around by naming the cache folder without the `.pt` suffix; possibly worth an upstream issue.

## Next steps

1. TensorRT FP16 and INT8 adapter on the 5070 Ti (stub in `runtimes.py`), the row people actually
   ship to and the one where the probe set has to catch a small rate.
2. Hard margin sensitivity (0.02, 0.05, 0.10) and a paired bootstrap interval on the mAP change.
3. YOLO26n end to end head as its own row; RF-DETR Nano as the second model family.
4. A low flip rate target (for example OpenVINO with an f16 hint, or ORT with graph optimisations
   off) to measure what the probe set catches when the effect is near 1%.
5. Segmentation and pose matchers; `--fail-above` exit code for CI.
