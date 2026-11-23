"""Runtime adapters.

Each adapter turns the same preprocessed batch (N, 3, H, W) float32 into the same raw head output
(N, 4 + classes, anchors). Decoding and NMS happen later in decode.py, identical for all of them.

Targets
-------
torch             PyTorch reference (the fused Ultralytics model, eval mode, CPU, float32)
onnx              ONNX Runtime CPU, float32, Ultralytics ONNX export
onnx-int8         ONNX Runtime CPU, static INT8 (QDQ, per channel weights, MinMax calibration),
                  box decoding tail of the head kept in float, the usual recipe for YOLO heads
onnx-int8-naive   the same quantizer with nothing excluded, to show what the default does
openvino          OpenVINO CPU, FP32 IR, inference precision forced to f32
openvino-default  OpenVINO CPU, FP32 IR, the plugin's default inference precision (bf16 on CPUs
                  that support it, which is what a user gets without asking)
openvino-int8     OpenVINO CPU, INT8 IR from the Ultralytics export (NNCF post training quantization)
tensorrt-fp16     stub, needs an NVIDIA GPU (see TensorRTStub)
tensorrt-int8     stub
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

THREADS = 8


class Runtime:
    name = "base"
    batch_size = 1

    def prepare(self, weights: Path, work: Path, calib: list[Path], imgsz: int) -> None:
        raise NotImplementedError

    def infer(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "batch_size": self.batch_size}


def _work_weights(weights: Path, work: Path) -> Path:
    """Copy the weights into the work dir so export artifacts land there, not next to the source."""
    work.mkdir(parents=True, exist_ok=True)
    dst = work / Path(weights).name
    if not dst.exists():
        shutil.copy2(weights, dst)
    return dst


def _raw_head(model) -> None:
    """Make the head return the raw (4 + classes, anchors) tensor even for NMS free heads.

    YOLO26 style heads have an end to end (one to one) branch that does top k inside the graph.
    exportdiff needs the one to many branch so NMS can run once in shared code.
    """
    head = model.model[-1] if hasattr(model, "model") else None
    if head is not None and getattr(head, "end2end", False):
        head.end2end = False


class TorchReference(Runtime):
    name = "torch"

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.model = None

    def prepare(self, weights, work, calib, imgsz):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(THREADS)
        yolo = YOLO(str(weights))
        m = yolo.model.float().eval()
        _raw_head(m)
        m = m.fuse(verbose=False) if hasattr(m, "fuse") else m
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m
        self.names = dict(yolo.names)

    def infer(self, batch):
        import torch

        with torch.inference_mode():
            y = self.model(torch.from_numpy(batch))
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.numpy()

    def info(self):
        import torch

        return {"name": self.name, "batch_size": self.batch_size, "torch": torch.__version__, "threads": THREADS}


def export_onnx(weights: Path, work: Path, imgsz: int) -> Path:
    """Ultralytics ONNX export, float32, static batch 1, NMS outside the graph. Cached in `work`."""
    from ultralytics import YOLO

    w = _work_weights(weights, work)
    out = w.with_suffix(".onnx")
    if out.exists():
        return out
    yolo = YOLO(str(w))
    _raw_head(yolo.model)
    kw = dict(format="onnx", imgsz=imgsz, batch=1, dynamic=False, simplify=True, half=False)
    p = yolo.export(**kw)  # nms left at None: raw head output, NMS runs in decode.py
    return Path(p)


class OnnxRuntimeCPU(Runtime):
    name = "onnx"

    def __init__(self):
        self.sess = None
        self.path = None

    def _session(self, path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.path = path

    def prepare(self, weights, work, calib, imgsz):
        self._session(export_onnx(weights, work, imgsz))

    def infer(self, batch):
        return self.sess.run(None, {self.inp: batch})[0]

    def info(self):
        import onnxruntime as ort

        return {"name": self.name, "onnxruntime": ort.__version__, "artifact": str(self.path), "threads": THREADS}


def head_tail_nodes(onnx_path: Path) -> list[str]:
    """Node names of the box decoding tail of an Ultralytics head (everything in the last module that
    is not one of its prediction convolutions, plus the DFL convolution). Mixing pixel scale boxes and
    0..1 scores in one INT8 tensor destroys the scores, so these stay in float."""
    import onnx

    g = onnx.load(str(onnx_path)).graph
    idx = [int(m.group(1)) for n in g.node for m in [re.match(r"^/model\.(\d+)/", n.name)] if m]
    if not idx:
        return []
    head = f"/model.{max(idx)}/"
    return [n.name for n in g.node
            if n.name.startswith(head) and (n.op_type != "Conv" or "dfl" in n.name.lower())]


class _CalibReader:
    def __init__(self, files: list[Path], input_name: str, imgsz: int):
        from .data import preprocess

        self._it = iter([{input_name: preprocess(f, imgsz)[0][None]} for f in files])

    def get_next(self):
        return next(self._it, None)


class OnnxRuntimeInt8(OnnxRuntimeCPU):
    name = "onnx-int8"
    naive = False

    def prepare(self, weights, work, calib, imgsz):
        import onnx
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        if not calib:
            raise ValueError(f"{self.name} needs calibration images (--calib)")
        fp32 = export_onnx(weights, work, imgsz)
        out = fp32.with_name(fp32.stem + ("_int8_naive.onnx" if self.naive else "_int8.onnx"))
        self.excluded = [] if self.naive else head_tail_nodes(fp32)
        if not out.exists():
            pre = fp32.with_name(fp32.stem + "_pre.onnx")
            if not pre.exists():
                quant_pre_process(str(fp32), str(pre))
            inp = onnx.load(str(pre)).graph.input[0].name
            quantize_static(
                str(pre), str(out), _CalibReader(calib, inp, imgsz),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=self.excluded,
            )
        self.n_calib = len(calib)
        self._session(out)

    def info(self):
        d = super().info()
        d.update({"calibration_images": self.n_calib, "excluded_nodes": len(self.excluded),
                  "recipe": "QDQ, S8S8, per channel weights, MinMax"})
        return d


class OnnxRuntimeInt8Naive(OnnxRuntimeInt8):
    name = "onnx-int8-naive"
    naive = True


def _calib_yaml(calib: list[Path], names: dict, work: Path) -> Path:
    """Ultralytics INT8 export wants a dataset yaml; point it at the calibration images."""
    d = work / "calib_images" / "images"
    d.mkdir(parents=True, exist_ok=True)
    for f in calib:
        t = d / f.name
        if not t.exists():
            shutil.copy2(f, t)
    y = work / "calib.yaml"
    lines = [f"path: {d.parent}", "train: images", "val: images", "names:"]
    lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    y.write_text("\n".join(lines) + "\n")
    return y


class OpenVINOCPU(Runtime):
    name = "openvino"
    int8 = False
    precision = "f32"  # None = plugin default

    def prepare(self, weights, work, calib, imgsz):
        import openvino as ov
        from ultralytics import YOLO

        w = _work_weights(weights, work)
        sub = w.parent / (w.stem + ("_int8" if self.int8 else "") + "_openvino_model")
        xmls = list(sub.glob("*.xml")) if sub.exists() else []
        if not xmls:
            yolo = YOLO(str(w))
            _raw_head(yolo.model)
            kw = dict(format="openvino", imgsz=imgsz, batch=1, dynamic=False, half=False)
            if self.int8:
                if not calib:
                    raise ValueError(f"{self.name} needs calibration images (--calib)")
                kw.update(int8=True, data=str(_calib_yaml(calib, dict(yolo.names), work)))
                self.n_calib = len(calib)
            p = Path(yolo.export(**kw))
            xmls = list(p.glob("*.xml"))
        self.xml = xmls[0]
        core = ov.Core()
        self.default_precision = str(core.get_property("CPU", "INFERENCE_PRECISION_HINT"))
        cfg = {"INFERENCE_NUM_THREADS": THREADS}
        if self.precision:
            cfg["INFERENCE_PRECISION_HINT"] = self.precision
        self.compiled = core.compile_model(str(self.xml), "CPU", cfg)
        self.used_precision = str(self.compiled.get_property("INFERENCE_PRECISION_HINT"))
        self.out = self.compiled.output(0)

    def infer(self, batch):
        return self.compiled(batch)[self.out]

    def info(self):
        import openvino as ov

        d = {"name": self.name, "openvino": ov.__version__, "artifact": str(self.xml), "threads": THREADS,
             "inference_precision": self.used_precision, "plugin_default_precision": self.default_precision}
        if self.int8:
            d["calibration_images"] = getattr(self, "n_calib", None)
        return d


class OpenVINODefault(OpenVINOCPU):
    name = "openvino-default"
    precision = None


class OpenVINOInt8(OpenVINOCPU):
    name = "openvino-int8"
    int8 = True


class TensorRTStub(Runtime):
    """Placeholder. TensorRT needs an NVIDIA GPU; the build box for this prototype has none.

    To implement: build an engine from export_onnx(...) with trtexec or the tensorrt Python API
    (FP16 flag, or INT8 with an IInt8EntropyCalibrator2 over the same calibration images), bind the
    single input and output0, and return output0 as (N, 4 + classes, anchors) float32 so decode.py
    handles it like every other target.
    """

    def __init__(self, name: str):
        self.name = name

    def prepare(self, weights, work, calib, imgsz):
        raise NotImplementedError(f"{self.name}: TensorRT adapter not implemented yet (needs an NVIDIA GPU)")


REGISTRY = {
    "onnx": OnnxRuntimeCPU,
    "onnx-int8": OnnxRuntimeInt8,
    "onnx-int8-naive": OnnxRuntimeInt8Naive,
    "openvino": OpenVINOCPU,
    "openvino-default": OpenVINODefault,
    "openvino-int8": OpenVINOInt8,
    "tensorrt-fp16": lambda: TensorRTStub("tensorrt-fp16"),
    "tensorrt-int8": lambda: TensorRTStub("tensorrt-int8"),
}


def make(name: str) -> Runtime:
    if name not in REGISTRY:
        raise KeyError(f"unknown target {name!r}; known: {', '.join(REGISTRY)}")
    return REGISTRY[name]()
