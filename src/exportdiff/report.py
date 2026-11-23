"""JSON result and a small static HTML report with the worst images drawn side by side."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

REF_COLOR = (40, 170, 70)
CHANGE_COLOR = (225, 30, 30)
RELABEL_COLOR = (200, 0, 200)
TILE_W = 480


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k != "union_flags"}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def write_json(result: dict, path: Path) -> None:
    path.write_text(json.dumps(_jsonable(result), indent=2))


def _draw(img_path: Path, dets, highlight: set, relabel: set, names: dict, title: str):
    from PIL import Image, ImageDraw

    im = Image.open(img_path).convert("RGB")
    scale = TILE_W / im.width
    im = im.resize((TILE_W, max(1, int(round(im.height * scale)))))
    dr = ImageDraw.Draw(im)
    for j, (b, s, c) in enumerate(zip(dets.boxes, dets.scores, dets.classes)):
        color = RELABEL_COLOR if j in relabel else CHANGE_COLOR if j in highlight else REF_COLOR
        width = 3 if (j in highlight or j in relabel) else 1
        x0, y0, x1, y1 = (float(v) * scale for v in b)
        dr.rectangle([x0, y0, x1, y1], outline=color, width=width)
        dr.text((x0 + 2, max(0, y0 - 11)), f"{names.get(int(c), c)} {s:.2f}", fill=color)
    dr.rectangle([0, 0, TILE_W, 14], fill=(0, 0, 0))
    dr.text((3, 1), title, fill=(255, 255, 255))
    return im


def draw_pair(img_path: Path, ref, new, diff, names: dict, target: str, out: Path) -> None:
    from PIL import Image

    left = _draw(img_path, ref, set(diff.vanished), {r for r, _ in diff.relabelled}, names,
                 f"reference  ({len(diff.vanished)} vanish)")
    right = _draw(img_path, new, set(diff.appeared), {c for _, c in diff.relabelled}, names,
                  f"{target}  ({len(diff.appeared)} appear, {len(diff.relabelled)} relabel)")
    canvas = Image.new("RGB", (TILE_W * 2 + 6, max(left.height, right.height)), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (TILE_W + 6, 0))
    canvas.save(out, quality=85)


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}%"


def write_html(result: dict, galleries: dict, path: Path) -> None:
    cfg = result["config"]
    rows = []
    ref_map = result.get("map", {}).get("torch", {})
    for t in result["targets"]:
        s = result["summary"].get(t)
        if s is None:
            rows.append(f"<tr><td>{html.escape(t)}</td><td colspan=8>{html.escape(result['errors'].get(t, ''))}</td></tr>")
            continue
        m = result.get("map", {}).get(t, {})
        dm = (m.get("map50_95", float("nan")) - ref_map.get("map50_95", float("nan"))) if m else float("nan")
        f = s["flip"]
        rows.append(
            "<tr>" + "".join(f"<td>{c}</td>" for c in [
                html.escape(t),
                f"{100 * m.get('map50_95', float('nan')):.2f}" if m else "n/a",
                f"{100 * dm:+.2f}" if m else "n/a",
                f"{f['k']} / {f['n']} = {_pct(f['rate'])}",
                f"{_pct(f['ci_exact'][0])} to {_pct(f['ci_exact'][1])}",
                "yes" if s["above_noise"] else "no",
                f"{s['hard_flip']['k']} = {_pct(s['hard_flip']['rate'])}",
                f"{s['mcnemar_vs_noise']['p']:.2g}",
                f"{s['probe']['flips']} / {s['probe']['size']} ({'caught' if s['probe']['caught'] else 'missed'})",
            ]) + "</tr>")
    nf = result["noise"]
    noise_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['k']} / {v['n']} = {_pct(v['rate'])}</td>"
        f"<td>{_pct(v['ci_exact'][0])} to {_pct(v['ci_exact'][1])}</td></tr>" for k, v in nf["runs"].items())
    gal = []
    for t, items in galleries.items():
        gal.append(f"<h3>{html.escape(t)}: worst {len(items)} images</h3>")
        for img_id, rel, ch in items:
            gal.append(f"<figure><img src='{html.escape(rel)}' loading='lazy'><figcaption>image {img_id}, "
                       f"{ch} box changes</figcaption></figure>")
    ref_line = (f"reference mAP50-95 {100 * ref_map['map50_95']:.2f}, mAP50 {100 * ref_map['map50']:.2f}"
                if ref_map else "no annotations given, mAP not computed")
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>exportdiff report</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:14px}}th{{background:#f3f3f3}}
figure{{margin:12px 0}}img{{max-width:100%;border:1px solid #ddd}}figcaption{{font-size:13px;color:#444}}</style>
</head><body>
<h1>exportdiff report</h1>
<p>model <b>{html.escape(str(cfg['model']))}</b>, {cfg['n_images']} images, deployment threshold {cfg['threshold']},
match IoU {cfg['match_iou']}, NMS IoU {cfg['nms_iou']}, image size {cfg['imgsz']}. {ref_line}.</p>
<h2>Answer changes per target</h2>
<table><tr><th>target</th><th>mAP50-95</th><th>change vs torch</th><th>images whose answer changed</th>
<th>95% exact CI</th><th>above noise floor</th><th>hard changes (score 0.05 or more past threshold, or class change)</th><th>McNemar p vs noise</th><th>200 image probe</th></tr>
{''.join(rows)}</table>
<h2>Noise floor (reference against itself)</h2>
<p>floor = {_pct(nf['floor'])} ({html.escape(nf['floor_run'])})</p>
<table><tr><th>self run</th><th>images changed</th><th>95% exact CI</th></tr>{noise_rows}</table>
<h2>Worst images</h2>
<p>Left: reference answer, boxes that vanish in the export in red. Right: export answer, boxes that
appear in red, class changes in magenta on both sides.</p>
{''.join(gal)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")
