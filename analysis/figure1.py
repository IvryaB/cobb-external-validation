"""Figure 1: two AASCE radiographs, each with the RadTools overlay (white tangents and
angle labels, as the service returns it) and the reference end-vertebra tilts from
the curated expert landmarks (dashed yellow) — one good agreement, one large error.

Usage: python figure1.py "<good image name>" "<bad image name>"
"""
import base64
import csv
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ground_truth as gt_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, 'radtools_run', 'aasce_train481_radtools.jsonl')


def end_vertebrae(pts):
    """Indices of the two vertebrae whose tilts give the largest Cobb angle, and the
    tilt segment (left-mid, right-mid) of every vertebra."""
    mids = []
    for i in range(0, 68, 4):
        left = (pts[i] + pts[i + 2]) / 2
        right = (pts[i + 1] + pts[i + 3]) / 2
        mids.append((left, right))
    vec = np.array([r - l for l, r in mids])
    cos = np.clip((vec @ vec.T) / np.outer(np.linalg.norm(vec, axis=1), np.linalg.norm(vec, axis=1)), 0, 1)
    ang = np.arccos(cos)
    a, b = np.unravel_index(np.argmax(ang), ang.shape)
    return sorted((int(a), int(b))), mids, float(ang[a, b] * 180 / np.pi)


def panel(name, runs, label):
    rec = runs[name]
    img = Image.open(io.BytesIO(base64.b64decode(rec['overlay']))).convert('RGB')
    xml = os.path.join(gt_mod.ZEN, os.path.splitext(name)[0] + '.xml')
    pts, _ = gt_mod.curated_points(xml)
    (i, j), mids, ref = end_vertebrae(pts)
    d = ImageDraw.Draw(img)
    w = max(3, img.width // 160)
    for k in (i, j):
        (lx, ly), (rx, ry) = mids[k]
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
        vx, vy = (rx - lx), (ry - ly)
        n = np.hypot(vx, vy)
        vx, vy = vx / n, vy / n
        half = img.width * 0.42
        x0, y0, x1, y1 = cx - vx * half, cy - vy * half, cx + vx * half, cy + vy * half
        segs = 14
        for s in range(segs):            # dashed
            if s % 2:
                continue
            t0, t1 = s / segs, (s + 1) / segs
            d.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
                   fill=(255, 210, 0), width=w)
    return img, ref, rec.get('max_angle')


def main():
    good, bad = sys.argv[1], sys.argv[2]
    runs = {}
    for line in open(RUN, encoding='utf-8'):
        rec = json.loads(line)
        runs[os.path.basename(rec['name'])] = rec
    panels = []
    for name, lab in ((good, '(a)'), (bad, '(b)')):
        im, ref, pred = panel(name, runs, lab)
        print(name, 'reference %.1f  radtools %s' % (ref, pred))
        panels.append(im)
    # keep the thoracolumbar spine: drop the top of the film and the pelvis
    panels = [p.crop((0, int(p.height * 0.05), p.width, int(p.height * 0.80))) for p in panels]
    h = 1100
    panels = [p.resize((int(p.width * h / p.height), h), Image.LANCZOS) for p in panels]
    # panel letters go on after the crop, or the crop takes them off
    font = ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf', 64)
    for p, lab in zip(panels, ('(a)', '(b)')):
        ImageDraw.Draw(p).text((16, 12), lab, fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    gap = 24
    sheet = Image.new('RGB', (sum(p.width for p in panels) + gap, h), 'white')
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + gap
    sheet.save(os.path.join(HERE, 'fig1_examples.png'), dpi=(300, 300))
    print('wrote analysis/fig1_examples.png', sheet.size)


if __name__ == '__main__':
    main()
