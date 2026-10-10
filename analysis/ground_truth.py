"""Reference Cobb angles for the 481 AASCE (SpineWeb Dataset 16) training images.

Two references:
  * original  — train_txt/angles.csv shipped with the dataset. Its columns are
                (MT, PT, TL): the FIRST is the largest angle. Read as (PT, MT, TL)
                it looked as if the port disagreed with the organisers on 460 images;
                in the right order it agrees to 0.1 deg on 466 of 481 triplets;
  * curated   — recomputed from the corrected landmarks published on Zenodo
                (doi:10.5281/zenodo.4413665), which fix shifted, flipped and swapped
                vertebra annotations in the original.

Angles are computed with the AASCE 2019 challenge algorithm (Python port used by
Yi et al., "Vertebra-focused landmark detection", ISBI 2020). The port is first
checked against angles.csv using the ORIGINAL landmarks: if it does not reproduce the
organisers' numbers, nothing downstream can be trusted.

Output: analysis/ground_truth.csv
"""
import csv
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KAG = os.path.join(ROOT, 'aasce_kaggle')
ZEN = os.path.join(ROOT, 'zenodo_curated', 'annotations_train')
VERTEBRAE = ['T%d' % i for i in range(1, 13)] + ['L%d' % i for i in range(1, 6)]


def is_s(mid_p_v):
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num - 2):
        term1 = (mid_p_v[i, 1] - mid_p_v[num - 1, 1]) / (mid_p_v[0, 1] - mid_p_v[num - 1, 1])
        term2 = (mid_p_v[i, 0] - mid_p_v[num - 1, 0]) / (mid_p_v[0, 0] - mid_p_v[num - 1, 0])
        ll.append(term1 - term2)
    ll = np.asarray(ll, np.float64)[:, None]
    ll_pair = ll @ ll.T
    return abs(ll_pair.sum() - np.abs(ll_pair).sum()) >= 1e-4


def cobb_angles(pts, image_h):
    """pts: 68 x 2, four corners per vertebra in the order TL, TR, BL, BR, T1..L5.
    Returns (PT, MT, TL) in degrees; MT is the largest angle between any two
    vertebral tilts."""
    pts = np.asarray(pts, np.float64)
    num_pts = pts.shape[0]
    vnum = num_pts // 4 - 1
    mid_p_v = (pts[0::2, :] + pts[1::2, :]) / 2
    mid_p = []
    for i in range(0, num_pts, 4):
        mid_p.append((pts[i, :] + pts[i + 2, :]) / 2)
        mid_p.append((pts[i + 1, :] + pts[i + 3, :]) / 2)
    mid_p = np.asarray(mid_p)
    vec_m = mid_p[1::2, :] - mid_p[0::2, :]
    dot_v = vec_m @ vec_m.T
    mod_v = np.sqrt((vec_m ** 2).sum(axis=1))[:, None]
    mod_v = mod_v @ mod_v.T
    angles = np.arccos(np.clip(dot_v / mod_v, 0.0, 1.0))
    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = int(np.argmax(maxt))
    mt = float(np.amax(maxt) / np.pi * 180)
    if not is_s(mid_p_v):
        pt = float(angles[0, pos2] / np.pi * 180)
        tl = float(angles[vnum, pos1[pos2]] / np.pi * 180)
    else:
        if (mid_p_v[pos2 * 2, 1] + mid_p_v[pos1[pos2] * 2, 1]) < image_h:
            angle2 = angles[pos2, :pos2 + 1]
            pt = float(np.max(angle2) / np.pi * 180)
            angle3 = angles[pos1[pos2], pos1[pos2]:vnum + 1]
            tl = float(np.max(angle3) / np.pi * 180)
        else:
            angle2 = angles[pos2, :pos2 + 1]
            pt = float(np.max(angle2) / np.pi * 180)
            pos1_1 = int(np.argmax(angle2))
            angle3 = angles[pos1_1, :pos1_1 + 1]
            tl = float(np.max(angle3) / np.pi * 180)
    return pt, mt, tl


def major_curve(pts):
    """End vertebrae (upper, lower index 0..16 = T1..L5) of the largest Cobb angle,
    its apex (the vertebra between them farthest from the line joining their
    centres) and the centre (x, y) of every vertebra."""
    centres = np.array([pts[i:i + 4].mean(axis=0) for i in range(0, 68, 4)])
    vec = np.array([((pts[i + 1] + pts[i + 3]) / 2) - ((pts[i] + pts[i + 2]) / 2) for i in range(0, 68, 4)])
    norm = np.linalg.norm(vec, axis=1)
    cos = np.clip((vec @ vec.T) / np.outer(norm, norm), 0, 1)
    ang = np.arccos(cos)
    a, b = np.unravel_index(np.argmax(ang), ang.shape)
    upper, lower = int(min(a, b)), int(max(a, b))
    p, q = centres[upper], centres[lower]
    d = q - p
    if lower - upper >= 2 and np.linalg.norm(d) > 0:
        inner = range(upper + 1, lower)
        dev = [abs(d[0] * (centres[k][1] - p[1]) - d[1] * (centres[k][0] - p[0])) / np.linalg.norm(d) for k in inner]
        apex = int(list(inner)[int(np.argmax(dev))])
    else:
        apex = (upper + lower) // 2
    return upper, lower, apex, centres


def read_rows(path):
    with open(path, encoding='utf-8') as f:
        return [r for r in csv.reader(f) if r]


def curated_points(xml_path):
    """Zenodo XML: per vertebra 'x1 y1 ... x4 y4' clockwise from top-left
    (TL, TR, BR, BL). Reordered to TL, TR, BL, BR."""
    root = ET.parse(xml_path).getroot()
    by_id = {}
    for ann in root.iter('annotation'):
        vals = [float(v) for v in ann.find('pos').text.split()]
        tl, tr, br, bl = (vals[0:2], vals[2:4], vals[4:6], vals[6:8])
        by_id[ann.get('id')] = [tl, tr, bl, br]
    missing = [v for v in VERTEBRAE if v not in by_id]
    if missing:
        return None, missing
    return np.array([p for v in VERTEBRAE for p in by_id[v]]), []


def main():
    names = [r[0] for r in read_rows(os.path.join(KAG, 'train_txt', 'filenames.csv'))]
    angles = [[float(x) for x in r] for r in read_rows(os.path.join(KAG, 'train_txt', 'angles.csv'))]
    lms = [[float(x) for x in r] for r in read_rows(os.path.join(KAG, 'train_txt', 'landmarks.csv'))]
    assert len(names) == len(angles) == len(lms) == 481, (len(names), len(angles), len(lms))

    out, dev_orig, levels = [], [], {}
    for name, ang, lm in zip(names, angles, lms):
        w, h = Image.open(os.path.join(KAG, 'train', name)).size
        # landmarks.csv: 68 normalised x values, then 68 normalised y values
        pts = np.stack([np.array(lm[:68]) * w, np.array(lm[68:]) * h], axis=1)
        o = cobb_angles(pts, h)
        # angles.csv order is (MT, PT, TL); cobb_angles() returns (PT, MT, TL)
        dev_orig.append(max(abs(o[1] - ang[0]), abs(o[0] - ang[1]), abs(o[2] - ang[2])))

        stem = os.path.splitext(name)[0]
        xml = os.path.join(ZEN, stem + '.xml')
        cur, cur_missing = (None, ['no file'])
        if os.path.exists(xml):
            cur, cur_missing = curated_points(xml)
        c = cobb_angles(cur, h) if cur is not None else (None, None, None)
        src = cur if cur is not None else pts
        up, lo, apex, centres = major_curve(src)
        levels[name] = {'centres': np.round(centres, 1).tolist(), 'upper': up, 'lower': lo, 'apex': apex,
                        'source': 'curated' if cur is not None else 'original'}
        out.append({
            'name': name, 'width': w, 'height': h,
            'orig_pt': ang[1], 'orig_mt': ang[0], 'orig_tl': ang[2], 'orig_max': max(ang),
            'check_pt': round(o[0], 3), 'check_mt': round(o[1], 3), 'check_tl': round(o[2], 3),
            'cur_pt': None if c[0] is None else round(c[0], 3),
            'cur_mt': None if c[1] is None else round(c[1], 3),
            'cur_tl': None if c[2] is None else round(c[2], 3),
            'cur_max': None if c[1] is None else round(max(c), 3),
            'cur_note': ','.join(cur_missing),
            'ref_upper': VERTEBRAE[up], 'ref_lower': VERTEBRAE[lo], 'ref_apex': VERTEBRAE[apex],
            'major_location': 'thoracic' if apex <= 10 else 'thoracolumbar/lumbar',
        })

    dev = np.array(dev_orig)
    print('port check vs angles.csv (max abs diff over PT/MT/TL): median %.4f, 95th %.4f, max %.4f; within 0.1 deg: %d/481'
          % (np.median(dev), np.percentile(dev, 95), dev.max(), int((dev < 0.1).sum())))
    have = [r for r in out if r['cur_max'] is not None]
    d = np.array([r['cur_max'] - r['orig_max'] for r in have])
    print('curated available: %d/481; curated - original (max angle): mean %.2f, |d|>5 deg: %d, |d|>10 deg: %d'
          % (len(have), d.mean(), int((abs(d) > 5).sum()), int((abs(d) > 10).sum())))
    with open(os.path.join(HERE, 'ground_truth.csv'), 'w', newline='', encoding='utf-8') as f:
        wri = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wri.writeheader()
        wri.writerows(out)
    import json
    json.dump(levels, open(os.path.join(HERE, 'levels.json'), 'w', encoding='utf-8'))
    print('wrote analysis/ground_truth.csv and levels.json')


if __name__ == '__main__':
    main()
