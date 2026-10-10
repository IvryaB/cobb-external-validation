"""How much is left to win by improving the segmentation?

The error analysis split the large Cobb errors into two: the post-processing (fixed
on branch feat/cobb-turning-points) and the mask itself, which on 38 of 67 lumbar
cases stops above the vertebra the reference calls the lower end.

Before training anything, measure the ceiling: run the SAME geometry on a perfect
mask — one built from the expert landmarks (each vertebra's quadrilateral, bridged
across the discs so the band is continuous from T1 to L5). Whatever MAE that gives
is the best any segmentation could do with this post-processing. The gap between it
and the model's masks is what fine-tuning could win, at most.

Usage: python cobb_oracle.py   ->  cobb_oracle.txt / cobb_oracle.json
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cobb_postproc as CP      # noqa: E402
import levels_analysis as LA    # noqa: E402


def oracle_mask(pts, shape):
    """A filled band over T1..L5 from the 17 x 4 corner landmarks.

    pts are ordered per vertebra as TL, TR, BL, BR (the challenge's own order), so
    the quadrilateral is TL, TR, BR, BL and the bridge to the next vertebra is
    BL, BR of this one and TL, TR of the next.
    """
    m = np.zeros(shape, np.uint8)
    quads = []
    for i in range(0, 68, 4):
        tl, tr, bl, br = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        quads.append(np.array([tl, tr, br, bl], np.int32))
    for i, q in enumerate(quads):
        cv2.fillPoly(m, [q], 1)
        if i + 1 < len(quads):
            nxt = quads[i + 1]
            bridge = np.array([q[3], q[2], nxt[1], nxt[0]], np.int32)   # BL, BR -> TR, TL
            cv2.fillPoly(m, [bridge], 1)
    return m


def measure_mask(mask, **variant):
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) < 10:
        return None
    xs = np.where(mask.any(axis=0))[0]
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    rec = {'h': mask.shape[0], 'w': mask.shape[1], 'box': [float(v) for v in box],
           'rows': [int(y) for y in ys],
           'left': [int(np.where(mask[y])[0].min()) for y in ys],
           'right': [int(np.where(mask[y])[0].max()) for y in ys]}
    return CP.measure(rec, **variant)


def main():
    recs = {}
    for line in open(CP.RUN, encoding='utf-8'):
        r = json.loads(line)
        recs[r['name']] = r
    ref = {r['name']: float(r['ref']) for r in csv.DictReader(open(os.path.join(HERE, 'results.csv'), encoding='utf-8'))}
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    levels = json.load(open(os.path.join(HERE, 'levels.json'), encoding='utf-8'))
    pts_of = LA.reference_points()

    variant = {'turning': True}          # the post-processing as it will be deployed
    rows = []
    for name, rec in recs.items():
        model = CP.measure(rec, **variant) if 'rows' in rec else None
        o = measure_mask(oracle_mask(pts_of[name], (rec['h'], rec['w'])), **variant)
        rows.append({'name': name, 'ref': ref[name], 'loc': gt[name]['major_location'],
                     'model': None if model is None else model['max'],
                     'oracle': None if o is None else o['max'],
                     'model_pair': None if model is None else model['pair'],
                     'oracle_pair': None if o is None else o['pair']})

    def stats(key, subset=None):
        sel = [r for r in rows if r[key] is not None and (subset is None or subset(r))]
        e = np.array([r[key] - r['ref'] for r in sel])
        a = np.abs(e)
        lower_short = 0
        for r in sel:
            cy = [c[1] for c in levels[r['name']]['centres']]
            lo = max(LA.level_of((l['p1'][1] + l['p2'][1]) / 2, cy) for l in r[key + '_pair'])
            if round(lo) - levels[r['name']]['lower'] <= -2 and r['loc'] != 'thoracic':
                lower_short += 1
        return {'n': len(sel), 'mae': float(a.mean()), 'median': float(np.median(a)),
                'bias': float(e.mean()), 'big10': int((a > 10).sum()),
                'within5': float(100 * (a <= 5).mean()), 'tl_lower_short': lower_short}

    out, J = [], {}
    for key in ('model', 'oracle'):
        for label, sub in (('all', None),
                           ('thoracic', lambda r: r['loc'] == 'thoracic'),
                           ('TL/lumbar', lambda r: r['loc'] != 'thoracic')):
            s = stats(key, sub)
            J['%s/%s' % (key, label)] = s
            out.append('%-7s %-10s n=%3d MAE %.2f median %.2f bias %+.2f >10: %2d within5 %.1f%% TL lower short %d'
                       % (key, label, s['n'], s['mae'], s['median'], s['bias'], s['big10'], s['within5'],
                          s['tl_lower_short']))

    paired = [r for r in rows if r['model'] is not None and r['oracle'] is not None]
    d = np.array([abs(r['oracle'] - r['ref']) - abs(r['model'] - r['ref']) for r in paired])
    rng = np.random.default_rng(7)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    J['paired'] = {'n': len(d), 'mean': float(d.mean()),
                   'ci': [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                   'better5': int((d < -5).sum()), 'worse5': int((d > 5).sum())}
    out.append('paired oracle - model: %+.2f deg (95%% CI %+.2f to %+.2f), n=%d; better by >5: %d, worse: %d'
               % (d.mean(), *J['paired']['ci'], len(d), J['paired']['better5'], J['paired']['worse5']))
    # how often the model's mask fails to reach the reference lower end vertebra
    short = 0
    for r in rows:
        rec = recs[r['name']]
        if 'rows' not in rec:
            continue
        cy = [c[1] for c in levels[r['name']]['centres']]
        if LA.level_of(max(rec['rows']), cy) < levels[r['name']]['lower']:
            short += 1
    J['mask_short_of_lower_end'] = short
    out.append('model mask ends above the reference lower end vertebra in %d of %d images' % (short, len(rows)))

    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(HERE, 'cobb_oracle.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(J, open(os.path.join(HERE, 'cobb_oracle.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
