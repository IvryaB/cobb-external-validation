"""What a trained segmentation actually buys, out of fold.

cobb_oracle.py measured the ceiling (a perfect mask built from the landmarks).
spine_unet.py trained a small net on those masks with group 5-fold CV and saved
its OUT-OF-FOLD predictions. This script runs the deployed geometry on those
predicted masks and compares three columns on the same films:

    deployed model mask  ->  what production does today
    U-Net (out of fold)  ->  what a trained-here segmentation gives
    perfect mask         ->  the ceiling

All three get the same out-of-fold offset recalibration, so no column is
flattered by a constant fitted on the films it is scored on.

Usage: [UNET_NPZ=<path>] python cobb_unet_eval.py  ->  cobb_unet_eval_<masks>.txt / .json
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cobb_oracle as CO          # noqa: E402
import cobb_postproc as CP        # noqa: E402
import cobb_recalibration as CR   # noqa: E402
import levels_analysis as LA      # noqa: E402

NPZ = os.environ.get('UNET_NPZ') or os.path.join(ROOT, 'radtools_run', 'unet_masks.npz')
TAG = os.path.splitext(os.path.basename(NPZ))[0]


def unet_mask(prob, shape):
    """Upsample the 384x128 probability map to the film and keep the main band."""
    import cv2
    m = Image.fromarray((np.asarray(prob, np.float32) * 255).astype(np.uint8))
    m = np.asarray(m.resize((shape[1], shape[0]), Image.BILINEAR))
    mask = (m > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == k).astype(np.uint8)
    return mask


def main():
    d = np.load(NPZ, allow_pickle=True)
    # npz members decompress on every access, so read the stack once.
    pred = d['pred']
    probs = {str(n): pred[i] for i, n in enumerate(d['names'])}

    recs = {}
    for line in open(CP.RUN, encoding='utf-8'):
        r = json.loads(line)
        recs[r['name']] = r
    ref = {r['name']: float(r['ref']) for r in csv.DictReader(open(os.path.join(HERE, 'results.csv'), encoding='utf-8'))}
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    levels = json.load(open(os.path.join(HERE, 'levels.json'), encoding='utf-8'))
    pts_of = LA.reference_points()

    names, cols = [], {'model': [], 'unet': [], 'perfect': []}
    reach = {'model': 0, 'unet': 0, 'perfect': 0}
    for n in sorted(recs):
        rec = recs[n]
        if 'rows' not in rec or n not in probs:
            continue
        shape = (rec['h'], rec['w'])
        um = unet_mask(probs[n], shape)
        om = CO.oracle_mask(pts_of[n], shape)
        a = CP.measure(rec, turning=True)
        b = CO.measure_mask(um, turning=True)
        c = CO.measure_mask(om, turning=True)
        if a is None or b is None or c is None:
            continue
        names.append(n)
        cols['model'].append(a['max'])
        cols['unet'].append(b['max'])
        cols['perfect'].append(c['max'])
        cy = [p[1] for p in levels[n]['centres']]
        lower = levels[n]['lower']
        bottoms = {'model': max(rec['rows']),
                   'unet': int(np.where(um.any(axis=1))[0].max()),
                   'perfect': int(np.where(om.any(axis=1))[0].max())}
        for k, y in bottoms.items():
            if LA.level_of(y, cy) >= lower:
                reach[k] += 1

    r = np.array([ref[n] for n in names])
    folds = CR.folds_for(names)
    tl = np.array([gt[n]['major_location'] != 'thoracic' for n in names])

    out, J = [], {'n': len(names)}
    rec_err = {}
    for key in ('model', 'unet', 'perfect'):
        p = np.array(cols[key])
        err = p - r
        e = CR.oof_recalibrated(err, folds)
        rec_err[key] = e
        J[key] = {'mae_raw': float(np.abs(err).mean()), 'bias': float(err.mean()),
                  'mae': float(np.abs(e).mean()), 'median': float(np.median(np.abs(e))),
                  'big10': int((np.abs(e) > 10).sum()),
                  'within5': float(100 * (np.abs(e) <= 5).mean()),
                  'mae_tl': float(np.abs(e[tl]).mean()), 'mae_th': float(np.abs(e[~tl]).mean()),
                  'reaches_lower_end': reach[key]}
        s = J[key]
        out.append('%-8s raw MAE %.2f (bias %+.2f) | recalibrated MAE %.2f median %.2f, >10: %2d, '
                   'within5 %.1f%%, TL %.2f, thoracic %.2f, reaches lower end %d/%d'
                   % (key, s['mae_raw'], s['bias'], s['mae'], s['median'], s['big10'],
                      s['within5'], s['mae_tl'], s['mae_th'], s['reaches_lower_end'], len(names)))

    rng = np.random.default_rng(23)
    for a, b in (('unet', 'model'), ('perfect', 'unet')):
        dd = np.abs(rec_err[a]) - np.abs(rec_err[b])
        boot = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
        ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        J['%s_vs_%s' % (a, b)] = {'mean': float(dd.mean()), 'ci': ci}
        out.append('%s vs %s: %+.2f deg (95%% CI %+.2f to %+.2f)' % (a, b, dd.mean(), *ci))

    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(HERE, 'cobb_unet_eval_%s.txt' % TAG), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(J, open(os.path.join(HERE, 'cobb_unet_eval_%s.json' % TAG), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
