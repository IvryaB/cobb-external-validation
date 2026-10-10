"""Out-of-fold recalibration: how much of each method's error is a fixed offset?

A mask built in a different style (vertebral bodies only, as the landmarks define
them) shifts every angle by roughly the same amount, and a shift is something a
calibration constant absorbs. Fitting that constant on the same films it is scored
on would flatter it, so it is fitted on 4 folds of the file-name groups and applied
to the fifth.

Usage: python cobb_recalibration.py   ->  cobb_recalibration.txt / .json
"""
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cobb_oracle as CO      # noqa: E402
import cobb_postproc as CP    # noqa: E402
import levels_analysis as LA  # noqa: E402


def group_of(name):
    m = re.match(r'sunhl-1th-(\d+-\w+-\d+)-(\d+)\s', name)
    return m.group(1) + '/' + m.group(2) if m else name


def folds_for(names, k=5, seed=3):
    groups = [group_of(n) for n in names]
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    fold = {g: i % k for i, g in enumerate(rng.permutation(uniq))}
    return np.array([fold[g] for g in groups])


def oof_recalibrated(err, folds, kind='offset'):
    """Errors after a constant (or a scale) fitted on the other folds."""
    out = np.zeros_like(err)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        out[te] = err[te] - err[tr].mean()
    return out


def main():
    recs = {}
    for line in open(CP.RUN, encoding='utf-8'):
        r = json.loads(line)
        recs[r['name']] = r
    ref = {r['name']: float(r['ref']) for r in csv.DictReader(open(os.path.join(HERE, 'results.csv'), encoding='utf-8'))}
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    pts_of = LA.reference_points()

    names, model, oracle = [], [], []
    for n in sorted(recs):
        rec = recs[n]
        if 'rows' not in rec:
            continue
        m = CP.measure(rec, turning=True)
        o = CO.measure_mask(CO.oracle_mask(pts_of[n], (rec['h'], rec['w'])), turning=True)
        if m is None or o is None:
            continue
        names.append(n)
        model.append(m['max'])
        oracle.append(o['max'])
    names = np.array(names)
    r = np.array([ref[n] for n in names])
    folds = folds_for(list(names))
    tl = np.array([gt[n]['major_location'] != 'thoracic' for n in names])

    out, J = [], {}
    for label, pred in (('model mask', np.array(model)), ('perfect mask', np.array(oracle))):
        err = pred - r
        rec_err = oof_recalibrated(err, folds)
        J[label] = {
            'n': len(err), 'mae': float(np.abs(err).mean()), 'bias': float(err.mean()),
            'mae_recal': float(np.abs(rec_err).mean()),
            'big10_recal': int((np.abs(rec_err) > 10).sum()),
            'within5_recal': float(100 * (np.abs(rec_err) <= 5).mean()),
            'mae_recal_tl': float(np.abs(rec_err[tl]).mean()),
            'mae_recal_th': float(np.abs(rec_err[~tl]).mean()),
        }
        s = J[label]
        out.append('%-13s MAE %.2f (bias %+.2f) | after out-of-fold recalibration: MAE %.2f, >10: %d, '
                   'within5 %.1f%%, TL %.2f, thoracic %.2f'
                   % (label, s['mae'], s['bias'], s['mae_recal'], s['big10_recal'], s['within5_recal'],
                      s['mae_recal_tl'], s['mae_recal_th']))

    d = np.abs(oof_recalibrated(np.array(oracle) - r, folds)) - np.abs(oof_recalibrated(np.array(model) - r, folds))
    rng = np.random.default_rng(11)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    J['paired_recal'] = {'mean': float(d.mean()), 'ci': [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
    out.append('paired, both recalibrated: perfect mask is %+.2f deg (95%% CI %+.2f to %+.2f) from the model mask'
               % (d.mean(), *J['paired_recal']['ci']))

    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(HERE, 'cobb_recalibration.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(J, open(os.path.join(HERE, 'cobb_recalibration.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
