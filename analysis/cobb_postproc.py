"""Offline bench for the RadTools/Duke Cobb post-processing.

The network only produces a spine mask. Everything after it — centreline, polynomial
fit, break points, most-tilted levels, angles — is plain geometry, and the error
analysis (levels_analysis.py) put most large errors there, at the choice of end
vertebrae. centrelines.jsonl holds the mask of every AASCE image as the model
produced it (row-wise left/right extent, the only thing the geometry reads), dumped
once in the ai-core pod. Here the geometry is rerun exactly as deployed
(scol_base.py = services/ai-core/scoliosis_inference.py with the sorted() fix) and
in variants, and each is scored against the landmark reference.

Usage: python cobb_postproc.py            -> cobb_postproc.txt / cobb_postproc.json
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scol_base as SC               # noqa: E402
import levels_analysis as LA         # noqa: E402

RUN = os.path.join(os.path.dirname(HERE), 'radtools_run', 'centrelines.jsonl')


def mask_of(rec):
    m = np.zeros((rec['h'], rec['w']), np.uint8)
    for y, lo, hi in zip(rec['rows'], rec['left'], rec['right']):
        m[y, lo:hi + 1] = 1
    return m


def steepest(res, image, break_points, x_fit, f_x, f_x_1d, x_offset, y_offset, break_i,
             tol_ratio=0.15, off_l=0, end_search=False, end_margin=0.03):
    """steepest_point_method without the drawing, plus the variant switches.

    end_search: when an interval has no interior peak of |f'| — find_peaks never
    returns the first or last sample, so a slope that keeps growing to the end of the
    curve has none — the deployed code falls back to the MIDDLE of the interval. For
    the lowest interval of a lumbar curve that puts the tangent well above the most
    tilted vertebra. With end_search the steepest sample of the interval is taken,
    ends included, except for the last ``end_margin`` of the curve where a degree-10
    polynomial swings freely.
    """
    vxs, vys, xcs, ycs, lines = [], [], [], [], []
    fx_dx = f_x_1d(x_fit)
    peaks, _ = find_peaks(abs(fx_dx))
    bp = (np.asarray(break_points) - x_offset) * res.ratio
    n = len(fx_dx)
    margin = int(n * end_margin)
    for i in range(len(bp) - 1):
        tol_l = (bp[i + 1] - bp[i]) * tol_ratio
        xs = np.array(x_fit)
        peak_now = peaks[(xs[peaks] > bp[i]) & (xs[peaks] < bp[i + 1])]
        if len(peak_now) == 0:
            if end_search:
                lo_i, hi_i = break_i[i], break_i[i + 1]
                lo_i, hi_i = max(lo_i, margin), min(hi_i, n - 1 - margin)
                if hi_i <= lo_i:
                    peak_now = np.array([int((break_i[i] + break_i[i + 1]) / 2)])
                else:
                    peak_now = np.array([lo_i + int(np.argmax(abs(fx_dx[lo_i:hi_i + 1])))])
            else:
                peak_now = np.array([int((break_i[i] + break_i[i + 1]) / 2)])
        peak_now = int(peak_now[abs(fx_dx)[peak_now].argmax()])
        mid = (bp[i] + bp[i + 1]) / 2
        off_x = min(int(off_l / np.sqrt(1 + fx_dx[peak_now] ** 2)), int(abs(x_fit[peak_now] - mid)))
        if x_fit[peak_now] > mid and i < (len(bp) - 2):
            peak_new = peak_now + off_x
        else:
            peak_new = max(peak_now - off_x, int(tol_l))
        m = np.mean(fx_dx[max(peak_new - int(tol_l), 0):peak_new + int(tol_l) + 1])
        if not np.isfinite(m):
            continue
        vx, vy = 1 / np.sqrt(1 + m ** 2), m / np.sqrt(1 + m ** 2)
        vxs.append(vx)
        vys.append(vy)
        x_c = int(x_fit[peak_now] / res.ratio) + x_offset
        y_c = int(f_x(x_fit[peak_now]) / res.ratio) + y_offset
        xcs.append(x_c)
        ycs.append(y_c)
        L = 60
        lines.append({'p1': [int(round(y_c - L * (-vx))), int(round(x_c - L * vy))],
                      'p2': [int(round(y_c + L * (-vx))), int(round(x_c + L * vy))]})
    angles = [float(np.degrees(np.arccos(np.clip(np.dot([vxs[i], vys[i]], [vxs[i + 1], vys[i + 1]]), -1, 1))))
              for i in range(len(vxs) - 1)]
    return angles, lines, list(zip(vxs, vys))


def measure(rec, trunk=0.02, break_tol=35, any_pair=False, turning=False, **variant):
    """any_pair: the main angle is the largest between ANY two tangents, not only
    consecutive ones — the reference's own definition (largest angle between any two
    vertebral tilts). A long C-curve that the break-point rules cut into two intervals
    is otherwise reported as two small angles."""
    if 'rows' not in rec:
        return None
    mask = mask_of(rec)
    res = SC._MaskAngle(np.zeros((1, 1, 3), np.uint8), mask, np.array(rec['box']))
    skeleton, px, py = SC._get_skeleton(mask, trunk)
    bp, _, x_fit, f_x, f_x_1d, xo, yo, bi = res.find_break_point(px, py, tol=break_tol)
    angles, lines, vecs = steepest(res, None, bp, x_fit, f_x, f_x_1d, xo, yo, bi, **variant)
    if not angles:
        return None
    if turning and len(vecs) > 2:
        # keep only the tangents where the tilt turns back: a run of tangents tilting
        # further the same way is one curve, and its inner tangents only split it
        th = [float(np.arctan2(vy, vx)) for vx, vy in vecs]
        keep = [0] + [k for k in range(1, len(th) - 1)
                      if (th[k] - th[k - 1]) * (th[k + 1] - th[k]) < 0] + [len(th) - 1]
        vk = [vecs[k] for k in keep]
        lk = [lines[k] for k in keep]
        ang = [float(np.degrees(np.arccos(np.clip(np.dot(vk[i], vk[i + 1]), -1, 1)))) for i in range(len(vk) - 1)]
        k = int(np.argmax(ang))
        return {'max': ang[k], 'pair': (lk[k], lk[k + 1]), 'angles': ang}
    if any_pair and len(vecs) > 2:
        best, pair = -1.0, (0, 1)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                a = float(np.degrees(np.arccos(np.clip(np.dot(vecs[i], vecs[j]), -1, 1))))
                if a > best:
                    best, pair = a, (i, j)
        return {'max': best, 'pair': (lines[pair[0]], lines[pair[1]]), 'angles': angles}
    k = int(np.argmax(angles))
    return {'max': max(angles), 'pair': (lines[k], lines[k + 1]), 'angles': angles}


def score(label, results, ref, levels, gt):
    """MAE etc. over images measured, plus failures and the level mechanism table."""
    ok = [(n, r) for n, r in results.items() if r is not None]
    err = np.array([r['max'] - ref[n] for n, r in ok])
    ae = np.abs(err)
    tl = np.array([gt[n]['major_location'] != 'thoracic' for n, r in ok])
    # end vertebrae of the largest angle, as levels_analysis maps them
    lower_short = 0
    for n, r in ok:
        cy = [c[1] for c in levels[n]['centres']]
        ys = sorted(LA.level_of((l['p1'][1] + l['p2'][1]) / 2, cy) for l in r['pair'])
        if round(ys[1]) - levels[n]['lower'] <= -2 and gt[n]['major_location'] != 'thoracic':
            lower_short += 1
    return {
        'label': label, 'n': len(ok), 'failed': len(results) - len(ok),
        'mae': float(ae.mean()), 'median': float(np.median(ae)), 'bias': float(err.mean()),
        'within5': float(100 * (ae <= 5).mean()), 'big10': int((ae > 10).sum()),
        'bias_tl': float(err[tl].mean()), 'mae_tl': float(ae[tl].mean()),
        'bias_th': float(err[~tl].mean()), 'mae_th': float(ae[~tl].mean()),
        'tl_lower_short': lower_short,
    }


def main():
    recs = {}
    for line in open(RUN, encoding='utf-8'):
        r = json.loads(line)
        recs[r['name']] = r
    res_csv = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'results.csv'), encoding='utf-8'))}
    ref = {n: float(r['ref']) for n, r in res_csv.items()}
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    levels = json.load(open(os.path.join(HERE, 'levels.json'), encoding='utf-8'))

    variants = [
        ('deployed (sorted fix)', {}),
        ('end search 3%', {'end_search': True, 'end_margin': 0.03}),
        ('any pair', {'any_pair': True}),
        ('any pair + end search 3%', {'any_pair': True, 'end_search': True, 'end_margin': 0.03}),
        ('turning points', {'turning': True}),
        ('turning points + end 3%', {'turning': True, 'end_search': True, 'end_margin': 0.03}),
        ('trunk 0', {'trunk': 0.0}),
        ('break tol 20', {'break_tol': 20}),
        ('any pair + end 3% + trunk 0', {'any_pair': True, 'end_search': True, 'end_margin': 0.03, 'trunk': 0.0}),
    ]
    out, J = [], {}
    base = None
    for label, v in variants:
        results = {n: measure(r, **v) for n, r in recs.items()}
        s = score(label, results, ref, levels, gt)
        if base is None:
            base = results
        else:
            changed = sum(1 for n in results if (results[n] or {}).get('max') != (base[n] or {}).get('max'))
            s['changed_vs_deployed'] = changed
        J[label] = s
        out.append('%-26s n=%d fail=%d MAE %.2f median %.2f bias %+.2f within5 %.1f%% >10: %d | '
                   'TL bias %+.2f MAE %.2f | TH bias %+.2f MAE %.2f | TL lower>=2 short %d%s'
                   % (label, s['n'], s['failed'], s['mae'], s['median'], s['bias'], s['within5'], s['big10'],
                      s['bias_tl'], s['mae_tl'], s['bias_th'], s['mae_th'], s['tl_lower_short'],
                      '' if 'changed_vs_deployed' not in s else ' | changed %d' % s['changed_vs_deployed']))
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(HERE, 'cobb_postproc.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(J, open(os.path.join(HERE, 'cobb_postproc.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
