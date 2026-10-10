"""Where does the method go wrong? Level-wise analysis of the RadTools tangents.

For every measured image the service returns the tangent segments it drew
(geometry.lines, pixel coordinates of the original image) and the angle between
each pair (geometry.angles). The midpoint of a segment is the point on the spine
centreline where the tilt was taken. It is placed on the reference spine as a
continuous level (0 = T1 ... 16 = L5; <0 above T1, >16 below L5) by interpolating
its y coordinate between the centres of the expert-annotated vertebrae.

For the pair that gives the largest angle this yields the method's end vertebrae,
which are compared with the reference end vertebrae of the largest curve.

Error decomposition. With the expert tilts taken at the vertebrae the method chose,
    RadTools - reference = (RadTools - reference at RadTools levels)      tilt term
                         + (reference at RadTools levels - reference)     level term
The level term is <= 0 by construction (the reference is the largest angle over all
pairs); it is the angle lost by choosing other end vertebrae. The tilt term is the
difference between a centreline tangent and an endplate tilt at the same levels.

Large errors (|error| > 10 deg) are sorted into one mechanism each, in this order:
  outside T1-L5     an end tangent lies more than half a level above T1 or below L5;
  other curve       the method's curve does not contain the reference apex;
  end shifted       it contains the apex, but an end is >= 2 levels off;
  tilt only         both ends within 1 level: the tilt itself is off.

Outputs: levels.txt, levels_summary.json, levels_per_image.csv, fig_levels.png
"""
import csv
import json
import os

import numpy as np
from PIL import Image

import ground_truth as G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, 'radtools_run', 'aasce_train481_geometry.jsonl')
NAMES = G.VERTEBRAE
BIG = 10.0


def reference_points():
    """name -> 68 x 2 landmarks (curated where available, else original)."""
    names = [r[0] for r in G.read_rows(os.path.join(G.KAG, 'train_txt', 'filenames.csv'))]
    lms = [[float(x) for x in r] for r in G.read_rows(os.path.join(G.KAG, 'train_txt', 'landmarks.csv'))]
    out = {}
    for name, lm in zip(names, lms):
        xml = os.path.join(G.ZEN, os.path.splitext(name)[0] + '.xml')
        pts = None
        if os.path.exists(xml):
            pts, _ = G.curated_points(xml)
        if pts is None:
            w, h = Image.open(os.path.join(G.KAG, 'train', name)).size
            pts = np.stack([np.array(lm[:68]) * w, np.array(lm[68:]) * h], axis=1)
        out[name] = pts
    return out


def tilt_angles(pts):
    """17 x 17 matrix of angles (deg) between the tilts of every two vertebrae,
    the same tilt definition as the challenge algorithm."""
    vec = np.array([((pts[i + 1] + pts[i + 3]) / 2) - ((pts[i] + pts[i + 2]) / 2) for i in range(0, 68, 4)])
    n = np.linalg.norm(vec, axis=1)
    return np.degrees(np.arccos(np.clip((vec @ vec.T) / np.outer(n, n), 0, 1)))


def level_of(y, cy):
    """Continuous vertebral level of image row y; linear extrapolation beyond T1/L5."""
    cy = np.asarray(cy, float)
    if y <= cy[0]:
        return (y - cy[0]) / (cy[1] - cy[0])
    if y >= cy[-1]:
        return 16 + (y - cy[-1]) / (cy[-1] - cy[-2])
    return float(np.interp(y, cy, np.arange(17)))


def lab(t):
    k = int(round(t))
    if k < 0:
        return 'C' if k < -1 else 'C7'
    if k > 16:
        return 'S'
    return NAMES[k]


def mechanism(err, up, lo, r_up, r_lo, r_apex):
    if abs(err) <= BIG:
        return ''
    if up < -0.5 or lo > 16.5:
        return 'outside T1–L5'
    if not (up - 0.5 <= r_apex <= lo + 0.5):
        return 'other curve'
    if abs(round(up) - r_up) >= 2 or abs(round(lo) - r_lo) >= 2:
        return 'end shifted'
    return 'tilt only'


def main():
    runs = {}
    for line in open(RUN, encoding='utf-8'):
        rec = json.loads(line)
        runs[os.path.basename(rec['name'])] = rec
    res = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'results.csv'), encoding='utf-8'))}
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    levels = json.load(open(os.path.join(HERE, 'levels.json'), encoding='utf-8'))
    pts_of = reference_points()

    rows, skipped = [], []
    for name, r in res.items():
        if r['failed'] == 'True':
            continue
        rec = runs.get(name)
        geo = (rec or {}).get('geometry') or {}
        if not geo.get('lines') or not geo.get('angles'):
            skipped.append(name)
            continue
        lines = {l['id']: l for l in geo['lines']}
        best = max(geo['angles'], key=lambda a: a['value'])
        lv = levels[name]
        cy = [c[1] for c in lv['centres']]
        ts = []
        for k in (best['a'], best['b']):
            p1, p2 = np.array(lines[k]['p1'], float), np.array(lines[k]['p2'], float)
            ts.append(level_of((p1[1] + p2[1]) / 2, cy))
        up, lo = sorted(ts)
        ref, pred = float(r['ref']), float(r['radtools'])
        ang = tilt_angles(pts_of[name])
        iu, il = int(np.clip(round(up), 0, 16)), int(np.clip(round(lo), 0, 16))
        ref_at = float(ang[iu, il])
        r_up, r_lo, r_apex = lv['upper'], lv['lower'], lv['apex']
        err = pred - ref
        rows.append({
            'name': name, 'ref': ref, 'radtools': pred, 'err': round(err, 3),
            'location': gt[name]['major_location'],
            'ref_upper': NAMES[r_up], 'ref_lower': NAMES[r_lo], 'ref_apex': NAMES[r_apex],
            'rt_upper': lab(up), 'rt_lower': lab(lo), 'rt_upper_level': round(up, 2), 'rt_lower_level': round(lo, 2),
            'd_upper': int(round(up)) - r_up, 'd_lower': int(round(lo)) - r_lo,
            'ref_at_rt_levels': round(ref_at, 3),
            'tilt_term': round(pred - ref_at, 3), 'level_term': round(ref_at - ref, 3),
            'mechanism': mechanism(err, up, lo, r_up, r_lo, r_apex),
        })

    n = len(rows)
    du = np.array([x['d_upper'] for x in rows])
    dl = np.array([x['d_lower'] for x in rows])
    err = np.array([x['err'] for x in rows])
    tilt = np.array([x['tilt_term'] for x in rows])
    lvl = np.array([x['level_term'] for x in rows])
    both = (np.abs(du) <= 1) & (np.abs(dl) <= 1)
    out, J = [], {'n': n, 'skipped_no_geometry': skipped}
    out.append('Measured images with tangent geometry: %d (skipped, no geometry: %d)' % (n, len(skipped)))
    out.append('End vertebrae of the largest curve, RadTools vs reference (levels; + = more caudal):')
    for key, d in (('upper', du), ('lower', dl)):
        exact, w1 = int((d == 0).sum()), int((np.abs(d) <= 1).sum())
        out.append('  %s end: exact %d (%.1f%%), within 1 level %d (%.1f%%), median offset %+.0f, mean %+.2f'
                   % (key, exact, 100 * exact / n, w1, 100 * w1 / n, np.median(d), d.mean()))
        J[key] = {'exact': exact, 'within1': w1, 'mean_offset': float(d.mean()), 'median_offset': float(np.median(d))}
    out.append('  both ends within 1 level: %d (%.1f%%)' % (both.sum(), 100 * both.mean()))
    J['both_within1'] = int(both.sum())
    for label, m in (('both ends within 1 level', both), ('at least one end off by >= 2', ~both)):
        a = np.abs(err[m])
        out.append('  %-30s n=%3d  MAE %.1f  median %.1f  bias %+.1f  |err|>10: %d'
                   % (label, m.sum(), a.mean(), np.median(a), err[m].mean(), int((a > BIG).sum())))
        J['mae_' + ('agree' if label.startswith('both') else 'disagree')] = {
            'n': int(m.sum()), 'mae': float(a.mean()), 'median': float(np.median(a)), 'bias': float(err[m].mean()),
            'big': int((a > BIG).sum())}

    out.append('')
    out.append('By location of the major curve (lower-end offset shows truncation at the lumbar end):')
    J['by_location'] = {}
    for loc in ('thoracic', 'thoracolumbar/lumbar'):
        m = np.array([x['location'] == loc for x in rows])
        d = {'n': int(m.sum()), 'upper_mean': float(du[m].mean()), 'lower_mean': float(dl[m].mean()),
             'upper_w1': float(100 * (np.abs(du[m]) <= 1).mean()), 'lower_w1': float(100 * (np.abs(dl[m]) <= 1).mean()),
             'lower_short2': float(100 * (dl[m] <= -2).mean()),
             'tilt_mean': float(tilt[m].mean()), 'level_mean': float(lvl[m].mean())}
        J['by_location'][loc] = d
        out.append('  %-22s n=%3d  upper within1 %.0f%% (mean %+.2f)  lower within1 %.0f%% (mean %+.2f), '
                   'lower end >=2 levels too cranial %.0f%%'
                   % (loc, d['n'], d['upper_w1'], d['upper_mean'], d['lower_w1'], d['lower_mean'], d['lower_short2']))
        out.append('  %-22s        error = tilt term %+.2f + level term %+.2f' % ('', d['tilt_mean'], d['level_mean']))

    out.append('')
    out.append('Error decomposition (all measured): mean error %+.2f = tilt term %+.2f + level term %+.2f'
               % (err.mean(), tilt.mean(), lvl.mean()))
    out.append('  MAE of the tilt term %.1f; mean level term %.1f (angle lost by choosing other end vertebrae)'
               % (np.abs(tilt).mean(), lvl.mean()))
    out.append('  tilt term when both ends agree: mean %+.2f, MAE %.1f' % (tilt[both].mean(), np.abs(tilt[both]).mean()))
    exact = (du == 0) & (dl == 0)
    r_tl = float(np.corrcoef(tilt, lvl)[0, 1])
    out.append('  both ends exact (n=%d): tilt term mean %+.2f, MAE %.1f; corr(tilt term, level term) r=%.2f'
               % (exact.sum(), tilt[exact].mean(), np.abs(err[exact]).mean(), r_tl))
    J['decomp'] = {'err_mean': float(err.mean()), 'tilt_mean': float(tilt.mean()), 'level_mean': float(lvl.mean()),
                   'tilt_mae': float(np.abs(tilt).mean()), 'tilt_mean_agree': float(tilt[both].mean()),
                   'tilt_mae_agree': float(np.abs(tilt[both]).mean()), 'n_exact': int(exact.sum()),
                   'tilt_mean_exact': float(tilt[exact].mean()), 'mae_exact': float(np.abs(err[exact]).mean()),
                   'r_tilt_level': r_tl}

    # "another curve" regardless of the size of the error: the measured curve does not contain the reference apex
    oth = np.array([not (x['rt_upper_level'] - 0.5 <= NAMES.index(x['ref_apex']) <= x['rt_lower_level'] + 0.5)
                    for x in rows])
    J['other_any'] = {'n': int(oth.sum()), 'mae': float(np.abs(err[oth]).mean()), 'big': int((np.abs(err[oth]) > BIG).sum()),
                      'median': float(np.median(np.abs(err[oth])))}
    for loc in ('thoracic', 'thoracolumbar/lumbar'):
        m = oth & np.array([x['location'] == loc for x in rows])
        J['other_any'][loc] = {'n': int(m.sum()), 'mae': float(np.abs(err[m]).mean()) if m.any() else None,
                               'big': int((np.abs(err[m]) > BIG).sum()),
                               'pct': float(100 * m.sum() / sum(x['location'] == loc for x in rows))}
    o = J['other_any']
    out.append('  another curve measured (any error): n=%d, MAE %.1f, median %.1f, >10: %d; thoracic %d (%.0f%%), '
               'TL/lumbar %d (%.0f%%)' % (o['n'], o['mae'], o['median'], o['big'], o['thoracic']['n'], o['thoracic']['pct'],
                                          o['thoracolumbar/lumbar']['n'], o['thoracolumbar/lumbar']['pct']))

    out.append('')
    big = [x for x in rows if x['mechanism']]
    out.append('Errors > %d° (n=%d): mechanism, count, mean error, locations' % (BIG, len(big)))
    J['mechanisms'] = {}
    for mech in ('outside T1–L5', 'other curve', 'end shifted', 'tilt only'):
        g = [x for x in big if x['mechanism'] == mech]
        if not g:
            J['mechanisms'][mech] = {'n': 0}
            continue
        e = np.array([x['err'] for x in g])
        tl = sum(x['location'] != 'thoracic' for x in g)
        under = int((e < 0).sum())
        out.append('  %-14s n=%2d  mean error %+.1f  underestimated %d  TL/lumbar major curve %d'
                   % (mech, len(g), e.mean(), under, tl))
        J['mechanisms'][mech] = {'n': len(g), 'mean_err': float(e.mean()), 'under': under, 'tl': tl,
                                 'examples': [x['name'] for x in sorted(g, key=lambda x: -abs(x['err']))[:3]]}
    other = [x for x in big if x['mechanism'] == 'other curve']
    if other:
        # was the measured curve above (cranial to) the reference apex, i.e. the thoracic one?
        above = sum(x['rt_lower_level'] < NAMES.index(x['ref_apex']) for x in other)
        out.append('  other curve: measured curve lies above the reference apex in %d of %d' % (above, len(other)))
        J['other_detail'] = {'above': above, 'n': len(other)}
    tilt_only = [x for x in big if x['mechanism'] == 'tilt only']
    if tilt_only:
        span = lambda x: NAMES.index(x['ref_lower']) - NAMES.index(x['ref_upper'])
        short = sum(span(x) <= 5 for x in tilt_only)
        upper_t = sum(NAMES.index(x['ref_upper']) <= 4 for x in tilt_only)
        J['tilt_only_detail'] = {
            'n': len(tilt_only), 'short': short, 'short_all_pct': 100 * float(np.mean([span(x) <= 5 for x in rows])),
            'upper_thoracic': upper_t,
            'upper_thoracic_all_pct': 100 * float(np.mean([NAMES.index(x['ref_upper']) <= 4 for x in rows])),
            'over': sum(x['err'] > 0 for x in tilt_only)}
        d = J['tilt_only_detail']
        out.append('  tilt only: curve of <=5 levels %d/%d (all curves %.0f%%), upper end T1-T5 %d/%d (all %.0f%%), '
                   'overestimated %d' % (short, d['n'], d['short_all_pct'], upper_t, d['n'], d['upper_thoracic_all_pct'],
                                         d['over']))
    shifted = [x for x in big if x['mechanism'] == 'end shifted']
    if shifted:
        lower_short = sum(x['d_lower'] <= -2 for x in shifted)
        upper_short = sum(x['d_upper'] >= 2 for x in shifted)
        out.append('  end shifted: lower end too cranial %d, upper end too caudal %d' % (lower_short, upper_short))
        J['shifted_detail'] = {'lower_short': lower_short, 'upper_short': upper_short}

    txt = '\n'.join(out)
    open(os.path.join(HERE, 'levels.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(J, open(os.path.join(HERE, 'levels_summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, 'levels_per_image.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(txt)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), dpi=300, gridspec_kw={'width_ratios': [1.25, 1]})
    offs = np.arange(-5, 6)
    width = 0.2
    series = []
    for loc, short in (('thoracic', 'Thoracic'), ('thoracolumbar/lumbar', 'TL/lumbar')):
        m = np.array([x['location'] == loc for x in rows])
        for end, d, hatch in (('upper', du, ''), ('lower', dl, '////')):
            v = np.clip(d[m], -5, 5)
            series.append(('%s, %s end' % (short, end), [100 * (v == o).mean() for o in offs], hatch, loc))
    colours = {'thoracic': '#9a9a9a', 'thoracolumbar/lumbar': '#303030'}
    for i, (label, vals, hatch, loc) in enumerate(series):
        a1.bar(offs + (i - 1.5) * width, vals, width, label=label, color='white' if hatch else colours[loc],
               edgecolor=colours[loc], hatch=hatch, linewidth=0.6)
    a1.set_xticks(offs)
    a1.set_xticklabels(['≤−5' if o == -5 else ('≥5' if o == 5 else ('%+d' % o if o else '0')) for o in offs])
    a1.set_xlabel('End vertebra, RadTools − reference (levels)')
    a1.set_ylabel('Images (%)')
    a1.legend(frameon=False, fontsize=7, loc='upper left')
    a1.text(-0.02, 1.02, '(a)', transform=a1.transAxes, va='bottom', ha='right', fontweight='bold')
    sev = [(0, 10), (10, 25), (25, 40), (40, 200)]
    labels = ['<10°', '10–25°', '25–40°', '≥40°']
    refs = np.array([x['ref'] for x in rows])
    tm = [tilt[(refs >= lo_) & (refs < hi_)].mean() for lo_, hi_ in sev]
    lm = [lvl[(refs >= lo_) & (refs < hi_)].mean() for lo_, hi_ in sev]
    em = [err[(refs >= lo_) & (refs < hi_)].mean() for lo_, hi_ in sev]
    x = np.arange(4)
    a2.bar(x - 0.18, tm, 0.34, color='#9a9a9a', label='Tilt term')
    a2.bar(x + 0.18, lm, 0.34, color='#303030', label='Level term')
    a2.plot(x, em, 'o-', color='black', ms=3, lw=0.8, label='Total (mean error)')
    a2.axhline(0, color='grey', lw=0.6)
    a2.set_xticks(x)
    a2.set_xticklabels(labels)
    a2.set_xlabel('Reference severity')
    a2.set_ylabel('Mean contribution (°)')
    lo_, hi_ = min(min(lm), min(em)), max(tm)
    a2.set_ylim(lo_ - 1.0, hi_ + 3.2)
    a2.legend(frameon=False, fontsize=7, loc='upper center', ncol=3, handlelength=1.2, columnspacing=0.8)
    a2.text(-0.02, 1.02, '(b)', transform=a2.transAxes, va='bottom', ha='right', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_levels.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
