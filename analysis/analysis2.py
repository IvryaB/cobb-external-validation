"""Article-level statistics on top of compare.py's results.csv.

  * bootstrap 95% CIs (2,000 resamples, fixed seed) for the headline metrics;
  * subgroups: severity, and location of the major curve (apex T1-T11 thoracic,
    T12-L5 thoracolumbar/lumbar) with Mann-Whitney / Kruskal-Wallis tests;
  * clinical thresholds 10, 25, 40 deg: sensitivity, specificity, accuracy, and a
    linearly weighted kappa over the four severity classes;
  * failures reported separately, plus a worst case that counts them as 0 deg.

Outputs: analysis2.txt, analysis2.json, fig_agreement.png (scatter + Bland-Altman),
fig_subgroups.png (|error| by severity and by curve location).
"""
import csv
import json
import os

import numpy as np
from scipy import stats

from compare import icc_2_1

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260921)
CUTS = [10, 25, 40]
CLASSES = ['<10°', '10–25°', '25–40°', '≥40°']


def cls(a):
    return int(np.searchsorted(CUTS, a, side='right'))


def weighted_kappa(a, b, k=4):
    o = np.zeros((k, k))
    for i, j in zip(a, b):
        o[i, j] += 1
    o /= o.sum()
    e = np.outer(o.sum(1), o.sum(0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    return 1 - (w * o).sum() / (w * e).sum(), o


def boot(fn, *arrays, n=2000):
    idx = np.arange(len(arrays[0]))
    vals = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        vals.append(fn(*[a[s] for a in arrays]))
    return np.percentile(vals, [2.5, 97.5])


def fmt_ci(v, ci, d=1):
    return '%.*f (%.*f–%.*f)' % (d, v, d, ci[0], d, ci[1])


def main():
    res = list(csv.DictReader(open(os.path.join(HERE, 'results.csv'), encoding='utf-8')))
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    for r in res:
        r['location'] = gt[r['name']]['major_location']
        r['ref'] = float(r['ref'])
        r['failed'] = r['failed'] == 'True'
        r['pred'] = None if r['failed'] else float(r['radtools'])
    ok = [r for r in res if not r['failed']]
    ref = np.array([r['ref'] for r in ok])
    pred = np.array([r['pred'] for r in ok])
    err = pred - ref
    ae = np.abs(err)
    out, J = [], {}

    head = {
        'mae': (ae.mean(), boot(lambda a: a.mean(), ae)),
        'median_ae': (np.median(ae), boot(lambda a: np.median(a), ae)),
        'within5': (100 * (ae <= 5).mean(), boot(lambda a: 100 * (a <= 5).mean(), ae)),
        'within10': (100 * (ae <= 10).mean(), boot(lambda a: 100 * (a <= 10).mean(), ae)),
        'bias': (err.mean(), boot(lambda e: e.mean(), err)),
        'icc': (icc_2_1(ref, pred), boot(lambda a, b: icc_2_1(a, b), ref, pred)),
    }
    out.append('Headline (n=%d of %d; %d failures), value (95%% bootstrap CI):' % (len(ok), len(res), len(res) - len(ok)))
    for k, (v, ci) in head.items():
        out.append('  %-10s %s' % (k, fmt_ci(v, ci, 3 if k == 'icc' else 1)))
    J['headline'] = {k: {'v': float(v), 'lo': float(ci[0]), 'hi': float(ci[1])} for k, (v, ci) in head.items()}
    rho, p_rho = stats.spearmanr(ref, ae)
    out.append('  |error| vs reference angle: Spearman rho %.2f, p=%.2g' % (rho, p_rho))
    J['spearman'] = {'rho': float(rho), 'p': float(p_rho)}
    sd = err.std(ddof=1)
    J['loa'] = [float(err.mean() - 1.96 * sd), float(err.mean() + 1.96 * sd)]

    # subgroups
    def sub(label, rows):
        g_ok = [r for r in rows if not r['failed']]
        a = np.abs(np.array([r['pred'] - r['ref'] for r in g_ok]))
        e = np.array([r['pred'] - r['ref'] for r in g_ok])
        ci = boot(lambda x: x.mean(), a)
        d = {'n': len(rows), 'failed': len(rows) - len(g_ok), 'mae': float(a.mean()), 'mae_lo': float(ci[0]),
             'mae_hi': float(ci[1]), 'median_ae': float(np.median(a)), 'bias': float(e.mean()),
             'within5': float(100 * (a <= 5).mean())}
        out.append('  %-24s n=%3d fail=%2d  MAE %s  median %.1f  bias %+.1f  within5 %.0f%%'
                   % (label, d['n'], d['failed'], fmt_ci(d['mae'], ci), d['median_ae'], d['bias'], d['within5']))
        return d, a

    out.append('')
    out.append('By severity of the reference (largest) angle:')
    J['severity'] = {}
    sev_ae = []
    for i, name in enumerate(CLASSES):
        rows = [r for r in res if cls(r['ref']) == i]
        J['severity'][name], a = sub(name, rows)
        sev_ae.append(a)
    kw = stats.kruskal(*sev_ae)
    out.append('  Kruskal-Wallis across classes: H=%.1f, p=%.2g' % (kw.statistic, kw.pvalue))
    J['kruskal'] = {'H': float(kw.statistic), 'p': float(kw.pvalue)}

    out.append('')
    out.append('By location of the major curve (apex):')
    J['location'] = {}
    loc_ae = []
    for name in ('thoracic', 'thoracolumbar/lumbar'):
        rows = [r for r in res if r['location'] == name]
        J['location'][name], a = sub(name, rows)
        loc_ae.append(a)
    mw = stats.mannwhitneyu(loc_ae[0], loc_ae[1], alternative='two-sided')
    out.append('  Mann-Whitney: U=%.0f, p=%.2g' % (mw.statistic, mw.pvalue))
    J['mannwhitney'] = {'U': float(mw.statistic), 'p': float(mw.pvalue)}
    # the location effect is in the SIGN of the error (lumbar curves are underestimated)
    signed = [np.array([r['pred'] - r['ref'] for r in res if r['location'] == name and not r['failed']])
              for name in ('thoracic', 'thoracolumbar/lumbar')]
    mws = stats.mannwhitneyu(signed[0], signed[1], alternative='two-sided')
    out.append('  signed error, Mann-Whitney: U=%.0f, p=%.2g' % (mws.statistic, mws.pvalue))
    J['mannwhitney_signed'] = {'U': float(mws.statistic), 'p': float(mws.pvalue)}
    # severity-matched: is the location effect just severity? compare within >=25 deg
    for name in ('thoracic', 'thoracolumbar/lumbar'):
        rows = [r for r in res if r['location'] == name and r['ref'] >= 25]
        J['location'][name + ' (>=25)'], _ = sub(name + ' (≥25°)', rows)

    # thresholds
    out.append('')
    out.append('Clinical thresholds (n=%d measured; worst case counts failures as 0°):' % len(ok))
    J['thresholds'] = {}
    for t in CUTS:
        pos, hat = ref >= t, pred >= t
        tp, tn = int((pos & hat).sum()), int((~pos & ~hat).sum())
        fp, fn = int((~pos & hat).sum()), int((pos & ~hat).sum())
        sens = tp / (tp + fn) if tp + fn else float('nan')
        spec = tn / (tn + fp) if tn + fp else float('nan')
        acc = (tp + tn) / len(ref)
        all_ref = np.array([r['ref'] for r in res])
        all_pred = np.array([0.0 if r['failed'] else r['pred'] for r in res])
        wc_sens = ((all_ref >= t) & (all_pred >= t)).sum() / max((all_ref >= t).sum(), 1)
        out.append('  >=%2d°: pos %3d neg %3d  sens %.1f%%  spec %.1f%%  acc %.1f%%  (worst-case sens %.1f%%)'
                   % (t, tp + fn, tn + fp, 100 * sens, 100 * spec, 100 * acc, 100 * wc_sens))
        J['thresholds'][str(t)] = {'pos': tp + fn, 'neg': tn + fp, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                                   'sens': sens, 'spec': spec, 'acc': acc, 'worst_sens': float(wc_sens)}
    kappa, conf = weighted_kappa([cls(a) for a in ref], [cls(a) for a in pred])
    same = sum(cls(a) == cls(b) for a, b in zip(ref, pred))
    out.append('  4-class agreement: %d/%d (%.1f%%), linearly weighted kappa %.2f' % (same, len(ref), 100 * same / len(ref), kappa))
    J['kappa'] = float(kappa)
    J['class_agreement'] = same / len(ref)
    conf_counts = np.zeros((4, 4), int)
    for a, b in zip(ref, pred):
        conf_counts[cls(a), cls(b)] += 1
    J['confusion'] = conf_counts.tolist()
    out.append('  confusion (rows reference, cols RadTools): ' + str(conf_counts.tolist()))

    # image type: posterior spinal instrumentation and in-brace films (visual screen, image_flags.csv).
    # The method's training data excluded spinal hardware, so instrumented spines are outside its intended use.
    flags = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'image_flags.csv'), encoding='utf-8'))}
    for r in res:
        f = flags[r['name']]
        r['type'] = 'instrumented' if f['instrumented'] == '1' else ('brace' if f['brace'] == '1' else 'plain')
    out.append('')
    out.append('By image type:')
    J['image_type'] = {}
    for name in ('plain', 'brace', 'instrumented'):
        rows = [r for r in res if r['type'] == name]
        g_ok = [r for r in rows if not r['failed']]
        e = np.array([r['pred'] - r['ref'] for r in g_ok])
        d = {'n': len(rows), 'failed': len(rows) - len(g_ok), 'mae': float(np.abs(e).mean()) if len(e) else None,
             'bias': float(e.mean()) if len(e) else None, 'errors': [round(float(x), 1) for x in e] if len(e) < 6 else None}
        J['image_type'][name] = d
        out.append('  %-13s n=%3d fail=%2d  MAE %s  bias %s%s'
                   % (name, d['n'], d['failed'], '%.1f' % d['mae'] if d['mae'] is not None else '-',
                      '%+.1f' % d['bias'] if d['bias'] is not None else '-',
                      ('  errors ' + str(d['errors'])) if d['errors'] else ''))
    fail_types = {}
    for r in res:
        if r['failed']:
            fail_types[r['type']] = fail_types.get(r['type'], 0) + 1
    J['failures_by_type'] = fail_types

    # sensitivity: intended-use population (no spinal instrumentation)
    iu = [r for r in res if r['type'] != 'instrumented']
    iu_ok = [r for r in iu if not r['failed']]
    rf = np.array([r['ref'] for r in iu_ok]); pf = np.array([r['pred'] for r in iu_ok])
    e = pf - rf; a = np.abs(e); sd_ = e.std(ddof=1)
    J['no_instrumentation'] = {
        'n': len(iu), 'failed': len(iu) - len(iu_ok),
        'mae': float(a.mean()), 'mae_ci': [float(x) for x in boot(lambda x: x.mean(), a)],
        'median_ae': float(np.median(a)), 'within5': float(100 * (a <= 5).mean()), 'within10': float(100 * (a <= 10).mean()),
        'bias': float(e.mean()), 'loa': [float(e.mean() - 1.96 * sd_), float(e.mean() + 1.96 * sd_)],
        'icc': float(icc_2_1(rf, pf)), 'icc_ci': [float(x) for x in boot(lambda x, y: icc_2_1(x, y), rf, pf)]}
    s = J['no_instrumentation']
    out.append('  without instrumented spines: n=%d, failed %d (%.1f%%), MAE %s, median %.1f, within5 %.1f%%, '
               'ICC %.3f, bias %+.1f, LoA %.1f to %.1f'
               % (s['n'], s['failed'], 100 * s['failed'] / s['n'], fmt_ci(s['mae'], s['mae_ci']), s['median_ae'],
                  s['within5'], s['icc'], s['bias'], s['loa'][0], s['loa'][1]))

    # images may share a source: file names carry a date and a case number (e.g. "...-21-Nov-2016-140 A AP").
    # Resample those groups instead of single images, which widens the CIs if images are correlated.
    import re
    def group(name):
        m = re.match(r'sunhl-1th-(\d+-\w+-\d+)-(\d+)\s', name)
        return m.group(1) + '/' + m.group(2) if m else name
    keys = sorted({group(r['name']) for r in ok})
    by = {k: [i for i, r in enumerate(ok) if group(r['name']) == k] for k in keys}
    vals_mae, vals_icc = [], []
    for _ in range(2000):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([by[keys[k]] for k in pick])
        vals_mae.append(ae[idx].mean())
        vals_icc.append(icc_2_1(ref[idx], pred[idx]))
    J['cluster'] = {'groups': len(keys), 'mae_ci': [float(x) for x in np.percentile(vals_mae, [2.5, 97.5])],
                    'icc_ci': [float(x) for x in np.percentile(vals_icc, [2.5, 97.5])]}
    out.append('  cluster bootstrap over %d file-name groups: MAE CI %.1f–%.1f, ICC CI %.3f–%.3f'
               % (len(keys), *J['cluster']['mae_ci'], *J['cluster']['icc_ci']))
    J['n_groups_all'] = len({group(r['name']) for r in res})

    txt = '\n'.join(out)
    open(os.path.join(HERE, 'analysis2.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(J, open(os.path.join(HERE, 'analysis2.json'), 'w', encoding='utf-8'), indent=1)
    print(txt)

    # figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8), dpi=300)
    a1.scatter(ref, pred, s=5, alpha=0.55, color='black', linewidths=0)
    lim = [0, max(ref.max(), pred.max()) * 1.05]
    a1.plot(lim, lim, color='grey', lw=0.8)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel('Reference Cobb angle (°)'); a1.set_ylabel('RadTools Cobb angle (°)')
    a1.text(0.03, 0.95, '(a)', transform=a1.transAxes, va='top', fontweight='bold')
    m = (ref + pred) / 2
    a2.scatter(m, err, s=5, alpha=0.55, color='black', linewidths=0)
    for y, ls in ((err.mean(), '-'), (J['loa'][0], '--'), (J['loa'][1], '--')):
        a2.axhline(y, color='grey', lw=0.8, ls=ls)
    a2.set_xlabel('Mean of methods (°)'); a2.set_ylabel('RadTools − reference (°)')
    a2.text(0.03, 0.95, '(b)', transform=a2.transAxes, va='top', fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_agreement.png')); plt.close(fig)

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.6, 2.6), dpi=300, gridspec_kw={'width_ratios': [2, 1.2]})
    b1.boxplot(sev_ae, tick_labels=CLASSES, flierprops={'markersize': 2})
    b1.axhline(5, color='grey', lw=0.8, ls='--')
    b1.set_ylabel('Absolute error (°)'); b1.set_xlabel('Reference severity')
    b1.text(0.02, 0.95, '(a)', transform=b1.transAxes, va='top', fontweight='bold')
    b2.boxplot(loc_ae, tick_labels=['Thoracic', 'TL/lumbar'], flierprops={'markersize': 2})
    b2.axhline(5, color='grey', lw=0.8, ls='--')
    b2.set_xlabel('Major curve apex')
    b2.text(0.04, 0.95, '(b)', transform=b2.transAxes, va='top', fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_subgroups.png')); plt.close(fig)


if __name__ == '__main__':
    main()
