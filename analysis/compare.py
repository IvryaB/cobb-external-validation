"""RadTools (segmentation-based Cobb angle) vs expert-landmark reference on AASCE.

Reference = the largest Cobb angle from the curated Zenodo landmarks (falls back to
the original angles.csv for the 3 images the curation does not cover). The original
reference is reported as a sensitivity analysis: how much of the "error" is label noise.

Outputs (analysis/):
  results.csv            one row per image
  metrics.txt            the numbers for the paper
  fig_scatter.png        RadTools vs reference, identity line
  fig_bland_altman.png   difference vs mean, bias and 95% limits of agreement
  fig_severity.png       absolute error by severity group
"""
import csv
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, 'radtools_run', 'aasce_train481_radtools.jsonl')
GROUPS = [(0, 10, '<10° (no scoliosis)'), (10, 25, '10–25° (mild)'),
          (25, 40, '25–40° (moderate)'), (40, 200, '>40° (severe)')]
# Five samples a Kaggle user reported as mis-annotated (discussion on the mirror)
FLAGGED = {'sunhl-1th-06-Jan-2017-182 A AP', 'sunhl-1th-25-Jul-2016-49 A AP',
           'sunhl-1th-25-Jul-2016-51 A AP', 'sunhl-1th-25-Jul-2016-61 A AP',
           'sunhl-1th-26-Jul-2016-76 C AP'}


def icc_2_1(a, b):
    """Two-way random effects, absolute agreement, single measurement (Shrout & Fleiss)."""
    x = np.stack([a, b], axis=1)
    n, k = x.shape
    grand = x.mean()
    msr = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def numbers(ref, pred):
    d = pred - ref
    ae = np.abs(d)
    sd = d.std(ddof=1)
    return {'n': int(len(ref)), 'mae': float(ae.mean()), 'sd_ae': float(ae.std(ddof=1)),
            'median_ae': float(np.median(ae)), 'p95_ae': float(np.percentile(ae, 95)),
            'within5': float(100 * (ae <= 5).mean()), 'within10': float(100 * (ae <= 10).mean()),
            'bias': float(d.mean()), 'loa_lo': float(d.mean() - 1.96 * sd), 'loa_hi': float(d.mean() + 1.96 * sd),
            'r': float(stats.pearsonr(ref, pred)[0]), 'icc': float(icc_2_1(ref, pred))}


def summarise(ref, pred, label):
    d = pred - ref
    ae = np.abs(d)
    bias, sd = d.mean(), d.std(ddof=1)
    r, _ = stats.pearsonr(ref, pred)
    smape = 100 * np.mean(np.abs(d) / ((np.abs(ref) + np.abs(pred)) + 1e-8))
    lines = [
        '%s (n=%d)' % (label, len(ref)),
        '  MAE %.2f° (SD %.2f), median AE %.2f°, 95th pct %.2f°' % (ae.mean(), ae.std(ddof=1), np.median(ae), np.percentile(ae, 95)),
        '  within 5°: %d (%.1f%%), within 10°: %d (%.1f%%)' % ((ae <= 5).sum(), 100 * (ae <= 5).mean(), (ae <= 10).sum(), 100 * (ae <= 10).mean()),
        '  bias %.2f°, 95%% LoA %.2f° to %.2f°' % (bias, bias - 1.96 * sd, bias + 1.96 * sd),
        '  Pearson r %.3f, ICC(2,1) %.3f, SMAPE(max angle) %.2f%%' % (r, icc_2_1(ref, pred), smape),
    ]
    return lines, bias, sd


def main():
    gt = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'ground_truth.csv'), encoding='utf-8'))}
    runs = {}
    for line in open(RUN, encoding='utf-8'):
        rec = json.loads(line)
        runs[os.path.basename(rec['name'])] = rec

    rows = []
    for name, g in gt.items():
        rec = runs.get(name)
        if rec is None:
            continue
        ref_cur = float(g['cur_max']) if g['cur_max'] not in ('', 'None', None) else None
        ref_orig = float(g['orig_max'])
        pred = rec.get('max_angle')
        rows.append({
            'name': name,
            'ref': ref_cur if ref_cur is not None else ref_orig,
            'ref_source': 'curated' if ref_cur is not None else 'original',
            'ref_orig': ref_orig,
            'radtools': pred,
            'radtools_all': ';'.join(str(a) for a in (rec.get('angles') or [])),
            'n_curves': len(rec.get('angles') or []),
            'failed': pred is None,
            'flagged_by_kaggle_user': os.path.splitext(name)[0] in FLAGGED,
        })
    for r in rows:
        r['abs_err'] = None if r['failed'] else round(abs(r['radtools'] - r['ref']), 3)
    with open(os.path.join(HERE, 'results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r['failed']]
    ref = np.array([r['ref'] for r in ok])
    pred = np.array([r['radtools'] for r in ok])
    ref_o = np.array([r['ref_orig'] for r in ok])
    out = ['Images run: %d of %d; no angle returned: %d (%.1f%%)'
           % (len(rows), len(gt), len(rows) - len(ok), 100 * (len(rows) - len(ok)) / max(len(rows), 1)),
           'Reference max angle: median %.1f°, IQR %.1f–%.1f°, range %.1f–%.1f°'
           % (np.median(ref), *np.percentile(ref, [25, 75]), ref.min(), ref.max()), '']
    main_lines, bias, sd = summarise(ref, pred, 'vs curated reference (primary)')
    out += main_lines + ['']
    out += summarise(ref_o, pred, 'vs original angles.csv (sensitivity)')[0] + ['']
    out.append('By severity (curated reference):')
    for lo, hi, label in GROUPS:
        grp = [r for r in rows if lo <= r['ref'] < hi]
        g_ok = [r for r in grp if not r['failed']]
        if not grp:
            continue
        ae = np.array([r['abs_err'] for r in g_ok]) if g_ok else np.array([np.nan])
        bias_g = np.mean([r['radtools'] - r['ref'] for r in g_ok]) if g_ok else np.nan
        out.append('  %-22s n=%3d  failed=%2d  MAE %.2f°  median AE %.2f°  bias %+.2f°  within 5° %.0f%%'
                   % (label, len(grp), len(grp) - len(g_ok), np.nanmean(ae), np.nanmedian(ae), bias_g,
                      100 * np.mean(ae <= 5) if g_ok else float('nan')))
    worst = sorted(ok, key=lambda r: -r['abs_err'])[:10]
    out += ['', 'Ten largest errors:'] + ['  %-40s ref %5.1f  radtools %5.1f  |err| %5.1f  curves %d'
                                          % (r['name'], r['ref'], r['radtools'], r['abs_err'], r['n_curves']) for r in worst]
    failed = [r for r in rows if r['failed']]
    out += ['', 'No angle returned:'] + ['  %-40s ref %5.1f' % (r['name'], r['ref']) for r in failed]
    flagged = [r for r in rows if r['flagged_by_kaggle_user']]
    out += ['', 'Samples reported as mis-annotated on Kaggle:'] + [
        '  %-40s curated %5.1f  original %5.1f  radtools %s' % (r['name'], r['ref'], r['ref_orig'], r['radtools'])
        for r in flagged]
    groups = {}
    for lo, hi, label in GROUPS:
        grp = [r for r in rows if lo <= r['ref'] < hi]
        g_ok = [r for r in grp if not r['failed']]
        if grp:
            groups[label] = {'n': len(grp), 'failed': len(grp) - len(g_ok),
                             'mae': float(np.mean([r['abs_err'] for r in g_ok])) if g_ok else None,
                             'bias': float(np.mean([r['radtools'] - r['ref'] for r in g_ok])) if g_ok else None,
                             'within5': float(100 * np.mean([r['abs_err'] <= 5 for r in g_ok])) if g_ok else None}
    summary = {'n_images': len(rows), 'n_failed': len(rows) - len(ok),
               'failed_refs': [r['ref'] for r in rows if r['failed']],
               'ref_median': float(np.median(ref)), 'ref_q1': float(np.percentile(ref, 25)),
               'ref_q3': float(np.percentile(ref, 75)), 'ref_min': float(ref.min()), 'ref_max': float(ref.max()),
               'curated': numbers(ref, pred), 'original': numbers(ref_o, pred), 'groups': groups}
    json.dump(summary, open(os.path.join(HERE, 'metrics.json'), 'w', encoding='utf-8'), indent=1)
    text = '\n'.join(out)
    open(os.path.join(HERE, 'metrics.txt'), 'w', encoding='utf-8').write(text + '\n')
    print(text)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})

    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=300)
    ax.scatter(ref, pred, s=6, alpha=0.6, color='black', linewidths=0)
    lim = [0, max(ref.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, color='grey', lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Reference Cobb angle (°)'); ax.set_ylabel('RadTools Cobb angle (°)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_scatter.png')); plt.close(fig)

    mean = (ref + pred) / 2
    diff = pred - ref
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=300)
    ax.scatter(mean, diff, s=6, alpha=0.6, color='black', linewidths=0)
    for y, ls in ((bias, '-'), (bias - 1.96 * sd, '--'), (bias + 1.96 * sd, '--')):
        ax.axhline(y, color='grey', lw=0.8, ls=ls)
    ax.text(mean.max(), bias + 1.96 * sd, ' +1.96 SD', va='bottom', ha='right', fontsize=7)
    ax.text(mean.max(), bias - 1.96 * sd, ' −1.96 SD', va='top', ha='right', fontsize=7)
    ax.set_xlabel('Mean of methods (°)'); ax.set_ylabel('RadTools − reference (°)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_bland_altman.png')); plt.close(fig)

    data, labels = [], []
    for lo, hi, label in GROUPS:
        v = [r['abs_err'] for r in ok if lo <= r['ref'] < hi]
        if v:
            data.append(v); labels.append(label.split(' ')[0])
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=300)
    ax.boxplot(data, tick_labels=labels, showfliers=True, flierprops={'markersize': 2})
    ax.axhline(5, color='grey', lw=0.8, ls='--')
    ax.set_ylabel('Absolute error (°)'); ax.set_xlabel('Reference severity')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_severity.png')); plt.close(fig)


if __name__ == '__main__':
    main()
