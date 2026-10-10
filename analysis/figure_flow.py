"""Figure 1: study flow (numbers from metrics.json, analysis2.json, run_stats.json)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, 'metrics.json'), encoding='utf-8'))
A = json.load(open(os.path.join(HERE, 'analysis2.json'), encoding='utf-8'))
R = json.load(open(os.path.join(HERE, 'run_stats.json'), encoding='utf-8'))
L = json.load(open(os.path.join(HERE, 'levels_summary.json'), encoding='utf-8'))

N, NF = M['n_images'], M['n_failed']
FT = A['failures_by_type']
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})
fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')


def box(x, y, w, h, text, bold_first=True, fc='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='black', lw=0.7))
    lines = text.split('\n')
    ax.text(x + w / 2, y + h / 2, '\n'.join(lines), ha='center', va='center', linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='black',
                                                                  shrinkA=0, shrinkB=0, mutation_scale=8))


box(14, 50, 44, 8, 'AASCE 2019 training set (SpineWeb Dataset 16)\n%d anteroposterior radiographs with vertebral landmarks' % N)
box(14, 37, 44, 7, 'RadTools scoliosis module (method of Li et al.)\nused unchanged, one run per image')
arrow(36, 49.6, 36, 44.4)
box(2, 21, 30, 9, 'Angle returned\nn = %d (%.1f%%)' % (N - NF, 100 * (N - NF) / N))
box(40, 17, 30, 13, 'No angle returned, n = %d\nposterior instrumentation: %d\nexternal brace: %d\nlow contrast or atypical field\nof view: %d (1 server error)'
    % (NF, FT.get('instrumented', 0), FT.get('brace', 0), FT.get('plain', 0)))
arrow(30, 36.6, 17, 30.4)
arrow(42, 36.6, 55, 30.4)
box(2, 3, 30, 11, 'Agreement analysis: n = %d\nLevel mapping of the tangents: n = %d' % (N - NF, L['n']))
arrow(17, 20.6, 17, 14.4)
box(76, 37, 22, 13, 'Challenge test set\n98 radiographs,\nno landmarks released\n\n%d of %d processed;\nnot analysed' % (R['test_ok'], R['test_n']), fc='#f2f2f2')
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(HERE, 'fig_flow.png'))
print('wrote fig_flow.png')
