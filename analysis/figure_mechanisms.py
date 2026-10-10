"""Figure 5: one radiograph per outcome — agreement, and each error mechanism found by
levels_analysis.py — with the RadTools overlay (white, as returned by the service) and
the tilts of the reference end vertebrae (dashed yellow, labelled).

Picks examples from levels_per_image.csv; writes fig_mechanisms.png and
fig_mechanisms.txt (the panel part of the legend).
"""
import base64
import csv
import io
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import levels_analysis as LA


def _font(names, size):
    """First available of the given TrueType files (Windows or Linux)."""
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    raise OSError('no font among %s' % names)


SANS_BOLD = ['C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf']
SERIF_BOLD = ['C:/Windows/Fonts/timesbd.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf']
# Radiology: AI allows at most four panels per figure; the instrumented example is described in the text
MAX_PANELS = int(os.environ.get('MAX_PANELS', '4'))

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = {r['name']: r for r in csv.DictReader(open(os.path.join(HERE, 'image_flags.csv'), encoding='utf-8'))}


def pick():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'levels_per_image.csv'), encoding='utf-8')))
    for r in rows:
        for k in ('ref', 'radtools', 'err', 'd_upper', 'd_lower', 'rt_upper_level', 'rt_lower_level'):
            r[k] = float(r[k])
    plain = [r for r in rows if FLAGS[r['name']]['instrumented'] == '0' and FLAGS[r['name']]['brace'] == '0']
    out = []
    good = [r for r in plain if r['d_upper'] == 0 and r['d_lower'] == 0 and abs(r['err']) < 1.5 and r['ref'] >= 35]
    good.sort(key=lambda r: abs(r['err']))
    if good:
        out.append(('agreement', good[0]))
    for mech in ('other curve', 'end shifted', 'tilt only', 'outside T1–L5'):
        g = [r for r in plain if r['mechanism'] == mech]
        if mech == 'end shifted':      # show the typical case: the lower end cut short
            g = [r for r in g if r['d_lower'] <= -2] or g
        if mech == 'other curve':      # the clear case: the whole measured curve lies above the reference one
            clean = [r for r in g if LA.NAMES.index(r['rt_lower']) <= LA.NAMES.index(r['ref_upper'])
                     if r['rt_lower'] in LA.NAMES]
            if clean:
                out.append((mech, max(clean, key=lambda r: r['ref'])))
                continue
        g.sort(key=lambda r: -abs(r['err']))
        if g:
            out.append((mech, g[len(g) // 3] if len(g) > 2 else g[0]))   # large but not the most extreme
    instr = [r for r in rows if FLAGS[r['name']]['instrumented'] == '1']
    instr.sort(key=lambda r: -abs(r['err']))
    if instr:
        out.append(('instrumented', instr[0]))
    return out


def dashed(d, p0, p1, fill, width, segs=16):
    for s in range(0, segs, 2):
        t0, t1 = s / segs, (s + 1) / segs
        d.line([tuple(p0 + (p1 - p0) * t0), tuple(p0 + (p1 - p0) * t1)], fill=fill, width=width)


def panel(rec, pts, row, levels):
    img = Image.open(io.BytesIO(base64.b64decode(rec['overlay']))).convert('RGB')
    ang = LA.tilt_angles(pts)
    up, lo = levels['upper'], levels['lower']
    d = ImageDraw.Draw(img)
    w = max(3, img.width // 170)
    font = _font(SANS_BOLD, max(22, img.width // 18))
    for k in (up, lo):
        i = 4 * k
        left = (pts[i] + pts[i + 2]) / 2
        right = (pts[i + 1] + pts[i + 3]) / 2
        c = (left + right) / 2
        v = (right - left) / np.linalg.norm(right - left)
        half = img.width * 0.30
        dashed(d, c - v * half, c + v * half, (255, 210, 0), w)
        tw = d.textlength(LA.NAMES[k], font=font)   # keep the whole label inside the image
        d.text((min(img.width - tw - 8, c[0] + half + 6), c[1] - 20), LA.NAMES[k], fill=(255, 210, 0), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    # crop to the spine between T1 and L5 (and the RadTools tangents), with a margin
    ys = [c[1] for c in levels['centres']]
    ys += [(l['p1'][1] + l['p2'][1]) / 2 for l in rec['geometry']['lines']]
    top = max(0, int(min(ys) - 0.06 * img.height))
    bottom = min(img.height, int(max(ys) + 0.06 * img.height))
    return img.crop((0, top, img.width, bottom)), float(ang[up, lo])


def main():
    runs = {}
    for line in open(LA.RUN, encoding='utf-8'):
        rec = json.loads(line)
        runs[os.path.basename(rec['name'])] = rec
    levels = json.load(open(os.path.join(HERE, 'levels.json'), encoding='utf-8'))
    pts_of = LA.reference_points()
    chosen = pick()[:MAX_PANELS]
    panels, legend = [], []
    letters = 'abcdefgh'
    titles = {'agreement': 'agreement', 'other curve': 'another curve measured',
              'end shifted': 'lower end vertebra cut short', 'tilt only': 'matching end vertebrae, tilt differs',
              'outside T1–L5': 'tangent outside T1–L5', 'instrumented': 'instrumented spine'}
    for (kind, row), letter in zip(chosen, letters):
        im, ref = panel(runs[row['name']], pts_of[row['name']], row, levels[row['name']])
        h = 1500
        im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
        font = _font(SERIF_BOLD, 72)
        ImageDraw.Draw(im).text((14, 10), '(%s)' % letter, fill=(255, 255, 255), font=font, stroke_width=3,
                                stroke_fill=(0, 0, 0))
        panels.append(im)
        legend.append('(%s) %s: reference %s° (%s–%s), RadTools %s° (%s–%s).'
                      % (letter, titles[kind].capitalize(), ('%.1f' % row['ref']),
                         row['ref_upper'], row['ref_lower'], ('%.1f' % row['radtools']),
                         row['rt_upper'], row['rt_lower']))
        print(letter, kind, row['name'], row['ref'], row['radtools'], row['err'])
    gap = 24
    sheet = Image.new('RGB', (sum(p.width for p in panels) + gap * (len(panels) - 1), 1500), 'white')
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + gap
    sheet.save(os.path.join(HERE, 'fig_mechanisms.png'), dpi=(300, 300))
    txt = ' '.join(legend)
    open(os.path.join(HERE, 'fig_mechanisms.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
