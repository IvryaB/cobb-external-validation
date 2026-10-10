"""
Scoliosis Cobb Angle Measurement module.

Uses MaskRCNN (ResNet50 backbone) to segment the spine from X-ray images
and calculates Cobb angles using the steepest point method.

Based on scoliosis_minimal/scoliosis_inference.py.
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
logging.getLogger('mmengine').setLevel(logging.ERROR)
logging.getLogger('mmdet').setLevel(logging.ERROR)

import os
import os.path as osp
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks

_default_weights = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
                            'neural_networks_weights', 'scoliosis_weights.pth')
if not osp.isfile(_default_weights) and osp.isfile('/models/scoliosis_weights.pth'):
    _default_weights = '/models/scoliosis_weights.pth'
WEIGHTS_PATH = os.environ.get('SCOLIOSIS_MODEL_PATH', _default_weights)

_MODEL = None


# ── Overlay style ────────────────────────────────────────────────────────────
# Endplate tangents only, white, thick. The hairline the overlay used to draw was
# invisible; the centre line running ALONG the bright vertebral bodies was the
# part that could not be made to read at any weight, so it is gone rather than
# fought with. What is left crosses the spine and continues onto darker tissue,
# where plain white is legible without a dark halo behind it. BGR, for OpenCV.
OVERLAY_COLOR = (255, 255, 255)
OVERLAY_OUTLINE = None             # set to (0, 0, 0) to put a dark halo back


def overlay_thickness(image):
    """Stroke width for the overlay, scaled to the film (3 px at 600 px)."""
    return max(3, int(round(3.2 * min(image.shape[:2]) / 600.0)))


def draw_overlay(image, centre_pts, tangents):
    """Paint the overlay: every outline first (if any), then every stroke.

    Two passes rather than outline-then-stroke per line — with a halo enabled,
    drawing it per line lays one line's dark edge over the next line's fill, a
    seam through every crossing.

    No dot at the tangent centres: at this weight it sat right on the endplate
    the angle is measured from.
    """
    thickness = overlay_thickness(image)
    passes = [(OVERLAY_COLOR, thickness)]
    if OVERLAY_OUTLINE is not None:
        passes.insert(0, (OVERLAY_OUTLINE, thickness + 1))
    for colour, width in passes:
        if centre_pts:
            pts = np.array(centre_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], False, colour, width, cv2.LINE_AA)
        for p1, p2 in tangents:
            cv2.line(image, p1, p2, colour, width, cv2.LINE_AA)
    return image


def _build_cfg():
    from mmengine.config import ConfigDict
    cfg = ConfigDict()
    cfg.model = ConfigDict(
        type='MaskRCNN',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=True,
            pad_size_divisor=32),
        backbone=dict(
            type='ResNet', depth=50, num_stages=4,
            out_indices=(0, 1, 2, 3), frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True, style='pytorch'),
        neck=dict(
            type='FPN', in_channels=[256, 512, 1024, 2048],
            out_channels=256, num_outs=5),
        rpn_head=dict(
            type='RPNHead', in_channels=256, feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator', scales=[8],
                ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[1., 1., 1., 1.]),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256, featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead', in_channels=256,
                fc_out_channels=1024, roi_feat_size=7, num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256, featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead', num_convs=4, in_channels=256,
                conv_out_channels=256, num_classes=1,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        train_cfg=None,
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000, max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100, mask_thr_binary=0.5)))
    cfg.classes = ('Spine',)
    cfg.test_dataloader = ConfigDict(
        batch_size=1, num_workers=0, persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=ConfigDict(
            type='CocoDataset', data_root='', ann_file='',
            data_prefix=dict(img=''), test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                dict(type='PackDetInputs')
            ],
            metainfo=dict(classes=('Spine',))))
    cfg.visualizer = ConfigDict(
        type='DetLocalVisualizer', name='visualizer',
        vis_backends=[dict(type='LocalVisBackend')])
    return cfg


def _load_model():
    global _MODEL
    if _MODEL is None:
        from mmdet.utils import register_all_modules
        from mmdet.apis import init_detector
        from mmengine.config import Config
        register_all_modules()
        cfg = Config(_build_cfg().to_dict())
        import torch as _torch  # local import to keep top of file clean
        _device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        # mmcv's compiled ops (e.g. nms) must also be CUDA-built. With a CPU-only
        # mmcv (vendored build without CUDA ops), running on GPU raises
        # "nms_impl: implementation for device cuda not found" — so fall back to
        # CPU when mmcv has no CUDA ops. Override with SCOLIOSIS_DEVICE.
        _env_dev = os.getenv('SCOLIOSIS_DEVICE')
        if _env_dev:
            _device = _env_dev
        elif _device == 'cuda':
            try:
                from mmcv.ops import get_compiling_cuda_version
                if str(get_compiling_cuda_version()).lower() in ('none', 'not available'):
                    _device = 'cpu'
            except Exception:
                _device = 'cpu'
        _MODEL = init_detector(cfg, WEIGHTS_PATH, device=_device)
    return _MODEL


def _get_skeleton(image, trunk=0.02):
    w, h = image.shape
    row_sum = image.sum(axis=1)
    skeleton = np.zeros_like(image)
    points_x, points_y = [], []
    for row in range(w):
        if row_sum[row] > 0:
            row_now = image[row]
            first = row_now.nonzero()[0].min()
            last = row_now.nonzero()[0].max()
            y = int((first + last) / 2)
            points_x.append(row)
            points_y.append(y)
            skeleton[row, y] = 1
    start = int(len(points_x) * trunk)
    end = int(len(points_x) * (1 - trunk))
    points_x = points_x[start:end]
    points_y = points_y[start:end]
    skeleton[:points_x[0], :] = 0
    skeleton[points_x[-1]:, :] = 0
    return skeleton, points_x, points_y


def _fit_curve(points_x, points_y, img_ratio=1, box=None, deg=10):
    if box is None:
        x_offset = np.min(points_x)
        y_offset = np.min(points_y)
    else:
        x_offset = np.array(box[1], dtype=int)
        y_offset = np.array(box[0], dtype=int)
    x_fit = (points_x - x_offset) * img_ratio
    y_fit = (points_y - y_offset) * img_ratio
    fit_para = np.polyfit(x_fit, y_fit, deg)
    f_x = np.poly1d(fit_para)
    # sorted(), not list(set()): a set of ints iterates in ascending order only while
    # every value is below its hash-table size. A normal film fills 0..~572 densely and
    # happened to come out sorted; a faint one with a short segmented spine rescales
    # by a ratio > 1, the values step by several pixels, and the order came out
    # interleaved — [0, 515, 5, 520, 10, ...]. Everything downstream (slopes, peaks,
    # break points) then ran on scrambled x, and AASCE sunhl-1th-10-Jan-2017-235 D
    # died with "cannot convert float NaN to integer" (HTTP 500, 2026-09-21).
    x_fit = sorted(set(int(i) for i in x_fit))
    y_hat = f_x(x_fit)
    return f_x, x_fit, y_hat, x_offset, y_offset


def _find_break_point(f_x, x_fit, tol=50, img_ratio=1):
    f_x_1d = np.polyder(f_x, 1)
    f_x_2d = np.polyder(f_x, 2)
    fx_dx = f_x_1d(x_fit)
    peaks, _ = find_peaks(abs(fx_dx))
    ffx_dx = f_x_2d(x_fit)
    break_i = [0]
    for i in range(len(fx_dx) - 1):
        if (fx_dx[i] * fx_dx[i + 1] <= 0) and (i - break_i[-1] > tol) and (x_fit[-1] - x_fit[i] > tol):
            break_i.append(i)
        elif i - break_i[-1] > 200:
            peak_2d = break_i[-1] + 1 + abs(ffx_dx[break_i[-1] + 1:i]).argmax()
            fx_peak = fx_dx[peak_2d]
            if peak_2d - break_i[-1] > tol and peak_2d < i - tol and abs(fx_peak) < 0.3:
                break_i.append(peak_2d)
    break_i.append(len(fx_dx) - 1)
    break_point = [x_fit[i] for i in break_i]
    return break_point, f_x_1d, break_i


def turning_points(vxs, vys):
    """Indices of the tangents to keep: both ends, and every tangent where the tilt
    turns back. A run of tangents tilting further the same way is one curve; its
    inner tangents only split it."""
    th = [math.atan2(float(np.ravel(vy)[0]), float(np.ravel(vx)[0])) for vx, vy in zip(vxs, vys)]
    if len(th) <= 2:
        return list(range(len(th)))
    return [0] + [k for k in range(1, len(th) - 1)
                  if (th[k] - th[k - 1]) * (th[k + 1] - th[k]) < 0] + [len(th) - 1]


class _MaskAngle:
    def __init__(self, image, mask, box=None):
        self.mask = mask
        self.image = image
        self.ratio = 572 / (box[3] - box[1]) if box is not None else 1
        self.box = box

    def mask2skeleton(self):
        return _get_skeleton(self.mask)

    def _fitskeleton(self, x_fit, y_hat, x_offset, y_offset):
        new_skeleton = np.zeros_like(self.mask)
        for i in range(len(x_fit)):
            new_skeleton[int(x_fit[i] / self.ratio) + x_offset, int(y_hat[i] / self.ratio) + y_offset] = 1
        return new_skeleton

    def find_break_point(self, points_x, points_y, tol=35):
        f_x, x_fit, y_hat, x_offset, y_offset = _fit_curve(points_x, points_y, self.ratio, self.box)
        break_point_x, f_x_1d, break_i = _find_break_point(f_x, x_fit, tol, self.ratio)
        break_point_x = [int(i / self.ratio + x_offset) for i in break_point_x]
        new_skeleton = self._fitskeleton(x_fit, y_hat, x_offset, y_offset)
        return break_point_x, new_skeleton, x_fit, f_x, f_x_1d, x_offset, y_offset, break_i

    def sp_skeleton(self, skeleton, break_point_x, tol_ratio=0.25):
        skes, start_x, end_x = [], [], []
        x_points = list(break_point_x)
        for i in range(len(x_points) - 1):
            ske_new = np.zeros_like(skeleton)
            tol_l = int((x_points[i + 1] - x_points[i]) * tol_ratio)
            if x_points[i + 1] - x_points[i] - 2 * tol_l < (50 / self.ratio):
                tol_l = int((x_points[i + 1] - x_points[i]) * (tol_ratio - 0.05))
            if np.sum(skeleton[x_points[i] + tol_l:x_points[i + 1] - tol_l, :]) > 20:
                ske_new[x_points[i] + tol_l:x_points[i + 1] - tol_l, :] = skeleton[x_points[i] + tol_l:x_points[i + 1] - tol_l, :]
                skes.append(ske_new)
                start_x.append(x_points[i])
                end_x.append(x_points[i + 1])
        return skes, start_x, end_x

    @staticmethod
    def _get_font(font_size):
        """Find a usable TrueType font."""
        for name in ["arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf",
                      "DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "C:/Windows/Fonts/arial.ttf"]:
            try:
                return ImageFont.truetype(name, font_size)
            except (OSError, IOError):
                continue
        try:
            return ImageFont.load_default(size=font_size)
        except TypeError:
            return ImageFont.load_default()

    def _mask_width_at_row(self, row):
        """Get the width of the spine mask at a given row (searches ±10 rows)."""
        for offset in range(11):
            for r in ([row + offset] if offset == 0 else [row + offset, row - offset]):
                if 0 <= r < self.mask.shape[0]:
                    cols = np.where(self.mask[r] > 0)[0]
                    if len(cols) > 0:
                        return int(cols.max() - cols.min())
        return 0

    def steepest_point_method(self, image, new_skeleton, break_points, x_fit, f_x,
                              f_x_1d, x_offset, y_offset, break_i, tol_ratio=0.1, off_l=10):
        vxs, vys, xcs, ycs = [], [], [], []
        # Endpoints of every endplate tangent, in standard image pixel space
        # (px=column, py=row) — the same coords the DICOM frame uses, so the
        # interactive editor can drag them over the cornerstone viewport.
        lines_px = []
        fx_dx = f_x_1d(x_fit)
        peaks, _ = find_peaks(abs(fx_dx))
        break_points = (break_points - x_offset) * self.ratio
        # Collected, not drawn: the whole overlay is painted below in one pass
        # per colour, so crossings stay solid.
        tangents = []

        for i in range(len(break_points) - 1):
            tol_l = (break_points[i + 1] - break_points[i]) * tol_ratio
            peak_now = peaks[(np.array(x_fit)[peaks] > break_points[i]) & (np.array(x_fit)[peaks] < break_points[i + 1])]
            if len(peak_now) == 0:
                peak_now = [int((break_i[i] + break_i[i + 1]) / 2)]
            if len(peak_now) > 0:
                peak_now = peak_now[abs(fx_dx)[peak_now].argmax()]
                mid = (break_points[i] + break_points[i + 1]) / 2
                off_x = min(int(off_l / np.sqrt(1 + fx_dx[peak_now] ** 2)), int(abs(x_fit[peak_now] - mid)))
                if x_fit[peak_now] > mid and i < (len(break_points) - 2):
                    peak_new = peak_now + off_x
                else:
                    peak_new = max(peak_now - off_x, int(tol_l))
                mean_fx_1d = np.mean(fx_dx[max(peak_new - int(tol_l), 0):peak_new + int(tol_l) + 1])
                if not np.isfinite(mean_fx_1d):
                    # A tangent that cannot be computed is left out, not turned into a
                    # 500 for the whole film.
                    continue
                vx = 1 / np.sqrt(1 + mean_fx_1d ** 2)
                vy = mean_fx_1d / np.sqrt(1 + mean_fx_1d ** 2)
                vxs.append(vx)
                vys.append(vy)
                x_c = int(x_fit[peak_now] / self.ratio) + x_offset
                y_c = int(f_x(x_fit[peak_now]) / self.ratio) + y_offset
                xcs.append(x_c)
                ycs.append(y_c)
                # Tangent line: ~1.3x vertebra width (half-length each side)
                mask_w = self._mask_width_at_row(x_c)
                line_len = max(15, int(mask_w * 0.65)) if mask_w > 0 else max(15, int(image.shape[1] / 25))
                x2, y2 = x_c + line_len * vy, y_c + line_len * (-vx)
                x1, y1 = x_c - line_len * vy, y_c - line_len * (-vx)
                tangents.append(((int(y1), int(x1)), (int(y2), int(x2))))
                # cv2 draws in (col, row) order — so (y*, x*) already IS (px, py).
                lines_px.append({
                    'id': len(lines_px),
                    'p1': [int(round(y1)), int(round(x1))],
                    'p2': [int(round(y2)), int(round(x2))],
                    'center': [int(round(y_c)), int(round(x_c))],
                })

        # One curve, one angle. The break-point rules can cut a long curve into several
        # intervals, each with its own steepest point; the tilt then keeps turning the
        # same way across three or more tangents, and one curve came out as two smaller
        # consecutive angles (AASCE sunhl-1th-10-Jan-2017-239 C: 37.0 deg, reference
        # 66.9; with this 69.1). Keep
        # only the tangents where the tilt turns back — the end vertebrae in Cobb's
        # sense. On the 481 AASCE films this changed 42 and lowered the MAE against
        # the landmark reference from 5.14 to 4.71 deg (paired 95% CI -0.69 to -0.19);
        # errors above 10 deg went from 56 to 42.
        keep = turning_points(vxs, vys)
        vxs, vys = [vxs[k] for k in keep], [vys[k] for k in keep]
        xcs, ycs = [xcs[k] for k in keep], [ycs[k] for k in keep]
        tangents = [tangents[k] for k in keep]
        lines_px = [dict(lines_px[k], id=i) for i, k in enumerate(keep)]

        image = draw_overlay(image, [], tangents)

        angles = []
        for i in range(len(vxs) - 1):
            v1, v2 = [vxs[i], vys[i]], [vxs[i + 1], vys[i + 1]]
            if hasattr(vxs[i], 'shape') and len(vxs[i].shape) > 0:
                v1 = [vxs[i][0], vys[i][0]]
                v2 = [vxs[i + 1][0], vys[i + 1][0]]
            angles.append(np.arccos(np.dot(v1, v2)) / np.pi * 180)
        return image, angles, xcs, ycs, lines_px

    def draw_angle_label(self, image, x_c1, y_c1, x_c2, y_c2, angle, is_max):
        img_scale = min(image.shape[:2]) / 600.0
        font_size = max(18, int(30 * img_scale))
        padding = max(2, int(3 * img_scale))
        radius = max(2, int(3 * img_scale))

        h, w = image.shape[:2]
        text = '{:.1f}\u00b0'.format(angle)
        # Position: well to the right of midpoint, away from tangent lines
        mid_y = int((y_c1 + y_c2) / 2)
        mid_x = int((x_c1 + x_c2) / 2)
        # Offset further: use mask width or fallback
        offset_px = max(int(40 * img_scale), int(w * 0.08))
        text_x = mid_y + offset_px
        text_y = mid_x - int(font_size / 2)

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = self._get_font(font_size)

        bbox = draw.textbbox((text_x, text_y), text, font=font)
        rect = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        # Clamp label to stay within image bounds
        if rect[2] > w:
            shift = rect[2] - w + 4
            text_x -= shift
            rect[0] -= shift
            rect[2] -= shift
        if rect[3] > h:
            shift = rect[3] - h + 4
            text_y -= shift
            rect[1] -= shift
            rect[3] -= shift
        if rect[0] < 0:
            shift = -rect[0] + 4
            text_x += shift
            rect[0] += shift
            rect[2] += shift
        if rect[1] < 0:
            shift = -rect[1] + 4
            text_y += shift
            rect[1] += shift
            rect[3] += shift

        # Semi-transparent dark background with white text (clean medical style)
        overlay = img_pil.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(rect, radius=radius, fill=(0, 0, 0), outline=(255, 255, 255), width=1)
        img_pil = Image.blend(img_pil, overlay, 0.7)
        draw = ImageDraw.Draw(img_pil)

        # White text, yellow for max angle
        color = (255, 220, 50) if is_max else (255, 255, 255)
        draw.text((text_x, text_y), text, font=font, fill=color)

        # Arrow-like connecting line from label to midpoint
        line_w = max(1, int(img_scale))
        draw.line([(mid_y, mid_x), (text_x - padding, text_y + font_size // 2)],
                  fill=(200, 200, 200), width=line_w)
        # Small arrowhead dot at midpoint
        arrow_r = max(2, int(2 * img_scale))
        draw.ellipse([mid_y - arrow_r, mid_x - arrow_r, mid_y + arrow_r, mid_x + arrow_r],
                     fill=(200, 200, 200))

        np.copyto(image, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
        return image

    @staticmethod
    def draw_summary_bar(image, angles):
        """Add a summary bar at the bottom with all angle values."""
        if not angles:
            return image
        h, w = image.shape[:2]
        img_scale = min(h, w) / 600.0
        bar_font_size = max(36, int(52 * img_scale))

        img_pil_dummy = Image.new('RGB', (10, 10))
        draw_dummy = ImageDraw.Draw(img_pil_dummy)
        font = _MaskAngle._get_font(bar_font_size)

        max_idx = angles.index(max(angles))
        parts = ['{:.1f}\u00b0'.format(a) for a in angles]
        summary = 'Cobb angles: ' + ' | '.join(parts)
        max_text = '  (max: {:.1f}\u00b0)'.format(angles[max_idx])

        bbox_main = draw_dummy.textbbox((0, 0), summary, font=font)
        bbox_max = draw_dummy.textbbox((0, 0), max_text, font=font)
        total_w = (bbox_main[2] - bbox_main[0]) + (bbox_max[2] - bbox_max[0])
        
        # Scale down if text is wider than image
        if total_w > w * 0.95:
            scale = (w * 0.95) / max(1.0, float(total_w))
            bar_font_size = max(12, int(bar_font_size * scale))
            font = _MaskAngle._get_font(bar_font_size)
            bbox_main = draw_dummy.textbbox((0, 0), summary, font=font)
            bbox_max = draw_dummy.textbbox((0, 0), max_text, font=font)
            total_w = (bbox_main[2] - bbox_main[0]) + (bbox_max[2] - bbox_max[0])

        bar_h = bar_font_size + max(30, int(36 * img_scale))

        # Extend image with dark bar at bottom
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        extended = np.vstack([image, bar])

        img_pil = Image.fromarray(cv2.cvtColor(extended, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        text_x = max(8, int((w - total_w) / 2))
        # Center vertically in the bar
        text_y = h + int((bar_h - (bbox_main[3] - bbox_main[1])) / 2) - bbox_main[1]

        draw.text((text_x, text_y), summary, font=font, fill=(255, 255, 255))
        sx = text_x + (bbox_main[2] - bbox_main[0])
        draw.text((sx, text_y), max_text, font=font, fill=(255, 220, 50))

        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return result


# ---------------------------------------------------------------------------
# Geometry helpers — shared by the raster path, the geometry the editor drags,
# and the PDF re-render on Approve. There is exactly ONE angle formula so the
# number the doctor sees while dragging equals the number printed in the report.
# ---------------------------------------------------------------------------

def _unit_canonical(p1, p2):
    """Unit direction of segment p1->p2 (px, py), oriented so px-component >= 0.

    Canonical orientation makes the angle between two endplate tangents
    independent of which end a handle was dragged — reproduces the model's own
    ``vx>=0`` convention, and stays well-defined for any user edit.
    """
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    n = math.hypot(dx, dy)
    if n == 0:
        return (1.0, 0.0)
    ux, uy = dx / n, dy / n
    if ux < 0:
        ux, uy = -ux, -uy
    return (ux, uy)


def cobb_pair_angle(line_a, line_b):
    """Cobb angle (deg) between two endplate tangents given as {'p1','p2'}."""
    ua = _unit_canonical(line_a['p1'], line_a['p2'])
    ub = _unit_canonical(line_b['p1'], line_b['p2'])
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    return math.degrees(math.acos(dot))


def build_geometry(lines_px, image_size):
    """Assemble the editor geometry dict from tangent endpoints.

    ``lines_px`` is the per-tangent list from ``steepest_point_method``;
    ``image_size`` is (width, height) in DICOM pixels. Angles are always derived
    from the lines via ``cobb_pair_angle`` so the geometry is self-consistent.
    """
    lines = [
        {'id': i, 'p1': list(ln['p1']), 'p2': list(ln['p2'])}
        for i, ln in enumerate(lines_px)
    ]
    angles = []
    for i in range(len(lines) - 1):
        angles.append({
            'a': lines[i]['id'],
            'b': lines[i + 1]['id'],
            'value': round(cobb_pair_angle(lines[i], lines[i + 1]), 1),
        })
    return {
        'image_size': [int(image_size[0]), int(image_size[1])],
        'lines': lines,
        'angles': angles,
    }


def render_cobb_overlay(image_bgr, geometry):
    """Redraw endplate tangents + angle labels from ``geometry`` onto an image.

    Used by the Approve path so the PDF reflects the doctor's edited lines using
    the exact same drawing helpers as the automatic overlay.
    """
    img = image_bgr.copy()
    lines = geometry.get('lines') or []
    if not lines:
        return img

    by_id = {}
    centers = {}
    tangents = []
    for ln in lines:
        p1, p2 = ln['p1'], ln['p2']
        by_id[ln['id']] = ln
        tangents.append(((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))))
        centers[ln['id']] = ((int(p1[0]) + int(p2[0])) // 2,
                             (int(p1[1]) + int(p2[1])) // 2)
    # Same two-pass helper as the automatic overlay — the doctor's edited lines
    # must come out of the report looking exactly like the ones they replaced.
    # (No centre line here: the geometry the editor sends back carries tangents
    # only.)
    draw_overlay(img, [], tangents)

    pairs = geometry.get('angles') or []
    values = []
    for pr in pairs:
        la, lb = by_id.get(pr['a']), by_id.get(pr['b'])
        values.append(cobb_pair_angle(la, lb) if (la and lb) else None)
    valid = [v for v in values if v is not None]
    max_val = max(valid) if valid else None

    helper = _MaskAngle(img, None, None)
    for pr, val in zip(pairs, values):
        if val is None:
            continue
        ca, cb = centers.get(pr['a']), centers.get(pr['b'])
        if not ca or not cb:
            continue
        # draw_angle_label takes centres as (x=row, y=col) — swap back from (col,row).
        helper.draw_angle_label(img, ca[1], ca[0], cb[1], cb[0], val,
                                is_max=(max_val is not None and abs(val - max_val) < 1e-6))
    return img


def predict_cobb_angle(image_input):
    """
    Run scoliosis Cobb angle prediction on an image.

    Args:
        image_input: file path (str), numpy array (BGR), or bytes.

    Returns:
        tuple: (annotated_image_np_bgr, angles_list, geometry)
            - annotated image as numpy array (BGR)
            - list of measured Cobb angles in degrees (empty if none found)
            - geometry dict {image_size, lines, angles} for the interactive
              editor; ``lines`` is empty when nothing was measured.
    """
    import mmcv
    from mmdet.apis import inference_detector

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, str):
        img = mmcv.imread(image_input)
    else:
        img = image_input.copy()

    original = img.copy()
    model = _load_model()
    result = inference_detector(model, img)

    pred_instances = result.pred_instances
    scores = pred_instances.scores.cpu().numpy()
    high_score_idx = np.where(scores > 0.8)[0]

    if len(high_score_idx) == 0:
        return img, [], build_geometry([], (img.shape[1], img.shape[0]))

    idx = high_score_idx[0]
    image_np = np.array(img)
    mask_np = pred_instances.masks[idx].cpu().numpy()
    box_np = pred_instances.bboxes[idx].cpu().numpy()

    res = _MaskAngle(image_np, mask_np, box_np)
    skeleton, points_x, points_y = res.mask2skeleton()
    break_points, new_skeleton, x_fit, f_x, f_x_1d, x_offset, y_offset, break_i = \
        res.find_break_point(points_x, points_y)
    _, _, _ = res.sp_skeleton(new_skeleton, break_points, tol_ratio=0.3)

    image_res, angles, xcs, ycs, lines_px = res.steepest_point_method(
        image_np, new_skeleton, break_points, x_fit, f_x, f_x_1d,
        x_offset, y_offset, break_i, tol_ratio=0.15, off_l=0)

    geometry = build_geometry(lines_px, (image_np.shape[1], image_np.shape[0]))

    if len(angles) == 0:
        return image_res, [], geometry

    max_angle_id = angles.index(max(angles))
    for i in range(len(angles)):
        res.draw_angle_label(image_res, xcs[i], ycs[i], xcs[i + 1], ycs[i + 1],
                             angles[i], i == max_angle_id)

    return image_res, angles, geometry
