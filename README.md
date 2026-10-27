# External validation of a segmentation-based Cobb angle method on the AASCE 2019 dataset

Code and per-image results for the manuscript *External Validation of a Segmentation-based Deep Learning
Method for Cobb Angle Measurement: Accuracy and Failure Modes on 481 Radiographs*.

The evaluated method is the open-source spine-segmentation method of Li et al. (BJR Artif Intell 2025;
2(1):ubaf009), run unchanged with the authors' published code and weights by a batch script. It was run on
all 481 anteroposterior radiographs of the AASCE 2019 training set (SpineWeb Dataset 16). The reference is the
largest Cobb angle computed from the curated vertebral-corner annotations of Kordon et al.
(doi:10.5281/zenodo.4413665).

## Contents

`data/` — results, one row or record per image:

| File | Content |
|---|---|
| `results.csv` | reference and the method's largest Cobb angle, failure flag, absolute error |
| `ground_truth.csv` | reference angles computed from the landmarks, end vertebrae, apex, curve location, image size |
| `image_flags.csv` | posterior instrumentation and external brace (visual screening) |
| `levels_per_image.csv` | end vertebrae chosen by the method vs the reference, error decomposition, mechanism |
| `tangents_train481.jsonl` | angles and tangent end points returned by the service (overlay images removed) |
| `mask_extents_train481.jsonl` | row-wise left/right extent of the spine mask produced by the deployed network |
| `aasce_test98_radtools.csv` | angles returned for the 98 challenge test images (no reference available) |
| `metrics.json`, `analysis2.json`, `levels_summary.json`, `levels.json`, `run_stats.json` | primary and level analyses |
| `cobb_postproc.json`, `cobb_turning_per_image.json`, `cobb_oracle.json`, `cobb_recalibration.json`, `cobb_unet_eval_unet_masks_gpu.json` | post hoc analyses |

The `radtools` and `radtools_all` columns, `aasce_test98_radtools.csv` and the `radtools_run/` folder keep
the names of the processing run; they hold the angles returned by the evaluated method.

`analysis/` — Python scripts (Python 3.12, NumPy, SciPy, pandas, Pillow, OpenCV; PyTorch for `spine_unet.py`):

| Script | Step |
|---|---|
| `ground_truth.py` | reference angles from the landmarks (challenge algorithm, implementation of Yi et al.) |
| `compare.py`, `analysis2.py` | agreement, bootstrap CIs, subgroups, thresholds |
| `levels_analysis.py` | tangent-to-vertebra mapping and error decomposition |
| `figure1.py`, `figure_flow.py`, `figure_mechanisms.py` | figures |
| `cobb_postproc.py`, `scol_turning.py`, `cobb_recalibration.py` | post hoc: end-vertebra rules on the deployed masks |
| `cobb_oracle.py`, `spine_unet.py`, `cobb_unet_eval.py` | post hoc: landmark-derived masks and the U-Net trained on them |

## Reproducing

1. Download the AASCE 2019 training images and landmarks (SpineWeb Dataset 16) and the curated annotations
   (Zenodo 4413665). The radiographs are not redistributed here.
2. Run `ground_truth.py` on the downloaded data first: it rebuilds `ground_truth.csv` with all columns,
   including the angles distributed with the dataset (`orig_*`). Those columns, and `ref_orig` in
   `results.csv`, are not included here because they are part of the dataset, not results of this study;
   they are needed only for the sensitivity analysis with the original annotations.
3. Copy the other files of `data/` next to the scripts (`analysis/`), and the two `.jsonl` files into
   `radtools_run/` one level up, as the scripts expect.
4. Run the scripts in the order of the table above.

`cobb_postproc.py` reruns the deployed geometry, which is derived from the code of Li et al.; that module
(`scol_base.py`) is not included here and is available from the original authors' code release.

## Licence

Code: MIT (see `LICENSE`). Derived data in `data/`: CC BY 4.0. The AASCE images and annotations remain under
their original terms.
