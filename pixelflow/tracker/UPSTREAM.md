# Upstream: roboflow/trackers

`SORT`, `ByteTrack`, `OCSORT`, `BoTSORT` and `CBIoU` are ported from [roboflow/trackers](https://github.com/roboflow/trackers),
Copyright (c) 2026 Roboflow, licensed under the Apache License 2.0, the license of PixelFlow too
(`LICENSE` and `NOTICE` at the repository root). PixelFlow does not depend on that package: the code is copied here and changed to
take and return PixelFlow `Detections` instead of `supervision.Detections`.

Last matched: commit `84569dcf06bdf9ecc2266ea14add22f4bd337280` (16 Nov 2026), branch `develop`.

## Files

The layout follows upstream's `src/trackers/`, without `core/`, so each file has one counterpart.

| PixelFlow | Upstream | Changes |
|-----------|----------|---------|
| `base.py` | `core/base.py` | Detections adapter (below); CLI registry, docstring parser and tuning search space removed |
| `bytetrack/tracker.py` | `core/bytetrack/tracker.py` | Detections adapter; class `ByteTrackTracker` renamed `ByteTrack`; docstrings say `None` for unconfirmed |
| `bytetrack/tracklet.py`, `bytetrack/utils.py` | same under `core/` | import paths |
| `sort/tracker.py` | `core/sort/tracker.py` | Detections adapter; class `SORTTracker` renamed `SORT`; deprecated `trackers` property removed; docstrings say `None` for unconfirmed |
| `sort/tracklet.py`, `sort/utils.py` | same under `core/` | import paths |
| `ocsort/tracker.py` | `core/ocsort/tracker.py` | Detections adapter; class `OCSORTTracker` renamed `OCSORT`; RGB frames |
| `ocsort/tracklet.py`, `ocsort/utils.py` | same under `core/` | import paths |
| `botsort/tracker.py` | `core/botsort/tracker.py` | Detections adapter; class `BoTSORTTracker` renamed `BoTSORT`; RGB frames; ReID (`reid_model`, `reid_config`) and deprecated `apply_cmc_batch` removed |
| `botsort/tracklet.py` | `core/botsort/tracklet.py` | ReID `feature_bank` removed |
| `botsort/utils.py` | `core/botsort/utils.py` | import paths |
| `cbiou/tracker.py` | `core/cbiou/tracker.py` | Detections adapter; class `CBIoUTracker` renamed `CBIoU` |
| `utils/cmc.py` | `utils/cmc.py` | frames are RGB (`COLOR_RGB2GRAY`); deprecated `CMC.apply_to_xyxy` removed; logger `pixelflow.tracker.cmc` |
| `utils/iou.py` | `utils/iou.py` | own `box_iou_batch` (same result as `supervision.box_iou_batch`) |
| `utils/base_tracklet.py`, `converters.py`, `kalman_filter.py`, `matrix.py`, `motion_models.py`, `predict_timing.py`, `scale_aware_tracklet.py`, `state_representations.py` | same | import paths, docstring links |

Not ported: `utils/detections.py` (its one function is folded into `BaseTracker._to_arrays`),
McByte, ReID (`reid/`), `motion/`, the deprecated `core/botsort/cmc.py` shim, the CLI, tuning,
datasets, evaluation, annotators.

## The Detections adapter

What differs from upstream, all in `base.py` and the `update` methods:

- Input is a PixelFlow `Detections`; each detection needs a `bbox`. A detection without a
  confidence counts as `1.0` (upstream: all or none, through `default_confidences`).
- Output is a copy of the input in the input's order (upstream may reorder rows). The input is
  not modified; mask arrays are shared with it, not copied. Every input detection is returned:
  BoT-SORT and C-BIoU ignore detections with a confidence of 0.1 or less, and upstream drops
  them from the output; here they come back with `tracker_id = None`.
- Frames passed for camera motion compensation are RGB, PixelFlow's image format (upstream: BGR).
- `frame_rate=None` counts as 30 (upstream: error), so `frame_rate=video.fps` works when a video
  or camera reports no rate.
- An unconfirmed detection has `tracker_id = None` (upstream: `-1`).
- Tracker IDs start at 1 (upstream: 0).
- `tracked_objects` returns a `Detections` with `bbox` and `tracker_id` only.

The tracking itself is unchanged. On the MOT17 validation public detections (21 sequences,
7,956 frames per setup), the port gives the same tracker ID as upstream for every detection on
every frame, for all five trackers: default settings, other state estimators and IoU variants,
another frame rate, timestamps with gaps and duplicates, and (on the 7 sequences' frames)
camera motion compensation in BoT-SORT and OC-SORT.

## Syncing

```bash
git clone https://github.com/roboflow/trackers && cd trackers
git log 84569dc..origin/develop -- src/trackers/core/base.py src/trackers/core/bytetrack \
    src/trackers/core/sort src/trackers/core/ocsort src/trackers/core/botsort \
    src/trackers/core/cbiou src/trackers/utils/base_tracklet.py src/trackers/utils/cmc.py \
    src/trackers/utils/converters.py src/trackers/utils/iou.py src/trackers/utils/kalman_filter.py \
    src/trackers/utils/matrix.py src/trackers/utils/motion_models.py \
    src/trackers/utils/predict_timing.py src/trackers/utils/scale_aware_tracklet.py \
    src/trackers/utils/state_representations.py src/trackers/utils/detections.py
```

Carry over each change by hand, keeping the adapter above. Then rerun the comparison against
upstream on MOT17 and update the commit here.
