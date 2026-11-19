# Upstream: roboflow/trackers

`ByteTrack` and `SORT` are ported from [roboflow/trackers](https://github.com/roboflow/trackers),
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
| `utils/iou.py` | `utils/iou.py` | own `box_iou_batch` (same result as `supervision.box_iou_batch`) |
| `utils/base_tracklet.py`, `converters.py`, `kalman_filter.py`, `motion_models.py`, `predict_timing.py`, `state_representations.py` | same | import paths, docstring links |

Not ported: `utils/detections.py` (its one function is folded into `BaseTracker._to_arrays`),
OC-SORT, BoT-SORT, C-BIoU, McByte, ReID, camera motion compensation, the CLI, tuning, datasets,
evaluation, annotators.

## The Detections adapter

What differs from upstream, all in `base.py` and the two `update` methods:

- Input is a PixelFlow `Detections`; each detection needs a `bbox`. A detection without a
  confidence counts as `1.0` (upstream: all or none, through `default_confidences`).
- Output is a copy of the input in the input's order (upstream may reorder rows). The input is
  not modified; mask arrays are shared with it, not copied.
- An unconfirmed detection has `tracker_id = None` (upstream: `-1`).
- Tracker IDs start at 1 (upstream: 0).
- `tracked_objects` returns a `Detections` with `bbox` and `tracker_id` only.

The tracking itself is unchanged. On the MOT17 validation public detections (21 sequences,
47,736 frames), the port gives the same tracker ID as upstream for every detection on every
frame, with default settings, the XCYCSR state estimator, GIoU and BIoU, another frame rate,
and timestamps with gaps and duplicates.

## Syncing

```bash
git clone https://github.com/roboflow/trackers && cd trackers
git log 84569dc..origin/develop -- src/trackers/core/base.py src/trackers/core/bytetrack \
    src/trackers/core/sort src/trackers/utils/base_tracklet.py src/trackers/utils/converters.py \
    src/trackers/utils/iou.py src/trackers/utils/kalman_filter.py \
    src/trackers/utils/motion_models.py src/trackers/utils/predict_timing.py \
    src/trackers/utils/state_representations.py src/trackers/utils/detections.py
```

Carry over each change by hand, keeping the adapter above. Then rerun the comparison against
upstream on MOT17 and update the commit here.
