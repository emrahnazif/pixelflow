"""
Unit tests for pixelflow.smoother.

The older classes pin regressions; TestPromises checks each promise of
`smooth`'s docstring.
"""

import warnings

import numpy as np
import pytest

import pixelflow as pf
from tests.helpers import black_canvas as blank, make_detections as detections


def fill(buffer, frames):
    """Push `frames` through `buffer`, yielding (raw, smoothed) once it is full."""
    out = []
    for detections_in in frames:
        raw, _ = buffer.update(detections_in, blank())
        if buffer.get_temporal_context() is not None:
            out.append((raw, pf.smooth(buffer)))
    return out


class TestGapHandling:
    """A tracker missing from the current frame is the interesting case."""

    def test_gap_frame_emits_each_tracker_once(self):
        """Regression: the gap was both smoothed and interpolated, emitting it twice.

        A tracker absent from the current frame but present either side belongs to
        the interpolation path alone. When the smoothing path also claimed it, one
        object came back as two rows sharing a tracker_id.
        """
        buffer = pf.Buffer(frames=5)
        frames = []
        for i in range(12):
            x = 100 + 10 * i
            frames.append(detections([[x, 50, x + 40, 90]], tracker_ids=[1]) if i != 5
                          else detections([]))

        for raw, smoothed in fill(buffer, frames):
            ids = [d.tracker_id for d in smoothed]
            assert len(ids) == len(set(ids)), f"duplicate tracker_id in {ids}"

    def test_tracker_ids_stay_unique_across_a_sequence(self):
        """The invariant, stated plainly: one id, one row, per frame."""
        buffer = pf.Buffer(frames=5)
        frames = []
        for i in range(20):
            boxes, ids = [], []
            if i != 7:
                boxes.append([100 + 10 * i, 50, 140 + 10 * i, 90]); ids.append(1)
            if i not in (11, 12):
                boxes.append([300 - 8 * i, 150, 340 - 8 * i, 190]); ids.append(2)
            frames.append(detections(boxes, tracker_ids=ids))

        for _, smoothed in fill(buffer, frames):
            ids = [d.tracker_id for d in smoothed]
            assert len(ids) == len(set(ids)), f"duplicate tracker_id in {ids}"


class TestUntrackedDetections:
    """Smoothing needs tracking; saying so beats returning nothing."""

    def test_warns(self):
        """Regression: untracked detections used to disappear without a word."""
        buffer = pf.Buffer(frames=5)
        for i in range(7):
            buffer.update(detections([[100 + 5 * i, 50, 140 + 5 * i, 90]]), blank())

        with pytest.warns(UserWarning, match="tracker_id"):
            pf.smooth(buffer)

    def test_returns_them_unsmoothed(self):
        """Nothing can be smoothed, so the frame comes back as it went in."""
        buffer = pf.Buffer(frames=5)
        for i in range(7):
            buffer.update(detections([[100 + 5 * i, 50, 140 + 5 * i, 90]]), blank())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert len(pf.smooth(buffer)) == 1

    def test_tracked_detections_do_not_warn(self):
        """The warning fires on the real mistake only."""
        buffer = pf.Buffer(frames=5)
        for i in range(7):
            buffer.update(detections([[100 + 5 * i, 50, 140 + 5 * i, 90]], tracker_ids=[1]), blank())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pf.smooth(buffer)


class TestSmoothing:
    """What the module is for."""

    def test_reduces_jitter(self):
        """A tracked box wobbling around a straight line comes back closer to it."""
        buffer = pf.Buffer(frames=5)
        true_x = [100 + 10 * i for i in range(12)]
        jitter = [0, 6, -6, 5, -5, 6, -6, 5, -5, 6, -6, 0]
        frames = [detections([[x + j, 50, x + j + 40, 90]], tracker_ids=[1])
                  for x, j in zip(true_x, jitter)]

        results = fill(buffer, frames)
        ideal = true_x[:len(results)]  # every frame comes out, from frame 0
        raw_error = np.mean([abs(raw[0].bbox[0] - x) for (raw, _), x in zip(results, ideal)])
        smoothed_error = np.mean([abs(sm[0].bbox[0] - x) for (_, sm), x in zip(results, ideal)])
        assert smoothed_error < raw_error

    def test_rejects_out_of_range_decay(self):
        """The weight decay is a documented range, enforced rather than assumed."""
        buffer = pf.Buffer(frames=5)
        with pytest.raises(ValueError, match="temporal_weight_decay"):
            pf.smooth(buffer, temporal_weight_decay=2.0)


# ============================================================================
# What smooth() promises (its docstring), one test per promise
# ============================================================================

def tracked(boxes, ids, **fields):
    """Detections with tracker ids and any extra fields set on every detection.

    Unlike `make_detections` (through `from_arrays`), it can set masks, keypoints,
    metadata, zones and times.
    """
    result = pf.Detections()
    for box, tracker_id in zip(boxes, ids):
        result.add_detection(pf.Detection(bbox=box, confidence=0.9, class_id=0, tracker_id=tracker_id, **fields))
    return result


def buffered(frames, size=5):
    buffer = pf.Buffer(frames=size)
    for results in frames:
        buffer.update(results, blank(4, 4))
    return buffer


class TestPromises:

    def test_empty_before_any_frame(self):
        buffer = pf.Buffer(frames=5)
        buffer.update(tracked([[0, 0, 10, 10]], [1]), blank(4, 4))
        assert len(pf.smooth(buffer)) == 0

    def test_weighted_average_of_the_window(self):
        xs = [0, 10, 20, 30, 40]
        buffer = buffered([tracked([[x, 0, x + 10, 10]], [1]) for x in xs])
        [result] = pf.smooth(buffer, temporal_weight_decay=0.5)
        weights = [0.25, 0.5, 1, 0.5, 0.25]
        expected = sum(w * x for w, x in zip(weights, xs)) / sum(weights)
        assert result.bbox[0] == pytest.approx(expected, abs=0.01)

    def test_confidence_is_averaged(self):
        frames = [tracked([[0, 0, 10, 10]], [1]) for _ in range(5)]
        frames[3][0].confidence = 0.4
        [result] = pf.smooth(buffered(frames), temporal_weight_decay=1.0)
        assert result.confidence == pytest.approx((0.9 * 4 + 0.4) / 5, abs=0.01)

    def test_every_detection_in_frame_order_untracked_unchanged(self):
        frames = [tracked([[50, 50, 60, 60], [x, 0, x + 10, 10]], [None, 1]) for x in range(5)]
        result = pf.smooth(buffered(frames))
        assert [d.tracker_id for d in result] == [None, 1]
        assert result[0].bbox == [50.0, 50.0, 60.0, 60.0]

    def test_keeps_every_field_and_shares_masks(self):
        mask = np.zeros((20, 20), dtype=bool)
        frames = [tracked([[0, 0, 10, 10]], [1], labels=["walking"], total_time=3.0,
                          first_seen_time=1.0, metadata={"camera": 2}, masks=[mask],
                          zones=[0], zone_names=["Door"]) for _ in range(5)]
        frames[2][0].line_crossings = [{"line_id": 0, "line_name": "Middle", "direction": "in"}]
        [result] = pf.smooth(buffered(frames))
        assert result.labels == ["walking"] and result.metadata == {"camera": 2}
        assert (result.total_time, result.first_seen_time) == (3.0, 1.0)
        assert result.zones == [0] and result.zone_names == ["Door"]
        assert result.line_crossings == [{"line_id": 0, "line_name": "Middle", "direction": "in"}]
        assert result.masks[0] is frames[2][0].masks[0]

    def test_returns_a_new_object_and_leaves_the_buffer_alone(self):
        frames = [tracked([[x, 0, x + 10, 10]], [1]) for x in (0, 10, 20, 30, 40)]
        buffer = buffered(frames)
        before = [[d.bbox for d in r] for r in buffer.get_buffer_contents()[0]]
        result = pf.smooth(buffer)
        result[0].bbox = [0, 0, 1, 1]
        assert result is not buffer.get_temporal_context()["current_results"]
        assert [[d.bbox for d in r] for r in buffer.get_buffer_contents()[0]] == before

    def test_edges_return_unsmoothed_copies(self):
        buffer = pf.Buffer(frames=5)
        for x in (0, 10, 20, 30):
            buffer.update(tracked([[x, 0, x + 10, 10]], [1]), blank(4, 4))
        for results, _ in buffer.flush():  # the last frames: fewer than 2 frames after them
            [result] = pf.smooth(buffer)
            assert result is not results[0]
            assert result.bbox == results[0].bbox

    def test_filled_in_by_distance_with_its_geometry(self):
        def frame(x, tracker_id):
            mask = np.zeros((20, 80), dtype=bool)
            mask[0:10, x:x + 10] = True
            kp = pf.KeyPoint(x=x + 5, y=5, id=0)
            return tracked([[x, 0, x + 10, 10]], [tracker_id], masks=[mask], keypoints=[kp],
                           metadata={"camera": 2})
        # Object 7 at x=0 two frames before and x=30 one frame after: at x=20 in between.
        frames = [frame(0, 7), frame(30, 8), frame(30, 8), frame(30, 7), frame(40, 8)]
        frames[0][0].line_crossings = [{"line_id": 0, "line_name": "Middle", "direction": "in"}]
        [filled] = [d for d in pf.smooth(buffered(frames)) if d.tracker_id == 7]
        assert filled.bbox == [20.0, 0.0, 30.0, 10.0]
        assert np.nonzero(filled.masks[0])[1].min() == 20
        assert filled.keypoints[0].x == 25
        assert filled.confidence == pytest.approx(0.9 * 0.85)
        assert filled.line_crossings == []
        assert filled.metadata == {"camera": 2, "interpolated": True, "interpolation_source": "buffer_smoother"}
        assert np.nonzero(frames[0][0].masks[0])[1].min() == 0  # the buffer's mask did not move

    def test_not_filled_without_a_sighting_on_both_sides(self):
        frames = [tracked([[0, 0, 10, 10]], [7]), tracked([[0, 0, 10, 10]], [7])] + \
                 [tracked([[50, 0, 60, 10]], [8]) for _ in range(3)]
        assert [d.tracker_id for d in pf.smooth(buffered(frames))] == [8]

    def test_mask_moved_left_is_cut_at_the_edge(self):
        def frame(x, tracker_id):
            mask = np.zeros((20, 40), dtype=bool)
            mask[0:10, x:x + 10] = True
            return tracked([[x, 0, x + 10, 10]], [tracker_id], masks=[mask])
        frames = [frame(30, 7), frame(5, 8), frame(5, 8), frame(0, 7), frame(5, 8)]
        [filled] = [d for d in pf.smooth(buffered(frames)) if d.tracker_id == 7]
        assert filled.masks[0].shape == (20, 40)
        assert filled.bbox == [10.0, 0.0, 20.0, 10.0]
        assert np.nonzero(filled.masks[0])[1].min() == 10

    def test_even_buffer_smooths_too(self):
        """Buffer(frames=4) holds 1 frame before the smoothed one and 2 after."""
        xs = [0, 10, 20, 30]
        with pytest.warns(UserWarning, match="even"):
            buffer = buffered([tracked([[x, 0, x + 10, 10]], [1]) for x in xs], size=4)
        [result] = pf.smooth(buffer, temporal_weight_decay=1.0)
        assert result.bbox[0] == pytest.approx(15.0)

    def test_filled_in_keypoints_keep_sub_pixel_positions(self):
        def frame(x, tracker_id):
            return tracked([[x, 0, x + 10, 10]], [tracker_id], keypoints=[pf.KeyPoint(x=x + 5, y=5, id=0)])
        # Object 7 at x=0 (offset -2) and x=31 (offset +1): two thirds of the way is x=20.67.
        frames = [frame(0, 7), frame(50, 8), frame(50, 8), frame(31, 7), frame(50, 8)]
        [filled] = [d for d in pf.smooth(buffered(frames)) if d.tracker_id == 7]
        assert filled.keypoints[0].x - filled.bbox[0] == pytest.approx(5, abs=0.01)


def test_shift_masks_cuts_at_the_canvas_edges():
    from pixelflow.masks import shift_masks
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    left = shift_masks([mask], -1, 0, (4, 4))[0]
    assert left.shape == (4, 4) and left[0:2, 0].all() and not left[:, 1:].any()
    off = shift_masks([mask], 3, 3, (4, 4))[0]
    assert off.sum() == 1 and off[3, 3]
    assert shift_masks([mask], 9, 9, (4, 4))[0].sum() == 0
