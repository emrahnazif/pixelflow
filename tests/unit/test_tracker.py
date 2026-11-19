"""
Unit and scenario tests for pixelflow.tracker.

The trackers are ported from roboflow/trackers. Their tracking output was
checked against upstream on the MOT17 validation set (identical IDs on every
frame, with and without camera motion compensation); these tests pin what the port adds -- PixelFlow Detections in and
out -- and the behaviour that matters to users, on synthetic sequences: a
tracker can pass every unit test and still switch identities the moment two
objects cross.

Scenarios use synthetic boxes rather than a real sequence, which keeps them
deterministic and fast. They are a regression net, not a benchmark.
"""

import warnings

import numpy as np
import pytest

import pixelflow as pf
from pixelflow.tracker.utils.iou import box_iou_batch
from tests.helpers import make_detections as detections_of

TRACKERS = [pf.tracker.ByteTrack, pf.tracker.SORT, pf.tracker.OCSORT, pf.tracker.BoTSORT, pf.tracker.CBIoU]


# ============================================================================
# Helpers
# ============================================================================

def box(x, y, w=50, h=50):
    """One [x1, y1, x2, y2] box at (x, y)."""
    return [x, y, x + w, y + h]


def moving(start_x, y, step, frames, w=50, h=50):
    """One object translating `step` px per frame."""
    return [box(start_x + step * i, y, w, h) for i in range(frames)]


def track_sequence(tracker, frames, scores=None):
    """Drive `frames` through `tracker`, returning the tracker_ids seen per frame."""
    seen = []
    for i, boxes in enumerate(frames):
        detections = detections_of(boxes, scores[i] if scores is not None else None)
        seen.append([d.tracker_id for d in tracker.update(detections)])
    return seen


def assigned(frame_ids):
    """The ids actually assigned in one frame, as a set (None means unconfirmed)."""
    return {i for i in frame_ids if i is not None}


# ============================================================================
# IoU
# ============================================================================

class TestBoxIoU:

    def test_values(self):
        a = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=float)
        b = np.array([[150, 150, 250, 250], [320, 320, 420, 420]], dtype=float)
        iou = box_iou_batch(a, b)
        assert iou.dtype == np.float32
        np.testing.assert_allclose(iou, [[2500 / 17500, 0], [0, 6400 / 13600]], rtol=1e-6)

    def test_empty_union_is_zero(self):
        point = np.array([[5, 5, 5, 5]], dtype=float)
        assert box_iou_batch(point, point)[0, 0] == 0


# ============================================================================
# PixelFlow Detections in and out
# ============================================================================

@pytest.mark.parametrize("tracker_class", TRACKERS)
class TestDetectionsAdapter:

    def test_input_is_not_modified(self, tracker_class):
        tracker = tracker_class(minimum_consecutive_frames=1)
        detections = detections_of([box(0, 0)])
        for _ in range(3):
            result = tracker.update(detections)
        assert result[0].tracker_id == 1
        assert detections[0].tracker_id is None
        assert result is not detections

    def test_order_and_fields_are_kept(self, tracker_class):
        # OC-SORT has no track_activation_threshold: high_conf_det_threshold decides.
        threshold = "high_conf_det_threshold" if tracker_class is pf.tracker.OCSORT else "track_activation_threshold"
        tracker = tracker_class(minimum_consecutive_frames=1, **{threshold: 0.1})
        boxes = [box(0, 0), box(200, 0), box(400, 0)]
        for _ in range(3):
            detections = detections_of(boxes, scores=[0.9, 0.2, 0.8])
            detections[1].class_name = "middle"
            result = tracker.update(detections)
        assert [d.bbox for d in result] == [d.bbox for d in detections]
        assert result[1].class_name == "middle"
        assert [d.confidence for d in result] == [0.9, 0.2, 0.8]

    def test_masks_are_shared_not_copied(self, tracker_class):
        mask = np.zeros((100, 100), dtype=bool)
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=box(0, 0), confidence=0.9, masks=[mask]))
        result = tracker_class().update(detections)
        assert result[0].masks[0] is mask
        assert result[0].masks is not detections[0].masks

    def test_ids_start_at_one(self, tracker_class):
        tracker = tracker_class(minimum_consecutive_frames=1)
        ids = track_sequence(tracker, [[box(0, 0), box(200, 0)]] * 3)
        assert assigned(ids[-1]) == {1, 2}

    def test_unconfirmed_is_none(self, tracker_class):
        """An object entering mid-video is unconfirmed on its first frame.

        Mid-video, because on the first frames OC-SORT confirms at once and BoT-SORT and
        C-BIoU give the first frame's objects an ID immediately, as their papers do.
        """
        tracker = tracker_class(minimum_consecutive_frames=3)
        ids = track_sequence(tracker, [[box(0, 0)]] * 6 + [[box(0, 0), box(300, 300)]])
        assert ids[-1][1] is None

    def test_missing_confidence_counts_as_one(self, tracker_class):
        tracker = tracker_class(minimum_consecutive_frames=1)
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=box(0, 0)))
        for _ in range(3):
            result = tracker.update(detections)
        assert result[0].tracker_id == 1

    def test_rejects_other_types(self, tracker_class):
        with pytest.raises(TypeError, match="Detections"):
            tracker_class().update(np.array([box(0, 0)]))

    def test_rejects_detection_without_box(self, tracker_class):
        detections = pf.Detections()
        detections.detections.append(pf.Detection(class_id=0, confidence=0.9))
        with pytest.raises(ValueError, match="bbox"):
            tracker_class().update(detections)

    def test_empty_detections(self, tracker_class):
        tracker = tracker_class()
        assert len(tracker.update(pf.Detections())) == 0
        track_sequence(tracker, [[box(0, 0)]] * 4)
        assert len(tracker.update(pf.Detections())) == 0

    def test_tracked_objects_include_missed_tracks(self, tracker_class):
        tracker = tracker_class(minimum_consecutive_frames=1)
        track_sequence(tracker, [[b] for b in moving(0, 100, 5, 5)])
        tracker.update(pf.Detections())
        objects = tracker.tracked_objects
        assert [d.tracker_id for d in objects] == [1]
        assert objects[0].confidence is None
        assert objects[0].bbox[0] > 20  # predicted past the last box at x=20

    def test_reset_restarts_ids(self, tracker_class):
        tracker = tracker_class(minimum_consecutive_frames=1)
        track_sequence(tracker, [[box(0, 0), box(200, 0)]] * 3)
        tracker.reset()
        ids = track_sequence(tracker, [[box(400, 400)]] * 3)
        assert assigned(ids[-1]) == {1}

    def test_frame_is_ignored_with_a_warning(self, tracker_class):
        if tracker_class is pf.tracker.BoTSORT:
            pytest.skip("BoT-SORT uses the frame for camera motion compensation")
        with pytest.warns(UserWarning, match="frame"):
            tracker_class().update(detections_of([box(0, 0)]), frame=np.zeros((10, 10, 3), np.uint8))

    def test_backwards_timestamp_skips_the_update(self, tracker_class):
        tracker = tracker_class(minimum_consecutive_frames=1)
        for i in range(3):
            tracker.update(detections_of([box(0, 0)]), timestamp=i / 30)
        with pytest.warns(UserWarning, match="earlier"):
            result = tracker.update(detections_of([box(0, 0)]), timestamp=0.0)
        assert result[0].tracker_id is None
        result = tracker.update(detections_of([box(0, 0)]), timestamp=3 / 30)
        assert result[0].tracker_id == 1

    def test_timestamp_gap_extrapolates_further(self, tracker_class):
        """A frame that arrives late is matched where the object is by then."""
        tracker = tracker_class(minimum_consecutive_frames=1)
        for i in range(6):
            tracker.update(detections_of([box(10 * i, 0)]), timestamp=i / 30)
        # 4 frames' time later, the object moved 40 px: still the same track.
        result = tracker.update(detections_of([box(90, 0)]), timestamp=9 / 30)
        assert result[0].tracker_id == 1


def test_bytetrack_defaults():
    tracker = pf.tracker.ByteTrack()
    assert tracker.track_activation_threshold == 0.7
    assert tracker.high_conf_det_threshold == 0.6
    assert tracker.minimum_consecutive_frames == 2
    assert tracker.maximum_frames_without_update == 30


def test_sort_defaults():
    tracker = pf.tracker.SORT()
    assert tracker.track_activation_threshold == 0.25
    assert tracker.minimum_consecutive_frames == 3
    assert tracker.minimum_iou_threshold == 0.3


def test_lost_track_buffer_scales_with_frame_rate():
    assert pf.tracker.ByteTrack(lost_track_buffer=30, frame_rate=60).maximum_frames_without_update == 60
    with pytest.raises(ValueError):
        pf.tracker.SORT(lost_track_buffer=-1)


# ============================================================================
# Scenarios -- the behaviour that actually matters
# ============================================================================

@pytest.mark.parametrize("tracker_class", TRACKERS)
class TestTrackingScenarios:
    """Whole sequences, asserting on identity rather than internals."""

    def test_single_object_keeps_one_id(self, tracker_class):
        seen = track_sequence(tracker_class(), [[b] for b in moving(0, 100, 8, 20)])
        settled = [assigned(f) for f in seen[5:]]
        assert all(s == settled[0] for s in settled)
        assert len(settled[0]) == 1

    def test_two_objects_get_distinct_ids(self, tracker_class):
        left = moving(0, 100, 8, 20)
        right = moving(400, 300, -8, 20)
        seen = track_sequence(tracker_class(), [[l, r] for l, r in zip(left, right)])
        assert len(assigned(seen[-1])) == 2

    def test_short_occlusion_preserves_identity(self, tracker_class):
        """A gap well inside lost_track_buffer must not mint a new id."""
        frames = [[b] for b in moving(0, 100, 10, 10)]
        frames += [[]] * 5
        frames += [[b] for b in moving(150, 100, 10, 10)]
        seen = track_sequence(tracker_class(), frames)
        before, after = assigned(seen[8]), assigned(seen[-1])
        assert before and after
        assert before == after

    def test_long_occlusion_starts_a_new_track(self, tracker_class):
        """Past the buffer the track is gone, and a fresh id beats a stale one."""
        frames = [[b] for b in moving(0, 100, 10, 10)]
        frames += [[]] * 15
        frames += [[b] for b in moving(500, 100, 10, 10)]
        seen = track_sequence(tracker_class(lost_track_buffer=5), frames)
        before, after = assigned(seen[8]), assigned(seen[-1])
        assert before and after
        assert before != after

    def test_crossing_objects_do_not_swap_ids(self, tracker_class):
        """Two objects passing through each other is the classic switch case."""
        frames = [[box(10 + 12 * i, 100), box(290 - 12 * i, 100)] for i in range(24)]
        seen = track_sequence(tracker_class(), frames)
        first = dict(zip(("left", "right"), seen[5]))
        last = dict(zip(("left", "right"), seen[-1]))
        assert None not in first.values()
        assert first == last

    @pytest.mark.parametrize("step", [5, 10, 20])
    def test_tracks_while_boxes_overlap(self, tracker_class, step):
        """Association is IoU-only, so it holds while boxes overlap enough frame to frame.

        At 20 px a step, two 50 px boxes overlap with IoU 0.43; at 30 px, 0.25, below
        SORT's default ``minimum_iou_threshold`` of 0.3.
        """
        seen = track_sequence(tracker_class(), [[b] for b in moving(100, 100, step, 15)])
        assert len(assigned(seen[-1])) == 1

    def test_no_overlap_is_never_confirmed(self, tracker_class):
        """Beyond the association limit the tracker reports nothing, not a stream of new ids."""
        seen = track_sequence(tracker_class(), [[b] for b in moving(100, 100, 60, 15)])
        assert assigned(seen[-1]) == set()

    def test_crowd_keeps_every_object_separate(self, tracker_class):
        """Eight objects in a grid, all moving together."""
        frames = [[box(60 * col + 4 * i, 60 * row) for row in range(2) for col in range(4)]
                  for i in range(12)]
        seen = track_sequence(tracker_class(), frames)
        assert len(assigned(seen[-1])) == 8

    def test_new_object_entering_gets_a_fresh_id(self, tracker_class):
        frames = [[box(8 * i, 100)] for i in range(10)]
        frames += [[box(80 + 8 * i, 100), box(400, 300)] for i in range(10)]
        seen = track_sequence(tracker_class(), frames)
        assert len(assigned(seen[9])) == 1
        assert len(assigned(seen[-1])) == 2

    def test_minimum_consecutive_frames_delays_confirmation(self, tracker_class):
        """A new object mid-video waits for its frames (see test_unconfirmed_is_none)."""
        frames = [[box(0, 0)]] * 6 + [[box(0, 0), b] for b in moving(300, 300, 5, 8)]
        seen = track_sequence(tracker_class(minimum_consecutive_frames=4), frames)
        assert seen[6 + 2][1] is None
        assert seen[-1][1] is not None

    def test_stationary_object_holds_its_id(self, tracker_class):
        seen = track_sequence(tracker_class(), [[box(100, 100)]] * 15)
        settled = [assigned(f) for f in seen[4:]]
        assert all(s == settled[0] for s in settled)
        assert len(settled[0]) == 1


def test_bytetrack_low_confidence_detections_sustain_a_track():
    """The second association stage is ByteTrack's whole point."""
    frames = [[b] for b in moving(0, 100, 8, 16)]
    scores = [[0.9]] * 6 + [[0.35]] * 5 + [[0.9]] * 5
    seen = track_sequence(pf.tracker.ByteTrack(), frames, scores=scores)
    assert assigned(seen[5]) == assigned(seen[8]) == assigned(seen[-1]) != set()


def test_bytetrack_low_confidence_alone_never_starts_a_track():
    seen = track_sequence(pf.tracker.ByteTrack(), [[box(0, 0)]] * 10, scores=[[0.5]] * 10)
    assert assigned(seen[-1]) == set()


# ============================================================================
# Camera motion compensation (BoT-SORT, OC-SORT)
# ============================================================================

def panning_video(frames=15, step=40):
    """A textured scene panning `step` px per frame, with two objects fixed in the scene.

    Each object moves `step` px per frame in the image, more than a 50 px box can move
    and still overlap itself: without camera motion compensation no track survives.
    """
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (400, 2000, 3), dtype=np.uint8)
    scene = np.repeat(np.repeat(scene[::8, ::8], 8, 0), 8, 1)
    for i in range(frames):
        x = 1200 - step * i                      # the camera moves right, the scene left
        image = np.ascontiguousarray(scene[:, x - 600:x + 40])
        boxes = [box(100 + step * i, 100), box(200 + step * i, 250)]
        yield image, detections_of(boxes)


def test_botsort_camera_pan_keeps_identities():
    tracker = pf.tracker.BoTSORT()
    seen = [[d.tracker_id for d in tracker.update(dets, frame=image)] for image, dets in panning_video()]
    assert all(ids == [1, 2] for ids in seen)


def test_botsort_without_cmc_loses_a_panning_scene():
    tracker = pf.tracker.BoTSORT(enable_cmc=False)
    seen = [[d.tracker_id for d in tracker.update(dets)] for _, dets in panning_video()]
    assert seen[0] == [1, 2]
    assert seen[-1] != [1, 2]  # the tracks were lost


def test_botsort_precomputed_transform():
    """A known camera motion can be passed instead of a frame."""
    tracker = pf.tracker.BoTSORT()
    shift = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]])
    seen = [[d.tracker_id for d in tracker.update(dets, cmc_transform=shift if i else None)]
            for i, (_, dets) in enumerate(panning_video())]
    assert all(ids == [1, 2] for ids in seen)


def test_botsort_rejects_frame_and_transform_together():
    with pytest.raises(ValueError, match="both"):
        pf.tracker.BoTSORT().update(detections_of([box(0, 0)]), frame=np.zeros((10, 10, 3), np.uint8),
                                    cmc_transform=np.eye(2, 3))


def test_ocsort_camera_pan_with_cmc():
    tracker = pf.tracker.OCSORT(enable_cmc=True)
    seen = [[d.tracker_id for d in tracker.update(dets, frame=image)] for image, dets in panning_video()]
    assert all(ids == seen[-1] for ids in seen[3:])
    assert len(assigned(seen[-1])) == 2


def test_ocsort_cmc_without_frame_warns_once():
    tracker = pf.tracker.OCSORT(enable_cmc=True)
    with pytest.warns(UserWarning, match="frame=None"):
        tracker.update(detections_of([box(0, 0)]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracker.update(detections_of([box(0, 0)]))


def test_cbiou_ignores_transform_with_a_warning():
    with pytest.warns(UserWarning, match="cmc_transform"):
        pf.tracker.CBIoU().update(detections_of([box(0, 0)]), cmc_transform=np.eye(2, 3))


def test_cbiou_buffer_tracks_beyond_plain_iou():
    """Buffered IoU matches boxes that no longer overlap at all, where IoU-only trackers fail."""
    frames = [[b] for b in moving(100, 100, 60, 15)]   # 60 px steps: 50 px boxes never overlap
    assert assigned(track_sequence(pf.tracker.ByteTrack(), frames)[-1]) == set()
    seen = track_sequence(pf.tracker.CBIoU(buffer_ratio_first=0.5, buffer_ratio_second=0.5), frames)
    assert all(ids == [1] for ids in seen)


@pytest.mark.parametrize("tracker_class", TRACKERS)
def test_low_confidence_detections_are_kept_in_order(tracker_class):
    """Every detection comes back in order; BoT-SORT and C-BIoU ignore ones at 0.1 or below."""
    tracker = tracker_class(minimum_consecutive_frames=1)
    for _ in range(3):
        result = tracker.update(detections_of([box(0, 0), box(200, 0)], scores=[0.9, 0.05]))
    assert [d.tracker_id for d in result] == [1, None]
    assert [d.confidence for d in result] == [0.9, 0.05]
