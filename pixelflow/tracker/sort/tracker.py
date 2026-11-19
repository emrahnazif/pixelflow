# ------------------------------------------------------------------------
# Ported from roboflow/trackers, src/trackers/core/sort/tracker.py at commit 84569dc.
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
# Changes for PixelFlow: PixelFlow Detections in and out; see pixelflow/tracker/UPSTREAM.md.
# ------------------------------------------------------------------------

import numpy as np
from scipy.optimize import linear_sum_assignment

from pixelflow.detections import Detections
from pixelflow.tracker.base import BaseTracker
from pixelflow.tracker.sort.tracklet import SORTTracklet
from pixelflow.tracker.sort.utils import _get_alive_tracklets
from pixelflow.tracker.utils.iou import BaseIoU, IoU
from pixelflow.tracker.utils.state_representations import (
    BaseStateEstimator,
    XYXYStateEstimator,
)


class SORT(BaseTracker):
    """In SORT, object tracking begins with high-confidence detections fed into a Kalman filter framework assuming
    uniform motion for state prediction across frames. Association occurs via IoU-based costs in the Hungarian
    algorithm, enforcing a threshold to filter weak matches and initialize new identities. Tracks persist only with
    consistent associations, terminating quickly to avoid erroneous propagation. This detection-driven approach
    underscores the importance of upstream detector performance in achieving competitive multi-object tracking results.
    Over time, SORT has become a cornerstone for evaluating motion-based improvements in the field.

    SORT's standout strength is its real-time capability, processing hundreds of frames
    per second while maintaining accuracy comparable to more complex offline methods. It
    performs well in controlled environments with reliable detections, minimizing
    computational demands. However, without mechanisms for re-identification, it incurs
    frequent identity switches during object reappearances post-occlusion. The linear
    motion assumption limits effectiveness in non-linear paths, such as those in sports
    or wildlife tracking. Ultimately, SORT's efficiency is offset by its sensitivity to
    environmental complexities, necessitating hybrid extensions for broader
    applicability.

    Args:
        lost_track_buffer: Non-negative `int` specifying number of 30 FPS frames
            to buffer when a track is lost. `0` deletes a confirmed track on the
            first missed frame. Increasing this value enhances occlusion
            handling but may increase ID switching for similar objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Must be positive; `None` means 30. Used to scale the lost track buffer for consistent
            tracking across different frame rates.
        track_activation_threshold: `float` specifying minimum detection
            confidence to create new tracks. Higher values reduce false
            positives but may miss low-confidence objects.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, their detections get `tracker_id = None`.
        minimum_iou_threshold: `float` specifying IoU threshold for associating
            detections to existing tracks. Higher values require more overlap.
        state_estimator_class: State estimator class to use for Kalman filter.
            `XCYCSRStateEstimator` for center-based representation or
            `XYXYStateEstimator` for corner-based representation.
        iou: IoU similarity metric instance to use for data association.
            Defaults to standard `IoU`. Can be replaced with any `BaseIoU`
            subclass (e.g. GIoU, DIoU, CIoU) to change how bounding-box
            similarity is computed during the association step.
            ``None`` (the default) means ``IoU()``.
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        state_estimator_class: type[BaseStateEstimator] = XYXYStateEstimator,
        iou: BaseIoU | None = None,
    ) -> None:
        self.maximum_frames_without_update = self._compute_maximum_frames_without_update(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
        )
        self.maximum_time_without_update: float = lost_track_buffer / 30.0
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()

        self._init_timestamp_state(frame_rate)

        # Active tracklets
        self.tracks: list[SORTTracklet] = []
        self._reset_id_allocator()

    def _get_associated_indices(
        self, iou_matrix: np.ndarray, detection_boxes: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections to tracks based on IOU.

        Args:
            iou_matrix: IOU cost matrix.
            detection_boxes: Detected bounding boxes in the form [x1, y1, x2, y2].

        Returns:
            matched: List of ``(track_index, detection_index)`` tuples for
                associations that meet the IoU threshold.
            unmatched_tracks: Sorted list of track indices not matched to any
                detection.
            unmatched_detections: Sorted list of detection indices not matched
                to any track.
        """
        matched_indices = []
        unmatched_tracklets = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detection_boxes)))

        if len(self.tracks) > 0 and len(detection_boxes) > 0:
            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            # Note that it uses a a modified Jonker-Volgenant algorithm with no
            # initialization instead of the Hungarian algorithm as mentioned in the
            # SORT paper.
            row_indices, col_indices = linear_sum_assignment(iou_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_tracklets.remove(row)
                    unmatched_detections.remove(col)

        # Return sorted lists for deterministic order across CPython versions.
        return (
            matched_indices,
            sorted(unmatched_tracklets),
            sorted(unmatched_detections),
        )

    def _spawn_new_tracklets(
        self,
        confidences: np.ndarray,
        detection_boxes: np.ndarray,
        unmatched_detections: list[int],
    ) -> None:
        for detection_idx in unmatched_detections:
            if confidences[detection_idx] >= self.track_activation_threshold:
                new_tracker = SORTTracklet(
                    detection_boxes[detection_idx],
                    state_estimator_class=self.state_estimator_class,
                )
                self.tracks.append(new_tracker)

    def update(
        self,
        detections: Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> Detections:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, IoU-based association, and initializes
        new tracks for unmatched high-confidence detections.

        Args:
            detections: `Detections` whose boxes are `[x1, y1, x2, y2]`. A
                detection without a confidence is treated as confidence `1.0`.
            frame: Ignored by SORT. If provided (not `None`), a warning is emitted.
            timestamp: Absolute time of the current frame in seconds, or ``None``
                for fixed-rate mode (``frame_step = 1.0`` per call).

        Returns:
            A copy of `detections`, in the same order, with `tracker_id` set on
            each detection of a confirmed track and `None` on the others.

        Raises:
            TypeError: If `detections` is not a `Detections`.
            ValueError: If a detection has no box, or a box contains NaN or inf.
                The check runs before any tracker state is advanced, so a failed
                call leaves the tracker unchanged.

        Warns:
            UserWarning: If ``frame`` is passed but SORT does not perform
                camera motion compensation (CMC), the frame is ignored.
        """
        detection_boxes, confidences = self._to_arrays(detections)
        self._warn_if_frame_unused(frame)
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._result(detections)

        if len(self.tracks) == 0 and len(detections) == 0:
            return self._result(detections)

        self._predict_tracklets(self.tracks, timing)

        # Ghost-ID prevention: budget-only filter before association.
        # Keeps immature tracks alive for matching; full lifecycle prune runs after.
        _budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        self._prune_lost_tracks(timing)

        predicted_boxes = np.array([t.get_state_bbox() for t in self.tracks]) if self.tracks else np.empty((0, 4))
        iou_matrix = self.iou.compute(predicted_boxes, detection_boxes)

        # Associate detections to tracklets based on IOU
        matched_indices, _unmatched_tracklets, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_boxes
        )

        # Update matched tracklets and record the det_idx -> tracklet mapping
        matched_tracklet_for_det: dict[int, SORTTracklet] = {}
        for row, col in matched_indices:
            self.tracks[row].update(detection_boxes[col])
            matched_tracklet_for_det[col] = self.tracks[row]

        self._spawn_new_tracklets(confidences, detection_boxes, unmatched_detections)

        # Full lifecycle prune: also removes immature+unmatched tracks
        self.tracks = _get_alive_tracklets(
            self.tracks,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
            _budget,
        )

        # Build tracker_ids from the recorded mapping (no deepcopy, no re-IoU)
        tracker_ids = np.full(len(detection_boxes), -1, dtype=int)
        for det_idx, tracklet in matched_tracklet_for_det.items():
            if tracklet.number_of_successful_updates >= self.minimum_consecutive_frames:
                if tracklet.tracker_id == -1:
                    tracklet.tracker_id = self._allocate_tracker_id()
                tracker_ids[det_idx] = tracklet.tracker_id

        # Return a copy rather than mutating the caller's object.
        return self._result(detections, range(len(tracker_ids)), tracker_ids)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.

        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        self._last_timestamp = None
        self._reset_id_allocator()
