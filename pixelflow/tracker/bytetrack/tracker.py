# ------------------------------------------------------------------------
# Ported from roboflow/trackers, src/trackers/core/bytetrack/tracker.py at commit 84569dc.
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
# Changes for PixelFlow: PixelFlow Detections in and out; see pixelflow/tracker/UPSTREAM.md.
# ------------------------------------------------------------------------

import numpy as np
from scipy.optimize import linear_sum_assignment

from pixelflow.detections import Detections
from pixelflow.tracker.base import BaseTracker
from pixelflow.tracker.bytetrack.tracklet import ByteTrackTracklet
from pixelflow.tracker.bytetrack.utils import _get_alive_tracklets
from pixelflow.tracker.utils.iou import BaseIoU, IoU
from pixelflow.tracker.utils.state_representations import (
    BaseStateEstimator,
    XYXYStateEstimator,
)


class ByteTrack(BaseTracker):
    """ByteTrack operates online by processing all detector outputs, categorizing them by confidence thresholds to
    enable a two-stage association process. High-score boxes are initially linked to tracklets via Kalman filter
    predictions and IoU-based Hungarian matching. Low-score boxes follow
    in a secondary matching phase using pure motion similarity to revive occluded tracks. Tracks without matches are
    kept briefly for potential re-association, preventing premature termination. This inclusive approach addresses
    common pitfalls in detection filtering, establishing ByteTrack as a flexible enhancer for existing tracking
    frameworks.

    ByteTrack excels in dense environments, where its low-score recovery mechanism
    minimizes missed detections and enhances overall trajectory completeness. It
    consistently improves performance across diverse datasets, demonstrating robustness
    and generalization. The tracker's speed remains competitive, facilitating
    integration into production pipelines. On the downside, it is highly dependent on
    detector quality, with performance drops in noisy or low-resolution inputs.
    Additionally, the motion-only secondary association may lead to erroneous matches
    in scenes with similar moving objects.

    Note:
        A detection without a confidence counts as fully confident (``1.0``).
        When no detection has one, ByteTrack works like SORT: a single IoU match,
        without the low-confidence recovery stage.

    Args:
        lost_track_buffer: Non-negative `int` specifying number of 30 FPS frames
            to buffer when a track is lost. `0` deletes a confirmed track on the
            first missed frame. Increasing this value enhances occlusion
            handling but may increase ID switching for disappearing objects.
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
        high_conf_det_threshold: `float` specifying threshold for separating
            high and low confidence detections in the two-stage association.
        state_estimator_class: State estimator class to use for Kalman filter.
            Defaults to `XYXYStateEstimator`. Can also use
            `XCYCSRStateEstimator` for center-based representation.
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
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
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
        self.high_conf_det_threshold = high_conf_det_threshold
        self.tracks: list[ByteTrackTracklet] = []
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()
        self._reset_id_allocator()

        self._init_timestamp_state(frame_rate)

    def update(
        self,
        detections: Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> Detections:
        """Update tracks state with new detections and return tracked objects.
        Performs Kalman filter prediction, two-stage association (high then low
        confidence), and initializes new tracks for unmatched detections.

        Args:
            detections: `Detections` whose boxes are `[x1, y1, x2, y2]`. A
                detection without a confidence is treated as confidence `1.0`:
                it bypasses the low-confidence stage, and if unmatched it can
                spawn a new track regardless of `track_activation_threshold`.
            frame: Ignored by ByteTrack. If provided (not `None`), a warning is
                emitted.
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
            UserWarning: If ``frame`` is passed but ByteTrack does not perform
                camera motion compensation (CMC), the frame is ignored.
        """
        detection_boxes, confidences = self._to_arrays(detections)
        self._warn_if_frame_unused(frame)
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._result(detections)

        if len(self.tracks) == 0 and len(detections) == 0:
            return self._result(detections)

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        self._predict_tracklets(self.tracks, timing)

        # Ghost-ID prevention: budget-only filter before association.
        # Keeps immature tracks alive for matching; full lifecycle prune runs after.
        _budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        self._prune_lost_tracks(timing)

        # Split indices by confidence threshold
        high_mask = confidences >= self.high_conf_det_threshold
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(~high_mask)[0]
        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]

        # Step 1: associate high-confidence detections to all tracks
        predicted_boxes = np.array([t.get_state_bbox() for t in self.tracks]) if self.tracks else np.empty((0, 4))
        iou_matrix = self.iou.compute(predicted_boxes, high_boxes)
        matched, unmatched_tracks, unmatched_high = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold)

        for row, col in matched:
            track = self.tracks[row]
            track.update(high_boxes[col])
            if (
                track.number_of_successful_consecutive_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = self._allocate_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        # Step 2: associate low-confidence detections to remaining tracks
        remaining_boxes = predicted_boxes[unmatched_tracks] if unmatched_tracks else np.empty((0, 4))
        iou_matrix = self.iou.compute(remaining_boxes, low_boxes)
        matched, _, unmatched_low = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold)

        for row, col in matched:
            track = remaining_tracks[row]
            track.update(low_boxes[col])
            if (
                track.number_of_successful_consecutive_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = self._allocate_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Unmatched low-confidence detections
        for det_local_idx in unmatched_low:
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Spawn new tracks from unmatched high-confidence detections
        self._spawn_new_tracks(
            detection_boxes,
            confidences,
            unmatched_high,
            high_indices,
            out_det_indices,
            out_tracker_ids,
        )

        # Full lifecycle prune: removes immature+unmatched and any remaining expired tracks
        self.tracks = _get_alive_tracklets(
            tracklets=self.tracks,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_frames_without_update=self.maximum_frames_without_update,
            maximum_time_without_update=_budget,
        )

        return self._result(detections, out_det_indices, out_tracker_ids)

    def _get_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        min_similarity_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections to tracks based on Similarity (IoU) using the Jonker-Volgenant algorithm approach with
        no initialization instead of the Hungarian algorithm as mentioned in the SORT paper, but it solves the
        assignment problem in an optimal way.

        Args:
            similarity_matrix: Similarity matrix between tracks (rows) and detections (columns).
            min_similarity_thresh: Minimum similarity threshold for a valid match.

        Returns:
            matched: List of ``(tracker_idx, detection_idx)`` tuples for
                associations that meet the similarity threshold.
            unmatched_tracks: Sorted list of track indices not matched to any
                detection.
            unmatched_detections: Sorted list of detection indices not matched
                to any track.
        """
        matched_indices = []
        n_tracks, n_detections = similarity_matrix.shape
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_detections))

        if n_tracks > 0 and n_detections > 0:
            row_indices, col_indices = linear_sum_assignment(similarity_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if similarity_matrix[row, col] >= min_similarity_thresh:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        # Return sorted lists for deterministic order across CPython versions.
        return matched_indices, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _spawn_new_tracks(
        self,
        detection_boxes: np.ndarray,
        confidences: np.ndarray,
        unmatched_high_local: list[int],
        high_indices: np.ndarray,
        out_det_indices: list[int],
        out_tracker_ids: list[int],
    ) -> None:
        for det_local_idx in unmatched_high_local:
            global_idx = int(high_indices[det_local_idx])
            # Every detection is returned (with tracker_id -1 when untracked),
            # matching the documented contract and SORT's behaviour. Only
            # detections clearing the activation threshold spawn a new track.
            out_det_indices.append(global_idx)
            out_tracker_ids.append(-1)
            if float(confidences[global_idx]) >= self.track_activation_threshold:
                self.tracks.append(
                    ByteTrackTracklet(
                        initial_bbox=detection_boxes[global_idx],
                        state_estimator_class=self.state_estimator_class,
                    )
                )

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.

        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        self._last_timestamp = None
        self._reset_id_allocator()
