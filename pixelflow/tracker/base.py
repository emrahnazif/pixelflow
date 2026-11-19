# ------------------------------------------------------------------------
# Ported from roboflow/trackers, src/trackers/core/base.py at commit 84569dc.
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
# Changes for PixelFlow: PixelFlow Detections in and out; CLI registry and tuning search space removed;
#   tracker IDs start at 1.
# ------------------------------------------------------------------------

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from pixelflow.detections import Detection, Detections
from pixelflow.tracker.utils.base_tracklet import BaseTracklet
from pixelflow.tracker.utils.predict_timing import PredictTiming


class TrackletProtocol(Protocol):
    """Contract every tracklet in ``BaseTracker.tracks`` must satisfy."""

    tracker_id: int

    def get_state_bbox(self) -> np.ndarray: ...


class BaseTracker(ABC):
    """Base class of the PixelFlow trackers.

    Internally a tracklet without an ID has ``tracker_id == -1``; in the
    ``Detections`` a tracker returns, that becomes ``tracker_id = None``.

    Attributes:
        tracks: List of alive tracklets after each `update()`. Each element
            must satisfy `TrackletProtocol` (exposes `.tracker_id: int` and
            `.get_state_bbox() -> np.ndarray`). Subclasses must initialise
            this as an empty list in `__init__`.
    """

    tracks: list[Any]
    maximum_frames_without_update: int
    maximum_time_without_update: float | None
    _next_track_id: int

    @staticmethod
    def _compute_maximum_frames_without_update(
        lost_track_buffer: int,
        frame_rate: float,
    ) -> int:
        """Scale positive lost-track buffers without changing explicit zero-buffer configs.

        Args:
            lost_track_buffer: Non-negative buffer length expressed in 30 FPS frames.
                Zero means no missed-frame grace period.
            frame_rate: Actual video frame rate in frames per second. Must be
                finite and strictly positive.

        Returns:
            Scaled maximum number of missed frames before a confirmed track expires.
            Returns zero when ``lost_track_buffer`` is zero; otherwise returns
            ``max(1, ceil(frame_rate / 30.0 * lost_track_buffer))`` to ensure at
            least one frame of grace for any positive buffer at any frame rate.

        Raises:
            ValueError: If ``lost_track_buffer`` is negative.
            ValueError: If ``frame_rate`` is not finite or is not strictly positive.
            ValueError: If the scaled product overflows to infinity for extreme inputs.

        Examples:
            >>> BaseTracker._compute_maximum_frames_without_update(30, 30.0)
            30
            >>> BaseTracker._compute_maximum_frames_without_update(30, 60.0)
            60
            >>> BaseTracker._compute_maximum_frames_without_update(3, 15.0)
            2
            >>> BaseTracker._compute_maximum_frames_without_update(0, 30.0)
            0
        """
        if lost_track_buffer < 0:
            raise ValueError("lost_track_buffer must be greater than or equal to 0")
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be a finite positive value")
        if lost_track_buffer == 0:
            return 0
        scaled = frame_rate / 30.0 * lost_track_buffer
        if not math.isfinite(scaled):
            raise ValueError("Scaled lost_track_buffer overflows: frame_rate / 30.0 * lost_track_buffer must be finite")
        return max(1, math.ceil(scaled))

    _frame_rate: float = 1.0
    _last_timestamp: float | None = None

    def _init_timestamp_state(self, frame_rate: float) -> None:
        """Register reference FPS and reset timestamp bookkeeping.

        Call from ``__init__`` on all concrete trackers.

        Args:
            frame_rate: Reference frames per second for bootstrap elapsed time.
        """
        self._frame_rate = frame_rate
        self._last_timestamp = None

    @staticmethod
    def _to_arrays(detections: Detections) -> tuple[np.ndarray, np.ndarray]:
        """Return the boxes ``(N, 4)`` and confidences ``(N,)`` of ``detections`` as arrays.

        A detection without a confidence counts as fully confident (``1.0``), like
        upstream's ``default_confidences``. Boxes are already valid and finite: the
        ``Detection.bbox`` setter checks them.

        Raises:
            TypeError: If ``detections`` is not a ``Detections``.
            ValueError: If a detection has no box (only possible when appended to
                ``.detections`` directly). The check runs before any tracker state
                is advanced, so a failed call leaves the tracker unchanged.
        """
        if not isinstance(detections, Detections):
            raise TypeError(f"update() takes a pixelflow Detections, got {type(detections).__name__}")
        if any(d.bbox is None for d in detections):
            raise ValueError("every detection needs a bbox to be tracked")
        boxes = np.array([d.bbox for d in detections], dtype=np.float64).reshape(-1, 4)
        confidences = np.array(
            [1.0 if d.confidence is None else d.confidence for d in detections], dtype=np.float64
        )
        return boxes, confidences

    @staticmethod
    def _result(detections: Detections, tracker_ids: np.ndarray | None = None) -> Detections:
        """Return a copy of ``detections`` with ``tracker_ids`` set, ``-1`` becoming ``None``.

        The input is not modified, and the result keeps its order. Mask arrays are shared
        with the input rather than copied: the tracker only sets ``tracker_id``, and copying
        full-image masks every frame cost about 9 ms for 30 masks at 1080p. With no
        ``tracker_ids``, every detection gets ``None`` (a skipped update).
        """
        result = Detections()
        for i, detection in enumerate(detections):
            copy = detection.copy(share_masks=True)
            track_id = -1 if tracker_ids is None else int(tracker_ids[i])
            copy.tracker_id = None if track_id == -1 else track_id
            result.add_detection(copy)
        return result

    def _warn_if_frame_unused(self, frame: np.ndarray | None) -> None:
        """Emit a UserWarning when a frame is passed to a tracker that ignores it.

        Subclasses that do not perform camera motion compensation should call this
        at the top of their ``update()`` implementation.

        Args:
            frame: Value passed to ``update(frame=...)``.
        """
        if frame is not None:
            warnings.warn(
                f"{type(self).__name__}.update() received a frame argument but does not use it.",
                UserWarning,
                stacklevel=3,
            )

    def _predict_timing(self, timestamp: float | None) -> PredictTiming:
        """Build predict timing from an optional timestamp.

        All timestamp ordering checks live here: fixed-rate mode, bootstrap,
        backwards (skip whole update), duplicate (skip predict only), normal gap.
        ``_last_timestamp`` advances only on bootstrap and strictly increasing times.
        """
        if timestamp is None:
            self._last_timestamp = None
            return PredictTiming(frame_step=1.0, elapsed_seconds=None)

        if not np.isfinite(timestamp):
            warnings.warn(
                f"{type(self).__name__}: timestamp {timestamp!r} is not finite; skipping update.",
                UserWarning,
                stacklevel=3,
            )
            return PredictTiming(frame_step=0.0, elapsed_seconds=None, skip_update=True)

        last = self._last_timestamp

        if last is None:
            # Bootstrap: no prior timestamp, so we cannot compute t - t_prev.
            # Use one nominal frame period (1 / frame_rate) so the first Kalman
            # step is frame_step=1.0 — matching fixed-rate behaviour rather than
            # using the absolute timestamp value (e.g. 37.2 s would break tuning).
            self._last_timestamp = timestamp
            elapsed = 1.0 / self._frame_rate
            return PredictTiming(
                frame_step=elapsed * self._frame_rate,
                elapsed_seconds=elapsed,
                frame_rate=self._frame_rate,
            )

        if timestamp < last:
            warnings.warn(
                f"{type(self).__name__}: timestamp {timestamp} is earlier than the "
                f"previous timestamp {last}. Skipping update; pass capture times in "
                "non-decreasing order.",
                UserWarning,
                stacklevel=3,
            )
            return PredictTiming(frame_step=0.0, elapsed_seconds=None, skip_update=True)

        if timestamp == last:
            warnings.warn(
                f"{type(self).__name__}: duplicate timestamp {timestamp}; skipping predict for this step.",
                UserWarning,
                stacklevel=3,
            )
            return PredictTiming(frame_step=0.0, elapsed_seconds=0.0)

        elapsed = timestamp - last
        self._last_timestamp = timestamp
        return PredictTiming(
            frame_step=elapsed * self._frame_rate,
            elapsed_seconds=elapsed,
            frame_rate=self._frame_rate,
        )

    def _predict_tracklets(
        self,
        tracklets: Sequence[BaseTracklet],
        timing: PredictTiming,
        *,
        return_predictions: bool = False,
    ) -> dict[BaseTracklet, np.ndarray]:
        """Predict all tracklets and optionally return states keyed by object identity.

        Args:
            tracklets: Tracklets to advance one predict step.
            timing: Predict-step timing; ``timing.skip_predict`` short-circuits
                to a no-op when the timestamp did not advance.
            return_predictions: When ``True``, return each tracklet's predicted
                bounding box keyed by the tracklet object itself. When
                ``False``, predict in place and discard the results.

        Returns:
            A dict mapping each tracklet object to its predicted bounding box
            when ``return_predictions`` is ``True`` and the predict step ran.
            The dict holds a strong reference to every key, so entries stay
            valid regardless of what the caller does with ``tracklets``
            afterwards. Returns ``{}`` on three distinct paths — a skipped
            predict step (``timing.skip_predict``), ``return_predictions=False``,
            or an empty ``tracklets`` argument — so callers should not infer
            "predict was skipped" from an empty dict alone; check
            ``timing.skip_predict`` directly for that.
        """
        if timing.skip_predict:
            return {}
        if return_predictions:
            return {tracklet: tracklet.predict(timing) for tracklet in tracklets}
        for tracklet in tracklets:
            tracklet.predict(timing)
        return {}

    def _lost_track_time_budget(
        self,
        timing: PredictTiming,
        seconds_budget: float | None,
    ) -> float | None:
        """Return the seconds lost-track budget when timestamps are in use."""
        return seconds_budget if timing.uses_elapsed_time else None

    def _prune_lost_tracks(self, timing: PredictTiming) -> None:
        """Remove tracks that exceed their lost-track budget (ghost-ID prevention).

        Applies a budget-only filter so immature tracks stay alive for matching. Call after ``_predict_tracklets`` and
        before association.

        At fixed frame rate (no timestamps) this is a no-op — the frame-count budget is enforced post-association,
        preserving the last-frame re-association opportunity that the original trackers relied on.  In variable-FPS mode
        the time budget can differ from the frame budget, so expired-by-time tracks are removed here before they can be
        matched and revived with a stale ID.
        """
        budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        if budget is None:
            return
        self.tracks = [
            t
            for t in self.tracks
            if BaseTracklet.within_lost_track_budget(
                t,
                maximum_frames_without_update=self.maximum_frames_without_update,
                maximum_time_without_update=budget,
            )
        ]

    @abstractmethod
    def update(
        self,
        detections: Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> Detections:
        """Process new detections and assign track IDs.

        Matches incoming detections to existing tracks, creates new tracks
        for unmatched detections, and handles track lifecycle management.

        Args:
            detections: Current frame detections; each needs a ``bbox``.
            frame: Current video frame, or ``None``. Used by trackers with camera
                motion compensation; the others warn and ignore it.
            timestamp: Absolute time of the current frame in seconds, or
                ``None`` for fixed-rate mode (``frame_step = 1.0`` per call).
                When provided, elapsed seconds are converted to Kalman frame
                units via ``* frame_rate``; pruning uses seconds directly. Must
                be non-decreasing in capture time. Passing ``None`` resets the
                internal timestamp anchor so the next timestamped call is
                treated as a fresh bootstrap.

        Returns:
            A copy of ``detections``, in the same order, with ``tracker_id`` set
            on each detection that belongs to a confirmed track and ``None`` on
            the others. When the update is skipped (backwards or non-finite
            timestamp), every ``tracker_id`` is ``None``.

        Raises:
            TypeError: If ``detections`` is not a ``Detections``.
            ValueError: If a detection has no box, or a box contains NaN or inf.

        Warns:
            UserWarning: If ``timestamp`` is earlier than the previous call
                (backwards order); the whole update is skipped. If ``timestamp``
                equals the previous call (duplicate); predict is skipped but
                association still runs on the last state.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal tracking state.

        Call between videos or when tracking should restart from scratch.
        """

    def _reset_id_allocator(self) -> None:
        """Restart this tracker instance's ID allocation from 1."""
        self._next_track_id = 1

    def _allocate_tracker_id(self) -> int:
        """Return the next tracker ID and advance the internal counter."""
        next_track_id = self._next_track_id
        self._next_track_id = next_track_id + 1
        return next_track_id

    @property
    def tracked_objects(self) -> Detections:
        """All confirmed alive tracks, with the box the Kalman filter predicts for each.

        Includes tracks not matched to a detection on the current frame (for example
        hidden or missed by the detector), until they have been lost for longer than
        ``lost_track_buffer``. Each detection has only ``bbox`` and ``tracker_id``:
        a predicted box has no confidence or class.
        """
        result = Detections()
        for tracklet in self.tracks:
            if tracklet.tracker_id != -1:
                bbox = [float(v) for v in tracklet.get_state_bbox()]
                result.add_detection(Detection(bbox=bbox, tracker_id=tracklet.tracker_id))
        return result
