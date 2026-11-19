"""
Temporal smoothing of tracked detections, using a Buffer's past and future frames.

``smooth`` averages each tracked object's box over the frames around the one being smoothed,
which removes the frame-to-frame jitter of detection boxes, and fills in an object the model
missed in that frame when it was found before and after it.
"""

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pixelflow.detections import Detection, Detections
from pixelflow.transforms.detections import _translate_detection

if TYPE_CHECKING:
    from pixelflow.buffer import Buffer

# One entry per sighting of a tracker id in the buffer: (offset from the smoothed frame, detection).
_Sighting = Tuple[int, Detection]


def smooth(buffer: 'Buffer', temporal_weight_decay: float = 0.8) -> Detections:
    """
    Smooth the detections of the frame the buffer last returned.

    Each tracked detection's box (and confidence) becomes a weighted average of the same
    object's boxes in the buffer: weight 1 for the frame itself, ``temporal_weight_decay`` for
    the frames next to it, ``temporal_weight_decay ** 2`` for the ones after, and so on. An
    object missing from the frame but found both before and after it is filled in.

    Args:
        buffer (Buffer): The buffer holding the frames' results, as ``Detections`` with
            tracker ids. ``update`` or ``flush`` must have returned a frame.
        temporal_weight_decay (float): Weight of each step away from the smoothed frame.
            Range: [0.1, 1.0]. Default is 0.8.

    Returns:
        Detections: A new ``Detections`` (the buffer is never changed) holding a copy of each
        detection of the frame, in order: tracked ones with box and confidence smoothed and
        every other field kept (mask arrays shared), untracked ones unsmoothed. Then the
        filled-in detections: copies of the object's nearest earlier detection, moved (box,
        masks, outline, keypoints) by distance between its nearest detections before and
        after, with 85% of their average confidence (None if either has none), no line
        crossings, and ``metadata["interpolated"] = True`` and
        ``metadata["interpolation_source"] = "buffer_smoother"``.

        Without a full window (near the start and end of a video), or with a warning when no
        detection in the buffer has a tracker id, the copies are unsmoothed; before the buffer
        has returned a frame, the result is empty. Tracker ids are assumed unique in a frame.

    Raises:
        ValueError: If ``temporal_weight_decay`` is outside [0.1, 1.0].

    Example:
        >>> import pixelflow as pf
        >>>
        >>> tracker = pf.tracker.ByteTrack()
        >>> buffer = pf.Buffer(frames=5)  # 2 past + current + 2 future
        >>> for frame in pf.VideoReader("video.mp4"):
        ...     results = tracker.update(model_results(frame))
        ...     results, frame = buffer.update(results, frame)
        ...     if frame is None:
        ...         continue
        ...     smoothed = pf.smooth(buffer)
        >>> for results, frame in buffer.flush():
        ...     smoothed = pf.smooth(buffer)  # the last frames come back unsmoothed

    See Also:
        Buffer : Temporal frame buffer for storing detection history
    """
    if not 0.1 <= temporal_weight_decay <= 1.0:
        raise ValueError(
            f"temporal_weight_decay must be between 0.1 and 1.0, got {temporal_weight_decay}"
        )

    context = buffer.get_temporal_context()
    if context is None:
        return Detections()
    current = context['current_results']

    # A full window: everything the buffer holds before and after the frame (an even buffer
    # holds one frame fewer before it than after it).
    if (len(context['past_results']) < buffer.buffer_size - 1 - buffer.delay
            or len(context['future_results']) < buffer.delay):
        return _copy_all(current)

    sightings = _sightings_by_tracker_id(context)
    if len(current) > 0 and not sightings:
        warnings.warn(
            "No detections have tracker_id. Smoothing matches each detection to "
            "itself across frames, which requires tracking. "
            "Make sure to use results = tracker.update(results) before buffering.",
            UserWarning,
            stacklevel=2,
        )
        return _copy_all(current)

    result = Detections()
    for detection in current:
        seen = sightings.get(detection.tracker_id)
        result.add_detection(
            _smoothed(detection, seen, temporal_weight_decay) if seen else detection.copy(share_masks=True)
        )

    present = {d.tracker_id for d in current}
    for tracker_id, seen in sightings.items():
        if tracker_id not in present:
            filled = _filled_in(seen)
            if filled is not None:
                result.add_detection(filled)
    return result


def _copy_all(detections: Detections) -> Detections:
    """A new ``Detections`` holding a copy of each detection (mask arrays shared)."""
    result = Detections()
    for detection in detections:
        result.add_detection(detection.copy(share_masks=True))
    return result


def _sightings_by_tracker_id(context: Dict) -> Dict[int, List[_Sighting]]:
    """Every tracked detection of the buffer, grouped by tracker id, with its frame offset.

    Offsets are negative for past frames (-1 is the frame just before), 0 for the smoothed
    frame and positive for future frames.
    """
    sightings: Dict[int, List[_Sighting]] = defaultdict(list)
    past = context['past_results']
    frames = list(enumerate(past, start=-len(past)))
    frames.append((0, context['current_results']))
    frames.extend(enumerate(context['future_results'], start=1))
    for offset, detections in frames:
        for detection in detections:
            if detection.tracker_id is not None and detection.bbox is not None:
                sightings[detection.tracker_id].append((offset, detection))
    return sightings


def _smoothed(detection: Detection, seen: List[_Sighting], decay: float) -> Detection:
    """A copy of ``detection`` with its box and confidence averaged over ``seen``."""
    weights = [decay ** abs(offset) for offset, _ in seen]
    total = sum(weights)
    bbox = [
        sum(w * d.bbox[i] for w, (_, d) in zip(weights, seen)) / total for i in range(4)
    ]
    rated = [(w, d.confidence) for w, (_, d) in zip(weights, seen) if d.confidence is not None]

    smoothed = detection.copy(share_masks=True)
    smoothed.bbox = bbox
    if rated:
        smoothed.confidence = sum(w * c for w, c in rated) / sum(w for w, _ in rated)
    return smoothed


def _filled_in(seen: List[_Sighting]) -> Optional[Detection]:
    """The detection expected in the smoothed frame between the nearest sightings around it.

    The box is placed by distance: with the object at offset -2 and +1, it goes two thirds of
    the way. Masks, outline and keypoints move with the box's center.
    """
    before = [s for s in seen if s[0] < 0]
    after = [s for s in seen if s[0] > 0]
    if not before or not after:
        return None
    t_before, previous = max(before, key=lambda s: s[0])
    t_after, following = min(after, key=lambda s: s[0])

    share = -t_before / (t_after - t_before)
    bbox = [p + (f - p) * share for p, f in zip(previous.bbox, following.bbox)]

    filled = previous.copy(share_masks=True)
    dx = (bbox[0] + bbox[2] - previous.bbox[0] - previous.bbox[2]) / 2
    dy = (bbox[1] + bbox[3] - previous.bbox[1] - previous.bbox[3]) / 2
    raster = next((m for m in filled.masks or () if hasattr(m, 'shape')), None)
    _translate_detection(filled, dx, dy, canvas_shape=None if raster is None else raster.shape)
    filled.bbox = bbox

    if previous.confidence is not None and following.confidence is not None:
        filled.confidence = (previous.confidence + following.confidence) / 2 * 0.85
    else:
        filled.confidence = None
    filled.line_crossings = []
    filled.metadata = {
        **(previous.metadata or {}),
        'interpolated': True,
        'interpolation_source': 'buffer_smoother',
    }
    return filled
