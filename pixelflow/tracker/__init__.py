"""Multi-object trackers: give each object the same ``tracker_id`` from frame to frame.

``SORT``, ``ByteTrack``, ``OCSORT``, ``BoTSORT`` and ``CBIoU`` are ported from
roboflow/trackers (Apache-2.0); see ``pixelflow/tracker/UPSTREAM.md`` for the commit
and what changed.

Example:
    >>> import pixelflow as pf
    >>> tracker = pf.tracker.ByteTrack()
    >>> for detections in per_frame_detections:  # doctest: +SKIP
    ...     detections = tracker.update(detections)
"""

from pixelflow.tracker.base import BaseTracker
from pixelflow.tracker.botsort.tracker import BoTSORT
from pixelflow.tracker.bytetrack.tracker import ByteTrack
from pixelflow.tracker.cbiou.tracker import CBIoU
from pixelflow.tracker.ocsort.tracker import OCSORT
from pixelflow.tracker.sort.tracker import SORT
from pixelflow.tracker.utils.iou import BIoU, BaseIoU, CIoU, DIoU, GIoU, IoU
from pixelflow.tracker.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)

__all__ = [
    "BaseTracker",
    "ByteTrack",
    "SORT",
    "OCSORT",
    "BoTSORT",
    "CBIoU",
    "BaseIoU",
    "IoU",
    "BIoU",
    "GIoU",
    "DIoU",
    "CIoU",
    "BaseStateEstimator",
    "XYXYStateEstimator",
    "XCYCSRStateEstimator",
    "XCYCWHStateEstimator",
]
