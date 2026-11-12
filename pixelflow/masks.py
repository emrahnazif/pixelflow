"""Raster masks to polygons.

Segmentation models answer in pixels - a boolean ``H x W`` raster per object - while
everything that stores, edits or transmits an outline wants a polygon. This module is
the bridge, and it settles the one question a bridge has to settle: where exactly a
polygon vertex sits relative to the pixels it encloses.

The convention is pixel corners. Pixel ``(row i, column j)`` covers the square from
``x = j`` to ``x = j + 1`` and ``y = i`` to ``y = i + 1``, and the polygon runs along
the outside edges of the pixels it encloses. A mask covering columns 10..19 therefore
spans ``x = 10`` to ``x = 20``, which is the same box ``[x1, y1, x2, y2]`` describes
for that object. Tracing through pixel centres instead - what ``cv2.findContours``
returns - shrinks every object by half a pixel on each side and turns a one-pixel-wide
line into a polygon of zero area.
"""

import cv2
import numpy as np

from pixelflow.validators import validate_segments

__all__ = ["mask_to_polygon", "simplify_polygon"]


def _trace_outline(region: np.ndarray) -> np.ndarray:
    """
    Trace the outer boundary of one connected region along its pixel corners.

    Returns ``(N, 2)`` int32 ``(x, y)`` points in doubled coordinates: pixel corners
    land on even numbers and the midpoints of pixel edges on odd ones, so the caller
    halves the result. Staying in integers until then keeps every vertex exact.
    """
    height, width = region.shape

    # Each pixel becomes a 3x3 block on a (2H+1, 2W+1) grid: its centre at an
    # odd/odd position, dilated out to its own corners. Neighbouring pixels share
    # the grid points on their common edge, so the filled shape is the region's
    # true footprint and its border passes through real pixel corners.
    grid = np.zeros((2 * height + 1, 2 * width + 1), dtype=np.uint8)
    grid[1::2, 1::2] = region
    grid = cv2.dilate(grid, np.ones((3, 3), dtype=np.uint8))

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len)[:, 0, :]

    # OpenCV's border follower steps diagonally across a concave corner, from one
    # edge midpoint to the next, because the corner itself has no background among
    # its four neighbours. Every diagonal step is such a cut.
    following = np.roll(points, -1, axis=0)
    diagonal = np.abs(following - points).sum(axis=1) == 2
    start, end = points[diagonal], following[diagonal]
    # Each midpoint has one even coordinate; the corner takes the even one of each.
    corners = np.where(start % 2 == 0, start, end)
    return np.insert(points, np.nonzero(diagonal)[0] + 1, corners, axis=0)


def _simplify_outline(outline: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Douglas-Peucker on a closed outline, with its four extreme vertices pinned.

    Simplifying the loop as a whole is free to drop the leftmost or topmost vertex
    whenever it lies within `epsilon` of the result, which pulls the polygon's
    bounding box inside the mask's. Cutting the loop at one vertex per side and
    simplifying each arc between them keeps those vertices, so the box around the
    polygon is exactly the box around the mask at any tolerance.
    """
    xs, ys = outline[:, 0], outline[:, 1]

    def extreme(primary, secondary):
        # First along one axis, ties broken along the other.
        candidates = np.flatnonzero(primary == primary.min())
        return int(candidates[np.argmin(secondary[candidates])])

    # Each side is a straight run of points; taking the end of the run rather than
    # any point on it pins a real corner, so a rectangle keeps exactly four.
    anchors = sorted([
        extreme(xs, ys),    # leftmost, then topmost
        extreme(ys, -xs),   # topmost, then rightmost
        extreme(-xs, -ys),  # rightmost, then bottommost
        extreme(-ys, xs),   # bottommost, then leftmost
    ])

    arcs = []
    for start, end in zip(anchors, anchors[1:] + [anchors[0] + len(outline)]):
        # Both end anchors included; the last arc wraps round to the first anchor.
        arc = outline.take(np.arange(start, end + 1), axis=0, mode="wrap")
        # An open curve keeps both end points; the last one opens the next arc.
        arcs.append(cv2.approxPolyDP(arc.reshape(-1, 1, 2), epsilon, False)[:-1, 0, :])
    return np.concatenate(arcs)


def mask_to_polygon(mask, tolerance: float = 1.5):
    """
    Convert one raster mask into one polygon outlining its largest region.

    Traces the outer boundary of the largest connected region along its pixel
    corners, then reduces the vertex count with Douglas-Peucker simplification.
    The result is in the shape `Detection.segments` holds, so it can be assigned
    directly.

    Args:
        mask: 2D array of shape `(H, W)`. Non-zero pixels are the object, so a
              boolean mask and a 0/255 uint8 mask behave the same. A probability
              map must be thresholded by the caller first.
        tolerance (float): Maximum distance in pixels a simplified edge may stray
                          from the traced outline. Range: [0.0, inf). `0` keeps
                          the exact pixel staircase and only removes points on
                          straight runs. Default is 1.5.

    Returns:
        Optional[List[List[float]]]: The polygon as `[[x, y], ...]` in the mask's own
            pixel coordinates, not closed (the first point is not repeated at the
            end). None if the mask is empty.

    Raises:
        ValueError: If `mask` is not 2D, or `tolerance` is negative.

    Example:
        >>> import numpy as np
        >>> import pixelflow as pf
        >>>
        >>> mask = np.zeros((480, 640), dtype=bool)
        >>> mask[100:200, 50:300] = True
        >>>
        >>> # A rectangle comes back as its four pixel corners
        >>> pf.mask_to_polygon(mask)
        [[50.0, 100.0], [50.0, 200.0], [300.0, 200.0], [300.0, 100.0]]
        >>>
        >>> # Exact outline, every stair of the pixel boundary kept
        >>> outline = pf.mask_to_polygon(mask, tolerance=0)
        >>>
        >>> # Assign straight to a detection
        >>> detection = pf.Detection(bbox=[50, 100, 300, 200])
        >>> detection.segments = pf.mask_to_polygon(mask)

    Notes:
        - Coordinates are pixel corners: a mask covering columns 10..19 spans
          x = 10 to x = 20, so the polygon's bounding box equals the mask's
        - One polygon per mask. Holes are filled and every region but the largest
          is dropped, because `segments` holds a single outline
        - Regions are 4-connected: pixels touching only at a corner are separate
          regions. A region that pinches to a single corner still yields one
          outline, which touches itself at that corner
        - Simplification only removes vertices, and never the outermost one on
          each side, so the remaining points are exact pixel corners and the
          bounding box is exact at any tolerance
        - The pixel staircase on a slanted edge is itself about one pixel of
          noise. Below a tolerance of roughly 1.5 the stairs survive and a large
          object keeps hundreds of points; at 1.5 it keeps tens
        - No object is too small: a one-pixel mask returns the four corners of
          that pixel at any tolerance, because the extreme vertices are kept

    Performance Notes:
        - One pass over the frame finds the occupied area; all other work is
          proportional to that area, not the image
        - In a 1920x1281 mask: well under 1 ms for a 20 px object, around 5 ms
          for one 1000 px across
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W), got shape {mask.shape}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")

    if mask.dtype != np.bool_:
        mask = mask != 0
    # A boolean array is already the 0/1 bytes OpenCV wants, so view it, don't copy it.
    binary = np.ascontiguousarray(mask).view(np.uint8)

    # Crop to the occupied part of the frame before anything else: an object is
    # usually a small part of the image, and every later step costs by the pixel.
    left, top, width, height = cv2.boundingRect(binary)
    if width == 0:
        return None
    binary = binary[top:top + height, left:left + width]

    # Label 0 is the background, so the regions start at 1.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height = (int(v) for v in stats[largest, :4])
    region = (labels[y:y + height, x:x + width] == largest).astype(np.uint8)

    outline = _simplify_outline(_trace_outline(region), 2.0 * tolerance)  # doubled, like the points
    return validate_segments(outline / 2.0 + (left + x, top + y))


def simplify_polygon(polygon_points, tolerance: float = 2.0, preserve_topology: bool = True):
    """
    Reduce the number of points in one polygon with Douglas-Peucker simplification.

    Args:
        polygon_points: The polygon as a sequence of (x, y) points. A closing point
                        that repeats the first one is accepted and dropped.
        tolerance (float): Maximum distance in pixels a simplified edge may stray from
                           the original outline (higher = more simplified).
        preserve_topology (bool): Accepted for backwards compatibility and ignored.
                                  Plain Douglas-Peucker can, on a thin shape and a
                                  large tolerance, produce an outline that crosses
                                  itself.

    Returns:
        List[List[float]]: The simplified polygon as ``[[x, y], ...]`` at the shared
            precision, not closed.
    """
    points = np.asarray(polygon_points, dtype=np.float32).reshape(-1, 2)
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) > 3:
        # Douglas-Peucker only ever drops vertices, so the survivors are the caller's
        # own points; validate_segments rounds the float32 artefacts back out of them.
        points = cv2.approxPolyDP(points.reshape(-1, 1, 2), float(tolerance), True)[:, 0, :]
    return validate_segments(points)
