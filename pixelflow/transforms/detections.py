"""
Detection-Aware Transformations.

Functions that transform both images and detections together, keeping coordinates synchronized.
All functions return (image, detections) tuple except crop_around_detections.

Transforms are non-destructive: the Detections passed in is never modified, and
the returned Detections is an independent deep copy.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Union, Dict, List, Optional
from ..detections.detections import Detections, Detection
from . import image as image_transforms
from ..masks import shift_masks

__all__ = [
    'rotate_detections',
    'flip_horizontal_detections',
    'flip_vertical_detections',
    'crop_detections',
    'crop_around_detections',
    'rotate_to_align',
    'update_bbox_from_keypoints',
    'add_padding'
]


def _bbox_copy(detection: Detection) -> Optional[List[float]]:
    """A copy of the detection's box, or None, for the transform history."""
    return list(detection.bbox) if detection.bbox is not None else None


def _record(detection: Detection, entry: Dict) -> None:
    """Append one entry to the detection's transform history, read by inverse_transforms."""
    if detection.metadata is None:
        detection.metadata = {}
    detection.metadata.setdefault('transforms', []).append(entry)


def _translate_detection(detection: Detection, dx: float, dy: float, canvas_shape=None) -> None:
    """Move every geometry of a detection by ``(dx, dy)``, in place.

    Raster masks are pasted onto a ``canvas_shape`` (``(height, width)``) canvas at
    ``(dx, dy)``, or kept as they are without one. Used by ``crop_detections``, by its
    inverse and by ``SlicedInference``.
    """
    if detection.bbox is not None:
        x1, y1, x2, y2 = detection.bbox
        detection.bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    if detection.keypoints:
        detection.keypoints = [kp.with_xy(kp.x + dx, kp.y + dy) for kp in detection.keypoints]
    if detection.segments:
        detection.segments = [[x + dx, y + dy] for x, y in detection.segments]
    detection.masks = shift_masks(detection.masks, dx, dy, canvas_shape)


def _warp_detection(detection: Detection, matrix: np.ndarray, out_size: Tuple[int, int]) -> None:
    """Apply a 2x3 affine matrix to every geometry of a detection, in place.

    The box becomes the upright envelope of its turned corners; keypoints, outlines and
    polygon masks are moved point by point; raster masks are warped onto an ``out_size``
    (``(height, width)``) canvas. Used by ``rotate_detections`` and by its inverse, with
    the inverted matrix.
    """
    apply = image_transforms._apply_affine

    if detection.bbox is not None:
        x1, y1, x2, y2 = detection.bbox
        corners = apply(matrix, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        detection.bbox = [min(xs), min(ys), max(xs), max(ys)]

    if detection.keypoints:
        points = apply(matrix, [(kp.x, kp.y) for kp in detection.keypoints])
        detection.keypoints = [kp.with_xy(x, y) for kp, (x, y) in zip(detection.keypoints, points)]

    if detection.segments:
        detection.segments = apply(matrix, detection.segments)

    if detection.masks is not None:
        height, width = out_size
        new_masks = []
        for mask in detection.masks:
            if isinstance(mask, np.ndarray):
                new_masks.append(cv2.warpAffine(
                    mask.astype(np.uint8), matrix, (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ).astype(bool))
            elif isinstance(mask, list):
                new_masks.append(apply(matrix, mask))
            else:
                new_masks.append(mask)
        detection.masks = new_masks


def rotate_detections(
    image: np.ndarray,
    detections: Detections,
    angle: float,
    center: Tuple[float, float] = None,
    fillcolor: Tuple[int, int, int] = None,
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Rotate image and update detection coordinates.

    Args:
        image: Input image (H, W, 3) RGB format
        detections: Detections to transform (not modified; new Detections are returned)
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Rotation center (x, y). If None, uses image center.
        fillcolor: Fill color for areas outside original image. If None, uses edge pixels.
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (rotated_image, rotated_detections) tuple

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Rotate 45 degrees
        rotated_img, rotated_dets = pf.transform.rotate_detections(
            image, detections, angle=45
        )
        ```

    Note:
        - Rotation is performed around image center by default
        - Uses edge pixel replication if fillcolor not specified
        - Returns new Detections; the ones passed in are not modified
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)

    rotated_img = image_transforms.rotate(image, angle, center=center, fillcolor=fillcolor)
    rotation_matrix, new_w, new_h = image_transforms._rotation_matrix(h, w, angle, center)

    # Work on a copy so the caller's Detections are never mutated.
    detections = detections.copy()

    for detection in detections:
        bbox_before = _bbox_copy(detection)
        _warp_detection(detection, rotation_matrix, (new_h, new_w))

        if track_metadata:
            _record(detection, {
                'type': 'rotate',
                'angle': angle,
                'center': center,
                'original_size': (h, w),
                # The matrix applied, so the inverse inverts exactly this one.
                'matrix': rotation_matrix.tolist(),
                # A rotated box is the upright envelope of the turned box, so geometry alone
                # cannot give the original back; these let the inverse restore it exactly
                # when the box was not changed in between.
                'bbox_before': bbox_before,
                'bbox_after': _bbox_copy(detection),
            })

    return rotated_img, detections


def flip_horizontal_detections(
    image: np.ndarray,
    detections: Detections,
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Flip image horizontally and update detection coordinates.

    Args:
        image: Input image (H, W, 3) RGB format
        detections: Detections to transform (not modified; new Detections are returned)
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (flipped_image, flipped_detections) tuple

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Flip horizontally
        flipped_img, flipped_dets = pf.transform.flip_horizontal_detections(
            image, detections
        )
        ```

    Note:
        - Left-right mirror transformation
        - Returns new Detections; the ones passed in are not modified
    """
    h, w = image.shape[:2]

    # Flip image using image.py function
    flipped_img = image_transforms.flip_horizontal(image)

    # Work on a copy so the caller's Detections are never mutated.
    detections = detections.copy()

    for detection in detections:
        # Transform bbox
        if detection.bbox is not None:
            x1, y1, x2, y2 = detection.bbox
            detection.bbox = [w - x2, y1, w - x1, y2]

        # Transform keypoints
        if detection.keypoints is not None:
            new_keypoints = []
            for kp in detection.keypoints:
                new_kp = kp.with_xy(w - kp.x, kp.y)
                new_keypoints.append(new_kp)
            detection.keypoints = new_keypoints

        # Transform segments
        if detection.segments is not None:
            detection.segments = [[w - x, y] for x, y in detection.segments]

        # Transform masks
        if detection.masks is not None:
            new_masks = []
            for mask in detection.masks:
                if isinstance(mask, np.ndarray):
                    flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype(bool)
                    new_masks.append(flipped_mask)
                else:
                    if isinstance(mask, list):
                        new_mask = [[w - x, y] for x, y in mask]
                        new_masks.append(new_mask)
                    else:
                        new_masks.append(mask)
            detection.masks = new_masks

        # Track metadata
        if track_metadata:
            _record(detection, {
                'type': 'flip_horizontal',
                'original_size': (h, w)
            })

    return flipped_img, detections


def flip_vertical_detections(
    image: np.ndarray,
    detections: Detections,
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Flip image vertically and update detection coordinates.

    Args:
        image: Input image (H, W, 3) RGB format
        detections: Detections to transform (not modified; new Detections are returned)
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (flipped_image, flipped_detections) tuple

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Flip vertically
        flipped_img, flipped_dets = pf.transform.flip_vertical_detections(
            image, detections
        )
        ```

    Note:
        - Top-bottom mirror transformation
        - Returns new Detections; the ones passed in are not modified
    """
    h, w = image.shape[:2]

    # Flip image using image.py function
    flipped_img = image_transforms.flip_vertical(image)

    # Work on a copy so the caller's Detections are never mutated.
    detections = detections.copy()

    for detection in detections:
        # Transform bbox
        if detection.bbox is not None:
            x1, y1, x2, y2 = detection.bbox
            detection.bbox = [x1, h - y2, x2, h - y1]

        # Transform keypoints
        if detection.keypoints is not None:
            new_keypoints = []
            for kp in detection.keypoints:
                new_kp = kp.with_xy(kp.x, h - kp.y)
                new_keypoints.append(new_kp)
            detection.keypoints = new_keypoints

        # Transform segments
        if detection.segments is not None:
            detection.segments = [[x, h - y] for x, y in detection.segments]

        # Transform masks
        if detection.masks is not None:
            new_masks = []
            for mask in detection.masks:
                if isinstance(mask, np.ndarray):
                    flipped_mask = cv2.flip(mask.astype(np.uint8), 0).astype(bool)
                    new_masks.append(flipped_mask)
                else:
                    if isinstance(mask, list):
                        new_mask = [[x, h - y] for x, y in mask]
                        new_masks.append(new_mask)
                    else:
                        new_masks.append(mask)
            detection.masks = new_masks

        # Track metadata
        if track_metadata:
            _record(detection, {
                'type': 'flip_vertical',
                'original_size': (h, w)
            })

    return flipped_img, detections


def crop_detections(
    image: np.ndarray,
    detections: Detections,
    bbox: List[float],
    track_metadata: bool = True
) -> Tuple[np.ndarray, Detections]:
    """
    Crop image to bounding box and update detection coordinates.

    Args:
        image: Input image (H, W, 3) RGB format
        detections: Detections to transform
        bbox: Crop region [x1, y1, x2, y2] in pixels
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        (cropped_image, cropped_detections) tuple

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Crop to region
        cropped_img, cropped_dets = pf.transform.crop_detections(
            image, detections, bbox=[100, 50, 500, 400]
        )
        ```

    Note:
        - The crop region is clipped to the image
        - Detections whose box does not overlap the crop are dropped
        - Boxes are translated to the crop's origin and cut at its edges
        - Keypoints, segments and polygon masks are translated, not cut: all keypoints are
          kept, so their positions in the list do not change
    """
    # The image crop is clipped to the image, so the detections are measured from the
    # clipped corner too: an unclipped x1 of -50 would shift every box 50 px off.
    x1, y1, x2, y2 = image_transforms._clip_box(bbox, *image.shape[:2])
    cropped_img = image[y1:y2, x1:x2]
    crop_w, crop_h = x2 - x1, y2 - y1

    cropped_detections = Detections()

    for detection in detections:
        # Keep a detection only if its box overlaps the crop by some area; one that merely
        # touches the edge would be clipped to nothing.
        if detection.bbox is not None:
            det_x1, det_y1, det_x2, det_y2 = detection.bbox
            if det_x2 <= x1 or det_x1 >= x2 or det_y2 <= y1 or det_y1 >= y2:
                continue

        new_detection = detection.copy()

        # Raster masks are cut to the crop; everything else is translated, not cut. Every
        # keypoint is kept, even outside the crop, so positions in the list do not change.
        if new_detection.masks is not None:
            new_detection.masks = [
                mask[y1:y2, x1:x2] if isinstance(mask, np.ndarray) else mask
                for mask in new_detection.masks
            ]
        _translate_detection(new_detection, -x1, -y1)

        # The box is cut at the crop's edges, so it describes the part of the object that
        # is in the cropped image.
        if new_detection.bbox is not None:
            bx1, by1, bx2, by2 = new_detection.bbox
            new_detection.bbox = [
                min(max(bx1, 0), crop_w), min(max(by1, 0), crop_h),
                min(max(bx2, 0), crop_w), min(max(by2, 0), crop_h),
            ]

        if track_metadata:
            _record(new_detection, {
                'type': 'crop',
                'bbox': [x1, y1, x2, y2],
                'original_size': image.shape[:2],
                # A box cut at the crop's edge cannot be uncut by geometry; these let the
                # inverse restore it exactly when the box was not changed in between.
                'bbox_before': _bbox_copy(detection),
                'bbox_after': _bbox_copy(new_detection),
            })

        cropped_detections.add_detection(new_detection)

    return cropped_img, cropped_detections


def crop_around_detections(
    image: np.ndarray,
    detections: Detections,
    padding: Union[float, Dict[str, float]] = 0.0
) -> List[np.ndarray]:
    """
    Crop image around each detection's bounding box.

    Args:
        image: Input image (H, W, 3) RGB format
        detections: Detections to crop around
        padding: Padding to add around bbox
            - Float: uniform padding as fraction of bbox shorter side
            - Dict: {'left': 0.2, 'right': 0.2, 'top': 0.1, 'bottom': 0.1}

    Returns:
        List of cropped images, one per detection (same order as detections)

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Crop around each detection with 20% padding
        crops = pf.transform.crop_around_detections(image, detections, padding=0.2)

        # Process each crop
        for crop, detection in zip(crops, detections):
            pf.save_image(f"{detection.class_name}.jpg", crop)
        ```

    Note:
        - Detections themselves are NOT modified (coordinates stay in original space)
        - Crop regions are clipped to image boundaries (no extension)
        - If detection has no bbox, it is skipped
        - No metadata tracking (detections unchanged)
    """
    crops = []
    h, w = image.shape[:2]

    for detection in detections:
        if detection.bbox is None:
            continue

        x1, y1, x2, y2 = detection.bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Calculate padding
        if isinstance(padding, dict):
            pad_left = padding.get('left', 0) * bbox_w
            pad_right = padding.get('right', 0) * bbox_w
            pad_top = padding.get('top', 0) * bbox_h
            pad_bottom = padding.get('bottom', 0) * bbox_h
        else:
            # Uniform padding based on shorter side
            reference = min(bbox_w, bbox_h)
            pad_amount = padding * reference
            pad_left = pad_right = pad_top = pad_bottom = pad_amount

        # Apply padding
        crop_x1 = x1 - pad_left
        crop_y1 = y1 - pad_top
        crop_x2 = x2 + pad_right
        crop_y2 = y2 + pad_bottom

        # Clip to image boundaries (no extension)
        crop_bbox = [
            max(0, crop_x1),
            max(0, crop_y1),
            min(w, crop_x2),
            min(h, crop_y2)
        ]

        # Crop directly using image.py function
        crop = image_transforms.crop(image, crop_bbox)
        crops.append(crop)

    return crops


def rotate_to_align(
    image: np.ndarray,
    detections: Detections,
    point1_name: str,
    point2_name: str,
    target_angle: float = 0.0,
    detection_index: int = 0
) -> Tuple[np.ndarray, Detections]:
    """
    Rotate image so two keypoints form specified angle.

    Internally calls rotate_detections() with calculated angle to align two keypoints
    to a target orientation.

    Args:
        image: Input image (H, W, 3) RGB format
        detections: Detections containing keypoints (not modified; new Detections are returned)
        point1_name: First keypoint name (e.g., 'p0')
        point2_name: Second keypoint name (e.g., 'p9')
        target_angle: Target angle in degrees (0 = horizontal, 90 = vertical)
        detection_index: Which detection to use for alignment (default: first)

    Returns:
        (rotated_image, rotated_detections) with metadata tracking

    Raises:
        ValueError: If keypoints not found or detection index invalid

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Align p0-p9 horizontally
        image, detections = pf.transform.rotate_to_align(
            image, detections, 'p0', 'p9', target_angle=0
        )
        ```

    Note:
        - Keypoints must exist in specified detection
        - Rotation is performed around image center
        - Metadata is automatically tracked
        - Returns new Detections; the ones passed in are not modified
    """
    # Get keypoints
    if detection_index >= len(detections.detections):
        raise ValueError(f"Detection index {detection_index} out of range")

    detection = detections.detections[detection_index]

    if detection.keypoints is None:
        raise ValueError("Detection has no keypoints")

    # Find keypoints by name
    point1 = None
    point2 = None

    for kp in detection.keypoints:
        if kp.name == point1_name:
            point1 = (kp.x, kp.y)
        if kp.name == point2_name:
            point2 = (kp.x, kp.y)

    if point1 is None:
        raise ValueError(f"Keypoint '{point1_name}' not found")
    if point2 is None:
        raise ValueError(f"Keypoint '{point2_name}' not found")

    # Calculate current angle
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    if dx == 0 and dy == 0:
        raise ValueError("Keypoints are at the same location")

    current_angle = math.degrees(math.atan2(dy, dx))

    # Calculate rotation needed
    rotation_angle = target_angle - current_angle

    return rotate_detections(image, detections, rotation_angle)


def update_bbox_from_keypoints(
    detections: Detections,
    keypoint_names: Optional[List[str]] = None,
    track_metadata: bool = True
) -> Detections:
    """
    Update detection bboxes based on keypoints for all detections.

    Calculates bounding box from specified keypoints (or all keypoints if not specified)
    and returns new Detections with each bbox replaced.

    Args:
        detections: Detections to update (not modified; new Detections are returned)
        keypoint_names: Which keypoints to include. If None, uses all keypoints.
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        New Detections with updated bboxes

    Example:
        ```python
        import pixelflow as pf

        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Update bbox from specific keypoints
        detections = pf.transform.update_bbox_from_keypoints(
            detections, keypoint_names=['p0', 'p9', 'p21', 'p3']
        )

        # Update bbox from all keypoints
        detections = pf.transform.update_bbox_from_keypoints(detections)
        ```

    Note:
        - Operates on ALL detections in the container
        - Skips detections without keypoints
        - If keypoint_names specified but not found, detection is skipped
        - Returns a new Detections; the input is left unmodified
    """
    # Work on a copy so the caller's Detections are never mutated.
    detections = detections.copy()

    for detection in detections:
        # Skip if no keypoints
        if detection.keypoints is None or len(detection.keypoints) == 0:
            continue

        # Filter keypoints
        if keypoint_names is not None:
            keypoints = [kp for kp in detection.keypoints if kp.name in keypoint_names]
            if not keypoints:
                # Skip this detection if none of the specified keypoints found
                continue
        else:
            keypoints = detection.keypoints

        # Calculate bbox from keypoints
        xs = [kp.x for kp in keypoints]
        ys = [kp.y for kp in keypoints]
        detection.bbox = [min(xs), min(ys), max(xs), max(ys)]

        # Track metadata
        if track_metadata:
            _record(detection, {
                'type': 'update_bbox_from_keypoints',
                'keypoint_names': keypoint_names
            })

    # A box rebuilt to nothing is no longer a detection; dropping it here keeps
    # the collection's guarantee that every member has geometry.
    return detections._drop_boxless()


def add_padding(
    detections: Detections,
    padding: Union[float, Dict[str, float]],
    reference: str = 'shorter',
    track_metadata: bool = True
) -> Detections:
    """
    Add padding to detection bboxes for all detections.

    Args:
        detections: Detections to pad (not modified; new Detections are returned)
        padding:
            - Float: uniform padding as fraction of reference dimension
            - Dict: {'left': 0.2, 'right': 0.2, 'top': 0.1, 'bottom': 0.1}
        reference: Which dimension to base percentage on
            - 'shorter': Use shorter side (bbox width or height)
            - 'longer': Use longer side
            - 'width': Always use bbox width
            - 'height': Always use bbox height
        track_metadata: If True, writes transform metadata to detection.metadata

    Returns:
        New Detections with padded bboxes

    Example:
        ```python
        import pixelflow as pf

        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # 10% uniform padding based on shorter side
        padded = pf.transform.add_padding(detections, padding=0.1, reference='shorter')

        # Asymmetric padding
        padded = pf.transform.add_padding(
            detections,
            padding={'left': 0.3, 'right': 0.2, 'top': 0, 'bottom': 0}
        )

        # 20% padding based on width
        padded = pf.transform.add_padding(detections, padding=0.2, reference='width')
        ```

    Note:
        - Operates on ALL detections in the container
        - Padding can extend bbox beyond image boundaries
        - Returns a new Detections; the input is left unmodified
    """
    # Work on a copy so the caller's Detections are never mutated.
    detections = detections.copy()

    for detection in detections:
        if detection.bbox is None:
            # No bbox, skip this detection
            continue

        x1, y1, x2, y2 = detection.bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Calculate reference dimension
        if reference == 'shorter':
            ref_dim = min(bbox_w, bbox_h)
        elif reference == 'longer':
            ref_dim = max(bbox_w, bbox_h)
        elif reference == 'width':
            ref_dim = bbox_w
        elif reference == 'height':
            ref_dim = bbox_h
        else:
            raise ValueError(f"Invalid reference: {reference}. Must be 'shorter', 'longer', 'width', or 'height'")

        # Calculate padding
        if isinstance(padding, dict):
            pad_left = padding.get('left', 0) * ref_dim
            pad_right = padding.get('right', 0) * ref_dim
            pad_top = padding.get('top', 0) * ref_dim
            pad_bottom = padding.get('bottom', 0) * ref_dim
        else:
            # Uniform padding
            pad_left = pad_right = pad_top = pad_bottom = padding * ref_dim

        # Apply padding
        detection.bbox = [
            x1 - pad_left,
            y1 - pad_top,
            x2 + pad_right,
            y2 + pad_bottom
        ]

        # Track metadata
        if track_metadata:
            _record(detection, {
                'type': 'add_padding',
                'padding': padding,
                'reference': reference,
                # In pixels, measured on the box before padding. The inverse cannot
                # recompute them from the padded box, whose sides are already longer.
                'pixels': [pad_left, pad_top, pad_right, pad_bottom],
            })

    # A box rebuilt to nothing is no longer a detection; dropping it here keeps
    # the collection's guarantee that every member has geometry.
    return detections._drop_boxless()
