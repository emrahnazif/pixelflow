"""
Automated Inverse Transformations.

Automatically reverses detection transforms by reading transform history from metadata.
Supports chained transforms with LIFO (last-in-first-out) reversal order.
"""

import warnings

import cv2
import numpy as np

from ..detections.detections import Detections
from .detections import _translate_detection, _warp_detection
from .image import _rotation_matrix

__all__ = ['inverse_transforms']


def inverse_transforms(
    detections: Detections,
    clear_history: bool = True
) -> Detections:
    """
    Automatically reverse all transforms to map detections to original image space.

    Reads transform history from detection metadata and applies inverse operations
    in reverse order (LIFO) to undo all transformations. Essential for production
    pipelines where transforms are applied before inference and results must be
    mapped back to original image coordinates.

    Args:
        detections: Detections with transform history in metadata['transforms']
        clear_history: If True, clears transform history after inverse (default: True)

    Returns:
        New Detections object with coordinates in original image space

    Example:
        ```python
        import pixelflow as pf

        image = pf.read_image("image.jpg")
        detections = pf.from_mayaku(predictor("image.jpg"), labels=predictor.class_names)

        # Apply transform chain
        img, dets = pf.transform.rotate_detections(image, detections, 45)
        img, dets = pf.transform.crop_detections(img, dets, [100, 50, 500, 400])
        img, dets = pf.transform.flip_horizontal_detections(img, dets)

        # Undo all of them, last first
        restored = pf.transform.inverse_transforms(dets)
        annotated = pf.annotate.box(image, restored)
        ```

        The history lives on the detections that went through the transforms. Results of
        a model run on the transformed image have none, so there is nothing to invert.

    Note:
        - Transforms are reversed in LIFO order (last applied, first undone)
        - Skips non-spatial transforms (e.g., update_bbox_from_keypoints)
        - Returns new Detections; the ones passed in are not modified
        - Boxes come back exactly when they were not changed between the transforms and
          the inverse. A box changed after a rotation comes back as the upright box around
          the turned-back box; one changed after a crop comes back cut where it was cut
        - Clear history to prevent double-inversion if applied twice
    """
    # Create deep copy to avoid modifying original
    inverse_detections = detections.copy()

    # Process each detection
    for detection in inverse_detections:
        # Check if detection has transform history
        if detection.metadata is None or 'transforms' not in detection.metadata:
            continue

        transforms = detection.metadata['transforms']
        if not transforms:
            continue

        # Reverse transforms in LIFO order (last applied, first undone)
        for transform in reversed(transforms):
            transform_type = transform.get('type')
            unchanged = _bbox_unchanged(detection.bbox, transform.get('bbox_after'))

            if transform_type == 'rotate':
                _inverse_rotate(detection, transform)

            elif transform_type == 'flip_horizontal':
                _inverse_flip_horizontal(detection, transform)

            elif transform_type == 'flip_vertical':
                _inverse_flip_vertical(detection, transform)

            elif transform_type == 'crop':
                _inverse_crop(detection, transform)

            elif transform_type == 'add_padding':
                _inverse_add_padding(detection, transform)

            elif transform_type == 'update_bbox_from_keypoints':
                # Skip - not a spatial transform, no inverse needed
                pass

            # Rotations and crops lose box information (envelope, cut). When the box is
            # still the one the transform produced, the recorded original is exact.
            if unchanged and transform.get('bbox_before') is not None:
                detection.bbox = list(transform['bbox_before'])

        # Clear transform history if requested
        if clear_history:
            detection.metadata['transforms'] = []

    return inverse_detections


def _bbox_unchanged(bbox, recorded, tolerance=1e-6):
    """Whether ``bbox`` is still the box a transform recorded as its output."""
    if bbox is None or recorded is None:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(bbox, recorded))


def _inverse_rotate(detection, metadata):
    """Undo a rotation by warping with the inverse of the matrix the rotation applied."""
    h, w = metadata['original_size']
    if 'matrix' in metadata:
        forward = np.array(metadata['matrix'], dtype=np.float64)
    else:
        # History recorded before the matrix was: rebuild it the way rotate_detections did.
        forward, _, _ = _rotation_matrix(h, w, metadata['angle'], metadata['center'])
    _warp_detection(detection, cv2.invertAffineTransform(forward), (h, w))


def _inverse_flip_horizontal(detection, metadata):
    """Inverse horizontal flip - flip is self-inverse."""
    original_size = metadata['original_size']
    h, w = original_size

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


def _inverse_flip_vertical(detection, metadata):
    """Inverse vertical flip - flip is self-inverse."""
    original_size = metadata['original_size']
    h, w = original_size

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


def _inverse_crop(detection, metadata):
    """Inverse crop - add the crop offset back; raster masks go back to the original size."""
    x1, y1 = metadata['bbox'][0], metadata['bbox'][1]
    _translate_detection(detection, x1, y1, canvas_shape=metadata.get('original_size'))


def _inverse_add_padding(detection, metadata):
    """Inverse add_padding - subtract the padding, in the pixels add_padding recorded."""
    if detection.bbox is None:
        return
    pixels = metadata.get('pixels')
    if pixels is None:
        # The padding cannot be recomputed from the padded box, whose sides are longer.
        warnings.warn(
            "add_padding history without its pixel amounts (written by an older version) "
            "cannot be undone exactly; the box is left padded.",
            stacklevel=3,
        )
        return
    pad_left, pad_top, pad_right, pad_bottom = pixels
    x1, y1, x2, y2 = detection.bbox
    detection.bbox = [x1 + pad_left, y1 + pad_top, x2 - pad_right, y2 - pad_bottom]
