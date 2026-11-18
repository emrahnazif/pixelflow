"""
Image and Detection Transformations.

Provides comprehensive transformation functions for images and detections,
organized into two categories:

1. Image Operations (image.py) - Image-only transforms (geometric + enhancement)
2. Detection Operations (detections.py) - Image + detection transforms together

## Usage Patterns

### Image-only transformations:
```python
import pixelflow as pf

# Geometric operations
resized = pf.transform.resize(image, width=640)
rotated = pf.transform.rotate(image, 45)
flipped = pf.transform.flip_horizontal(image)
cropped = pf.transform.crop(image, [100, 50, 500, 400])

# Enhancement operations
enhanced = pf.transform.clahe(image)
gray = pf.transform.to_grayscale(image)
```

### Detection-aware transformations:
```python
# Transform both image and detections together
rotated_img, rotated_detections = pf.transform.rotate_detections(
    image, detections, angle=45
)

flipped_img, flipped_detections = pf.transform.flip_horizontal_detections(
    image, detections
)

cropped_img, cropped_detections = pf.transform.crop_detections(
    image, detections, bbox=[100, 50, 500, 400]
)
```

### Additional detection utilities:
```python
# Align keypoints to horizontal
aligned_img, aligned_detections = pf.transform.rotate_to_align(
    image, detections, 'p0', 'p9', target_angle=0
)

# Add padding to detection bboxes
padded_detections = pf.transform.add_padding(detections, padding=0.1)

# Update bbox from keypoints
updated_detections = pf.transform.update_bbox_from_keypoints(
    detections, keypoint_names=['p0', 'p9', 'p21', 'p3']
)
```

### Automated inverse transformations:
```python
# Apply transform chain; each one is recorded on the detections it moved
img, dets = pf.transform.rotate_detections(image, detections, 45)
img, dets = pf.transform.crop_detections(img, dets, [100, 50, 500, 400])

# Undo all of them, last first, to get coordinates in the original image
restored = pf.transform.inverse_transforms(dets)
annotated = pf.annotate.box(image, restored)

# The history lives on those detections: results of a model run on `img` have none,
# so inverse_transforms cannot map them back.
```
"""

# Import all functions from submodules
from .image import (
    # Geometric operations
    resize,
    rotate,
    flip_horizontal,
    flip_vertical,
    crop,

    # Enhancement operations
    clahe,
    to_grayscale,
    auto_contrast,
    normalize,
    gamma_correction,
    standardize
)

from .detections import (
    rotate_detections,
    flip_horizontal_detections,
    flip_vertical_detections,
    crop_detections,
    crop_around_detections,
    rotate_to_align,
    update_bbox_from_keypoints,
    add_padding
)

from .inverse import (
    inverse_transforms
)

__all__ = [
    # Image-only geometric operations
    'resize',
    'rotate',
    'flip_horizontal',
    'flip_vertical',
    'crop',

    # Image-only enhancement operations
    'clahe',
    'to_grayscale',
    'auto_contrast',
    'normalize',
    'gamma_correction',
    'standardize',

    # Detection-aware operations (always return tuple)
    'rotate_detections',
    'flip_horizontal_detections',
    'flip_vertical_detections',
    'crop_detections',
    'crop_around_detections',

    # Additional detection utilities
    'rotate_to_align',
    'update_bbox_from_keypoints',
    'add_padding',

    # Inverse transformations
    'inverse_transforms',
]
