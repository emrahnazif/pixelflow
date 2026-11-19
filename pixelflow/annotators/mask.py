from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..detections import Detections

import cv2
import numpy as np
from ..colors import _get_color_for_prediction


def mask(
    frame: np.ndarray,
    detections: 'Detections',
    opacity: float = 0.5,
    colors: Optional[List[tuple]] = None
) -> np.ndarray:
    """
    Overlay segmentation masks on a video frame with support for both binary masks and polygon formats.
    
    Provides efficient mask visualization using optimized single-pass rendering and OpenCV's 
    accelerated blending operations. Supports both binary mask arrays and polygon coordinate 
    lists, with automatic format detection and conversion for seamless integration.
    
    Args:
        frame (np.ndarray): Input video frame in RGB format (height, width, 3).
                           Modified in-place with overlaid masks.
        detections (Detections): Detections object containing mask data.
                                Each detection's masks attribute can contain binary arrays
                                or polygon coordinate lists.
        opacity (float): Opacity level for blending masks with the frame.
                        Range: [0.0, 1.0]. Default is 0.5 (semi-transparent).
                        Value of 1.0 creates opaque masks, 0.0 makes them invisible.
        colors (Optional[List[tuple]]): List of RGB color tuples to override default colors.
                                       Colors are mapped to unique class_ids in order of appearance.
                                       If None, uses default ColorManager colors.
        
    Returns:
        np.ndarray: Frame with masks overlaid using the specified opacity.
                   The input frame is modified in-place for memory efficiency.
    
    Raises:
        ValueError: If binary mask dimensions do not match frame dimensions.
        AttributeError: If detection objects lack required 'masks' attribute.
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Load image and get segmentation predictions
        >>> image = cv2.imread("path/to/image.jpg")
        >>> model = YOLO("yolo11n-seg.pt")  # Segmentation model
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> detections = pf.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Apply masks with default semi-transparent overlay
        >>> annotated = pf.annotators.mask(image, detections)
        >>> 
        >>> # Use custom colors for specific classes
        >>> custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        >>> annotated = pf.annotators.mask(image, detections, colors=custom_colors)
        >>> 
        >>> # Create opaque masks for clear segmentation boundaries
        >>> annotated = pf.annotators.mask(image, detections, opacity=1.0)
        >>> 
        >>> # Subtle overlay for background preservation
        >>> annotated = pf.annotators.mask(image, detections, opacity=0.2)
    
    Notes:
        - Input frame is modified in-place for memory efficiency
        - Supports both binary mask arrays (boolean or uint8) and polygon coordinate lists
        - Binary masks must match frame dimensions exactly
        - Polygon coordinates are automatically converted to binary masks using cv2.fillPoly
        - Where masks overlap, the later detection's colour is drawn; each pixel is
          blended once
        - Empty or None mask data is automatically skipped

    Performance Notes:
        - Drawing and blending run inside each mask's bounding rectangle, so the cost
          follows the area the masks cover, not the frame size times the mask count
        - Direct pixel assignment used for full opacity (opacity=1.0) to bypass blending
    """
    # Every step works inside each mask's bounding rectangle, not the whole frame: an
    # object usually covers a small part of the frame, and full-frame boolean indexing
    # per mask was the cost (about 7 ms per mask at 1080p).
    height, width = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    has_mask = np.zeros((height, width), dtype=bool)
    rects = []  # (y0, y1, x0, x1) of each drawn mask

    # Draw all masks to the overlay; where masks overlap, the later one's colour wins
    for result in detections:
        if result.masks is None:
            continue

        color = _get_color_for_prediction(result, colors)

        for mask_data in result.masks:
            local = None  # the mask inside its rectangle

            if isinstance(mask_data, np.ndarray):
                # Binary mask format
                if mask_data.shape[:2] != (height, width):
                    raise ValueError(f"Mask dimensions {mask_data.shape[:2]} do not match frame dimensions {frame.shape[:2]}.")
                binary_mask = mask_data if mask_data.dtype == bool else mask_data.astype(bool)
                x, y, w, h = cv2.boundingRect(np.ascontiguousarray(binary_mask).view(np.uint8))
                if w and h:
                    local = binary_mask[y:y + h, x:x + w]

            elif isinstance(mask_data, list) and len(mask_data) > 0:
                # Polygon format - rasterise into a canvas the size of its rectangle
                points = np.array(mask_data, dtype=np.int32)
                if len(points.shape) == 2 and points.shape[1] == 2:
                    x = int(np.clip(points[:, 0].min(), 0, width))
                    y = int(np.clip(points[:, 1].min(), 0, height))
                    w = int(np.clip(points[:, 0].max() + 1, 0, width)) - x
                    h = int(np.clip(points[:, 1].max() + 1, 0, height)) - y
                    if w > 0 and h > 0:
                        canvas = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(canvas, [(points - (x, y)).reshape((-1, 1, 2))], 1)
                        local = canvas.astype(bool)

            if local is not None:
                overlay[y:y + h, x:x + w][local] = color
                has_mask[y:y + h, x:x + w] |= local
                rects.append((y, y + h, x, x + w))

    # Blend rectangle by rectangle. Clearing has_mask after each one keeps a pixel
    # shared by overlapping rectangles from being blended twice, so every masked pixel
    # is blended once, from the original frame and the final overlay.
    for y0, y1, x0, x1 in rects:
        region = frame[y0:y1, x0:x1]
        hit = has_mask[y0:y1, x0:x1]
        if not hit.any():
            continue
        if opacity >= 1.0:
            # Direct copy for full opacity
            region[hit] = overlay[y0:y1, x0:x1][hit]
        else:
            blended = cv2.addWeighted(region, 1 - opacity, overlay[y0:y1, x0:x1], opacity, 0)
            region[hit] = blended[hit]
        hit[:] = False

    return frame