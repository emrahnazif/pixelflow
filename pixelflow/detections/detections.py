"""
Core Detection Data Structures for Computer Vision Processing.

Provides unified data structures for representing detection results from various
computer vision frameworks, including bounding boxes, masks, keypoints, and tracking
information. Designed for efficient processing and seamless integration across
different ML frameworks and visualization tools.
"""

import copy as copy_module
import io
import math
import json
import base64
import numpy as np
from PIL import Image
from pixelflow.validators import (validate_bbox,
                                  validate_segments,
                                  round_coord,
                                  round_to_decimal)
from pixelflow.masks import mask_to_polygon, simplify_polygon
from pixelflow.serialization import constructor_fields, own
from typing import (List,
                    Iterator, Optional, Union, Any, Dict)

__all__ = ["KeyPoint", "Detection", "Detections"]


# Mask dtypes Pillow round-trips losslessly through PNG. Pillow's 'I' mode
# (int8/int16/int32/uint32) is deliberately excluded: saving it as PNG is
# deprecated and removed in Pillow 13. Arrays outside this set - float
# probability maps, int64, stacked 3D masks, empty arrays - use the raw
# fallback in Detection.to_dict().
_PNG_MASK_DTYPES = frozenset(("bool", "uint8", "uint16"))


def _is_raster_mask(mask: Any) -> bool:
    """Whether an entry of `Detection.masks` is a pixel raster rather than a polygon."""
    return isinstance(mask, np.ndarray)


def _is_png_encodable(mask: np.ndarray) -> bool:
    """Whether a mask array can be losslessly PNG-encoded by Pillow."""
    if mask.size == 0 or str(mask.dtype) not in _PNG_MASK_DTYPES:
        return False
    if mask.ndim == 2:
        return True
    # Multi-channel is only representable as 8-bit RGB/RGBA.
    return mask.ndim == 3 and mask.dtype == np.uint8 and mask.shape[2] in (3, 4)


# Object-oriented approach instead of a NumPy array-based approach
# Let's see how it goes


class KeyPoint:
    """
    One landmark of a detected object, in the shape :class:`Detection` uses for its class.

    A keypoint is identified by its `id`: the position it occupies in the vocabulary the model
    was trained on. That number comes out of the weights and is always right. Its `name` is
    metadata about that number, and metadata has to come from somewhere -- the weights, or the
    caller. Where neither supplies it, `name` is None rather than a plausible guess: a hand model
    with 21 landmarks and a pose model with 17 both emit an index 5, and calling that
    "left_shoulder" because a pose model's fifth landmark is one would be wrong in a way nobody
    would think to check.

    Args:
        x (float): X coordinate in pixels from the image's left edge.
        y (float): Y coordinate in pixels from the image's top edge.
        id (int): Index of this landmark in the model's keypoint vocabulary.
        name (Optional[str]): Human-readable name, when the model or caller supplies one.
        confidence (Optional[float]): The score the model emitted for this landmark.

    Example:
        >>> import pixelflow as pf
        >>>
        >>> # Named, because the caller knew the vocabulary
        >>> nose = pf.KeyPoint(x=320, y=240, id=0, name="nose", confidence=0.98)
        >>>
        >>> # Unnamed: the index still identifies it exactly
        >>> fifth = pf.KeyPoint(x=150, y=200, id=5, confidence=0.12)
        >>> fifth.name is None
        True

    Notes:
        - Coordinates keep sub-pixel precision, rounded to two decimals. Annotators cast
          to int themselves when they draw.
        - `confidence` is the model's raw score, not a boolean. Deciding what counts as visible
          is the caller's threshold to pick; annotators take `min_confidence` for exactly this.
        - `id` is meaningful only within the model that produced it. Two models' index 5 are
          unrelated unless they share a vocabulary.
    """

    def __init__(self, x: float, y: float, id: int, name: Optional[str] = None,
                 confidence: Optional[float] = None):
        # __init__ is the choke point: with_xy funnels every transform through it.
        # round_coord raises on a non-finite coordinate, which is deliberate -
        # unlike bbox there is no None convention for consumers to guard against.
        self.x = round_coord(x)
        self.y = round_coord(y)
        self.id = id
        self.name = name
        self.confidence = confidence

    def with_xy(self, x: float, y: float) -> "KeyPoint":
        """
        Return a copy of this keypoint moved to `(x, y)`, keeping its identity.

        Geometric transforms move a landmark without changing which landmark it is, so
        `id`, `name` and `confidence` carry across untouched. Having one place that says
        what "the same keypoint, somewhere else" means keeps the eight transform call
        sites from drifting apart when a field is added.

        Args:
            x (float): New X coordinate in pixels.
            y (float): New Y coordinate in pixels.

        Returns:
            KeyPoint: A new keypoint at `(x, y)` with this one's id, name and confidence.
        """
        return KeyPoint(x, y, self.id, self.name, self.confidence)

    def _clone(self) -> "KeyPoint":
        """
        Copy this keypoint verbatim, skipping validation.

        The coordinates were rounded on the way in, so re-running round_coord
        on them is a guaranteed no-op. Detection.copy() clones every landmark
        of every detection, which made that no-op measurable on pose workloads.
        """
        twin = object.__new__(KeyPoint)
        twin.__dict__.update(self.__dict__)
        return twin

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert KeyPoint to dictionary format for JSON serialization and storage.

        Returns:
            Dict[str, Any]: The keypoint's x, y, id, name and confidence. `name` is None when
                nothing supplied one, so a consumer can tell "unnamed" from a real name.

        Example:
            >>> import pixelflow as pf
            >>> pf.KeyPoint(100, 200, 0, "nose", 0.9).to_dict()
            {'x': 100.0, 'y': 200.0, 'id': 0, 'name': 'nose', 'confidence': 0.9}
        """
        return {
            # round_coord already returned native floats, so no numpy scalars
            # can reach here.
            "x": self.x,
            "y": self.y,
            "id": int(self.id) if self.id is not None else None,
            "name": self.name,
            "confidence": float(self.confidence) if self.confidence is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyPoint":
        """
        Rebuild a KeyPoint from the dictionary `to_dict()` produces.

        Args:
            data (Dict[str, Any]): A keypoint dictionary. `x`, `y` and `id` are
                                  required; `name` and `confidence` default to None.
                                  Unknown keys are ignored.

        Returns:
            KeyPoint: A new keypoint with the dictionary's values.

        Raises:
            TypeError: If `x`, `y` or `id` is missing.
            ValueError: If a coordinate is not a finite number.

        Example:
            >>> import pixelflow as pf
            >>> keypoint = pf.KeyPoint(100, 200, 0, "nose", 0.9)
            >>> restored = pf.KeyPoint.from_dict(keypoint.to_dict())
            >>> restored.to_dict() == keypoint.to_dict()
            True
        """
        return cls(**own(constructor_fields(cls, data)))


class Detection:
    """
    Unified representation of a single object detection with comprehensive metadata and tracking.
    
    Stores complete detection information including bounding boxes, segmentation masks,
    keypoints, classification data, tracking information, and spatial analytics. Provides
    a standardized interface for detection data across different ML frameworks with automatic
    validation and type conversion for consistent data handling.
    
    Args:
        inference_id (Optional[str]): Unique identifier for the inference session or batch.
        bbox (Optional[List[float]]): Bounding box coordinates in XYXY format [x1, y1, x2, y2].
                                     Validated by validate_bbox: kept as floats rounded to two
                                     decimals, or None if the box is malformed or non-finite.
        masks (Optional[List[Any]]): Segmentation masks, either boolean rasters or polygon
                                    point lists depending on the source model. Unlike bbox and
                                    segments this field is not yet normalized to one shape.
        segments (Optional[List[List[float]]]): Object outline as one polygon, normalized by
                                    validate_segments to [[x, y], ...] at two decimals, or None.
                                    Coordinates follow the pixel-corner convention bbox uses:
                                    pixel (row i, column j) covers x from j to j + 1 and y
                                    from i to i + 1. Polygons a model computed itself (the
                                    Ultralytics converter passes them through) only
                                    approximate this; masks_to_segments() makes them exact.
        keypoints (Optional[List[KeyPoint]]): List of KeyPoint objects for pose/structure data.
        class_id (Optional[Union[int, str]]): Numeric or string class identifier from model.
        class_name (Optional[str]): Human-readable class name (e.g., "person", "vehicle").
        text (Optional[str]): Free-form string this instance carries -- what an OCR engine read
                             inside the box, or a region caption. Distinct from `class_name`,
                             which names one class from a fixed vocabulary the model was trained
                             on; `text` is per-instance content the model produced, drawn from no
                             vocabulary. A detection can have both, either, or neither.
        labels (Optional[List[str]]): Additional classification labels or attributes.
        confidence (Optional[float]): Detection confidence score [0.0-1.0], rounded by
                                     `round_to_decimal` to the shared CONFIDENCE_DECIMALS
                                     precision so every result type reports it identically.
        tracker_id (Optional[int]): Unique tracking identifier for multi-frame object tracking.
        metadata (Optional[Dict[str, Any]]): Additional custom metadata and framework-specific data.
        zones (Optional[List[str]]): List of zone identifiers the detection intersects.
                                   Defaults to empty list if None.
        zone_names (Optional[List[str]]): Human-readable names for intersected zones.
                                         Defaults to empty list if None.
        line_crossings (Optional[List[Dict]]): Line crossing events for this detection.
                                             Defaults to empty list if None.
        first_seen_time (Optional[float]): Timestamp when detection first appeared (Unix time).
        total_time (float): Total duration since first detection in seconds. Default is 0.0.
    
    Raises:
        ValueError: If bbox coordinates are invalid or out of expected format.
        TypeError: If keypoints contain non-KeyPoint objects.
    
    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> import cv2
        >>> 
        >>> # Basic detection from ML framework
        >>> image = cv2.imread("image.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> results = pf.from_ultralytics(outputs)
        >>> 
        >>> # Create basic detection with bounding box
        >>> detection = pf.Detection(
        ...     bbox=[100, 50, 200, 150],
        ...     class_name="person",
        ...     confidence=0.85
        ... )
        >>> 
        >>> # Create detection with tracking and zones
        >>> tracked_detection = pf.Detection(
        ...     bbox=[150, 75, 250, 175],
        ...     class_name="vehicle",
        ...     confidence=0.92,
        ...     tracker_id=42,
        ...     zones=["parking_lot"],
        ...     first_seen_time=1234567890.5
        ... )
        >>> 
        >>> # Detection with keypoints for pose estimation
        >>> nose_point = pf.KeyPoint(250, 120, id=0, name="nose", confidence=0.98)
        >>> pose_detection = pf.Detection(
        ...     bbox=[200, 100, 300, 400],
        ...     class_name="person",
        ...     keypoints=[nose_point]
        ... )

    Notes:
        - Bounding box coordinates are automatically validated using validate_bbox function,
          which preserves sub-pixel precision to two decimal places
        - Confidence scores are automatically rounded using round_to_decimal for precision
        - Zone and line crossing lists are initialized as empty lists if None provided
        - Compatible with all major ML framework outputs through converter functions
        - Supports in-place mask simplification for performance optimization
    """
    
    def __init__(self,
                 inference_id: Optional[str] = None,
                 bbox: Optional[List[float]] = None,
                 masks: Optional[List[Any]] = None,
                 segments: Optional[List[Any]] = None,
                 keypoints: Optional[List[KeyPoint]] = None,
                 class_id: Optional[Union[int, str]] = None,
                 class_name: Optional[str] = None,
                 labels: Optional[List[str]] = None,
                 confidence: Optional[float] = None,
                 tracker_id: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 zones: Optional[List[str]] = None,
                 zone_names: Optional[List[str]] = None,
                 line_crossings: Optional[List[Dict]] = None,
                 first_seen_time: Optional[float] = None,
                 total_time: float = 0.0,
                 text: Optional[str] = None):
        self.inference_id = inference_id
        self.bbox = bbox  # goes through the validating property setter below
        self.masks = masks
        self.segments = segments
        self.keypoints = keypoints if keypoints is not None else None
        self.class_id = class_id
        self.class_name = class_name
        self.text = text
        self.labels = labels
        self.confidence = round_to_decimal(confidence)
        self.tracker_id = tracker_id

        self.metadata = metadata

        self.zones = zones if zones is not None else []  # List of zone IDs
        self.zone_names = zone_names if zone_names is not None else []  # List of zone names
        self.line_crossings = line_crossings if line_crossings is not None else []  # List of line crossing events
        self.first_seen_time = first_seen_time  # Timestamp/frame when first detected
        self.total_time = total_time  # Total time since first detection (in seconds)

    @property
    def bbox(self) -> Optional[List[float]]:
        """Bounding box in XYXY format, or None if never set or rejected as invalid."""
        return self._bbox

    @bbox.setter
    def bbox(self, value: Optional[List[float]]) -> None:
        """Validate on every assignment, not just at construction.

        Transforms rewrite this attribute in place (flips, rotations, crops), so
        validating only in __init__ would let transformed boxes drift off the
        precision contract and let a non-finite coordinate through. Note that
        mutating the list itself - bbox[0] = x - still bypasses this, as it does
        for any Python attribute.
        """
        self._bbox = validate_bbox(value)

    @property
    def segments(self) -> Optional[List[List[float]]]:
        """Polygon outline as a list of [x, y] points, or None if never set."""
        return self._segments

    @segments.setter
    def segments(self, value) -> None:
        """Normalize on every assignment; see validate_segments for the shape."""
        self._segments = validate_segments(value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Detection to dictionary format for JSON serialization and API export.

        Transforms the Detection object into a comprehensive dictionary representation
        suitable for JSON export, API responses, database storage, and data analysis.
        Handles nested KeyPoint objects and maintains data type consistency. Automatically
        encodes numpy binary masks as base64-wrapped PNGs for JSON compatibility and converts
        all NumPy types to native Python types for JSON serialization.

        Returns:
            Dict[str, Any]: Dictionary containing all detection fields with keypoints
                          converted to dictionaries, numpy masks PNG-encoded as base64,
                          numpy types converted to native Python types, and proper
                          type formatting for JSON serialization.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Basic detection serialization
            >>> detection = pf.Detection(bbox=[100, 50, 200, 150], class_name="car")
            >>> data = detection.to_dict()
            >>> import json
            >>> json_str = json.dumps(data, indent=2)  # Now JSON-safe!
            >>>
            >>> # Detection with keypoints serialization
            >>> keypoint = pf.KeyPoint(250, 120, id=0, name="nose", confidence=0.98)
            >>> detection_with_pose = pf.Detection(
            ...     bbox=[200, 100, 300, 400],
            ...     keypoints=[keypoint],
            ...     tracker_id=42
            ... )
            >>> data = detection_with_pose.to_dict()
            >>>
            >>> # Export to file
            >>> with open("detection.json", "w") as f:
            ...     json.dump(data, f, indent=2)

        Notes:
            - Keypoints are recursively converted to dictionaries using their to_dict() method
            - Numpy binary masks are PNG-compressed, then base64-encoded, with metadata
            - Masks PNG cannot represent (float, int64, stacked, empty) use format 'raw'
            - Polygon masks remain as coordinate lists (already JSON-compatible)
            - All NumPy numeric types (int32, float32, etc.) converted to native Python types
            - All optional fields are included even if None for consistent API responses
            - Output is fully JSON-serializable (can use json.dumps() directly)
            - Lists and dictionaries such as `metadata`, `zones` and `segments` are
              the detection's own objects, not copies. Treat the result as read-only,
              or rebuild with `from_dict()`, which copies them
        """
        # Helper function to convert numpy types to native Python types
        def to_python_type(value):
            """Convert numpy types to native Python types for JSON serialization."""
            if value is None:
                return None
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [to_python_type(v) for v in value]
            return value
        # Process masks - PNG-encode numpy arrays, keep polygons as-is
        serializable_masks = None
        if self.masks:
            serializable_masks = []
            for mask in self.masks:
                if _is_raster_mask(mask):
                    if _is_png_encodable(mask):
                        # Binary mask - PNG-encode, then base64 for JSON
                        # transport. PNG compresses masks losslessly; encoding
                        # the raw bitmap instead costs ~600x the payload on
                        # full-frame masks.
                        buffer = io.BytesIO()
                        Image.fromarray(mask).save(buffer, format='PNG')
                        mask_format, mask_bytes = 'png', buffer.getvalue()
                    else:
                        # Not representable as an image - fall back to the
                        # uncompressed buffer so every dtype stays serializable.
                        mask_format, mask_bytes = 'raw', mask.tobytes()
                    serializable_masks.append({
                        'format': mask_format,
                        'data': base64.b64encode(mask_bytes).decode('utf-8'),
                        'shape': list(mask.shape),
                        'dtype': str(mask.dtype)
                    })
                else:
                    # Polygon format - keep as-is (already JSON-compatible)
                    serializable_masks.append({
                        'format': 'polygon',
                        'data': mask
                    })

        return {
            "inference_id": self.inference_id,
            "bbox": to_python_type(self.bbox),
            "masks": serializable_masks,
            "segments": self.segments,  # already plain lists of floats
            "keypoints": [kp.to_dict() for kp in self.keypoints] if self.keypoints is not None else None,
            "class_id": to_python_type(self.class_id),
            "class_name": self.class_name,
            "text": self.text,
            "labels": self.labels,
            "confidence": to_python_type(self.confidence),
            "tracker_id": to_python_type(self.tracker_id),
            "metadata": self.metadata,
            "zones": self.zones,
            "zone_names": self.zone_names,
            "line_crossings": self.line_crossings,
            "first_seen_time": to_python_type(self.first_seen_time),
            "total_time": to_python_type(self.total_time)
        }

    @staticmethod
    def decode_mask(mask_dict: Dict[str, Any]) -> Union[np.ndarray, List]:
        """
        Decode mask from dictionary format (png, raw or polygon).

        Utility function to decode masks that were serialized using to_dict().
        Handles PNG-encoded masks, uncompressed raw masks, and polygon formats.

        Args:
            mask_dict (Dict[str, Any]): Dictionary with 'format' and 'data' keys.
                                       Format can be 'png', 'raw' or 'polygon'.

        Returns:
            Union[np.ndarray, List]: Decoded mask as numpy array (for png/raw)
                                    or list of coordinates (for polygon).

        Raises:
            ValueError: If mask format is unknown or invalid.
            KeyError: If required dictionary keys are missing.

        Example:
            >>> import pixelflow as pf
            >>> import json
            >>>
            >>> # Serialize detection with mask
            >>> detection = pf.Detection(masks=[np.ones((100, 100), dtype=bool)])
            >>> data = detection.to_dict()
            >>>
            >>> # Export to JSON and load back
            >>> json_str = json.dumps(data)
            >>> loaded_data = json.loads(json_str)
            >>>
            >>> # Decode mask
            >>> mask_dict = loaded_data['masks'][0]
            >>> decoded_mask = pf.Detection.decode_mask(mask_dict)
            >>> print(decoded_mask.shape)  # (100, 100)
            >>>
            >>> # Polygon mask decoding
            >>> polygon_dict = {'format': 'polygon', 'data': [(0, 0), (100, 0), (100, 100)]}
            >>> polygon = pf.Detection.decode_mask(polygon_dict)
            >>> print(polygon)  # [(0, 0), (100, 0), (100, 100)]

        Notes:
            - For png format, requires 'data' and 'dtype' keys
            - For raw format, requires 'data', 'shape', and 'dtype' keys
            - For polygon format, returns the 'data' field directly
            - PNG carries its own dimensions, so 'shape' is informational there
            - Compatible with all masks serialized by to_dict() or to_json()
        """
        if mask_dict['format'] == 'png':
            # Decode PNG-encoded binary mask
            png_bytes = base64.b64decode(mask_dict['data'])
            mask = np.array(Image.open(io.BytesIO(png_bytes)))
            return mask.astype(mask_dict['dtype'])
        elif mask_dict['format'] == 'raw':
            # Uncompressed buffer - wrapping in bytearray keeps the decoded
            # array writable, which np.frombuffer alone does not.
            mask_bytes = bytearray(base64.b64decode(mask_dict['data']))
            mask = np.frombuffer(mask_bytes, dtype=mask_dict['dtype'])
            return mask.reshape(mask_dict['shape'])
        elif mask_dict['format'] == 'polygon':
            # Polygon format - return as-is
            return mask_dict['data']
        else:
            raise ValueError(f"Unknown mask format: {mask_dict['format']}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        """
        Rebuild a Detection from the dictionary `to_dict()` produces.

        The inverse of `to_dict()`: masks are decoded back into arrays, keypoints
        become `KeyPoint` objects again, and every other field is handed to the
        constructor. Reads stored or transmitted results without the caller having
        to know which fields a detection has or how its masks were encoded.

        Args:
            data (Dict[str, Any]): A detection dictionary, as written by `to_dict()`
                                  or parsed from `to_json()`. Unknown keys are
                                  ignored and missing keys take the constructor's
                                  defaults, so dictionaries written by other
                                  pixelflow versions still read.

        Returns:
            Detection: A new detection, independent of `data`.

        Raises:
            ValueError: If a mask uses a format `decode_mask` does not know, or a
                        keypoint coordinate is not a finite number.
            TypeError: If a keypoint is missing `x`, `y` or `id`.

        Example:
            >>> import numpy as np
            >>> import pixelflow as pf
            >>>
            >>> mask = np.zeros((480, 640), dtype=bool)
            >>> mask[100:200, 50:300] = True
            >>> detection = pf.Detection(bbox=[50, 100, 300, 200], masks=[mask],
            ...                          class_name="car", confidence=0.9)
            >>>
            >>> # Round trip through a plain dictionary
            >>> restored = pf.Detection.from_dict(detection.to_dict())
            >>> np.array_equal(restored.masks[0], mask)
            True
            >>>
            >>> # Partial dictionaries read too; the rest take defaults
            >>> pf.Detection.from_dict({"bbox": [10, 20, 30, 40]}).class_name is None
            True
            >>>
            >>> # Keys this version does not know are ignored
            >>> pf.Detection.from_dict({"bbox": [10, 20, 30, 40], "added_later": 1}).bbox
            [10.0, 20.0, 30.0, 40.0]

        Notes:
            - Values pass through the same validation as any new detection: an
              invalid `bbox` becomes None, coordinates round to two decimals and
              confidence to three. Values that came out of `to_dict()` were already
              rounded, so they come back unchanged
            - Raster masks come back with the pixels, dtype and shape they were
              written with
            - A mask entry that is not an encoded dictionary is taken as already
              decoded: a polygon list is copied, an array is kept as it is
            - Lists and dictionaries are copied, so changing the restored
              detection never reaches `data`, or the detection `data` came from
        """
        fields = constructor_fields(cls, data)
        # Each keypoint is rebuilt from its own dictionary below, and copies it there.
        keypoints = fields.pop("keypoints", None)
        fields = own(fields)

        if fields.get("masks") is not None:
            fields["masks"] = [cls.decode_mask(mask) if isinstance(mask, dict) else mask
                               for mask in fields["masks"]]
        if keypoints is not None:
            fields["keypoints"] = [KeyPoint.from_dict(kp) for kp in keypoints]

        return cls(**fields)

    def simplify_masks(self, tolerance: float = 2.0, preserve_topology: bool = True) -> None:
        """
        Simplify polygon masks to reduce complexity while preserving essential shape characteristics.
        
        Applies Douglas-Peucker polygon simplification algorithm to reduce the number of
        vertices in mask polygons while maintaining visual fidelity. Optimizes memory
        usage and processing speed for downstream operations without significant accuracy loss.
        
        Args:
            tolerance (float): Simplification tolerance in pixels. Higher values create
                             more simplified polygons with fewer vertices. 
                             Range: [0.1, 10.0]. Default is 2.0.
            preserve_topology (bool): Whether to preserve polygon topology during
                                    simplification to avoid self-intersections and holes.
                                    Default is True.
        
        Raises:
            ValueError: If tolerance is outside the valid range [0.1, 10.0].
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create detection with complex polygon mask
            >>> complex_polygon = [[100, 100], [101, 100], [102, 101], [200, 200]]
            >>> detection = pf.Detection(masks=[complex_polygon])
            >>> 
            >>> # Simplify with default settings
            >>> detection.simplify_masks()
            >>> 
            >>> # Aggressive simplification for performance
            >>> detection.simplify_masks(tolerance=5.0, preserve_topology=False)
            >>> 
            >>> # Conservative simplification for accuracy
            >>> detection.simplify_masks(tolerance=0.5, preserve_topology=True)
        
        Notes:
            - Modifies masks in-place for memory efficiency and performance
            - Uses Shapely's Douglas-Peucker algorithm for geometric simplification
            - Only processes polygon-format masks; binary/raster masks are unchanged
            - Higher tolerance values result in more aggressive vertex reduction
            - Topology preservation prevents self-intersections but may retain more vertices
        
        Performance Notes:
            - Significant speedup for complex polygons with hundreds of vertices
            - Memory usage reduced proportionally to vertex count reduction
            - Recommended for real-time applications with complex segmentation masks
        """
        if self.masks:
            # A raster is not a polygon: it passes through untouched, as documented.
            self.masks = [
                mask if _is_raster_mask(mask)
                else simplify_polygon(mask, tolerance, preserve_topology)
                for mask in self.masks
            ]

    def masks_to_segments(self, tolerance: float = 1.5, keep_masks: bool = True) -> bool:
        """
        Fill `segments` with the outline of this detection's raster mask.

        Converts the raster in `masks` to one polygon with `mask_to_polygon` and
        stores it in `segments`. Lets a result from any segmentation model be drawn,
        stored or sent as a polygon without carrying the raster along.

        Args:
            tolerance (float): Simplification tolerance in pixels, passed to
                              `mask_to_polygon`. Range: [0.0, inf). Default is 1.5.
            keep_masks (bool): Whether to keep the raster in `masks` after converting.
                              False drops it to save memory. Default is True.

        Returns:
            bool: True if `segments` was set, False if the detection was left unchanged.

        Raises:
            ValueError: If `tolerance` is negative.

        Example:
            >>> import numpy as np
            >>> import pixelflow as pf
            >>>
            >>> mask = np.zeros((480, 640), dtype=bool)
            >>> mask[100:200, 50:300] = True
            >>> detection = pf.Detection(bbox=[50, 100, 300, 200], masks=[mask])
            >>>
            >>> # Fill segments, keep the raster for drawing
            >>> detection.masks_to_segments()
            True
            >>> detection.segments
            [[50.0, 100.0], [50.0, 200.0], [300.0, 200.0], [300.0, 100.0]]
            >>>
            >>> # Finer outline, raster dropped
            >>> detection.masks_to_segments(tolerance=0.5, keep_masks=False)
            True
            >>> detection.masks is None
            True

        Notes:
            - Modifies the detection in-place
            - A detection with no mask, a polygon mask, or an empty raster is left
              unchanged
            - Existing `segments` are replaced when a raster is present: the raster
              is what the model produced, and a polygon derived from it follows the
              pixel-corner convention exactly
            - Several rasters of the same shape are treated as parts of one object
              and merged before tracing; `segments` still receives one outline
            - With `keep_masks=False` only a converted raster is dropped. A raster
              that produced no polygon stays, so no geometry is lost silently
        """
        if not self.masks:
            return False

        # Only a 2D raster can be traced; stacked or multi-channel arrays are left alone.
        rasters = [m for m in self.masks if _is_raster_mask(m) and m.ndim == 2]
        if not rasters:
            return False
        rasters = [m for m in rasters if m.shape == rasters[0].shape]

        polygon = mask_to_polygon(np.logical_or.reduce(rasters), tolerance=tolerance)
        if polygon is None:
            return False

        self.segments = polygon
        if not keep_masks:
            converted = {id(m) for m in rasters}
            self.masks = [m for m in self.masks if id(m) not in converted] or None
        return True

    def copy(self) -> "Detection":
        """
        Create a deep copy of this detection.

        Creates a new Detection instance with deep copies of all fields to ensure
        complete independence from the original. Useful for immutable transform
        operations and safe data manipulation workflows.

        Returns:
            Detection: New detection instance with all fields deep copied.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Create and copy detection
            >>> detection = pf.Detection(bbox=[100, 50, 200, 150], class_name="car")
            >>> copied = detection.copy()
            >>>
            >>> # Modifications don't affect original
            >>> copied.bbox[0] = 150
            >>> print(detection.bbox)  # [100, 50, 200, 150] - unchanged
            >>> print(copied.bbox)     # [150, 50, 200, 150] - modified
            >>>
            >>> # Use in transform workflows
            >>> new_detection = detection.copy()
            >>> new_detection.bbox = transform_bbox(new_detection.bbox)

        Notes:
            - All list and dict fields are deep copied for complete independence
            - Keypoints are deep copied to ensure complete independence from original
            - Useful for immutable transform operations in detection processing
        """
        duplicate = Detection(
            inference_id=self.inference_id,
            bbox=None,
            masks=copy_module.deepcopy(self.masks) if self.masks else None,
            segments=None,
            keypoints=[kp._clone() for kp in self.keypoints] if self.keypoints else None,
            class_id=self.class_id,
            class_name=self.class_name,
            text=self.text,
            labels=self.labels.copy() if self.labels else None,
            confidence=self.confidence,
            tracker_id=self.tracker_id,
            metadata=self.metadata.copy() if self.metadata else None,
            zones=self.zones.copy() if self.zones else [],
            zone_names=self.zone_names.copy() if self.zone_names else [],
            line_crossings=copy_module.deepcopy(self.line_crossings) if self.line_crossings else [],
            first_seen_time=self.first_seen_time,
            total_time=self.total_time
        )
        # Already validated on the way in, so bypass the setter rather than
        # re-running validate_bbox: transforms copy then rewrite, and paying
        # for validation twice per detection showed up on their hot path.
        duplicate._bbox = self._bbox.copy() if self._bbox else None
        duplicate._segments = [p.copy() for p in self._segments] if self._segments else None
        return duplicate


class Detections:
    """
    Container for multiple Detection objects with comprehensive filtering and processing capabilities.
    
    Provides a unified interface for managing collections of detections with support
    for iteration, indexing, filtering, zone management, serialization, and bulk operations.
    Implements standard Python container protocols and includes dynamically attached filter
    methods for zero-overhead detection processing and analysis workflows.
    
    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> import cv2
        >>> 
        >>> # Create detections from ML framework
        >>> image = cv2.imread("image.jpg")
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> detections = pf.from_ultralytics(outputs)
        >>> 
        >>> # Manual creation and management
        >>> detections = pf.Detections()
        >>> detection1 = pf.Detection(bbox=[100, 50, 200, 150], class_name="person")
        >>> detection2 = pf.Detection(bbox=[300, 100, 400, 200], class_name="car")
        >>> detections.add_detection(detection1)
        >>> detections.add_detection(detection2)
        >>> 
        >>> # Container operations
        >>> print(f"Found {len(detections)} objects")
        >>> for detection in detections:
        ...     print(f"Class: {detection.class_name}")
        >>> 
        >>> # Apply filters (dynamically attached methods)
        >>> high_conf = detections.filter_by_confidence(0.8)
        >>> people_only = detections.filter_by_class_id("person")
        >>> chained = detections.filter_by_confidence(0.7).filter_by_size(min_area=1000)
    
    Notes:
        - Implements standard Python container protocols (__len__, __iter__, __getitem__)
        - Filter methods are dynamically attached from filters module for zero overhead
        - Supports method chaining for complex filtering workflows
        - Zone management integration for spatial filtering and analytics
        - Bulk operations for mask simplification and serialization
        - Compatible with all major ML framework outputs through converter functions
    """
    
    def __init__(self):
        self.detections: List[Detection] = []

    def show(self) -> None:
        """
        Display annotated image with detections (placeholder for future implementation).
        
        Notes:
            - Placeholder method for future visualization features
            - Will integrate with annotation and display modules
        """
        # Display the annotated image
        pass

    def add_detection(self, detection: Detection) -> None:
        """
        Add a Detection object to the collection.
        
        Appends a Detection instance to the internal collection, enabling batch
        processing and filtering operations on the complete detection set.
        
        A detection that does not describe an object is dropped rather than stored:
        one whose `bbox` is None -- never set, or rejected by `validate_bbox` as
        non-finite, inverted or enclosing no pixels -- and one whose confidence is
        NaN, which no threshold can compare against and which strict JSON parsers
        refuse. Dropping at the boundary is what lets every consumer downstream
        assume a detection has a box, rather than each re-deciding what to do
        about one that does not.

        Args:
            detection (Detection): Detection object to add to the collection.
                                 Must be a valid Detection instance.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Create container and add detection
            >>> detections = pf.Detections()
            >>> detection = pf.Detection(bbox=[100, 50, 200, 150])
            >>> detections.add_detection(detection)
            >>> 
            >>> # Add multiple detections
            >>> detection2 = pf.Detection(bbox=[200, 100, 300, 200], class_name="car")
            >>> detections.add_detection(detection2)
            >>> print(f"Total detections: {len(detections)}")
        """
        # Geometry was already judged by validate_bbox on assignment, which leaves
        # None for anything that does not enclose pixels.
        if detection.bbox is None:
            return
        if detection.confidence is not None and math.isnan(detection.confidence):
            return

        self.detections.append(detection)
    
    def _drop_boxless(self) -> "Detections":
        """
        Remove detections whose box was invalidated by an in-place update.

        `add_detection` judges a detection on the way in, but a transform rewrites
        `bbox` on detections already stored, where nothing re-checks them. One that
        shrinks a box to nothing - padding by a negative fraction, a box rebuilt
        from a single keypoint - leaves it None, and the detection would otherwise
        stay in the collection: still counted, with no geometry, and fatal to any
        consumer that assumes it has one.

        Returns:
            Detections: This collection, for chaining.
        """
        self.detections = [d for d in self.detections if d.bbox is not None]
        return self

    def update_zones(self, zone_manager: Any) -> 'Detections':
        """
        Update all detections with zone intersection information for spatial analytics.
        
        Processes each detection in the collection to determine which spatial zones
        it intersects, updating the zones and zone_names fields for spatial filtering
        and analytics workflows.
        
        Args:
            zone_manager (Any): ZoneManager instance to check zone intersections.
                              If None, no zone updates are performed.
        
        Returns:
            Detections: Returns self for method chaining with updated zone information.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Setup zones and update detections
            >>> zone_manager = pf.zones.ZoneManager()
            >>> zone_manager.add_polygon_zone("parking", [(0, 0), (100, 0), (100, 100), (0, 100)])
            >>> detections.update_zones(zone_manager)
            >>> 
            >>> # Chain with filtering
            >>> parking_detections = detections.update_zones(zone_manager).filter_by_zones(["parking"])
        
        Notes:
            - Modifies detection objects in-place for memory efficiency
            - Zone intersection uses bounding box center point by default
            - Compatible with all zone types (polygon, circular, rectangular)
        """
        if zone_manager is not None:
            zone_manager.update(self)
        return self

    def __len__(self):
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]


    def simplify(self, tolerance: float = 2.0, preserve_topology: bool = True) -> 'Detections':
        """
        Simplify polygon masks for all detections in the collection for performance optimization.
        
        Applies polygon simplification to all detections with mask data, reducing
        vertex count while preserving essential shape characteristics. Optimizes
        memory usage and processing speed for bulk detection operations.
        
        Args:
            tolerance (float): Simplification tolerance in pixels. Higher values create
                             more simplified polygons. Range: [0.1, 10.0]. Default is 2.0.
            preserve_topology (bool): Whether to preserve polygon topology during
                                    simplification. Default is True.
        
        Returns:
            Detections: Returns self for method chaining with simplified masks.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Default simplification
            >>> detections.simplify()
            >>> 
            >>> # Aggressive simplification for performance
            >>> detections.simplify(tolerance=5.0, preserve_topology=False)
            >>> 
            >>> # Chain with other operations
            >>> processed = detections.simplify().filter_by_confidence(0.7)
        
        Notes:
            - Applies simplification to all detections with mask data in the collection
            - Calls Detection.simplify_masks() on each detection instance
            - Performance improvement scales with collection size and mask complexity
        """
        for detection in self.detections:
            detection.simplify_masks(tolerance=tolerance, preserve_topology=preserve_topology)
        return self

    def masks_to_segments(self, tolerance: float = 1.5, keep_masks: bool = True) -> 'Detections':
        """
        Fill `segments` from the raster mask of every detection in the collection.

        Turns the output of any segmentation model into polygons: each detection's
        raster in `masks` is traced along its pixel corners, simplified, and stored in
        `segments`. Every source then shares one polygon convention, whichever model
        produced it.

        Args:
            tolerance (float): Simplification tolerance in pixels. Higher values give
                              fewer points. Range: [0.0, inf). Default is 1.5.
            keep_masks (bool): Whether to keep the rasters in `masks` after
                              converting. False drops them to save memory.
                              Default is True.

        Returns:
            Detections: Returns self for method chaining.

        Raises:
            ValueError: If `tolerance` is negative.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # SAM-style output: boxes and boolean (N, H, W) masks
            >>> detections = pf.from_arrays(boxes, scores, masks=masks)
            >>> detections.masks_to_segments()
            >>> detections[0].segments  # [[x, y], ...]
            >>>
            >>> # Draw outlines, which read `segments`
            >>> image = pf.annotate.polygon(image, detections)
            >>>
            >>> # Keep only polygons, e.g. before serializing
            >>> payload = detections.masks_to_segments(keep_masks=False).to_json()
            >>>
            >>> # Finer outlines at the cost of more points
            >>> detections.masks_to_segments(tolerance=0.5)

        Notes:
            - Modifies detections in-place; order and every other field are untouched
            - Detections without a raster mask, or with an empty one, are left unchanged
            - Existing `segments` are replaced where a raster is present, which also
              brings Ultralytics' own polygons onto the pixel-corner convention
            - One outline per detection: holes and all but the largest region are
              dropped. See `mask_to_polygon` for the exact rules

        Performance Notes:
            - A few milliseconds per detection at full-HD; cost follows object size
        """
        for detection in self.detections:
            detection.masks_to_segments(tolerance=tolerance, keep_masks=keep_masks)
        return self

    def copy(self) -> "Detections":
        """
        Create a deep copy of all detections in the collection.

        Creates a new Detections container with deep copies of all Detection objects,
        ensuring complete independence from the original collection. Enables safe
        immutable operations and transform workflows without modifying original data.

        Returns:
            Detections: New detections container with copied detection objects.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Create and copy detections
            >>> detections = model.predict(image)
            >>> copied = detections.copy()
            >>>
            >>> # Modifications don't affect original
            >>> copied[0].bbox[0] = 150
            >>> print(detections[0].bbox[0])  # Original unchanged
            >>> print(copied[0].bbox[0])      # 150 - modified
            >>>
            >>> # Use in immutable transform workflows
            >>> transformed = detections.copy()
            >>> for detection in transformed:
            ...     detection.bbox = transform_bbox(detection.bbox)
            >>>
            >>> # Chain with filters
            >>> filtered = detections.copy().filter_by_confidence(0.7)

        Notes:
            - All Detection objects are deep copied using Detection.copy()
            - Completely independent from original collection
            - Safe for concurrent operations and immutable patterns
            - Useful for transform operations that modify detection data
        """
        new_detections = Detections()
        for detection in self.detections:
            new_detections.add_detection(detection.copy())
        return new_detections

    def to_json(self) -> str:
        """
        Convert all detections to JSON string format for export and storage.
        
        Transforms the entire detection collection into a formatted JSON string
        suitable for file export, API responses, and data interchange workflows.
        
        Returns:
            str: JSON string representation of all detections with proper indentation
                and formatting for readability.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Export to JSON string
            >>> json_data = detections.to_json()
            >>> 
            >>> # Save to file
            >>> with open("detections.json", "w") as f:
            ...     f.write(json_data)
            >>> 
            >>> # API response format
            >>> response_data = {"detections": json.loads(detections.to_json())}
        
        Notes:
            - Uses 4-space indentation for readability
            - Keypoints are recursively converted to dictionaries
            - Compatible with standard JSON parsing libraries
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert all detections to list of dictionaries for programmatic processing.
        
        Transforms the detection collection into a list of dictionary representations,
        providing structured data access for analysis, filtering, and integration workflows.
        
        Returns:
            List[Dict[str, Any]]: List of detection dictionaries with all fields
                                including nested keypoint data and metadata.
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Convert to dictionary format
            >>> data = detections.to_dict()
            >>> 
            >>> # Programmatic access
            >>> for detection_dict in data:
            ...     print(f"Class: {detection_dict['class_name']}")
            ...     print(f"Confidence: {detection_dict['confidence']}")
            >>> 
            >>> # Data analysis workflow
            >>> import pandas as pd
            >>> df = pd.DataFrame(data)
            >>> high_conf = df[df['confidence'] > 0.8]
        
        Notes:
            - Each detection is converted using its to_dict() method
            - Keypoints are recursively converted to nested dictionaries
            - Suitable for pandas DataFrame conversion and data analysis
        """
        return [detection.to_dict() for detection in self.detections]

    @classmethod
    def from_dict(cls, items: List[Dict[str, Any]]) -> "Detections":
        """
        Rebuild a Detections collection from the list `to_dict()` produces.

        The inverse of `to_dict()`. Reads results that were stored or sent as plain
        data back into a collection that filters, annotates and converts like one
        that came straight from a model.

        Args:
            items (List[Dict[str, Any]]): Detection dictionaries, as written by
                                         `to_dict()`. Each is read with
                                         `Detection.from_dict`.

        Returns:
            Detections: A new collection holding the detections in the order given.

        Raises:
            ValueError: If a mask uses an unknown format, or a keypoint coordinate
                        is not a finite number.
            TypeError: If a keypoint is missing `x`, `y` or `id`.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> detections = pf.from_arrays(boxes, scores, masks=masks)
            >>>
            >>> # Store as plain data, read back later
            >>> stored = detections.to_dict()
            >>> restored = pf.Detections.from_dict(stored)
            >>> len(restored) == len(detections)
            True
            >>>
            >>> # A restored result behaves like any other
            >>> confident = restored.filter_by_confidence(0.5)
            >>> restored.masks_to_segments()
            >>>
            >>> # Hand-written rows work too
            >>> pf.Detections.from_dict([{"bbox": [10, 20, 30, 40], "class_name": "car"}])

        Notes:
            - Everything `to_dict()` writes comes back: boxes, masks, segments,
              keypoints, classes, text, tracking, zones and timing
            - Detections are added through `add_detection`, so a dictionary without
              a valid `bbox`, or with a NaN confidence, is dropped as it would be
              from any converter. Nothing `to_dict()` wrote is affected: a
              collection never holds such a detection
            - Raster masks are decoded eagerly, one array per mask
        """
        detections = cls()
        for item in items:
            detections.add_detection(Detection.from_dict(item))
        return detections

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "Detections":
        """
        Rebuild a Detections collection from the JSON string `to_json()` produces.

        Args:
            text (Union[str, bytes]): JSON array of detection dictionaries.

        Returns:
            Detections: A new collection holding the detections in the order given.

        Raises:
            json.JSONDecodeError: If `text` is not valid JSON.
            ValueError: If a mask uses an unknown format, or a keypoint coordinate
                        is not a finite number.

        Example:
            >>> import pixelflow as pf
            >>>
            >>> # Write results to disk, read them back in another process
            >>> with open("results.json", "w") as f:
            ...     f.write(detections.to_json())
            >>> with open("results.json") as f:
            ...     restored = pf.Detections.from_json(f.read())
            >>>
            >>> # Or straight from an HTTP response body
            >>> restored = pf.Detections.from_json(response.content)

        Notes:
            - Parses the JSON and hands the result to `from_dict`; see it for what
              is restored and what is dropped
        """
        return cls.from_dict(json.loads(text))

    def to_json_with_metrics(self) -> str:
        """
        Convert detections to JSON with additional analytics metrics for comprehensive reporting.
        
        Generates a JSON representation that includes both detection data and computed
        analytics metrics for comprehensive reporting and analysis workflows.
        
        Returns:
            str: JSON string with detections and computed metrics including counts,
                confidence statistics, and spatial analytics (placeholder for future features).
        
        Example:
            >>> import pixelflow as pf
            >>> 
            >>> # Export with analytics
            >>> json_with_stats = detections.to_json_with_metrics()
            >>> 
            >>> # Save comprehensive report
            >>> with open("detection_report.json", "w") as f:
            ...     f.write(json_with_stats)
        
        Notes:
            - Currently identical to to_json(), future versions will include analytics
            - Planned metrics: detection counts, confidence distributions, zone statistics
            - Will include class distribution, temporal analytics, and spatial statistics
            - Designed for comprehensive reporting and dashboard integration
        """
        detections_dict = [detection.to_dict() for detection in self.detections]
        return json.dumps(detections_dict, indent=4)


# Import filter methods and attach them to Detections class for zero overhead
from .filters import (
    filter_by_confidence,
    filter_by_class_id,
    remap_class_ids,
    filter_by_size,
    filter_by_dimensions,
    filter_by_aspect_ratio,
    filter_by_zones,
    filter_by_position,
    filter_by_relative_size,
    filter_by_tracking_duration,
    filter_by_first_seen_time,
    filter_tracked_objects,
    remove_duplicates,
    filter_overlapping,
)


# Attach filter methods directly to Detections class - zero overhead method injection
Detections.filter_by_confidence = filter_by_confidence
Detections.filter_by_class_id = filter_by_class_id
Detections.remap_class_ids = remap_class_ids
Detections.filter_by_size = filter_by_size
Detections.filter_by_dimensions = filter_by_dimensions
Detections.filter_by_aspect_ratio = filter_by_aspect_ratio
Detections.filter_by_zones = filter_by_zones
Detections.filter_by_position = filter_by_position
Detections.filter_by_relative_size = filter_by_relative_size
Detections.filter_by_tracking_duration = filter_by_tracking_duration
Detections.filter_by_first_seen_time = filter_by_first_seen_time
Detections.filter_tracked_objects = filter_tracked_objects
Detections.remove_duplicates = remove_duplicates
Detections.filter_overlapping = filter_overlapping