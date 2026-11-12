"""
Unit tests for pixelflow.detections module.

Tests the core Detection, Detections, and KeyPoint classes including
creation, serialization, and basic operations.
"""

import inspect

import pytest
import numpy as np
import json
import pixelflow as pf


# ============================================================================
# KeyPoint Tests
# ============================================================================

class TestKeyPoint:
    """Tests for the KeyPoint class."""

    def test_keypoint_creation(self):
        """Test basic keypoint creation."""
        kp = pf.KeyPoint(x=100, y=200, id=0, name="nose", confidence=0.9)
        assert kp.x == 100
        assert kp.y == 200
        assert kp.name == "nose"
        assert kp.id == 0
        assert kp.confidence == 0.9

    def test_keypoint_to_dict(self, sample_keypoint):
        """Test keypoint serialization to dict."""
        data = sample_keypoint.to_dict()
        assert data["x"] == 100
        assert data["y"] == 200
        assert data["name"] == "nose"
        assert data["id"] == 0
        assert data["confidence"] == 0.9

    def test_keypoint_numpy_conversion(self):
        """Test keypoint with numpy numeric types."""
        kp = pf.KeyPoint(
            x=np.int64(150),
            y=np.float32(250.5),
            name="point",
            id=0,
            confidence=0.9
        )
        data = kp.to_dict()
        # Native Python types, not numpy scalars, so the payload is JSON-safe.
        assert type(data["x"]) is float
        assert type(data["y"]) is float
        assert json.dumps(data, allow_nan=False)

    def test_keypoint_preserves_subpixel_precision(self):
        """Landmarks keep their fraction; truncating biased them toward the origin."""
        kp = pf.KeyPoint(x=10.7, y=20.3, id=0)
        assert (kp.x, kp.y) == (10.7, 20.3)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_keypoint_rejects_non_finite(self, bad):
        """A non-finite landmark fails at construction rather than downstream.

        Unlike bbox there is no None-coordinate convention for consumers to
        guard against, so this raises instead of degrading.
        """
        with pytest.raises(ValueError):
            pf.KeyPoint(x=bad, y=0.0, id=0)


# ============================================================================
# Detection Tests
# ============================================================================

class TestDetection:
    """Tests for the Detection class."""

    def test_detection_creation_minimal(self):
        """Test detection with minimal required fields."""
        det = pf.Detection(bbox=[100, 100, 200, 200])
        assert det.bbox == [100, 100, 200, 200]
        assert det.confidence is None
        assert det.class_id is None
        assert det.class_name is None

    def test_detection_creation_full(self):
        """Test detection with all common fields."""
        det = pf.Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            class_id=0,
            class_name="person",
            labels=["person 0.95"]
        )
        assert det.bbox == [100, 100, 200, 200]
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.labels == ["person 0.95"]

    def test_detection_with_mask(self):
        """Test detection with binary mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        det = pf.Detection(
            bbox=[100, 100, 200, 200],
            masks=[mask]
        )
        assert len(det.masks) == 1
        assert det.masks[0].shape == (100, 100)
        assert det.masks[0].dtype == bool

    def test_detection_with_keypoints(self, sample_keypoints):
        """Test detection with keypoints."""
        det = pf.Detection(
            bbox=[50, 50, 150, 200],
            keypoints=sample_keypoints
        )
        assert len(det.keypoints) == 5
        assert det.keypoints[0].name == "nose"


    def test_detection_with_tracking(self):
        """Test detection with tracking information."""
        det = pf.Detection(
            bbox=[100, 100, 200, 200],
            tracker_id=42,
            first_seen_time=1.5,
            total_time=3.2
        )
        assert det.tracker_id == 42
        assert det.first_seen_time == 1.5
        assert det.total_time == 3.2

    def test_detection_to_dict_basic(self, sample_detection):
        """Test detection serialization to dict."""
        data = sample_detection.to_dict()
        assert data["bbox"] == [100, 100, 200, 200]
        assert data["confidence"] == 0.95
        assert data["class_id"] == 0
        assert data["class_name"] == "person"

    def test_detection_to_dict_with_mask(self, sample_detection_with_mask):
        """Test detection with mask serialization."""
        data = sample_detection_with_mask.to_dict()
        assert "masks" in data
        assert len(data["masks"]) == 1
        # Mask should be a PNG-encoded payload
        assert isinstance(data["masks"][0], dict)
        assert "data" in data["masks"][0]
        assert "shape" in data["masks"][0]

    def test_detection_decode_mask(self):
        """Test mask decoding from PNG."""
        # Create and encode mask
        original_mask = np.zeros((50, 50), dtype=bool)
        original_mask[10:40, 10:40] = True

        det = pf.Detection(bbox=[0, 0, 50, 50], masks=[original_mask])
        data = det.to_dict()

        # Decode mask
        decoded_mask = pf.Detection.decode_mask(data["masks"][0])
        assert decoded_mask.shape == original_mask.shape
        assert decoded_mask.dtype == bool
        assert np.array_equal(decoded_mask, original_mask)

    def test_detection_mask_uses_png_format(self):
        """Masks serialize as PNG, not a raw bitmap."""
        det = pf.Detection(
            bbox=[0, 0, 50, 50], masks=[np.zeros((50, 50), dtype=bool)]
        )
        assert det.to_dict()["masks"][0]["format"] == "png"

    def test_detection_decode_mask_uint8_roundtrip(self):
        """uint8 masks round-trip through PNG without dtype drift."""
        original_mask = np.zeros((40, 60), dtype=np.uint8)
        original_mask[5:35, 10:50] = 255

        det = pf.Detection(bbox=[0, 0, 60, 40], masks=[original_mask])
        decoded_mask = pf.Detection.decode_mask(det.to_dict()["masks"][0])

        assert decoded_mask.dtype == np.uint8
        assert np.array_equal(decoded_mask, original_mask)

    @pytest.mark.parametrize(
        "mask,expected_format",
        [
            (np.zeros((20, 30), dtype=bool), "png"),
            (np.zeros((20, 30), dtype=np.uint8), "png"),
            (np.zeros((20, 30), dtype=np.uint16), "png"),
            (np.zeros((20, 30, 3), dtype=np.uint8), "png"),
            # Pillow's 'I' mode PNG save is removed in Pillow 13 -> raw.
            (np.zeros((20, 30), dtype=np.int32), "raw"),
            (np.zeros((20, 30), dtype=np.int64), "raw"),
            (np.zeros((20, 30), dtype=np.float32), "raw"),
            (np.zeros((20, 30), dtype=np.float64), "raw"),
            (np.zeros((5, 20, 30), dtype=bool), "raw"),
            (np.zeros((0, 0), dtype=bool), "raw"),
        ],
    )
    def test_detection_mask_roundtrip_all_dtypes(self, mask, expected_format):
        """Every dtype stays serializable; PNG where possible, raw otherwise."""
        rng = np.random.default_rng(0)
        if mask.size:
            mask = (rng.random(mask.shape) * 200).astype(mask.dtype)

        det = pf.Detection(bbox=[0, 0, 30, 20], masks=[mask])
        encoded = det.to_dict()["masks"][0]
        decoded = pf.Detection.decode_mask(encoded)

        assert encoded["format"] == expected_format
        assert decoded.shape == mask.shape
        assert decoded.dtype == mask.dtype
        assert np.array_equal(decoded, mask)
        # Callers mutate masks; np.frombuffer alone would return a read-only view.
        assert decoded.flags.writeable

    def test_detection_mask_payload_is_compressed(self):
        """Guard against regressing to a raw 1-byte-per-pixel bitmap."""
        height, width = 1080, 1920
        mask = np.zeros((height, width), dtype=bool)
        mask[100:900, 200:1500] = True

        det = pf.Detection(bbox=[200, 100, 1500, 900], masks=[mask])
        payload = det.to_dict()["masks"][0]["data"]

        # A raw bitmap would base64 to ~2.8 MB; PNG must stay far below that.
        assert len(payload) < height * width / 100

    def test_detection_to_dict_is_json_serializable(self, sample_detection):
        """Detection exposes to_dict(); to_json() lives on the Detections container."""
        assert not hasattr(sample_detection, "to_json")

        data = json.loads(json.dumps(sample_detection.to_dict()))
        assert data["bbox"] == [100, 100, 200, 200]
        assert data["confidence"] == 0.95

    def test_detection_bbox_validates_on_reassignment(self):
        """Writing to .bbox validates, not just constructing.

        Transforms rewrite this attribute in place, so validating only in
        __init__ would let transformed boxes escape the precision contract.
        """
        det = pf.Detection(bbox=[0, 0, 10, 10])
        det.bbox = [1.23456, 2.0, 3.0, 4.0]
        assert det.bbox == [1.23, 2.0, 3.0, 4.0]

    def test_detection_bbox_rejects_non_finite_on_reassignment(self):
        """A non-finite coordinate written post-construction is rejected too."""
        det = pf.Detection(bbox=[0, 0, 10, 10])
        det.bbox = [0.0, 0.0, float("nan"), 10.0]
        assert det.bbox is None

    def test_detection_bbox_accepts_none(self):
        # Legal on a Detection, which is the mid-build state a converter passes
        # through; a Detections container rejects it. See TestDetectionValidity.
        """Clearing the box stays legal."""
        det = pf.Detection(bbox=[0, 0, 10, 10])
        det.bbox = None
        assert det.bbox is None

    @pytest.mark.parametrize(
        "raw",
        [
            [[10.7, 20.3], [50.9, 60.4]],                              # ultralytics
            [(10.7, 20.3), (50.9, 60.4)],                              # florence-2
            np.array([[10.7, 20.3], [50.9, 60.4]], dtype=np.float32),  # raw model output
        ],
        ids=["list", "tuple", "float32"],
    )
    def test_detection_segments_normalize_to_one_shape(self, raw):
        """Every shape a converter produces is stored identically.

        Callers used to branch on which of the three they had been handed, and
        the branches disagreed - one raised, one silently dropped the polygon.
        """
        det = pf.Detection(bbox=[0, 0, 60, 60], segments=raw)
        assert det.segments == [[10.7, 20.3], [50.9, 60.4]]

    def test_detection_segments_validate_on_reassignment(self):
        """Transforms rewrite this attribute, so writes validate like bbox writes."""
        det = pf.Detection(bbox=[0, 0, 60, 60], segments=[[1.0, 2.0]])
        det.segments = np.array([[10.7, 20.3]], dtype=np.float32)
        assert det.segments == [[10.7, 20.3]]

    def test_detection_bbox_is_json_safe(self):
        """A validated bbox always survives a strict JSON encoder."""
        det = pf.Detection(bbox=[10.7, 20.3, 110.9, 220.4])
        assert json.dumps(det.to_dict()["bbox"], allow_nan=False) == "[10.7, 20.3, 110.9, 220.4]"

    def test_detection_bbox_survives_copy(self):
        """copy() round-trips through the property without losing precision."""
        det = pf.Detection(bbox=[10.7, 20.3, 110.9, 220.4])
        assert det.copy().bbox == [10.7, 20.3, 110.9, 220.4]


# ============================================================================
# Detections Container Tests
# ============================================================================

class TestDetections:
    """Tests for the Detections container class."""

    def test_detections_creation_empty(self):
        """Test creating empty Detections container."""
        dets = pf.Detections()
        assert len(dets) == 0

    def test_detections_add_detection(self, empty_detections, sample_detection):
        """Test adding detection to container."""
        empty_detections.add_detection(sample_detection)
        assert len(empty_detections) == 1

    def test_detections_iteration(self, sample_detections):
        """Test iterating over detections."""
        count = 0
        for det in sample_detections:
            assert isinstance(det, pf.Detection)
            count += 1
        assert count == 4

    def test_detections_indexing(self, sample_detections):
        """Test accessing detections by index."""
        first = sample_detections[0]
        assert isinstance(first, pf.Detection)
        assert first.bbox == [100, 100, 200, 200]

    def test_detections_len(self, sample_detections):
        """Test length of detections container."""
        assert len(sample_detections) == 4

    def test_detections_bool(self, sample_detections, empty_detections):
        """Test boolean conversion of detections."""
        assert bool(sample_detections) is True
        assert bool(empty_detections) is False

    def test_detections_to_list(self, sample_detections):
        """Test converting detections to list of dicts."""
        det_list = [d.to_dict() for d in sample_detections]
        assert len(det_list) == 4
        assert all(isinstance(d, dict) for d in det_list)

    def test_detections_to_json(self, sample_detections):
        """Test JSON serialization of detections container."""
        json_str = sample_detections.to_json()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 4

    def test_detections_class_names_property(self, sample_detections):
        """Test extracting class names from detections."""
        class_names = [d.class_name for d in sample_detections if d.class_name]
        assert "person" in class_names
        assert "car" in class_names
        assert "bicycle" in class_names

    def test_detections_confidence_scores_property(self, sample_detections):
        """Test extracting confidence scores from detections."""
        confidences = [d.confidence for d in sample_detections if d.confidence is not None]
        assert len(confidences) == 4
        assert max(confidences) == 0.95
        assert min(confidences) == 0.45

    def test_detections_bboxes_property(self, sample_detections):
        """Test extracting bboxes from detections."""
        bboxes = [d.bbox for d in sample_detections if d.bbox is not None]
        assert len(bboxes) == 4
        assert bboxes[0] == [100, 100, 200, 200]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestDetectionEdgeCases:
    """Test edge cases and error handling."""

    def test_detection_with_none_values(self):
        """Test detection with explicitly None values."""
        det = pf.Detection(
            bbox=[0, 0, 10, 10],
            confidence=None,
            class_id=None,
            class_name=None
        )
        assert det.confidence is None
        assert det.class_id is None
        assert det.class_name is None

    def test_detection_empty_masks_list(self):
        """Test detection with empty masks list."""
        det = pf.Detection(bbox=[0, 0, 10, 10], masks=[])
        assert det.masks == []

    def test_detection_empty_keypoints_list(self):
        """Test detection with empty keypoints list."""
        det = pf.Detection(bbox=[0, 0, 10, 10], keypoints=[])
        assert det.keypoints == []

    def test_detections_slicing(self, sample_detections):
        """Test slicing detections container."""
        subset = sample_detections[1:3]
        # Slicing should work like a list
        assert len(subset) == 2


# ============================================================================
# What counts as a detection at all
# ============================================================================

class TestDetectionValidity:
    """Garbage geometry never becomes a detection.

    Every consumer downstream -- filters, zones, annotators, the tracker --
    assumes a detection has a box that encloses pixels. That assumption is only
    safe because nothing else can get in, so this is where it is pinned.
    """

    @pytest.mark.parametrize("bbox,reason", [
        ([float("nan"), 10, 60, 60], "non-finite coordinate"),
        ([60, 60, 10, 10], "inverted corners"),
        ([60, 10, 10, 60], "inverted on x only"),
        ([50, 50, 50, 90], "zero width"),
        ([50, 50, 90, 50], "zero height"),
        ([50, 50, 50, 50], "zero area"),
    ])
    def test_invalid_geometry_never_becomes_a_detection(self, bbox, reason):
        """Regression: an inverted box computed a positive area from two negative
        sides, so it passed every size filter while scoring zero IoU against
        everything -- a detection that inflated counts and that no deduplication
        could remove."""
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=bbox, confidence=0.9))
        assert len(detections) == 0, f"{reason} was accepted"

    @pytest.mark.parametrize("bbox", [
        [10, 10, 60, 60],
        [-20, -20, 30, 30],      # partly off-frame is legitimate
        [0, 0, 1e6, 1e6],        # a full-frame detection is legitimate
        [10.5, 10.5, 60.25, 60.25],
    ])
    def test_valid_geometry_is_kept(self, bbox):
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=bbox, confidence=0.9))
        assert len(detections) == 1

    def test_nan_confidence_never_becomes_a_detection(self):
        """NaN compares False against every threshold and is not valid JSON."""
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=[10, 10, 60, 60],
                                              confidence=float("nan")))
        assert len(detections) == 0

    def test_missing_confidence_is_kept(self):
        """Absent is not the same as corrupt: from_easyocr paragraphs have none."""
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=[10, 10, 60, 60]))
        assert len(detections) == 1

    @pytest.mark.parametrize("confidence", [1.5, -0.2])
    def test_out_of_range_confidence_is_kept(self, confidence):
        """A miscalibrated score is a scoring problem, not a reason to discard
        an object the model genuinely located."""
        detections = pf.Detections()
        detections.add_detection(pf.Detection(bbox=[10, 10, 60, 60],
                                              confidence=confidence))
        assert len(detections) == 1

    def test_converters_drop_invalid_rows(self):
        """The boundary holds for the path real callers use."""
        result = pf.from_arrays([[10, 10, 60, 60], [60, 60, 10, 10], [50, 50, 50, 50]],
                                scores=[0.9, 0.9, 0.9], class_ids=[0, 0, 0])
        assert len(result) == 1
        assert result[0].bbox == [10.0, 10.0, 60.0, 60.0]


# ============================================================================
# Reading to_dict() back
# ============================================================================

def full_detection():
    """A detection with every field set, already at pixelflow's stored precision."""
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:30, 20:50] = True
    return pf.Detection(
        inference_id="run-7",
        bbox=[20.25, 10, 50, 30.5],
        masks=[mask],
        segments=[[20, 10], [50, 10], [50, 30], [20, 30]],
        keypoints=[pf.KeyPoint(25.5, 12, 0, "nose", 0.91), pf.KeyPoint(40, 22.75, 1)],
        class_id=3,
        class_name="cat",
        labels=["indoor", "pet"],
        confidence=0.876,
        tracker_id=7,
        metadata={"source": "cam-1", "tags": [1, 2]},
        zones=["z1"],
        zone_names=["Entrance"],
        line_crossings=[{"line": "l1", "direction": "in"}],
        first_seen_time=1.5,
        total_time=2.5,
        text="hello",
    )


class TestFromDict:
    """Test Detection.from_dict, Detections.from_dict and Detections.from_json."""

    def test_keypoint_round_trip(self, sample_keypoint):
        restored = pf.KeyPoint.from_dict(sample_keypoint.to_dict())

        assert isinstance(restored, pf.KeyPoint)
        assert restored.to_dict() == sample_keypoint.to_dict()

    def test_keypoint_optional_fields_default_and_unknown_keys_are_ignored(self):
        restored = pf.KeyPoint.from_dict({"x": 1, "y": 2, "id": 4, "visible": True})

        assert (restored.x, restored.y, restored.id) == (1.0, 2.0, 4)
        assert restored.name is None and restored.confidence is None

    def test_keypoint_without_a_position_is_refused(self):
        with pytest.raises(TypeError):
            pf.KeyPoint.from_dict({"id": 0, "name": "nose"})
        with pytest.raises(ValueError):
            pf.KeyPoint.from_dict({"x": float("nan"), "y": 1, "id": 0})

    def test_brief_round_trip_box_mask_keypoints_class_confidence(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[10:40, 20:70] = True
        detections = pf.from_arrays(
            boxes=[[20, 10, 70, 40]], scores=[0.83], class_ids=[1], masks=mask[None],
            keypoints=np.array([[[30.5, 15.25, 0.9], [60.0, 35.0, 0.4]]]),
            labels=["cat", "dog"],
        )

        restored = pf.Detections.from_dict(detections.to_dict())

        assert len(restored) == 1
        original, back = detections[0], restored[0]
        assert back.bbox == original.bbox == [20.0, 10.0, 70.0, 40.0]
        assert back.masks[0].dtype == bool and np.array_equal(back.masks[0], mask)
        assert [kp.to_dict() for kp in back.keypoints] == [kp.to_dict() for kp in original.keypoints]
        assert all(isinstance(kp, pf.KeyPoint) for kp in back.keypoints)
        assert (back.class_id, back.class_name, back.confidence) == (1, "dog", 0.83)

    def test_every_field_round_trips(self):
        original = full_detection()

        restored = pf.Detection.from_dict(original.to_dict())

        assert restored.to_dict() == original.to_dict()
        assert np.array_equal(restored.masks[0], original.masks[0])

    def test_every_field_round_trips_through_json(self):
        detections = pf.Detections()
        detections.add_detection(full_detection())

        restored = pf.Detections.from_json(detections.to_json())

        assert restored.to_dict() == detections.to_dict()
        assert pf.Detections.from_json(detections.to_json().encode()).to_dict() == detections.to_dict()

    def test_to_dict_and_the_constructor_name_the_same_fields(self):
        """from_dict reads the constructor's arguments; this keeps to_dict in step with them."""
        assert set(pf.Detection().to_dict()) == set(inspect.signature(pf.Detection).parameters)
        assert set(pf.KeyPoint(0, 0, 0).to_dict()) == set(inspect.signature(pf.KeyPoint).parameters)

    @pytest.mark.parametrize("mask", [
        np.random.default_rng(0).random((37, 53)) > 0.5,
        np.random.default_rng(1).random((20, 20)).astype(np.float32),
    ], ids=["png", "raw"])
    def test_raster_masks_come_back_with_pixels_dtype_and_shape(self, mask):
        """One mask per encoding; the codec's own dtype matrix is tested with decode_mask."""
        detection = pf.Detection(bbox=[0, 0, 10, 10], masks=[mask])

        restored = pf.Detection.from_dict(json.loads(json.dumps(detection.to_dict())))

        assert restored.masks[0].dtype == mask.dtype
        assert restored.masks[0].shape == mask.shape
        assert np.array_equal(restored.masks[0], mask)

    def test_polygon_mask_round_trips(self):
        polygon = [[1.0, 2.0], [5.0, 2.0], [5.0, 9.0]]
        detection = pf.Detection(bbox=[1, 2, 5, 9], masks=[polygon])

        restored = pf.Detection.from_dict(detection.to_dict())

        assert restored.masks == [polygon]
        assert restored.masks[0] is not polygon

    def test_masks_already_decoded_are_accepted(self):
        raster = np.ones((4, 4), dtype=bool)
        polygon = [[0, 0], [4, 0], [4, 4]]

        restored = pf.Detection.from_dict({"bbox": [0, 0, 4, 4], "masks": [raster, polygon]})

        assert restored.masks[0] is raster
        assert restored.masks[1] == polygon and restored.masks[1] is not polygon

    def test_a_container_field_added_later_is_copied_without_being_listed(self):
        """The copy is by type, not by field name, so new fields are safe by default."""
        from pixelflow.serialization import constructor_fields, own

        class Future:
            def __init__(self, known=None, added_later=None):
                pass

        data = {"known": 1, "added_later": {"tags": [1, 2]}, "unknown": 3}
        fields = own(constructor_fields(Future, data))

        assert fields == {"known": 1, "added_later": {"tags": [1, 2]}}
        assert fields["added_later"] is not data["added_later"]
        assert fields["added_later"]["tags"] is not data["added_later"]["tags"]

    def test_unknown_mask_format_is_refused(self):
        with pytest.raises(ValueError, match="Unknown mask format"):
            pf.Detection.from_dict({"bbox": [0, 0, 4, 4], "masks": [{"format": "rle", "data": ""}]})

    def test_unknown_keys_are_ignored(self):
        data = full_detection().to_dict()
        data["added_by_a_newer_version"] = {"anything": 1}

        assert pf.Detection.from_dict(data).to_dict() == full_detection().to_dict()

    def test_missing_keys_take_the_constructor_defaults(self):
        assert pf.Detection.from_dict({}).to_dict() == pf.Detection().to_dict()

        partial = pf.Detection.from_dict({"bbox": [1, 2, 3, 4], "class_name": "car"})
        assert partial.bbox == [1.0, 2.0, 3.0, 4.0]
        assert partial.class_name == "car"
        assert partial.masks is None and partial.keypoints is None
        assert partial.zones == [] and partial.total_time == 0.0

    def test_restored_detection_shares_nothing_with_its_source(self):
        original = full_detection()

        restored = pf.Detection.from_dict(original.to_dict())
        restored.metadata["tags"].append(3)
        restored.zones.append("z2")
        restored.labels.append("extra")
        restored.line_crossings[0]["direction"] = "out"
        restored.masks[0][:] = False

        assert original.to_dict() == full_detection().to_dict()
        assert original.masks[0].any()

    def test_values_are_validated_like_any_new_detection(self):
        restored = pf.Detection.from_dict({"bbox": [10.123456, 20, 30, 40], "confidence": 0.87654})

        assert restored.bbox == [10.12, 20.0, 30.0, 40.0]
        assert restored.confidence == 0.877
        assert pf.Detection.from_dict({"bbox": [30, 20, 10, 40]}).bbox is None

    def test_collection_keeps_order_and_count(self, sample_detections):
        restored = pf.Detections.from_dict(sample_detections.to_dict())

        assert isinstance(restored, pf.Detections)
        assert len(restored) == len(sample_detections)
        assert restored.to_dict() == sample_detections.to_dict()

    def test_collection_drops_rows_that_describe_no_object(self):
        restored = pf.Detections.from_dict([
            {"bbox": [1, 2, 3, 4], "class_name": "kept"},
            {"class_name": "no box"},
            {"bbox": [1, 2, 3, 4], "confidence": float("nan")},
        ])

        assert [d.class_name for d in restored] == ["kept"]

    def test_empty_collection(self):
        assert len(pf.Detections.from_dict([])) == 0
        assert len(pf.Detections.from_json("[]")) == 0
        assert len(pf.Detections.from_json(pf.Detections().to_json())) == 0

    def test_report_with_metrics_reads_too(self, sample_detections):
        """Its rows carry extra keys, which are ignored like any other unknown key."""
        rows = json.loads(sample_detections.to_json_with_metrics())

        assert pf.Detections.from_dict(rows).to_dict() == sample_detections.to_dict()

    def test_restored_result_behaves_like_a_fresh_one(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[10:40, 20:70] = True
        stored = pf.from_arrays([[20, 10, 70, 40]], [0.9], masks=mask[None]).to_json()

        restored = pf.Detections.from_json(stored).masks_to_segments()

        assert sorted(restored[0].segments) == [[20.0, 10.0], [20.0, 40.0], [70.0, 10.0], [70.0, 40.0]]
        assert len(restored.filter_by_confidence(0.5)) == 1

