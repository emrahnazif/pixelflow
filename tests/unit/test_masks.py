"""
Unit tests for raster-mask to polygon conversion.

Covers pf.mask_to_polygon.
"""

import cv2
import numpy as np
import pytest

import pixelflow as pf
from pixelflow.validators import validate_segments


def polygon_bbox(polygon):
    """Axis-aligned box around a polygon, as [x1, y1, x2, y2]."""
    points = np.asarray(polygon)
    return [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]


def mask_bbox(mask):
    """The box a mask occupies in pixel-corner coordinates: the far edge is exclusive."""
    rows, cols = np.nonzero(mask)
    return [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]


def polygon_area(polygon):
    """Shoelace area of a polygon."""
    points = np.asarray(polygon)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def distance_to_outline(points, polygon):
    """Distance from each point to the nearest edge of a closed polygon."""
    points = np.asarray(points, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    starts, ends = polygon, np.roll(polygon, -1, axis=0)
    edge = ends - starts                                    # (E, 2)
    offset = points[:, None, :] - starts[None, :, :]        # (P, E, 2)
    t = (offset * edge).sum(-1) / (edge * edge).sum(-1)
    nearest = starts[None] + np.clip(t, 0, 1)[..., None] * edge[None]
    return np.linalg.norm(points[:, None, :] - nearest, axis=-1).min(axis=1)


def smooth_blob(height=1281, width=1920):
    """A large three-lobed blob, the shape of a typical full-frame object mask."""
    rows, cols = np.mgrid[0:height, 0:width]
    dy, dx = rows - height / 2, cols - width / 2
    radius = 400 + 50 * np.sin(3 * np.arctan2(dy, dx))
    return np.hypot(dy, dx) < radius


# ============================================================================
# mask_to_polygon
# ============================================================================

class TestMaskToPolygon:
    """Test the single-mask conversion."""

    def test_rectangle_gives_its_four_pixel_corners(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[5:15, 10:20] = True  # columns 10..19, rows 5..14

        polygon = pf.mask_to_polygon(mask)

        assert len(polygon) == 4
        assert sorted(polygon) == [[10.0, 5.0], [10.0, 15.0], [20.0, 5.0], [20.0, 15.0]]

    @pytest.mark.parametrize("tolerance", [0, 0.5, 1.5, 5.0, 25.0])
    def test_bounding_box_equals_the_mask_box_at_any_tolerance(self, tolerance):
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 100), (90, 60), 20, 0, 360, 1, -1)

        polygon = pf.mask_to_polygon(mask, tolerance=tolerance)

        assert polygon_bbox(polygon) == mask_bbox(mask)

    def test_concave_corner_is_exact(self):
        """An L-shape has one inner corner, which OpenCV's tracer cuts diagonally."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:4] = True
        mask[6:8, 2:8] = True

        polygon = pf.mask_to_polygon(mask, tolerance=0)

        assert sorted(polygon) == sorted(
            [[2.0, 2.0], [4.0, 2.0], [4.0, 6.0], [8.0, 6.0], [8.0, 8.0], [2.0, 8.0]]
        )

    def test_exact_outline_encloses_exactly_the_mask_area(self):
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(mask, (60, 60), 41, 1, -1)

        polygon = pf.mask_to_polygon(mask, tolerance=0)

        assert polygon_area(polygon) == mask.sum()

    def test_ring_keeps_only_the_outer_outline(self):
        disc = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(disc, (30, 30), 25, 1, -1)
        ring = disc.copy()
        cv2.circle(ring, (30, 30), 10, 0, -1)

        # The hole changes nothing: the ring and the solid disc trace identically.
        assert pf.mask_to_polygon(ring) == pf.mask_to_polygon(disc)
        assert polygon_area(pf.mask_to_polygon(ring, tolerance=0)) == disc.sum()

    def test_two_blobs_give_the_larger_one(self):
        mask = np.zeros((50, 80), dtype=bool)
        mask[5:10, 5:10] = True
        mask[20:40, 30:70] = True

        polygon = pf.mask_to_polygon(mask)

        assert polygon_bbox(polygon) == [30, 20, 70, 40]

    def test_pixels_touching_only_at_a_corner_are_separate_regions(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:3, 1:3] = True
        mask[3:5, 3:6] = True  # larger, meets the first only at corner (3, 3)

        polygon = pf.mask_to_polygon(mask, tolerance=0)

        assert polygon_bbox(polygon) == [3, 3, 6, 5]

    def test_empty_mask_gives_none(self):
        assert pf.mask_to_polygon(np.zeros((20, 20), dtype=bool)) is None
        assert pf.mask_to_polygon(np.zeros((0, 0), dtype=bool)) is None

    @pytest.mark.parametrize("tolerance", [0, 1.5, 10.0])
    def test_one_pixel_mask_gives_that_pixels_square(self, tolerance):
        """Documented choice: a valid polygon, never None, however large the tolerance."""
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 4] = True

        polygon = pf.mask_to_polygon(mask, tolerance=tolerance)

        assert sorted(polygon) == [[4.0, 3.0], [4.0, 4.0], [5.0, 3.0], [5.0, 4.0]]

    def test_one_pixel_wide_line_keeps_its_area(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5, 2:18] = True

        polygon = pf.mask_to_polygon(mask)

        assert polygon_area(polygon) == 16
        assert polygon_bbox(polygon) == [2, 5, 18, 6]

    def test_mask_filling_the_whole_image(self):
        polygon = pf.mask_to_polygon(np.ones((30, 40), dtype=bool))

        assert sorted(polygon) == [[0.0, 0.0], [0.0, 30.0], [40.0, 0.0], [40.0, 30.0]]

    def test_uint8_255_mask_matches_boolean(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[5:25, 10:30] = True

        assert pf.mask_to_polygon(mask.astype(np.uint8) * 255) == pf.mask_to_polygon(mask)

    def test_output_is_already_in_segments_form(self):
        """Plain [[x, y], ...] floats that validate_segments leaves untouched."""
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 100), (90, 60), 20, 0, 360, 1, -1)

        polygon = pf.mask_to_polygon(mask, tolerance=0.5)

        assert validate_segments(polygon) == polygon
        assert all(type(value) is float for point in polygon for value in point)
        assert pf.Detection(segments=polygon).segments == polygon

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError, match="2D"):
            pf.mask_to_polygon(np.zeros((3, 20, 20), dtype=bool))
        with pytest.raises(ValueError, match="tolerance"):
            pf.mask_to_polygon(np.ones((5, 5), dtype=bool), tolerance=-1)


class TestMaskToPolygonRealSize:
    """Test a full-frame mask of a large smooth object."""

    @pytest.fixture(scope="class")
    def blob(self):
        return smooth_blob()

    @pytest.fixture(scope="class")
    def exact(self, blob):
        return pf.mask_to_polygon(blob, tolerance=0)

    @pytest.mark.parametrize("tolerance", [0.5, 1.0, 1.5, 3.0])
    def test_simplified_polygon_stays_within_tolerance(self, blob, exact, tolerance):
        polygon = pf.mask_to_polygon(blob, tolerance=tolerance)

        assert distance_to_outline(exact, polygon).max() <= tolerance + 1e-6

    def test_default_tolerance_gives_tens_of_points(self, blob, exact):
        polygon = pf.mask_to_polygon(blob)

        assert len(exact) > 1000  # the staircase it started from
        assert len(polygon) < 200

    def test_area_is_preserved(self, blob):
        polygon = pf.mask_to_polygon(blob)

        assert polygon_area(polygon) == pytest.approx(blob.sum(), rel=1e-3)
