"""
Regression tests for the utilities fixed while writing their docs.

Each test pins one defect found by running the documented examples: an inverse that
did not invert, a buffer that lost frames, a slicer that left masks in slice space.
"""

import contextlib
import warnings

import numpy as np
import pytest
from PIL import Image

import pixelflow as pf
from tests.helpers import make_detections


def frame(height=480, width=640):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def boxes(detections):
    return [np.round(d.bbox, 6).tolist() for d in detections]


BOXES = [[100, 200, 300, 400], [400, 50, 600, 300]]


# ---------------------------------------------------------------------------
# inverse_transforms
# ---------------------------------------------------------------------------

class TestInverseTransforms:

    @pytest.mark.parametrize("angle", [15, 30, 90, -45, 180])
    def test_rotation_round_trip_is_exact(self, angle):
        """Regression: a box rotated 30 degrees came back hundreds of pixels off."""
        source = make_detections(BOXES)
        _, rotated = pf.transform.rotate_detections(frame(), source, angle=angle)
        assert boxes(pf.transform.inverse_transforms(rotated)) == boxes(source)

    @pytest.mark.parametrize("angle", [15, 30, -60])
    def test_rotation_inverts_keypoints_and_segments(self, angle):
        """Points have no envelope to lose: geometry alone must bring them back."""
        source = make_detections([[100, 100, 200, 200]])
        source[0].keypoints = [pf.KeyPoint(x=120, y=130, id=0, name="a"),
                               pf.KeyPoint(x=180, y=190, id=1, name="b")]
        source[0].segments = [[100, 100], [200, 100], [200, 200], [100, 200]]
        _, rotated = pf.transform.rotate_detections(frame(), source, angle=angle)
        back = pf.transform.inverse_transforms(rotated)[0]
        # Coordinates are stored to 2 decimals, once after each of the two rotations.
        assert np.allclose([[k.x, k.y] for k in back.keypoints], [[120, 130], [180, 190]], atol=0.02)
        assert np.allclose(back.segments, source[0].segments, atol=0.02)

    def test_rotation_inverts_a_raster_mask(self):
        image = frame()
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[200:300, 250:400] = True
        source = make_detections([[250, 200, 400, 300]])
        source[0].masks = [mask]
        _, rotated = pf.transform.rotate_detections(image, source, angle=30)
        back = pf.transform.inverse_transforms(rotated)[0].masks[0]
        assert back.shape == mask.shape
        assert (back == mask).mean() > 0.99

    def test_box_edited_after_rotation_is_inverted_by_geometry(self):
        """No exact record applies to a changed box; it must still land near the object."""
        source = make_detections([[100, 200, 300, 400]])
        _, rotated = pf.transform.rotate_detections(frame(), source, angle=30)
        rotated[0].bbox = [v + 0.5 for v in rotated[0].bbox]
        back = pf.transform.inverse_transforms(rotated)[0].bbox
        # The envelope of the envelope contains the original box and is centred on it.
        assert back[0] <= 100 and back[1] <= 200 and back[2] >= 300 and back[3] >= 400
        assert abs((back[0] + back[2]) / 2 - 200) < 2 and abs((back[1] + back[3]) / 2 - 300) < 2

    @pytest.mark.parametrize("padding,reference", [
        (0.1, "shorter"), (0.25, "longer"), (0.2, "width"), (0.3, "height"),
        ({"left": 0.3, "right": 0.1, "top": 0.2, "bottom": 0}, "shorter"),
    ])
    def test_padding_round_trip_is_exact(self, padding, reference):
        """Regression: padding was recomputed from the padded box, which came back smaller."""
        source = make_detections(BOXES)
        padded = pf.transform.add_padding(source, padding, reference=reference)
        assert np.allclose(boxes(pf.transform.inverse_transforms(padded)), boxes(source))

    def test_padding_history_without_pixels_warns_and_keeps_the_box(self):
        """Old history cannot be undone exactly; saying so beats a wrong box."""
        source = make_detections(BOXES)
        padded = pf.transform.add_padding(source, 0.1)
        for detection in padded:
            del detection.metadata["transforms"][-1]["pixels"]
        with pytest.warns(UserWarning, match="pixel amounts"):
            back = pf.transform.inverse_transforms(padded)
        assert boxes(back) == boxes(padded)

    def test_rotation_history_without_matrix_is_rebuilt(self):
        """History recorded before the matrix was still inverts exactly."""
        source = make_detections([[100, 200, 300, 400]])
        source[0].keypoints = [pf.KeyPoint(x=150, y=250, id=0, name="a")]
        _, rotated = pf.transform.rotate_detections(frame(), source, angle=30)
        del rotated[0].metadata["transforms"][-1]["matrix"]
        back = pf.transform.inverse_transforms(rotated)[0]
        assert np.allclose([back.keypoints[0].x, back.keypoints[0].y], [150, 250], atol=0.02)

    def test_crop_cut_boxes_come_back_whole(self):
        source = make_detections([[-0, 100, 300, 300], [500, 300, 700, 470]])
        _, cropped = pf.transform.crop_detections(frame(), source, [100, 50, 600, 400])
        assert boxes(pf.transform.inverse_transforms(cropped)) == boxes(source)

    def test_crop_restores_raster_masks_to_image_size(self):
        """Regression: a cropped mask stayed the size of the crop."""
        image = frame()
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[120:220, 150:260] = True
        source = make_detections([[150, 120, 260, 220]])
        source[0].masks = [mask]
        _, cropped = pf.transform.crop_detections(image, source, [100, 50, 600, 400])
        assert cropped[0].masks[0].shape == (350, 500)
        back = pf.transform.inverse_transforms(cropped)[0].masks[0]
        assert back.shape == mask.shape and (back == mask).all()

    def test_chain_round_trip_is_exact(self):
        source = make_detections(BOXES)
        image, out = pf.transform.rotate_detections(frame(), source, angle=25)
        image, out = pf.transform.crop_detections(image, out, [50, 50, 700, 600])
        image, out = pf.transform.flip_horizontal_detections(image, out)
        out = pf.transform.add_padding(out, 0.15)
        assert np.allclose(boxes(pf.transform.inverse_transforms(out)), boxes(source))


# ---------------------------------------------------------------------------
# crop_detections
# ---------------------------------------------------------------------------

class TestCropDetections:

    def test_boxes_are_cut_at_the_crop(self):
        """Regression: a box crossing the crop kept coordinates outside it (x = -600)."""
        source = make_detections([[0, 100, 300, 300], [500, 300, 700, 470]])
        crop, out = pf.transform.crop_detections(frame(), source, [100, 50, 600, 400])
        h, w = crop.shape[:2]
        for x1, y1, x2, y2 in boxes(out):
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert boxes(out) == [[0, 50, 200, 250], [400, 250, 500, 350]]

    def test_a_box_only_touching_the_crop_is_dropped(self):
        source = make_detections([[0, 0, 100, 100]])
        _, out = pf.transform.crop_detections(frame(), source, [100, 0, 300, 200])
        assert len(out) == 0

    def test_crop_region_outside_the_image_is_clipped(self):
        """Regression: x1 < 0 shifted every box by that amount, off the cropped pixels."""
        image = frame()
        source = make_detections([[50, 60, 150, 160]])
        crop, out = pf.transform.crop_detections(image, source, [-40, -30, 300, 300])
        assert crop.shape[:2] == (300, 300)
        assert boxes(out) == [[50, 60, 150, 160]]

    def test_all_keypoints_are_kept(self):
        """Regression: keypoints outside the crop were dropped, shifting the list."""
        source = make_detections([[100, 100, 300, 300]])
        source[0].keypoints = [pf.KeyPoint(x=120, y=120, id=0, name="nose"),
                               pf.KeyPoint(x=290, y=290, id=1, name="ankle")]
        _, out = pf.transform.crop_detections(frame(), source, [100, 100, 200, 200])
        assert [k.name for k in out[0].keypoints] == ["nose", "ankle"]
        assert [(k.x, k.y) for k in out[0].keypoints] == [(20, 20), (190, 190)]


# ---------------------------------------------------------------------------
# gamma_correction
# ---------------------------------------------------------------------------

def test_gamma_above_one_brightens():
    """The docstring said the opposite of the code; the code is the contract."""
    image = frame()
    assert pf.transform.gamma_correction(image, 2.0).mean() > image.mean()
    assert pf.transform.gamma_correction(image, 0.5).mean() < image.mean()
    assert "brightens" in pf.transform.gamma_correction.__doc__.split(">1")[1][:12]


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

def numbered(i):
    return np.full((4, 4, 3), i, dtype=np.uint8)


class TestBufferEveryFrame:

    @pytest.mark.parametrize("size,count", [(5, 10), (3, 7), (1, 4), (4, 9), (7, 3)])
    def test_every_frame_comes_out_once_in_order(self, size, count):
        """Regression: the first and last `delay` frames were never returned."""
        expect_warning = pytest.warns(UserWarning) if size % 2 == 0 else contextlib.nullcontext()
        with expect_warning:
            buffer = pf.Buffer(frames=size)
        out = []
        for i in range(count):
            results, image = buffer.update(i, numbered(i))
            if image is not None:
                assert results == image[0, 0, 0]
                out.append(results)
        out += [results for results, _ in buffer.flush()]
        assert out == list(range(count))

    def test_nothing_ready_returns_none_not_a_black_frame(self):
        """Regression: warm-up returned a black frame, written to videos as real output."""
        buffer = pf.Buffer(frames=5)
        assert buffer.update("r0", numbered(9)) == (None, None)
        assert buffer.update("r1", numbered(9)) == (None, None)
        results, image = buffer.update("r2", numbered(9))
        assert results == "r0" and image is not None

    def test_delay(self):
        buffer = pf.Buffer(frames=5)
        returned = [buffer.update(i, numbered(i))[0] for i in range(8)]
        assert returned == [None, None, 0, 1, 2, 3, 4, 5]

    def test_context_follows_the_returned_frame(self):
        buffer = pf.Buffer(frames=5)
        assert buffer.get_temporal_context() is None
        buffer.update(0, numbered(0)); buffer.update(1, numbered(1)); buffer.update(2, numbered(2))
        context = buffer.get_temporal_context()
        assert (context["past_results"], context["current_results"], context["future_results"]) == ([], 0, [1, 2])
        for i in range(3, 6):
            buffer.update(i, numbered(i))
        context = buffer.get_temporal_context()
        assert (context["past_results"], context["current_results"], context["future_results"]) == ([1, 2], 3, [4, 5])
        seen = []
        for results, _ in buffer.flush():
            context = buffer.get_temporal_context()
            seen.append((context["current_results"], context["future_results"]))
        assert seen == [(4, [5]), (5, [])]

    def test_update_after_flush_starts_a_new_video(self):
        buffer = pf.Buffer(frames=3)
        for i in range(4):
            buffer.update(i, numbered(i))
        list(buffer.flush())
        assert buffer.update(100, numbered(100)) == (None, None)
        assert buffer.update(101, numbered(101))[0] == 100

    def test_flush_twice_yields_nothing_more(self):
        buffer = pf.Buffer(frames=3)
        for i in range(4):
            buffer.update(i, numbered(i))
        assert [r for r, _ in buffer.flush()] == [3]
        assert list(buffer.flush()) == []

    def test_is_full(self):
        buffer = pf.Buffer(frames=3)
        buffer.update(0, numbered(0)); buffer.update(1, numbered(1))
        assert not buffer.is_full
        buffer.update(2, numbered(2))
        assert buffer.is_full


class TestSmoothAtTheEdges:

    def test_edge_frames_come_back_raw_and_interior_frames_smoothed(self):
        buffer = pf.Buffer(frames=5)
        jitter = [0, 6, -6, 5, -5, 6, -6, 5, -5, 0]
        frames = [make_detections([[100 + 10 * i + j, 50, 140 + 10 * i + j, 90]], tracker_ids=[1])
                  for i, j in enumerate(jitter)]
        pairs = []
        for detections in frames:
            raw, image = buffer.update(detections, numbered(0))
            if image is not None:
                pairs.append((raw, pf.smooth(buffer)))
        for raw, _ in buffer.flush():
            pairs.append((raw, pf.smooth(buffer)))
        assert len(pairs) == len(frames)
        for index, (raw, smoothed) in enumerate(pairs):
            same = raw[0].bbox == smoothed[0].bbox
            assert same == (index < 2 or index >= len(frames) - 2), index


# ---------------------------------------------------------------------------
# SlicedInference
# ---------------------------------------------------------------------------

class TestSlicedInference:

    def test_masks_segments_keypoints_are_moved_to_the_image(self):
        """Regression: only boxes were shifted; masks stayed slice-sized."""
        image = frame(480, 640)

        def detect(tile):
            mask = np.zeros(tile.shape[:2], dtype=bool)
            mask[10:20, 10:30] = True
            detections = make_detections([[10, 10, 30, 20]])
            detections[0].masks = [mask]
            detections[0].segments = [[10, 10], [30, 10], [30, 20], [10, 20]]
            detections[0].keypoints = [pf.KeyPoint(x=15, y=15, id=0, name="p")]
            return detections

        slicer = pf.SlicedInference(slice_width=320, slice_height=240,
                                    overlap_ratio_w=0, overlap_ratio_h=0)
        result = slicer.predict(image, detect)
        assert len(result) == 4
        for detection in result:
            x1, y1, x2, y2 = detection.bbox
            mask = detection.masks[0]
            assert mask.shape == (480, 640)
            ys, xs = np.nonzero(mask)
            assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (x1, y1, x2, y2)
            assert detection.segments[0] == [x1, y1]
            assert (detection.keypoints[0].x, detection.keypoints[0].y) == (x1 + 5, y1 + 5)

    def test_non_detections_return_raises(self):
        """Regression: a raw model output became empty Detections without a word."""
        slicer = pf.SlicedInference(slice_width=320, slice_height=240)
        with pytest.raises(TypeError, match="Detections"):
            slicer.predict(frame(), lambda tile: [[0, 0, 10, 10]])

    def test_min_slice_area_ratio_warns(self):
        with pytest.warns(DeprecationWarning, match="min_slice_area_ratio"):
            pf.SlicedInference(min_slice_area_ratio=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pf.SlicedInference()

    def test_ios_threshold_is_used(self):
        """Regression: any value below 0.9 behaved like 0.9."""
        def detect(tile):
            # A box and a smaller one 64% inside it, same class, same slice.
            return make_detections([[0, 0, 100, 100], [50, 50, 130, 130]], scores=[0.9, 0.8])

        image = frame(200, 200)
        loose = pf.SlicedInference(slice_width=200, slice_height=200, ios_threshold=0.9)
        strict = pf.SlicedInference(slice_width=200, slice_height=200, ios_threshold=0.3)
        assert len(loose.predict(image, detect)) == 2
        assert len(strict.predict(image, detect)) == 1

    def test_full_overlap_does_not_hang(self):
        """Regression: overlap 1.0 made the step 0, and generate_slices never advanced."""
        slicer = pf.SlicedInference(slice_width=100, slice_height=100,
                                    overlap_ratio_w=1.0, overlap_ratio_h=1.0)
        assert slicer.step_width >= 1 and slicer.step_height >= 1
        assert len(slicer.generate_slices(200, 200)) > 0


# ---------------------------------------------------------------------------
# from_pil
# ---------------------------------------------------------------------------

class TestFromPil:

    @pytest.mark.parametrize("mode", ["RGB", "L", "P", "I;16", "CMYK"])
    def test_every_mode_becomes_rgb(self, mode):
        """Regression: RGBA gave 4 channels and L gave 2 dimensions."""
        array = pf.from_pil(Image.new(mode, (8, 6)))
        assert array.shape == (6, 8, 3) and array.dtype == np.uint8

    def test_alpha_is_dropped_with_a_warning(self):
        with pytest.warns(UserWarning, match="alpha"):
            array = pf.from_pil(Image.new("RGBA", (8, 6), (10, 20, 30, 128)))
        assert array.shape == (6, 8, 3) and tuple(array[0, 0]) == (10, 20, 30)

    def test_round_trip(self):
        image = frame(60, 80)
        assert (pf.from_pil(pf.to_pil(image)) == image).all()
