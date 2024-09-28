import pathlib

import numpy as np
import pytest

import align
import common

OUTSIDE = 0
MAYBE_INSIDE = 112
INSIDE = 128
SMILE = 255

INSIDE_PIXELS = 63 * 174


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("pad", [0, 1, 2])
@pytest.mark.parametrize("kind", ["simple", "smile1", "smile2", "smile3"])
def test(kind, crop, pad):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, _ = common.read_test_data(test_dir / kind)
    raw_img = np.pad(raw_img, ((0, pad), (0, 0)))

    unaligned_rgb, coords = align.extract_channel_plates(raw_img, crop=crop)

    # Sanity-check the returned types
    assert isinstance(unaligned_rgb, (list, tuple))
    assert len(unaligned_rgb) == 3
    for u in unaligned_rgb:
        common.assert_value_is_ndarray(u)
        common.assert_dtypes_compatible(u.dtype, np.float64)

    assert u.ndim == 2
    uh, uw = u.shape
    for u in unaligned_rgb:
        # Extracted channels must all have the same shape
        common.assert_shapes_match(u.shape, (uh, uw))

    assert isinstance(coords, (list, tuple))
    assert len(coords) == 3
    for c in coords:
        common.assert_value_is_ndarray(c)
        common.assert_dtypes_compatible(c.dtype, np.int64)
        common.assert_shapes_match(c.shape, (2,))

    # Sanity-check the returned coordinates
    h, w = raw_img.shape
    for y, x in coords:
        assert 0 <= y < h, "Crop coords are out of bounds"
        assert 0 <= x < w, "Crop coords are out of bounds"

    channel_order = np.argsort([c[0] for c in coords])
    assert (channel_order == [2, 1, 0]).all(), (
        "Returned channels in the wrong order, expected RGB "
        "(note that Prokudin-Gorsky images are in BGR order)"
    )

    # Check that the returned coordinates and crop match
    for u, (y, x) in zip(unaligned_rgb, coords):
        assert (u == raw_img[y : y + uh, x : x + uw]).all(), (
            "Returned coordinates and crop correspond to "
            "different parts of the source image"
        )

    if crop:
        # Check that the crop wasn't too small or too large
        for u in unaligned_rgb:
            o = [OUTSIDE]
            i = [INSIDE, SMILE]
            a = INSIDE_PIXELS
            assert (
                np.isin(255 * u, o).sum() == 0
            ), "Crop shouldn't contain any 'outside' pixels"
            assert (
                np.isin(255 * u, i).sum() == a
            ), "Crop should contain all 'inside' pixels"
    else:
        # Check that the non-cropped channel extraction didn't apply any crop
        for u in unaligned_rgb:
            common.assert_shapes_match(u.shape, (111, 222))
