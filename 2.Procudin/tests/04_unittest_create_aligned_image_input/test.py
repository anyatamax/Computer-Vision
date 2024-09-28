import pathlib

import numpy as np
import pytest

import align
import common


@pytest.mark.parametrize("kind", ["smile1", "smile2", "smile3"])
def test_smile(kind):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_dir / kind)
    r_to_g = g_point - r_point
    b_to_g = g_point - b_point
    shift = np.array([111, 0])
    channels = tuple(raw_img.reshape(3, 111, 222)[::-1])
    coords = [shift * (2 - i) for i in range(3)]
    span_hi = np.max([[0, 0], r_to_g + shift, b_to_g - shift], axis=0)
    span_lo = np.min([[0, 0], r_to_g + shift, b_to_g - shift], axis=0)
    span = span_hi - span_lo

    aligned_image = align.create_aligned_image(channels, coords, r_to_g, b_to_g)

    # Check that the produced image has the correct size
    common.assert_value_is_ndarray(aligned_image)
    common.assert_dtypes_compatible(aligned_image.dtype, np.float64)
    common.assert_shapes_match(aligned_image.shape, (111 - span[0], 222 - span[1], 3))

    r_smile, g_smile, b_smile = (aligned_image == 1).transpose(2, 0, 1)

    # Check that the smiley face pixels are correctly aligned
    # (the smile should be completely white on the visualization)
    common.assert_ndarray_equal(actual=r_smile, correct=r_smile)
    common.assert_ndarray_equal(actual=b_smile, correct=b_smile)
