import pathlib

import pytest

import common
import pipeline


@pytest.mark.parametrize("kind", ["smile_large", "smile_far_large"])
def test_smile(kind):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_dir / kind)

    r_to_g, b_to_g, aligned_image = pipeline.align_image(raw_img, method="fourier")

    r_pred = g_point - r_to_g
    b_pred = g_point - b_to_g

    # Check that the coordinates are correctly aligned
    common.assert_ndarray_equal(actual=r_pred, correct=r_point)
    common.assert_ndarray_equal(actual=b_pred, correct=b_point)

    r_smile, g_smile, b_smile = (aligned_image == 1).transpose(2, 0, 1)

    # Check that the smiley face pixels are correctly aligned
    # (the smile should be completely white on the visualization)
    common.assert_ndarray_equal(actual=r_smile, correct=r_smile)
    common.assert_ndarray_equal(actual=b_smile, correct=b_smile)


@pytest.mark.parametrize("num", range(10))
def test(num):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_dir / f"{num:02}")

    r_to_g, b_to_g, _ = pipeline.align_image(raw_img, method="fourier")

    r_pred = g_point - r_to_g
    b_pred = g_point - b_to_g

    r_error = abs(r_pred - r_point).sum()
    b_error = abs(b_pred - b_point).sum()

    assert r_error + b_error <= 10
