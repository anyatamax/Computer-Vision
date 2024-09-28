import pathlib

import numpy as np
import pytest

import align
import common


@pytest.mark.parametrize("kind", ["simple", "smile1", "smile2", "smile3"])
@pytest.mark.parametrize("crop", range(10, 24))
def test(kind, crop):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, points = common.read_test_data(test_dir / kind)
    shift = np.array([111, 0])
    raw_img = raw_img.reshape(3, 111, 222)
    r_im, g_im, b_im = [c[crop:-crop, crop:-crop] for c in raw_img][::-1]
    r_pt, g_pt, b_pt = [p - crop - shift * (2 - i) for i, p in enumerate(points)]

    # Alignment with self should be zero
    for ai in [r_im, g_im, b_im]:
        pr_a_to_a = align.find_relative_shift_pyramid(ai, ai)
        common.assert_ndarray_equal(actual=pr_a_to_a, correct=np.array([0, 0]))

    # Alignment with green channel should match gt point difference
    ai = g_im
    gt_a = g_pt
    for bi, gt_b in [
        (r_im, r_pt),
        (b_im, b_pt),
    ]:
        gt_a_to_b = gt_b - gt_a
        pr_a_to_b = align.find_relative_shift_pyramid(ai, bi)
        common.assert_ndarray_equal(actual=pr_a_to_b, correct=gt_a_to_b)

        # Sanity check: fliping the arguments should return the opposite shift
        gt_b_to_a = gt_a - gt_b
        pr_b_to_a = align.find_relative_shift_pyramid(bi, ai)
        common.assert_ndarray_equal(actual=pr_b_to_a, correct=gt_b_to_a)
