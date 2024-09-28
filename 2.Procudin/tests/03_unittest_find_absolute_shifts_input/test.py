import pathlib
from unittest import mock

import numpy as np
import pytest

import align
import common


def _direct_call_not_allowed(*_, **__):
    __tracebackhide__ = True
    raise AssertionError(
        "You shouldn't call the find_relative_shift_* functions directly, "
        "use the provided find_relative_shift_fn argument instead"
    )


@mock.patch(
    "align.find_relative_shift_pyramid",
    new=_direct_call_not_allowed,
)
@mock.patch(
    "align.find_relative_shift_fourier",
    new=_direct_call_not_allowed,
)
def test_mocked():
    gt_r_to_g_abs = np.array([-274, -18])
    gt_b_to_g_abs = np.array([164, 16])
    coords = (
        np.array([622, 674]),
        np.array([335, 664]),
        np.array([161, 659]),
    )
    r_to_g_rel = np.array([13, -8])
    b_to_g_rel = np.array([-10, 11])

    crops = ("r", "g", "b")
    call_count = [0]

    def _mock_shift_fn(a, b):
        rel = {
            ("r", "g"): r_to_g_rel,
            ("g", "r"): -r_to_g_rel,
            ("b", "g"): b_to_g_rel,
            ("g", "b"): -b_to_g_rel,
        }.get((a, b), None)
        if rel is None:
            raise AssertionError(
                "You should only call the find_relative_shift_fn function with pairs of "
                "provided crops as arguments (one green channel + one other channel)"
            )
        call_count[0] += 1
        return rel

    (
        pr_r_to_g_abs,
        pr_b_to_g_abs,
    ) = align.find_absolute_shifts(
        crops,
        coords,
        _mock_shift_fn,
    )

    assert (
        call_count[0] == 2
    ), "You should call the provided find_relative_shift_fn function exactly twice"

    common.assert_ndarray_equal(actual=pr_r_to_g_abs, correct=gt_r_to_g_abs)
    common.assert_ndarray_equal(actual=pr_b_to_g_abs, correct=gt_b_to_g_abs)


@pytest.mark.parametrize("kind", ["simple", "smile1", "smile2", "smile3"])
@pytest.mark.parametrize("crop", range(10, 24))
def test(kind, crop):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_dir / kind)
    shift = np.array([111, 0])
    raw_img = raw_img.reshape(3, 111, 222)
    crops = [c[crop:-crop, crop:-crop] for c in raw_img][::-1]
    coords = [crop + shift * (2 - i) for i in range(3)]
    gt_r_to_g = g_point - r_point
    gt_b_to_g = g_point - b_point

    shift_fn = align.find_relative_shift_pyramid
    pr_r_to_g, pr_b_to_g = align.find_absolute_shifts(crops, coords, shift_fn)

    # Check that the absolute shifts were calculated correctly
    common.assert_ndarray_equal(actual=pr_r_to_g, correct=gt_r_to_g)
    common.assert_ndarray_equal(actual=pr_b_to_g, correct=gt_b_to_g)
