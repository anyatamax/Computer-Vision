import pathlib

import pytest

import common
import pipeline


@pytest.mark.parametrize("num", range(10))
def test(num):
    test_dir = pathlib.Path(__file__).resolve().parent
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_dir / f"{num:02}")

    r_to_g, b_to_g, _ = pipeline.align_image(raw_img, method="pyramid")

    r_pred = g_point - r_to_g
    b_pred = g_point - b_to_g

    r_error = abs(r_pred - r_point).sum()
    b_error = abs(b_pred - b_point).sum()

    assert r_error + b_error <= 10
