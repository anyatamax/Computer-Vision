import time

from bayer import improved_interpolation
from common import assert_ndarray_equal, assert_time_limit, get_test_images


def test():
    elapsed = 0
    r = slice(2, -2), slice(2, -2)
    for img_filename, raw_img, gt_img in get_test_images(__file__):

        start = time.time()
        img = improved_interpolation(raw_img)
        finish = time.time()
        elapsed += finish - start

        assert_ndarray_equal(
            actual=img[r],
            correct=gt_img[r],
            atol=1,
            err_msg=f"Testing on img {img_filename} failed",
        )

    assert_time_limit(actual=elapsed, limit=30.0)
