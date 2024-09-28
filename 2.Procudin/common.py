import json
import os

import numpy as np
from PIL import Image


def read_test_data(test_path):
    test_path = str(test_path)
    raw_img = np.array(Image.open(test_path + ".png")) / 255.0

    if os.path.exists(test_path + ".npy"):
        gt_points = np.load(test_path + ".npy")
    else:
        gt_points = None

    return raw_img, gt_points


def save_image(out_path, img_fp32):
    if img_fp32 is None:
        print(f"No image data provided for {out_path!r}, skipping...")
        return

    img_u8 = (255 * img_fp32).clip(0, 255).round().astype(np.uint8)
    Image.fromarray(img_u8).save(out_path)


def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    error = f"Value should be an instance of np.ndarray, but it is {type(value)}."
    assert isinstance(value, (np.ndarray, np.generic)), error


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    error = (
        "The dtypes of actual value and correct value are not the same "
        "and can't be safely converted.\n"
        f"actual.dtype={actual_dtype}, correct.dtype={correct_dtype}"
    )
    assert np.can_cast(actual_dtype, correct_dtype, casting="same_kind"), error
    assert np.can_cast(correct_dtype, actual_dtype, casting="same_kind"), error


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    error = (
        "The shapes of actual value and correct value are not the same.\n"
        f"actual.shape={actual_shape}, correct.shape={correct_shape}"
    )
    assert len(actual_shape) == len(correct_shape), error
    assert actual_shape == correct_shape, error


def assert_ndarray_equal(*, actual, correct, rtol=0, atol=1e-6, err_msg=""):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)
    np.testing.assert_allclose(
        actual,
        correct,
        atol=atol,
        rtol=rtol,
        verbose=True,
        err_msg=err_msg,
    )
